"""One standard report shape for every run: business understanding -> HITL
clarification history -> EDA -> feature transforms -> feature engineering
-> model -> explainability -> diagnostics -> fairness -> classification
metrics -> confusion matrix -> ROC curve -> PR curve -> known limitations ->
reflection. Same sections every time so reports are comparable across runs
without re-reading each one to find where a number lives.

Everything here is deterministic from what's already on disk for this
run_id (source CSV via meta["source_path"], train/test.csv, meta.json,
pipeline.pkl) — no dependency on conversation history, so calling
generate_report(run_id) alone, in a fresh session, reproduces the same
report. The one thing intentionally left out is rendered plot images:
those live in the separate visualization-agent (shared RUNS_DIR file
contract, see core.py's comment) and are spliced in by the calling agent
per the `reporting` skill, rather than duplicating matplotlib/shap-plotting
code into this agent."""

import json

import joblib
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .core import _load_meta, _load_split, _run_dir, positive_index, positive_label, mcp
from .data_cleaning import _reasoned_drops
from .eda import detect_outliers, eda
from .feat_engineering import _reasoned_features
from .gates import compute_readiness, measured_metric

# ponytail: fixed cap, not adaptive to test-fold size — raise if a report
# needs finer curve resolution than 20 points gives.
MAX_CURVE_POINTS = 20


def _thin(values, n=MAX_CURVE_POINTS):
    values = list(values)
    if len(values) <= n:
        return [round(float(v), 4) for v in values]
    step = max(1, len(values) // n)
    return [round(float(v), 4) for v in values[::step]]


def _build_executive_summary(
    meta: dict, train, metrics: dict, readiness: dict
) -> list[str]:
    """Rule-based, not a hallucinated business judgment: every claim here
    traces to a field already computed elsewhere in the report. Manager
    -facing — headline number, top driver with direction, the risk
    caveats that keep a clean-looking metric from being over-trusted, and
    a ship/hold call anchored in the overfitting/stability gates the rest
    of this agent already computes."""
    lines = ["## Executive Summary"]
    model_name = meta.get("model") or "no model trained yet"
    tuned = " (tuned)" if meta.get("best_params") else " (baseline)"
    accuracy = (metrics.get("classification_report") or {}).get("accuracy")
    auc_block = metrics.get("auc") or {}
    roc = auc_block.get("roc_auc") or auc_block.get("roc_auc_macro")
    pr = auc_block.get("pr_auc") or auc_block.get("pr_auc_macro")

    # Lead with the metric that means something for THIS target. On wangiri
    # (2.5% positive) the old headline read "98.1% accuracy, ROC-AUC 0.9845" —
    # both true, both flattering, and both nearly unrelated to whether the
    # model finds fraud, which PR-AUC put at 0.62. Always predicting the
    # majority class would have scored 97.5% accuracy. A manager reading only
    # the first line should not come away with the opposite of the finding.
    counts = train[meta["target"]].value_counts()
    imbalance_ratio = float(counts.min() / counts.max())
    majority_rate = float(counts.max() / counts.sum())
    imbalanced = imbalance_ratio < 0.2

    headline = f"**{model_name}{tuned}**"
    if imbalanced and pr is not None:
        headline += f" — PR-AUC {pr} on held-out test data (the metric that matters at this class balance)"
    elif accuracy is not None:
        headline += f" — {round(accuracy * 100, 1)}% accuracy on held-out test data"
    if roc is not None:
        headline += f", ROC-AUC {roc}"
    lines.append(headline + ".")
    if imbalanced and accuracy is not None:
        beats_majority = accuracy > majority_rate
        lines.append(
            f"- **Read accuracy with care:** {round(accuracy * 100, 1)}% accuracy "
            + (
                f"sounds high, but always predicting the majority class alone scores "
                f"{round(majority_rate * 100, 1)}% on this target."
                if beats_majority
                # Below the majority rate is not a failure here — a model tuned to
                # catch a 2.5% minority necessarily gives up some raw accuracy, and
                # saying "sounds high, but" about a number that is actually LOWER
                # than the do-nothing baseline would misread its own evidence.
                else f"is BELOW the {round(majority_rate * 100, 1)}% you get by always predicting the majority "
                "class — expected for a model tuned to catch the minority, and the reason accuracy is the "
                "wrong yardstick here, not a sign the model is worse."
            )
            + (
                f" ROC-AUC ({roc}) is similarly flattered by the {round(imbalance_ratio * 100, 1)}% minority "
                f"rate; PR-AUC ({pr}) is the honest headline."
                if roc is not None and pr is not None
                else ""
            )
        )

    explain = meta.get("explain")
    if explain and explain.get("feature_importance"):
        top_feature, top_value = explain["feature_importance"][0]
        direction = (explain.get("direction") or {}).get(top_feature)
        if direction:
            pos_class = explain.get("positive_class")
            lines.append(
                f"- **Top driver:** `{top_feature}` — {direction} the predicted likelihood of "
                f"`{meta['target']}={pos_class}` the most (mean |SHAP| {top_value})."
            )
        else:
            lines.append(
                f'- **Top driver:** `{top_feature}` (importance {top_value}, magnitude only — direction needs `method="shap"`).'
            )
    else:
        lines.append(
            "- **Top driver:** not available — `explain_model` hasn't been run for this run yet."
        )

    risks = []
    if metrics.get("overfitting_warning"):
        risks.append(
            f"CV-to-test gap ({metrics.get('cv_to_test_gap')}) exceeds the overfitting threshold"
        )
    stability = meta.get("stability")
    if stability and stability.get("high_variance_warning"):
        risks.append(
            f"CV score is unstable across random splits (std={stability.get('std')})"
        )
    if imbalanced:
        risks.append(
            f"target is imbalanced (minority class {round(imbalance_ratio * 100, 1)}% of training rows)"
        )
    # domain-notes.md explicitly says to flag a large ROC/PR divergence rather
    # than quietly reporting the higher one.
    if roc is not None and pr is not None and roc - pr > 0.15:
        risks.append(
            f"ROC-AUC ({roc}) and PR-AUC ({pr}) diverge by {round(roc - pr, 3)} — the model ranks well overall "
            "but is much weaker at the minority class specifically; trust PR-AUC here"
        )
    if train.shape[0] < 1000:
        risks.append(
            f"small training set ({train.shape[0]} rows) — metrics have wide uncertainty"
        )
    lines.append(
        f"- **Risk caveats:** {'; '.join(risks) if risks else 'none flagged'}."
    )

    # The recommendation comes from the deterministic gate, NOT from whichever
    # warnings happen to be present. The previous version said "✅ Proceed —
    # metrics are stable" whenever meta["stability"] was absent, because None
    # is falsy: a run that never checked stability was indistinguishable from
    # one that passed. Missing evidence now reads as missing, never as clean.
    recommendation = {
        "blocked": f"⚠ **Hold** — {len(readiness['failed_gates'])} gate(s) failed: "
        f"{', '.join(readiness['failed_gates'])}. See Run Readiness below.",
        "incomplete": f"⚠ **Incomplete** — required evidence was never produced: "
        f"{', '.join(readiness['not_run_gates'])}. Absence of evidence is not a pass; "
        "run these before treating this report as final.",
        "ready": "✅ **Proceed to human review** — every mechanical gate passed. This certifies execution "
        "correctness only; the modelling choices and business framing still need a human.",
    }[readiness["overall_status"]]
    lines.append(f"- **Recommendation:** {recommendation}")
    return lines


def _build_readiness_section(readiness: dict) -> list[str]:
    """Put the gate result where a reviewer sees it before any metric — a
    number from an incomplete run should never be read as a finished one."""
    symbol = {"pass": "✅", "fail": "❌", "not_run": "⬜", "not_applicable": "➖"}
    lines = [
        "## Run Readiness (deterministic gates)",
        "",
        f"**Status: `{readiness['overall_status']}`** — {readiness['interpretation']}",
        "",
        f"**Mechanical completeness: {readiness['gates_passed']}/{readiness['gates_total']} gates "
        f"({readiness['mechanical_completeness']:.0%})** — the fraction of deterministic checks satisfied. "
        "This measures execution correctness only. It is not a confidence score for the model's accuracy, "
        "and it makes no claim that the modelling approach or business framing is correct.",
        "",
        "| Gate | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for name, check in readiness["checks"].items():
        evidence = str(check["evidence"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{name}` | {symbol.get(check['status'], '?')} {check['status']} | {evidence} |"
        )
    return lines


def _build_success_criteria_section(meta: dict, readiness: dict) -> list[str]:
    """Reads the bar recorded up front and compares it to what was measured.
    Never derives a bar: if none was recorded, that is what it says. A run
    once reported 'ROC-AUC 0.822 vs the 0.85 target' for a target nobody
    set — the threshold has to come from the record or not exist."""
    target = ((meta.get("business_understanding") or {}).get("success_target")) or {}
    check = readiness["checks"]["success_criteria"]
    lines = ["## Success Criteria"]
    if not target:
        lines += [
            "_No measurable success threshold was recorded for this run._",
            "",
            "The model is therefore reported against no pass/fail bar — the metrics below stand on their own. "
            "No threshold has been inferred, and none should be read into this report.",
        ]
        return lines
    measured = measured_metric(meta, target["metric"])
    lines += [
        f"- **Recorded bar:** `{target['metric']} {target['direction']} {target['threshold']}` "
        f"(set via `record_business_context`, before modelling)",
        f"- **Rationale:** {target.get('rationale') or 'none recorded'}",
        f"- **Measured:** {round(float(measured), 4) if measured is not None else 'not computable for this run'}",
        f"- **Verdict:** {'✅ met' if check['status'] == 'pass' else '❌ not met' if check['status'] == 'fail' else '⬜ ' + check['status']}",
    ]
    return lines


def _build_execution_log_section(meta: dict) -> list[str]:
    """What actually ran, from the ledger every tool writes automatically.

    Exists because "those tools were not available in this session" was once
    offered as the reason a run skipped fairness, stability, error analysis and
    reflection — in a process where those tools were registered and callable.
    A claim about what the agent could or did do is checkable here."""
    log = meta.get("execution_log") or []
    lines = ["## Execution Log"]
    if not log:
        lines.append("_no tool calls recorded for this run._")
        return lines
    counts, failures = {}, []
    for entry in log:
        counts[entry["tool"]] = counts.get(entry["tool"], 0) + 1
        if not entry.get("ok"):
            failures.append(f"`{entry['tool']}` — {entry.get('error')}")
    lines.append(
        f"- **{len(log)} tool calls** across {len(counts)} distinct tools: "
        + ", ".join(
            f"`{t}`×{n}" if n > 1 else f"`{t}`" for t, n in sorted(counts.items())
        )
    )
    if failures:
        lines.append(f"- **{len(failures)} call(s) returned an error:**")
        lines += [f"  - {f}" for f in dict.fromkeys(failures)]
    else:
        lines.append("- No tool call returned an error.")
    return lines


def _build_business_understanding_section(meta: dict) -> list[str]:
    bu = meta.get("business_understanding")
    lines = ["## Business Understanding & Assumptions"]
    if not bu:
        lines.append(
            "_not recorded — `record_business_context` was not called for this run._"
        )
        return lines
    lines.append(f"- **Domain:** {bu.get('domain') or 'not declared'}")
    lines.append(
        f"- **Business objective:** {bu.get('business_objective') or 'not stated'}"
    )
    lines.append(
        f"- **Target definition:** {bu.get('target_definition') or 'not stated'}"
    )
    lines.append(
        f"- **Label provenance:** {bu.get('label_provenance') or 'not recorded — how the labels were produced, and whether the negatives are confirmed, is unknown for this run'}"
    )
    lines.append(
        f"- **Success criteria:** {bu.get('success_criteria') or 'not stated'}"
    )
    assumptions = bu.get("assumptions") or []
    if assumptions:
        lines.append("- **Documented assumptions (proceeded without asking):**")
        for a in assumptions:
            lines.append(f"  - {a}")
    else:
        lines.append("- **Documented assumptions:** none recorded")
    return lines


def _build_hitl_history_section(meta: dict) -> list[str]:
    clarifications = (
        (meta.get("business_understanding") or {}).get("clarifications")
    ) or []
    lines = ["## HITL Clarification History"]
    if not clarifications:
        lines.append(
            "_no clarifying questions were raised for this run — either the business framing was "
            "unambiguous, or the ambiguity was resolved as a documented assumption instead (see "
            "Business Understanding above)._"
        )
        return lines
    for i, c in enumerate(clarifications, 1):
        lines.append(f"{i}. **Q:** {c.get('question', '?')}")
        lines.append(f"   **A:** {c.get('answer', '(no answer recorded)')}")
        if c.get("impact"):
            lines.append(f"   **Impact on the plan:** {c['impact']}")
    return lines


def _build_operating_point_section(meta: dict) -> list[str]:
    """The threshold that ships. Absent this section a reader assumes 0.5,
    and on an imbalanced target 0.5 is a different model."""
    lines = ["## Operating point"]
    op = meta.get("operating_point")
    if not op:
        lines += [
            "_No threshold was tuned — this model decides at sklearn's default 0.5._",
            "",
            "On an imbalanced target that is rarely the right cutoff, and it is the threshold the exported "
            "pickle will use. Call `tune_threshold` if the operating point matters.",
        ]
        return lines
    default = op.get("default_0.5") or {}
    ci = op.get("ci") or {}
    selected = op.get("selected_on") or {}

    def at_op(key):
        return (
            f"{op.get(key)} [{ci[key][0]}, {ci[key][1]}]"
            if key in ci
            else f"{op.get(key)}"
        )

    lines += [
        f"- **Threshold:** `{op.get('threshold')}` — chosen by `{op.get('chosen_by')}`",
        f"- **Chosen on:** {selected.get('validation') or '_the test fold itself (run predates validation-based tuning) — the numbers below are optimistic_'}",
        f"- **Measured on:** {op.get('measured_on') or 'the held-out test fold'}",
        "- **This threshold travels with the exported model; `predict` applies it automatically.**",
        "",
        "| | At the operating point (95% CI) | At the 0.5 default |",
        "|---|---|---|",
        f"| Precision | {at_op('precision')} | {default.get('precision')} |",
        f"| Recall | {at_op('recall')} | {default.get('recall')} |",
        f"| F1 | {at_op('f1')} | {default.get('f1')} |",
        f"| Share of rows flagged | {op.get('alert_rate')} | {default.get('alert_rate')} |",
        "",
        "Share of rows flagged is the review workload this threshold creates — the number a queue is sized "
        "against, which precision and recall between them never state outright.",
    ]
    return lines


def _build_label_rule_section(meta: dict) -> list[str]:
    lines = ["## Label-rule screen"]
    check = meta.get("label_rule_check")
    if not check:
        lines.append(
            "_Not run._ Nothing has tested whether a single-column rule reproduces these labels, so the "
            "possibility that the target was rule-generated — and that the model has rediscovered the rule — "
            "remains open."
        )
        return lines
    best = check.get("best_single_column_rule") or {}
    flag = "⚠ SUSPICIOUS" if check.get("label_rule_suspicion") else "✅ clear"
    lines += [
        f"- **Verdict:** {flag}",
        f"- **Best single-column rule:** depth-{best.get('tree_depth')} tree on `{best.get('column')}` — "
        f"PR-AUC {best.get('pr_auc')} vs the model's {check.get('model_pr_auc')} "
        f"(**{check.get('rule_to_model_ratio')}** of it)",
        f"- {check.get('interpretation')}",
    ]
    if check.get("top_5"):
        lines += ["", "| Column | Depth | Rule PR-AUC |", "|---|---|---|"]
        lines += [
            f"| `{r['column']}` | {r['depth']} | {r['pr_auc']} |"
            for r in check["top_5"]
        ]
    return lines


def _build_backtest_section(meta: dict) -> list[str]:
    lines = ["## Forward backtest"]
    bt = meta.get("backtest")
    if not bt:
        lines.append(
            "_Not run._ The held-out test fold shows the model works on data it did not train on; it cannot "
            "show the model still works on a later period. In an adversarial domain those differ."
        )
        return lines
    change = bt.get("pr_auc_change")
    lines += [
        f"- **Source:** `{bt.get('source')}` — {bt.get('n_rows')} rows, positive rate {bt.get('positive_rate')}",
        f"- **Threshold applied:** {bt.get('threshold_applied')}",
        f"- **PR-AUC:** {bt.get('pr_auc')} vs {bt.get('held_out_pr_auc')} held out "
        + (f"(**{change:+}**)" if change is not None else ""),
        f"- **Precision / recall / alert rate:** {bt.get('precision')} / {bt.get('recall')} / {bt.get('alert_rate')}",
        f"- {bt.get('note')}",
    ]
    return lines


def _build_diagnostics_section(meta: dict) -> list[str]:
    lines = ["## Diagnostics", "", "### Data leakage check"]
    leakage = meta.get("leakage")
    if not leakage:
        lines.append("_not run — call `detect_data_leakage` first_")
    elif leakage.get("clear"):
        lines.append("- Clear — no leakage signal found.")
    else:
        lines.append("- **Flagged:**")
        if leakage.get("near_perfect_correlation_with_target"):
            lines.append(
                f"  - near-perfect correlation with target: {leakage['near_perfect_correlation_with_target']}"
            )
        if leakage.get("id_like_columns"):
            lines.append(f"  - ID-like columns: {leakage['id_like_columns']}")
        if leakage.get("high_cardinality_categorical_columns"):
            lines.append(
                f"  - high-cardinality categorical (likely identifier): {leakage['high_cardinality_categorical_columns']}"
            )
        if leakage.get("name_flagged_columns"):
            lines.append(f"  - name-pattern flagged: {leakage['name_flagged_columns']}")

    lines += ["", "### Baseline comparison"]
    baseline = meta.get("baseline_comparison")
    if not baseline:
        lines.append("_not run — call `train_baseline` first_")
    else:
        baseline_acc = (baseline.get("classification_report") or {}).get("accuracy")
        model_metrics = meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
        model_acc = (model_metrics.get("classification_report") or {}).get("accuracy")
        lines.append(
            f"- Dummy baseline accuracy: {round(baseline_acc, 4) if baseline_acc is not None else 'n/a'}"
        )
        lines.append(
            f"- Current model accuracy: {round(model_acc, 4) if model_acc is not None else 'n/a'}"
        )
        if baseline_acc is not None and model_acc is not None:
            lines.append(
                "- ✅ Beats the dummy baseline."
                if model_acc > baseline_acc
                else "- ⚠ Does NOT clearly beat the dummy baseline — a stop-and-review signal."
            )

    lines += ["", "### Models compared"]
    comparison = meta.get("model_comparison")
    if not comparison:
        lines.append("_not run — call `compare_models` to see the ranked alternatives_")
    else:
        for r in comparison["ranked"]:
            marker = (
                " ← recommended by CV"
                if r["model"] == comparison["recommended"]
                else ""
            )
            # .get fallbacks: runs created before model selection became
            # metric-adaptive stored these under cv_f1_weighted_* and are still
            # on disk — an old run must stay renderable, not crash the report.
            metric_name = r.get("cv_metric", "f1_weighted")
            mean = r.get("cv_score_mean", r.get("cv_f1_weighted_mean"))
            std = r.get("cv_score_std", r.get("cv_f1_weighted_std"))
            lines.append(f"- `{r['model']}`: CV {metric_name} {mean} ± {std}{marker}")
        chosen = meta.get("model")
        if chosen and chosen != comparison.get("recommended"):
            lines.append(
                f"- **Note:** the model actually trained (`{chosen}`) differs from CV's top-ranked "
                f"recommendation (`{comparison['recommended']}`) — verify this was a deliberate choice."
            )

    lines += ["", "### Error analysis"]
    error = meta.get("error_analysis")
    if not error:
        lines.append("_not run — call `error_analysis` first_")
    elif error.get("n_errors") == 0:
        lines.append("- No misclassifications on the held-out test fold.")
    else:
        lines.append(
            f"- {error['n_errors']} errors on the test fold ({round(error['error_rate'] * 100, 1)}%). "
            "Most-confident wrong predictions are in this run's meta.json under `error_analysis.top_errors`."
        )

    lines += ["", "### Segment performance"]
    segments = meta.get("segment_analysis")
    if not segments:
        lines.append(
            "_not run — call `analyze_segments` if a business-meaningful segment column exists_"
        )
    else:
        for col, seg_result in segments.items():
            lines.append(f"- **By `{col}`:**")
            for value, stats in seg_result["segments"].items():
                if "note" in stats:
                    lines.append(f"  - `{value}` (n={stats['n']}): {stats['note']}")
                else:
                    lines.append(
                        f"  - `{value}` (n={stats['n']}): F1={round(stats.get('f1-score', 0), 4)}, "
                        f"precision={round(stats.get('precision', 0), 4)}, recall={round(stats.get('recall', 0), 4)}"
                    )
    return lines


def _build_fairness_section(meta: dict) -> list[str]:
    fairness = meta.get("fairness")
    lines = ["## Fairness & Bias Assessment"]
    if not fairness:
        lines.append(
            "_not run — call `assess_fairness(run_id, protected_attribute)` if the dataset has a "
            "protected/sensitive attribute; if none applies to this domain, say so explicitly in the "
            "reflection step rather than leaving this silently blank._"
        )
        return lines
    lines.append(
        f"- **Protected attribute:** `{fairness['protected_attribute']}` (positive outcome: `{fairness['positive_label']}`)"
    )
    if fairness.get("note"):
        lines.append(f"- {fairness['note']}")
    for value, g in fairness["groups"].items():
        if g.get("insufficient_sample"):
            lines.append(
                f"  - `{value}`: n={g['n']} — insufficient sample, excluded from parity metrics"
            )
        else:
            lines.append(
                f"  - `{value}`: n={g['n']}, selection rate={g['selection_rate']}, TPR={g['true_positive_rate']}"
            )
    metrics = fairness.get("metrics")
    if metrics:
        lines.append(
            f"- **Demographic parity difference:** {metrics['demographic_parity_difference']}"
        )
        lines.append(
            f"- **Disparate impact ratio:** {metrics['disparate_impact_ratio']} (below 0.8 is the standard four-fifths-rule concern threshold)"
        )
        lines.append(
            f"- **Equal opportunity difference (TPR gap):** {metrics['equal_opportunity_difference']}"
        )
        flags = fairness.get("flags") or {}
        lines.append(
            f"- ⚠ **Flagged:** {', '.join(k for k, v in flags.items() if v)}"
            if any(flags.values())
            else "- No fairness flags raised at the configured thresholds."
        )
    lines.append(f"- **Limitations:** {'; '.join(fairness.get('limitations', []))}")
    return lines


# Always-true scope limitations, independent of this run's evidence — see
# PRODUCTION_READINESS.md §6, kept here so a reviewer reading only report.md
# (not that separate doc) still sees them.
STATIC_LIMITATIONS = [
    "Model zoo is limited to logistic regression, random forest, and XGBoost — no LightGBM/CatBoost or stacking/blending.",
    "Hyperparameter search is a fast Optuna TPE pass (8 trials default), not an exhaustive search.",
    "No nested cross-validation — the same CV split used for model selection is the one reported (a mild optimism bias).",
    "High-cardinality legitimate categoricals go through plain one-hot encoding — no target/frequency encoding option.",
]


def _build_limitations_section(meta: dict, train) -> list[str]:
    dynamic = []
    metrics = meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
    if metrics.get("overfitting_warning"):
        dynamic.append(
            f"CV-to-test gap ({metrics.get('cv_to_test_gap')}) exceeds the overfitting threshold — generalization to new data is uncertain."
        )
    stability = meta.get("stability")
    if stability and stability.get("high_variance_warning"):
        dynamic.append(
            f"CV score is unstable across random splits (std={stability.get('std')}) — the reported metric may not replicate on a different split."
        )
    counts = train[meta["target"]].value_counts()
    imbalance_ratio = float(counts.min() / counts.max())
    if imbalance_ratio < 0.2:
        dynamic.append(
            f"Target is imbalanced (minority class {round(imbalance_ratio * 100, 1)}% of training rows) — minority-class metrics carry more uncertainty than the headline accuracy suggests."
        )
    if train.shape[0] < 1000:
        dynamic.append(
            f"Small training set ({train.shape[0]} rows) — all metrics have wide uncertainty."
        )
    fairness = meta.get("fairness")
    if fairness:
        dynamic.extend(fairness.get("limitations", []))
    else:
        dynamic.append(
            "No fairness/bias assessment was run for this model — unknown whether it treats any subgroup disparately."
        )
    reflection = meta.get("reflection")
    if reflection and reflection.get("critical_issues"):
        dynamic.extend(reflection["critical_issues"])
    if not meta.get("business_understanding"):
        dynamic.append(
            "Business objective/target definition/success criteria were not recorded for this run."
        )
    calibration = meta.get("calibration")
    operating_point = meta.get("operating_point")
    if calibration is None:
        dynamic.append(
            "Probability calibration was never measured (check_calibration not run) — predicted probabilities are "
            "a ranking score here, not verified probabilities."
        )
    elif calibration.get("miscalibration_warning"):
        dynamic.append(
            f"Model is miscalibrated (ECE {calibration.get('expected_calibration_error')}, Brier "
            f"{calibration.get('brier_score')}) — read its outputs as a ranking, not as probabilities."
        )
    if operating_point and operating_point.get("calibration_caveat"):
        dynamic.append(operating_point["calibration_caveat"])
    if meta.get("positive_label_inferred") and meta.get("positive_label") is not None:
        dynamic.append(
            f"The positive class (`{meta['positive_label']}`) was INFERRED, not declared — every positive-class "
            "number here (recall/precision on the positive class, PR-AUC, the operating point, fairness TPR, SHAP) "
            "describes that label. If the event being predicted is the other one, re-run prepare_dataset with "
            "positive_label set."
        )

    lines = [
        "## Known Limitations",
        "",
        "**Run-specific (from this run's actual evidence):**",
    ]
    lines += [f"- {d}" for d in dynamic] if dynamic else ["- none flagged"]
    lines += ["", "**Always-true scope limitations of this agent:**"]
    lines += [f"- {s}" for s in STATIC_LIMITATIONS]
    return lines


def _build_reflection_section(meta: dict) -> list[str]:
    reflection = meta.get("reflection")
    lines = ["## Reflection & Self-Critique"]
    if not reflection:
        lines.append(
            "_not recorded — call `record_reflection` before the report is considered final for human sign-off._"
        )
        return lines
    lines.append(f"- **Overall status:** {reflection['overall_status']}")
    lines.append("- **Checks:**")
    for name, check in reflection["checks"].items():
        lines.append(
            f"  - `{name}`: **{check.get('status')}** — {check.get('evidence') or '(no evidence given)'}"
        )
    if reflection.get("critical_issues"):
        lines.append("- **Critical issues:**")
        lines += [f"  - {issue}" for issue in reflection["critical_issues"]]
    if reflection.get("recommended_actions"):
        lines.append("- **Recommended next steps:**")
        lines += [f"  - {action}" for action in reflection["recommended_actions"]]
    return lines


def _build_charts_section(run_id: str, meta: dict) -> list[str]:
    """The visualization agent's plots of THIS fit, linked relative to
    report.md so the links hold in the run dir and in MLflow alike. Charts
    of an earlier fit (older pipeline_version) are not shown."""
    pv = meta.get("pipeline_version", 0)
    charts = sorted((_run_dir(run_id) / "charts").glob(f"pv{pv}_*.png"))
    lines = ["## Evaluation charts", ""]
    if not charts:
        return lines + [
            f"_none rendered for this fit (pipeline v{pv}) — the `evaluation_charts` gate stays "
            "not_run until the visualization agent draws them_"
        ]
    return lines + [f"![{c.stem.split('_', 1)[1]}](charts/{c.name})" for c in charts]


@mcp.tool()
def generate_report(run_id: str) -> str:
    """Full markdown report for a run: Executive Summary, Run Readiness,
    Business Understanding & Assumptions, Success Criteria, HITL
    Clarification History, EDA, feature transforms, feature engineering,
    model, explainability (SHAP/permutation), Diagnostics (leakage/baseline/
    models-compared/error-analysis/segments), Fairness & Bias Assessment,
    classification metrics, confusion matrix, ROC curve, PR curve, Known
    Limitations, Reflection & Self-Critique, Execution Log — always in that
    order. Deterministic from meta.json/source CSV/pipeline.pkl alone
    (recomputes EDA stats from the original file, replays .predict/
    .predict_proba on the held-out test fold — nothing here is refit).
    Self-contained: a human reviewer needs only this report, not the
    conversation that produced it, to validate the run.

    Read-only — it never blocks, but it cannot be used to make an unfinished
    run look finished either. The Run Readiness section states every
    deterministic gate's status up front, and the Executive Summary's
    ship/hold call is taken from that gate rather than from whichever
    warnings happen to be present, so a run that skipped its checks reads as
    `incomplete`, not as clean. The Execution Log lists every tool call that
    actually happened.

    Writes report.md next to the run's other artifacts and returns its
    content. Doesn't render plot images — see the `reporting` skill for
    splicing in the visualization agent's charts, which is a required step of
    an autonomous run, not an optional flourish."""
    meta = _load_meta(run_id)
    metrics = meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
    train, test, _ = _load_split(run_id)
    auc = metrics.get("auc") or {}

    readiness = compute_readiness(run_id)
    lines = [f"# Classification Report — run `{run_id}`", ""]
    lines += _build_executive_summary(meta, train, metrics, readiness)
    lines += [""] + _build_readiness_section(readiness)
    lines += [""] + _build_business_understanding_section(meta)
    lines += [""] + _build_success_criteria_section(meta, readiness)
    lines += [""] + _build_hitl_history_section(meta)

    # 1. EDA — recomputed from the original file, before any cleaning.
    lines += ["", "## EDA"]
    try:
        raw_eda = json.loads(eda(meta["source_path"]))
        missing = {k: v for k, v in raw_eda["missing_pct"].items() if v > 0}
        lines.append(f"- **Raw shape:** {tuple(raw_eda['shape'])}")
        lines.append(f"- **Missingness:** {missing or 'none'}")
    except Exception as e:
        lines.append(
            f"_source file unavailable for recompute ({meta.get('source_path')}): {e}_"
        )
    target = meta["target"]
    counts = train[target].value_counts()
    lines.append(
        f"- **Training-fold class balance:** {counts.to_dict()} (ratio {round(float(counts.min() / counts.max()), 4)})"
    )
    outliers = {
        k: v
        for k, v in json.loads(detect_outliers(run_id))[
            "outlier_counts_by_column"
        ].items()
        if v > 0
    }
    lines.append(f"- **Outliers (|z|>3, training fold):** {outliers or 'none'}")

    # 2. Feature transforms — what data-cleaning changed, and why.
    lines += ["", "## Feature transforms"]
    dropped = _reasoned_drops(meta)
    lines.append("- **Dropped columns:**" if dropped else "- **Dropped columns:** none")
    for col, reason in dropped.items():
        lines.append(f"  - `{col}` — {reason}")
    imputation = meta.get("imputation") or {}
    lines.append(
        "- **Imputed (training-fold value, applied to both folds):**"
        if imputation
        else "- **Imputed:** none"
    )
    for col, value in imputation.items():
        lines.append(f"  - `{col}` → `{value}`")
    if meta.get("datetime_features"):
        lines.append(f"- **Datetime columns decomposed:** {meta['datetime_features']}")
    if meta.get("frequency_encoded"):
        lines.append(
            "- **Frequency-encoded (training-fold value share, applied to both folds):**"
        )
        for col, info in meta["frequency_encoded"].items():
            lines.append(
                f"  - `{col}` → `{info.get('new_column')}` "
                f"({info.get('distinct_values_in_train')} distinct in train; "
                f"{info.get('test_rows_with_unseen_value_pct')}% of test rows hold a value never seen in train)"
            )
    lines.append(
        f"- **SMOTE:** {'on, ratio=' + str(meta.get('smote_sampling_strategy')) if meta.get('use_smote') else 'off'}"
    )
    split_type = meta.get("split_type") or (
        "grouped" if meta.get("group_column") else "random"
    )
    split_detail = {
        "temporal": f"forward-chained on `{meta.get('time_column')}` — training data is strictly older than test data",
        "grouped": f"grouped on `{meta.get('group_column')}` — no entity appears in both folds",
        "random": "stratified random split by row",
    }[split_type]
    lines.append(f"- **Split:** {split_type} — {split_detail}")
    if meta.get("immature_rows_dropped"):
        lines.append(
            f"  - **Unsettled labels excluded:** {meta['immature_rows_dropped']} rows after "
            f"{meta.get('immature_after')} — their outcome window had not closed"
        )
    source = meta.get("data_source")
    if source:
        lines.append(
            f"- **Data source:** `{source.get('source')}`"
            + (f", query `{source['query']}`" if source.get("query") else "")
            + f" — {source.get('rows')} rows fetched {source.get('fetched_at')}"
            + (" (**truncated at the row cap**)" if source.get("truncated_at_max_rows") else "")
        )
    if meta.get("entity_overlap_pct") is not None:
        lines.append(
            f"  - **Entity overlap across the time boundary:** {meta['entity_overlap_pct']}% of test rows "
            f"belong to an entity also present in training. Legitimate for a production scorer that sees "
            f"returning entities, but metrics on those rows are not evidence about unseen ones."
        )

    # 3. Feature engineering — business features applied, with formula.
    lines += ["", "## Feature engineering"]
    engineered = _reasoned_features(meta)
    if not engineered:
        lines.append("_none applied_")
    else:
        for name, entry in engineered.items():
            lines.append(f"- `{name}` = `{entry.get('formula')}`")
            rationale = entry.get("rationale")
            lines.append(
                f"  - **Why:** {rationale}"
                if rationale
                else "  - **Why:** _no rationale recorded — the formula alone does not show whether this "
                "feature encodes a real mechanism or a coincidence_"
            )

    # 4. Model
    lines += [
        "",
        "## Model",
        f"- **Source:** `{meta.get('source_path')}`",
        f"- **Target:** `{target}`",
        f"- **Train shape:** {train.shape}",
        # Split provenance is rendered once, in Feature transforms — this used
        # to print a second, group-only version here, which said "grouped by
        # caller" for a run that was actually split temporally.
        f"- **Split:** see Feature transforms above",
        f"- **Model:** {meta.get('model')}"
        + (" (tuned)" if meta.get("best_params") else " (baseline)"),
    ]
    if meta.get("best_params"):
        lines += [
            "",
            "### Best hyperparameters",
            f"```\n{json.dumps(meta['best_params'], indent=2)}\n```",
        ]

    # 5. Explainability — last explain_model result, if one has been run.
    lines += ["", "## Explainability"]
    explain = meta.get("explain")
    if not explain:
        lines.append("_not run yet — call explain_model first_")
    else:
        direction = explain.get("direction") or {}
        lines.append(
            f"- **Method:** {explain['method']}"
            + (
                f" (positive class: `{explain.get('positive_class')}`)"
                if explain["method"] == "shap"
                else ' (global ranking only, no direction — use method="shap" for direction)'
            )
        )
        lines.append("- **Top features:**")
        for feature, value in explain["feature_importance"]:
            note = (
                f" — {direction[feature]} the prediction"
                if feature in direction
                else ""
            )
            lines.append(f"  - `{feature}`: {value}{note}")

    # 5b. Diagnostics — leakage, baseline delta, model comparison, errors, segments.
    lines += [""] + _build_operating_point_section(meta)
    lines += [""] + _build_label_rule_section(meta)
    lines += [""] + _build_diagnostics_section(meta)
    lines += [""] + _build_backtest_section(meta)

    # 5c. Fairness & bias — distinct from segment performance above.
    lines += [""] + _build_fairness_section(meta)

    # 6. Classification metrics
    ci = metrics.get("ci") or {}

    def with_ci(name, value):
        return (
            f"{value} (95% CI [{ci[name][0]}, {ci[name][1]}])"
            if name in ci
            else f"{value}"
        )

    lines += [
        "",
        "## Classification metrics",
        "_Held-out test fold, scored once; intervals are 95% bootstrap over its rows._",
        f"- ROC-AUC: {with_ci('roc_auc', auc.get('roc_auc', auc.get('roc_auc_macro', 'n/a')))}",
        f"- PR-AUC: {with_ci('pr_auc', auc.get('pr_auc', auc.get('pr_auc_macro', 'n/a')))}",
        f"- CV-to-test gap: {metrics.get('cv_to_test_gap', 'n/a')}"
        + (" ⚠ overfitting" if metrics.get("overfitting_warning") else ""),
    ]
    if metrics.get("cv_scheme"):
        lines.append(f"- Cross-validation: {metrics['cv_scheme']}")
    cv_n_rows = metrics.get("cv_computed_on_n_rows")
    if cv_n_rows is not None and cv_n_rows < train.shape[0]:
        lines.append(
            f"- CV score computed on a sample of {cv_n_rows} rows that keeps the split's structure (train fold has {train.shape[0]} — sampled for speed at this scale, final model is fit on all of it)"
        )
    weighted = (metrics.get("classification_report") or {}).get("weighted avg", {})
    if weighted:
        lines += [
            f"- Precision (weighted, at the 0.5 default): {round(weighted.get('precision', 0), 4)}",
            f"- Recall (weighted, at the 0.5 default): {round(weighted.get('recall', 0), 4)}",
            f"- F1 (weighted, at the 0.5 default): {round(weighted.get('f1-score', 0), 4)}",
        ]

    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    lines += ["", "## Confusion matrix", "## ROC curve", "## Precision-recall curve"]
    if not pipeline_path.exists():
        lines[-3:] = [
            "## Confusion matrix\n\n_no fitted model for this run yet — call train_model first_"
        ]
    else:
        pipeline = joblib.load(pipeline_path)
        X_test, y_test = test.drop(columns=[target]), test[target]
        labels = sorted(y_test.unique())
        # At the threshold the model ships with (export bundles it, serving
        # applies it) — the matrix a reviewer signs off must be the deployed
        # decision rule, not pipeline.predict()'s 0.5.
        op = meta.get("operating_point") or {}
        if (
            op.get("threshold") is not None
            and op.get("pipeline_version") == meta.get("pipeline_version")
            and len(labels) == 2
            and hasattr(pipeline, "predict_proba")
        ):
            pos_idx = positive_index(meta, y=y_test)
            positive, negative = (
                pipeline.classes_[pos_idx],
                pipeline.classes_[1 - pos_idx],
            )
            proba = pipeline.predict_proba(X_test)[:, pos_idx]
            y_pred = [
                positive if p >= float(op["threshold"]) else negative for p in proba
            ]
            cutoff = f"at the shipped threshold {op['threshold']} on P({positive})"
        else:
            y_pred = pipeline.predict(X_test)
            cutoff = "at the 0.5 default (no current tuned operating point)"
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        lines[-3] = (
            f"## Confusion matrix\n\n{cutoff}; labels={[str(x) for x in labels]}\n```\n{cm.tolist()}\n```"
        )

        if len(labels) == 2 and hasattr(pipeline, "predict_proba"):
            pos_label = positive_label(meta, y_test)
            proba = pipeline.predict_proba(X_test)[:, positive_index(meta, y=y_test)]
            fpr, tpr, _ = roc_curve(y_test, proba, pos_label=pos_label)
            precision, recall, _ = precision_recall_curve(
                y_test, proba, pos_label=pos_label
            )
            lines[-2] = "## ROC curve\n\n" + f"fpr={_thin(fpr)}\ntpr={_thin(tpr)}"
            lines[-1] = (
                "## Precision-recall curve\n\n"
                + f"precision={_thin(precision)}\nrecall={_thin(recall)}"
            )
        else:
            lines[-2] = (
                "## ROC curve\n\n_skipped — curves are computed for binary targets only_"
            )
            lines[-1] = (
                "## Precision-recall curve\n\n_skipped — curves are computed for binary targets only_"
            )

    lines += [""] + _build_charts_section(run_id, meta)
    lines += [""] + _build_limitations_section(meta, train)
    lines += [""] + _build_reflection_section(meta)
    lines += [""] + _build_execution_log_section(meta)

    report = "\n".join(lines) + "\n"
    (_run_dir(run_id) / "report.md").write_text(report)
    return report
