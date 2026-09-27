"""The deterministic layer: what is TRUE about a run, computed from its
artifacts, independent of anything the agent says about it.

Every other module answers "what did we find?". This one answers "is this run
actually finished, and is it safe to believe?" — and it answers from
meta.json/train.csv/pipeline.pkl only, never from a claim the model wrote.

Why this exists, concretely. Three real failures, none of which more prompting
would have caught:

  1. An identifier column (`Ticket`, 0.79 uniqueness ratio) reached the model,
     one-hot expanded into hundreds of columns, and dominated SHAP. The human
     noticed, not the agent. -> `no_identifier_in_model` / `explainability_clean`
  2. A report announced "Success criteria not met: ROC-AUC 0.822 vs the 0.85
     target" for a 0.85 target the user never set. The bar was invented at
     report time. -> `success_criteria`, which reads a threshold recorded up
     front or reports that none exists, and never derives one.
  3. A report skipped fairness/stability/error-analysis/reflection and
     explained that the tools "were not available in this session's toolset".
     They were registered and callable in the same process. -> every gate
     distinguishes not_run from pass, and the execution ledger in core.py makes
     the availability claim checkable.

The cardinal rule here: **missing evidence is never passing evidence.** The old
executive summary printed "✅ Proceed — metrics are stable" whenever
`meta["stability"]` was absent, because `None` is falsy. A run that skipped the
stability check read exactly like a run that passed it.
"""

import json

import pandas as pd

from .core import (
    ECE_CONCERN,
    IMBALANCED_RATIO,
    _load_meta,
    _load_split,
    _run_dir,
    _save_meta,
    mcp,
    split_keys,
)
from .data_cleaning import identifier_columns

# Gates that must actively pass before a run counts as complete. A gate that
# legitimately does not apply resolves to "not_applicable" WITH a recorded
# reason (fairness on a dataset with no protected attribute); it never
# silently disappears.
REQUIRED_GATES = (
    "business_context",
    "leakage_screened",
    "label_rule_screened",
    "feature_engineering_considered",
    "no_identifier_in_model",
    "baseline_beaten",
    "no_overfitting",
    "stability_checked",
    "calibration_checked",
    "explainability_run",
    "explainability_clean",
    "error_analysis",
    "fairness_assessed",
    "reflection_recorded",
    "evaluation_charts",
    "evidence_ordering",
    "success_criteria",
)

PASS, FAIL, NOT_RUN, NOT_APPLICABLE = "pass", "fail", "not_run", "not_applicable"

# The plots a reviewer signs off on. The visualization agent writes each one
# to <run>/charts/pv<N>_<kind>.png, N = the pipeline_version it was drawn
# from, so a chart of an earlier fit never satisfies the gate for a later one.
REQUIRED_CHARTS = ("confusion_matrix", "roc_curve", "pr_curve")

# Where a measurable success threshold can point. Anything outside this set is
# rejected at record time rather than silently ignored at report time.
SUPPORTED_SUCCESS_METRICS = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "f1_weighted",
    "precision_weighted",
    "recall_weighted",
    "f1_positive",
    "precision_positive",
    "recall_positive",
)


def _current_metrics(meta: dict) -> dict:
    return meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}


def measured_metric(meta: dict, metric: str):
    """Pull one named metric out of this run's CURRENT held-out test results.
    Returns None when the metric genuinely isn't computable for this run (e.g.
    roc_auc on a multiclass target) — the caller reports that, rather than
    substituting a different metric that happens to exist."""
    metrics = _current_metrics(meta)
    report = metrics.get("classification_report") or {}
    auc = metrics.get("auc") or {}
    if metric == "roc_auc":
        return auc.get("roc_auc", auc.get("roc_auc_macro"))
    if metric == "pr_auc":
        return auc.get("pr_auc", auc.get("pr_auc_macro"))
    if metric == "accuracy":
        return report.get("accuracy")
    if metric.endswith("_weighted"):
        key = {"f1": "f1-score"}.get(
            metric[: -len("_weighted")], metric[: -len("_weighted")]
        )
        return (report.get("weighted avg") or {}).get(key)
    if metric.endswith("_positive"):
        key = {"f1": "f1-score"}.get(
            metric[: -len("_positive")], metric[: -len("_positive")]
        )
        # "_positive" means the run's RECORDED positive class — the same one
        # assess_fairness, tune_threshold and explain_model use. classification_report
        # keys are stringified labels, so match on str(). Falling back to the highest
        # key preserves the old behaviour for runs predating positive_label, but a
        # recorded label always wins: on a {"churn","no_churn"} target the highest
        # key is the NEGATIVE class, and this gate would otherwise pass or fail the
        # run against the recall of not-churning.
        class_keys = [
            k for k in report if k not in ("accuracy", "macro avg", "weighted avg")
        ]
        if not class_keys:
            return None
        pos = meta.get("positive_label")
        chosen = next(
            (k for k in class_keys if pos is not None and k == str(pos)),
            max(class_keys),
        )
        return (report.get(chosen) or {}).get(key)
    return None


def _encoded_features_tracing_to(
    columns: set[str], features: list[str], train: pd.DataFrame
) -> dict:
    """Map SHAP/permutation feature names back to the source column they came
    from. OneHotEncoder(verbose_feature_names_out=False) emits "{col}_{value}",
    so a flagged categorical shows up as a family of derived names.

    The prefix test is restricted to non-numeric source columns on purpose:
    numeric columns pass through the encoder unchanged and only ever appear
    under their exact name, so without that guard a flagged `PER_TO` would
    falsely capture the unrelated real feature `PER_TO_3`."""
    hits = {}
    for col in columns:
        expanded = col in train.columns and not pd.api.types.is_numeric_dtype(
            train[col]
        )
        matched = [
            f for f in features if f == col or (expanded and f.startswith(f"{col}_"))
        ]
        if matched:
            hits[col] = matched
    return hits


def _check_evaluation_charts(run_id: str, meta: dict) -> dict:
    """Were the review plots drawn from THIS fit? Nobody signs off a model
    from a table of ROC coordinates. Not a FAIL when missing — absence of
    evidence, like every other unrun gate."""
    if not meta.get("model"):
        return _gate(NOT_RUN, "no fitted model yet")
    pv = meta.get("pipeline_version", 0)
    missing = [
        k
        for k in REQUIRED_CHARTS
        if not (_run_dir(run_id) / "charts" / f"pv{pv}_{k}.png").exists()
    ]
    if missing:
        return _gate(
            NOT_RUN,
            f"evaluation charts not rendered for the current fit (pipeline v{pv}): {missing}. "
            "The visualization agent renders them (plot_confusion_matrix / plot_roc_curve / "
            "plot_pr_curve with this run_id); the orchestrator requests them after this agent "
            "replies — this agent cannot render them itself, so do not loop on this gate.",
        )
    return _gate(PASS, f"{list(REQUIRED_CHARTS)} rendered from pipeline v{pv}")


def _check_evidence_ordering(meta: dict) -> dict:
    """Every other gate asks whether evidence EXISTS. This one asks whether it
    describes the model that is actually here.

    The failure it catches is invisible to the rest: a step runs, the pipeline
    is later refit, and the stored result keeps looking perfectly reasonable
    while describing a model that no longer exists. `_invalidate_model_derived`
    clears the results that are known to be model-derived on refit — this is
    the backstop for the ordering that clearing cannot express, in particular
    a leakage screen that ran before the columns it screened were changed.
    """
    log = meta.get("execution_log") or []
    order = [e.get("tool") for e in log if e.get("ok")]

    def last(tool):
        return len(order) - 1 - order[::-1].index(tool) if tool in order else None

    problems = []
    leak, train_at = last("detect_data_leakage"), last("train_model")
    for mutator in (
        "apply_drop_columns",
        "apply_features",
        "apply_custom_feature",
        "apply_entity_features",
        "apply_peer_features",
        "apply_frequency_encoding",
        "apply_imputation",
        "apply_datetime_features",
    ):
        changed = last(mutator)
        if leak is not None and changed is not None and changed > leak:
            problems.append(
                f"detect_data_leakage ran before `{mutator}` changed the feature set — it screened columns that "
                "are no longer the ones being modelled; re-run it"
            )
    if leak is None and train_at is not None:
        problems.append(
            "a model was trained without any leakage screen having run at all"
        )

    op = meta.get("operating_point")
    if op and op.get("pipeline_version") != meta.get("pipeline_version"):
        problems.append(
            f"the operating point was tuned against pipeline version {op.get('pipeline_version')} but the run is "
            f"now on version {meta.get('pipeline_version')} — the threshold describes a model that was refit "
            "since; re-run tune_threshold"
        )

    if problems:
        return _gate(FAIL, "; ".join(problems))
    if not order:
        return _gate(
            NOT_RUN, "no execution log on this run — ordering cannot be verified"
        )
    return _gate(
        PASS,
        "every recorded check describes the current feature set and the current fitted pipeline",
    )


def _gate(status: str, evidence: str) -> dict:
    return {"status": status, "evidence": evidence}


def _evaluate_success_criteria(meta: dict) -> dict:
    """Compare the measured metric against the threshold recorded BEFORE
    modelling. Never derives a threshold, never rounds one into existence:
    if nothing was recorded, that is the finding."""
    bu = meta.get("business_understanding") or {}
    target = bu.get("success_target") or {}
    metric, threshold = target.get("metric"), target.get("threshold")
    if not metric or threshold is None:
        return _gate(
            NOT_RUN,
            "no measurable success threshold was recorded for this run — record_business_context was called "
            "without success_metric/success_threshold, so there is no bar to judge the model against. "
            "A threshold must NOT be invented at report time.",
        )
    # Judge the model that SHIPS. The stored metrics are computed at
    # sklearn's 0.5; when a threshold was tuned, that is not the model anyone
    # deploys, and the two can straddle the bar — 0.968 recall at 0.5 versus
    # exactly 0.900 at the tuned point is the difference between comfortable
    # and no margin at all.
    op = meta.get("operating_point")
    at = (
        "any threshold (ranking metric)"
        if metric in ("roc_auc", "pr_auc")
        else "the default 0.5 threshold"
    )
    measured = interval = None
    if op and metric in ("recall_positive", "precision_positive", "f1_positive"):
        base = metric.rsplit("_", 1)[0]
        measured, interval = op.get(base), (op.get("ci") or {}).get(base)
        at = f"the run's operating point (threshold {op.get('threshold')}, chosen by {op.get('chosen_by')})"
    if measured is None:
        measured = measured_metric(meta, metric)
        interval = (_current_metrics(meta).get("ci") or {}).get(metric)
    if measured is None:
        return _gate(
            NOT_RUN,
            f"success metric '{metric}' is not computable for this run's target/model",
        )
    direction = target.get("direction", ">=")
    why = f"set up front via record_business_context: {target.get('rationale') or 'no rationale recorded'}"
    if not interval:
        # Runs evaluated before intervals were recorded (or multiclass).
        met = measured >= threshold if direction == ">=" else measured <= threshold
        return _gate(
            PASS if met else FAIL,
            f"{metric} = {round(float(measured), 4)} at {at} {'meets' if met else 'MISSES'} the recorded bar "
            f"({direction} {threshold}) by {round(abs(float(measured) - threshold), 4)} — point estimate only, "
            f"no confidence interval recorded; {why}",
        )
    # Judged on the 95% interval, not the point: a verdict the test fold
    # cannot actually distinguish is not a pass and not a fail.
    lo, hi = interval
    clears = lo >= threshold if direction == ">=" else hi <= threshold
    misses = hi < threshold if direction == ">=" else lo > threshold
    stated = f"{metric} = {round(float(measured), 4)} (95% CI [{lo}, {hi}]) at {at}"
    if clears or misses:
        return _gate(
            PASS if clears else FAIL,
            f"{stated} {'meets' if clears else 'MISSES'} the recorded bar ({direction} {threshold}) with the "
            f"whole interval on that side; {why}",
        )
    return _gate(
        NOT_RUN,
        f"INCONCLUSIVE — {stated}: the interval straddles the bar ({direction} {threshold}), so this test fold "
        "cannot tell whether the model meets it. More labelled test rows narrow the interval; re-tuning against "
        f"the same rows does not. {why}",
    )


def compute_readiness(run_id: str) -> dict:
    """Every required gate's status for this run, computed from artifacts.

    overall_status:
      blocked   — at least one gate actively FAILED (evidence of a problem)
      incomplete— no failures, but required gates never ran (absence of evidence)
      ready     — every required gate passed or is a justified not_applicable
    """
    meta = _load_meta(run_id)
    train, _, _ = _load_split(run_id)
    target = meta["target"]
    metrics = _current_metrics(meta)
    checks: dict[str, dict] = {}

    bu = meta.get("business_understanding") or {}
    checks["business_context"] = (
        _gate(
            PASS,
            f"domain='{bu.get('domain') or 'not declared'}', objective and target definition recorded",
        )
        if bu.get("business_objective") and bu.get("target_definition")
        else _gate(
            NOT_RUN,
            "record_business_context was not called — no business objective or target definition on file",
        )
    )

    leakage = meta.get("leakage")
    # A column that almost decides the label alone, still in the fold the
    # model trains on, is positive evidence — not a note for the report.
    near_perfect = (
        set(
            leakage.get("near_perfect_predictors")
            or leakage.get("near_perfect_correlation_with_target")
            or []
        )
        if leakage
        else set()
    )
    live_near_perfect = sorted(
        c
        for c in near_perfect
        if c in train.columns
        and c not in (meta.get("acknowledged_identifiers") or {})
        and c not in split_keys(meta)  # kept for validation, dropped by the pipeline
    )
    if not leakage:
        checks["leakage_screened"] = _gate(
            NOT_RUN, "detect_data_leakage was never called for this run"
        )
    elif live_near_perfect:
        checks["leakage_screened"] = _gate(
            FAIL,
            f"near-perfect single-column predictor(s) still in the training fold: {live_near_perfect} — one column "
            "that nearly decides the label alone is almost always a copy/proxy of it or a field recorded after the "
            "outcome. Drop it (apply_drop_columns) and retrain, or record the domain reason it is a genuine "
            "feature via acknowledge_identifier_column.",
        )
    else:
        checks["leakage_screened"] = _gate(
            PASS,
            "detect_data_leakage ran; "
            + (
                "no signal found"
                if leakage.get("clear")
                else f"flagged {list(leakage.get('identifier_columns') or {})}"
                + (
                    f", possible post-event {leakage['possible_post_event_columns']}"
                    if leakage.get("possible_post_event_columns")
                    else ""
                )
            ),
        )

    # The gate the Ticket incident is named after: a strong identifier that is
    # still a column of the training fold the model was fit on.
    acknowledged = set(meta.get("acknowledged_identifiers") or {})
    live_identifiers = {
        c: s
        for c, s in identifier_columns(train, target).items()
        if s["strength"] == "strong" and c not in split_keys(meta)
    }
    unacknowledged = {
        c: s for c, s in live_identifiers.items() if c not in acknowledged
    }
    if unacknowledged:
        checks["no_identifier_in_model"] = _gate(
            FAIL,
            "identifier-shaped columns are still in the training fold and will be encoded as model features: "
            + "; ".join(f"`{c}` ({s['reason']})" for c, s in unacknowledged.items())
            + " — drop them (apply_drop_columns) and retrain, or record an explicit justification via "
            "acknowledge_identifier_column if one is genuinely a feature here",
        )
    elif live_identifiers:
        checks["no_identifier_in_model"] = _gate(
            PASS,
            f"identifier-shaped columns kept with recorded justification: {sorted(live_identifiers)}",
        )
    else:
        checks["no_identifier_in_model"] = _gate(
            PASS, "no identifier-shaped column remains in the training fold"
        )

    # Compare on PR-AUC when the target is imbalanced, accuracy otherwise.
    # Accuracy alone makes this gate nearly unfailable on a skewed target: at
    # wangiri's 2.5% positive rate a model that predicts "never fraud" scores
    # 97.5% and comfortably beats the stratified dummy's 95.2%, so the gate
    # would certify a model that has learned nothing. On PR-AUC that same
    # model scores ~0.025 and fails, which is the point of having the gate.
    baseline = meta.get("baseline_comparison")
    counts = train[target].value_counts()
    imbalanced = float(counts.min() / counts.max()) < IMBALANCED_RATIO
    metric_name = "pr_auc" if imbalanced else "accuracy"
    model_score = measured_metric(meta, metric_name)
    base_score = None
    if baseline:
        base_score = (
            (baseline.get("auc") or {}).get("pr_auc")
            if imbalanced
            else (baseline.get("classification_report") or {}).get("accuracy")
        )
    if baseline is None:
        checks["baseline_beaten"] = _gate(
            NOT_RUN,
            "train_baseline was never called — no reference point for 'better than guessing'",
        )
    elif model_score is None or base_score is None:
        checks["baseline_beaten"] = _gate(
            NOT_RUN, f"no {metric_name} available for both the model and the baseline"
        )
    else:
        checks["baseline_beaten"] = _gate(
            PASS if model_score > base_score else FAIL,
            f"model {metric_name} {round(model_score, 4)} vs dummy baseline {round(base_score, 4)}"
            + (
                " (compared on PR-AUC because the target is imbalanced — accuracy would pass a "
                "model that never predicts the minority class)"
                if imbalanced
                else ""
            ),
        )

    if not metrics:
        checks["no_overfitting"] = _gate(
            NOT_RUN, "no model has been trained for this run"
        )
    elif metrics.get("overfitting_warning"):
        checks["no_overfitting"] = _gate(
            FAIL,
            f"CV-to-test gap {metrics.get('cv_to_test_gap')} exceeds the overfitting threshold",
        )
    else:
        checks["no_overfitting"] = _gate(
            PASS, f"CV-to-test gap {metrics.get('cv_to_test_gap')} is within threshold"
        )

    stability = meta.get("stability")
    if stability is None:
        checks["stability_checked"] = _gate(
            NOT_RUN,
            "check_model_stability was never called — the reported score may be one split's luck",
        )
    elif stability.get("high_variance_warning"):
        checks["stability_checked"] = _gate(
            FAIL, f"CV score varies across random splits (std={stability.get('std')})"
        )
    else:
        checks["stability_checked"] = _gate(
            PASS, f"score is stable across repeated splits (std={stability.get('std')})"
        )

    # Only gate calibration on a run that actually ships a threshold-based
    # decision. A model whose output is read as a ranking (top-N review
    # queue) doesn't need calibrated probabilities, and a gate that fires
    # where it doesn't apply teaches people to ignore gates.
    calibration = meta.get("calibration")
    operating_point = meta.get("operating_point")
    if calibration is None and operating_point is None:
        checks["calibration_checked"] = _gate(
            NOT_RUN,
            "check_calibration was never called — unknown whether this model's predicted probabilities mean what "
            "they say. Required before any threshold is tuned against them.",
        )
    elif calibration is None:
        checks["calibration_checked"] = _gate(
            FAIL,
            f"a decision threshold ({operating_point.get('threshold')}) was tuned on probabilities whose "
            "calibration was never measured — the cutoff's promised precision/recall rests on scores nobody "
            "verified are probabilities; run check_calibration",
        )
    elif calibration.get("miscalibration_warning") and operating_point is not None:
        checks["calibration_checked"] = _gate(
            FAIL,
            f"ECE {calibration.get('expected_calibration_error')} exceeds {ECE_CONCERN} AND a threshold "
            f"({operating_point.get('threshold')}) was tuned on those probabilities — run calibrate_model, then "
            "re-run tune_threshold",
        )
    elif calibration.get("miscalibration_warning"):
        checks["calibration_checked"] = _gate(
            NOT_APPLICABLE,
            f"ECE {calibration.get('expected_calibration_error')} exceeds {ECE_CONCERN}, but this run ships no "
            "tuned threshold — the scores are used as a ranking, where calibration doesn't change the decision. "
            "Calibrate before anyone reads a probability off this model as a probability.",
        )
    else:
        checks["calibration_checked"] = _gate(
            PASS,
            f"ECE {calibration.get('expected_calibration_error')}, Brier {calibration.get('brier_score')}"
            + (
                f" (calibrated with {calibration.get('calibrated_by')})"
                if calibration.get("calibrated_by")
                else ""
            ),
        )

    explain = meta.get("explain")
    if not explain:
        checks["explainability_run"] = _gate(NOT_RUN, "explain_model was never called")
        checks["explainability_clean"] = _gate(
            NOT_RUN, "no explainability output to check for identifier dominance"
        )
    else:
        features = [f for f, _ in explain.get("feature_importance") or []]
        checks["explainability_run"] = _gate(
            PASS,
            f"{explain.get('method')} importances computed over {len(features)} top features",
        )
        polluted = _encoded_features_tracing_to(set(live_identifiers), features, train)
        if polluted and not all(c in acknowledged for c in polluted):
            checks["explainability_clean"] = _gate(
                FAIL,
                "top features trace back to identifier columns, so the ranking reflects memorised rows rather than "
                f"drivers: {json.dumps(polluted, default=str)} — this is the signal that reached a human as 'the SHAP "
                "Ticket_* columns'; drop the column and retrain",
            )
        else:
            checks["explainability_clean"] = _gate(
                PASS,
                "no top feature traces back to an unacknowledged identifier column",
            )

    checks["error_analysis"] = (
        _gate(
            PASS,
            f"{meta['error_analysis'].get('n_errors')} misclassifications inspected on the test fold",
        )
        if meta.get("error_analysis")
        else _gate(NOT_RUN, "error_analysis was never called")
    )

    fairness = meta.get("fairness")
    fairness_na = meta.get("fairness_not_applicable")
    if fairness:
        flags = [k for k, v in (fairness.get("flags") or {}).items() if v]
        checks["fairness_assessed"] = _gate(
            PASS,
            f"assess_fairness ran on `{fairness['protected_attribute']}`"
            + (
                f" — flagged {flags} (surfaced, not silently passed)"
                if flags
                else " — no flags at configured thresholds"
            ),
        )
    elif fairness_na:
        checks["fairness_assessed"] = _gate(
            NOT_APPLICABLE, f"declared not applicable: {fairness_na}"
        )
    else:
        checks["fairness_assessed"] = _gate(
            NOT_RUN,
            "assess_fairness was never called and no justification for skipping it was recorded — "
            "call it on the protected attribute, or record why none applies via declare_fairness_not_applicable",
        )

    rule_check = meta.get("label_rule_check")
    if not rule_check:
        checks["label_rule_screened"] = _gate(
            NOT_RUN,
            "check_label_rule was never called — nothing has asked whether a single-column rule reproduces these "
            "labels, which is how a rule-generated target passes every other check while inflating the score",
        )
    elif rule_check.get("label_rule_suspicion"):
        best = rule_check.get("best_single_column_rule") or {}
        finding = (
            f"a depth-{best.get('tree_depth')} rule on `{best.get('column')}` alone reaches "
            f"{rule_check.get('rule_to_model_ratio')} of the model's PR-AUC"
        )
        acknowledged = meta.get("label_rule_acknowledged")
        # A high ratio has two possible causes and the tool cannot tell them
        # apart: a rule-generated label, or a domain where one physical driver
        # genuinely dominates. So it demands an answer rather than asserting
        # a defect — the same written-justification escape hatch the
        # identifier gate uses, for the same reason.
        checks["label_rule_screened"] = (
            _gate(
                PASS,
                f"{finding} — acknowledged as a genuine single-driver problem: {acknowledged}",
            )
            if acknowledged
            else _gate(
                FAIL,
                f"{finding}. Either a rule on that column generated these labels — in which case every metric here "
                "measures agreement with the rule — or the domain genuinely has one dominant driver. Ask the label "
                "owner, then drop the column or record the answer via acknowledge_label_rule.",
            )
        )
    else:
        checks["label_rule_screened"] = _gate(
            PASS,
            f"best single-column rule reaches {rule_check.get('rule_to_model_ratio')} of the model's PR-AUC — "
            "the model is doing work no single threshold reproduces",
        )

    engineered = meta.get("engineered_features") or {}
    fe_na = meta.get("feature_engineering_not_applicable")
    if engineered:
        checks["feature_engineering_considered"] = _gate(
            PASS, f"{len(engineered)} business feature(s) applied: {sorted(engineered)}"
        )
    elif fe_na:
        checks["feature_engineering_considered"] = _gate(
            NOT_APPLICABLE, f"declared not applicable: {fe_na}"
        )
    else:
        checks["feature_engineering_considered"] = _gate(
            NOT_RUN,
            "no features were engineered and no reason was recorded — a run that skipped the step is "
            "indistinguishable from one where nothing applied. Apply features, or record why none do via "
            "declare_feature_engineering_not_applicable",
        )

    reflection = meta.get("reflection")
    if not reflection:
        checks["reflection_recorded"] = _gate(
            NOT_RUN, "record_reflection was never called"
        )
    elif reflection.get("contradictions"):
        checks["reflection_recorded"] = _gate(
            FAIL,
            f"reflection contradicts this run's artifacts: {reflection['contradictions']}",
        )
    else:
        checks["reflection_recorded"] = _gate(
            PASS,
            f"reflection recorded, self-assessed status '{reflection.get('overall_status')}'",
        )

    checks["evaluation_charts"] = _check_evaluation_charts(run_id, meta)
    checks["evidence_ordering"] = _check_evidence_ordering(meta)
    checks["success_criteria"] = _evaluate_success_criteria(meta)

    failed = [k for k, v in checks.items() if v["status"] == FAIL]
    not_run = [k for k, v in checks.items() if v["status"] == NOT_RUN]
    passed = [k for k, v in checks.items() if v["status"] in (PASS, NOT_APPLICABLE)]
    overall = "blocked" if failed else ("incomplete" if not_run else "ready")

    return {
        "run_id": run_id,
        "overall_status": overall,
        "gates_passed": len(passed),
        "gates_total": len(REQUIRED_GATES),
        # Deliberately a fraction of deterministic checks, not a probability.
        # It says "this much of the mechanical contract is satisfied" — it is
        # NOT a claim that the modelling choices or the business framing are
        # right, which no gate here can measure.
        "mechanical_completeness": round(len(passed) / len(REQUIRED_GATES), 3),
        "failed_gates": failed,
        "not_run_gates": not_run,
        "checks": checks,
        "interpretation": {
            "blocked": "At least one gate found positive evidence of a problem. Do not ship; fix and re-run.",
            "incomplete": "No failures, but required evidence is missing. Absence of evidence is NOT a pass — run the missing steps.",
            "ready": "Every mechanical gate passed. This covers execution correctness only; methodological and business judgment still need human review.",
        }[overall],
    }


@mcp.tool()
def check_readiness(run_id: str) -> str:
    """Deterministic readiness gate — the answer to "is this run actually
    finished and safe to believe", computed from this run's artifacts rather
    than from anything the agent asserts.

    Returns every required gate's status (pass / fail / not_run /
    not_applicable) with the evidence behind it, plus overall_status:
      `blocked`    — a gate found positive evidence of a problem
      `incomplete` — required evidence is missing (NOT the same as passing)
      `ready`      — every mechanical gate passed

    Call before generate_report and before export_model. generate_report
    renders this at the top of the report and export_model refuses a blocked
    run, so a run that skipped its checks cannot be presented as a clean one.
    `mechanical_completeness` is the fraction of deterministic gates satisfied
    — execution correctness only, never a claim that the modelling choices or
    the business framing are correct."""
    return json.dumps(compute_readiness(run_id))


@mcp.tool()
def acknowledge_label_rule(run_id: str, justification: str) -> str:
    """Record that check_label_rule's finding is expected — that one column
    dominating is a real property of the domain, not a sign that a rule wrote
    the labels. The only way past the `label_rule_screened` gate without
    removing the column.

    Legitimate use: the labels are analyst-investigated (so no rule could
    have generated them) and the dominant column is the phenomenon's actual
    physical driver. Illegitimate use: making a blocked run pass without
    having asked anyone where the labels came from. If you have not put the
    question to whoever produced the labels, you do not yet have a
    justification — you have an assumption."""
    meta = _load_meta(run_id)
    if not justification.strip():
        return json.dumps(
            {
                "error": "a justification is required — this gate cannot be waived silently"
            }
        )
    if not meta.get("label_rule_check"):
        return json.dumps(
            {
                "error": "run check_label_rule first — there is no finding to acknowledge yet"
            }
        )
    meta["label_rule_acknowledged"] = justification
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, "label_rule_acknowledged": justification})


@mcp.tool()
def acknowledge_identifier_column(run_id: str, column: str, justification: str) -> str:
    """Record that an identifier-shaped column is deliberately being kept as a
    feature, with the domain reason why. This is the ONLY way past the
    `no_identifier_in_model` / `explainability_clean` gates without dropping
    the column — the escape hatch is deliberately a written justification on
    the record rather than a silent override, so a reviewer sees the decision
    and who made it.

    Use only when the column genuinely carries signal in this domain (a
    product SKU with real repeat structure, say). Do NOT use it to get a
    blocked run to pass: a ticket/order/invoice number keeps no signal a
    one-hot encoder can generalise from."""
    meta = _load_meta(run_id)
    if not justification.strip():
        return json.dumps(
            {
                "error": "a justification is required — this gate cannot be waived silently"
            }
        )
    train, _, _ = _load_split(run_id)
    if column not in train.columns:
        return json.dumps({"error": f"no column '{column}' in the training fold"})
    meta.setdefault("acknowledged_identifiers", {})[column] = justification
    _save_meta(run_id, meta)
    return json.dumps(
        {"run_id": run_id, "acknowledged": column, "justification": justification}
    )


@mcp.tool()
def declare_fairness_not_applicable(run_id: str, reason: str) -> str:
    """Record why no fairness assessment applies to this run (no protected or
    proxy attribute in the data, non-person-level prediction unit, ...).

    The `fairness_assessed` gate treats a skipped assessment as missing
    evidence, not as a pass. This turns "we thought about it and here is why
    it does not apply" into a recorded, reviewable statement, which is a
    different thing from silence — and keeps the report honest either way."""
    meta = _load_meta(run_id)
    if not reason.strip():
        return json.dumps({"error": "a reason is required"})
    meta["fairness_not_applicable"] = reason
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, "fairness_not_applicable": reason})
