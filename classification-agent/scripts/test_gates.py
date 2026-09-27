"""Regression tests for the deterministic enforcement layer.

Every test here is named after a failure that actually shipped in a report, so
that the specific way each one slipped through stays closed:

  identifier gap ......... `Ticket` (0.79 uniqueness ratio, ~544 distinct)
                           cleared both cardinality bars, reached the model,
                           and dominated SHAP. A human caught it, twice.
  missing == passing ..... a run that never checked stability printed
                           "✅ Proceed — metrics are stable", because
                           meta["stability"] was None and None is falsy.
  invented threshold ..... a report announced "ROC-AUC 0.822 vs the 0.85
                           target" for a target nobody ever set.
  unfalsifiable skip ..... a report explained that reporting/mlops tools
                           "were not available in this session's toolset"
                           while they were registered in the same process.
  reflection ratified .... reflection marked data_quality_and_leakage "clear"
                           because the human had already patched the leak.

Run: python test_gates.py
"""

import os as _os
import tempfile as _tempfile

# Tests never write into the real data/ (runs, artifacts, predictions) that
# the orchestrator reports as "recent runs"; CI may point this elsewhere.
_os.environ.setdefault(
    "AGENTIC_ML_DATA_DIR", _tempfile.mkdtemp(prefix="agentic-ml-test-")
)
# ...nor register test models into the real MLflow registry (the UI on :5000).
_os.environ.setdefault(
    "MLFLOW_TRACKING_URI", "file:" + _tempfile.mkdtemp(prefix="agentic-ml-test-mlruns-")
)

import json
import shutil

import numpy as np
import pandas as pd

import mcp_server as m
from mcp_server.data_cleaning import identifier_columns


def _csv(df: pd.DataFrame, tmp: str) -> str:
    import tempfile

    fd, path = tempfile.mkstemp(suffix=f"_{tmp}.csv")
    df.to_csv(path, index=False)
    return path


def _frame(n=400, seed=0) -> pd.DataFrame:
    """Learnable but deliberately NOT separable — ~15% of labels are flipped so
    the model lands well short of a perfect ROC-AUC. A fixture the model can
    score 1.0 on cannot tell a met success threshold from an unmet one."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    label = (signal > 0).astype(int)
    flip = rng.random(n) < 0.15
    label = np.where(flip, 1 - label, label)
    return pd.DataFrame(
        {
            "signal": signal,
            "amount": rng.normal(50, 10, n),
            "channel": rng.choice(
                ["web", "mobile", "atm"], n
            ),  # legitimate low-card categorical
            "region": rng.choice(
                [f"r{i}" for i in range(8)], n
            ),  # legitimate mid-card categorical
            "label": label,
        }
    )


# --------------------------------------------------------------------------
# 1. The identifier gap: the exact shape that reached the model twice.
# --------------------------------------------------------------------------
def identifier_ratio_gap() -> None:
    n = 400
    df = _frame(n)
    # ~0.75 ratio, 300 distinct: clears near-unique (0.98) AND the absolute
    # high-cardinality bar (1000). This is the Ticket/order-no/invoice-no shape.
    df["ticket_ref"] = [f"T{i // 2 + 1000}" for i in range(n)]
    flagged = identifier_columns(df, "label")

    assert "ticket_ref" in flagged, (
        "moderate-cardinality identifier must be caught — this is the Ticket bug"
    )
    assert flagged["ticket_ref"]["rule"] == "identifier_ratio"
    assert flagged["ticket_ref"]["strength"] == "strong"
    assert 0.5 <= flagged["ticket_ref"]["ratio"] < 0.98, (
        "must be caught in the band BETWEEN the two old bars"
    )

    # Adversarial: legitimate features must not be dragged in with it.
    for legit in ("signal", "amount", "channel", "region"):
        assert legit not in flagged, f"false positive on legitimate feature `{legit}`"

    # Adversarial: a continuous NUMERIC column is naturally near-unique-ish but
    # is not an identifier — the non-numeric guard is what keeps `Fare`-shaped
    # columns out of the ratio rule.
    numeric_heavy = pd.DataFrame(
        {"price": np.linspace(0, 1, 300), "label": [0, 1] * 150}
    )
    assert "price" not in identifier_columns(numeric_heavy, "label"), (
        "continuous numeric must not trip the ratio rule"
    )

    # Adversarial: a small categorical at a high ratio stays clean — one-hot on
    # 12 distinct values is harmless, which is what MIN_DISTINCT protects.
    tiny = pd.DataFrame({"grp": [f"g{i}" for i in range(12)] * 2, "label": [0, 1] * 12})
    assert "grp" not in identifier_columns(tiny, "label"), (
        "small categorical must not trip the ratio rule"
    )

    # Both entry points must agree — they used to disagree (detect_data_leakage
    # demanded 100% uniqueness where propose_drop_columns wanted 98%).
    path = _csv(df, "idgap")
    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    proposals = json.loads(m.propose_drop_columns(run_id))["proposals"]
    leakage = json.loads(m.detect_data_leakage(run_id))
    assert "ticket_ref" in proposals, "propose_drop_columns must flag it"
    assert "ticket_ref" in leakage["strong_identifier_columns"], (
        "detect_data_leakage must flag the same column"
    )
    assert not leakage["clear"]
    shutil.rmtree(m._run_dir(run_id))
    print(
        "  identifier_ratio_gap: the Ticket shape is caught by both entry points, no false positives"
    )


# --------------------------------------------------------------------------
# 2. Missing evidence must never read as a pass.
# --------------------------------------------------------------------------
def missing_evidence_is_not_a_pass() -> None:
    path = _csv(_frame(), "bare")
    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    # The leakage screen is the one check that must precede training —
    # training without it is a FAILED ordering (blocked), a different finding
    # from the missing evidence this test is about.
    m.detect_data_leakage(run_id)
    m.train_model(run_id, "random_forest")

    readiness = json.loads(m.check_readiness(run_id))
    assert readiness["overall_status"] == "incomplete", (
        "a run that skipped every check is not 'ready'"
    )
    assert readiness["overall_status"] != "ready"
    for skipped in (
        "stability_checked",
        "error_analysis",
        "fairness_assessed",
        "reflection_recorded",
    ):
        assert readiness["checks"][skipped]["status"] == "not_run", (
            f"{skipped} must read as not_run"
        )
    assert readiness["mechanical_completeness"] < 1.0

    report = m.generate_report(run_id)
    assert "⚠ **Incomplete**" in report, (
        "the summary must not call an unchecked run shippable"
    )
    assert "✅ **Proceed" not in report, (
        "REGRESSION: missing evidence rendered as Proceed"
    )
    assert "Run Readiness" in report and "stability_checked" in report

    # And the same run, once every gate is actually satisfied, must flip to ready.
    m.train_baseline(run_id)
    m.check_model_stability(run_id, n_repeats=2)
    m.explain_model(run_id, method="permutation")
    m.error_analysis(run_id)
    m.detect_data_leakage(run_id)
    m.check_label_rule(run_id)
    # The fixture's label IS a threshold on `signal` — the screen is right to
    # flag it; a human's recorded answer is the only way past it.
    m.acknowledge_label_rule(
        run_id, "synthetic fixture: signal is the designed driver of the label"
    )
    m.check_calibration(run_id)
    m.declare_feature_engineering_not_applicable(
        run_id, "synthetic fixture — no domain to engineer from"
    )
    m.declare_fairness_not_applicable(
        run_id, "synthetic fixture, no person-level attribute"
    )
    m.record_business_context(
        run_id,
        "test objective",
        "label = synthetic positive",
        "beat the baseline",
        success_metric="roc_auc",
        success_threshold="0.5",
    )
    m.record_reflection(
        run_id,
        checks=json.dumps(
            {
                k: {"status": "clear", "evidence": "checked"}
                for k in m.diagnostics.REFLECTION_CHECKS
            }
        ),
    )
    # Nobody signs off a model unseen: without the review charts of THIS fit
    # the run is still incomplete. (The visualization agent writes these.)
    readiness = json.loads(m.check_readiness(run_id))
    assert "evaluation_charts" in readiness["not_run_gates"], readiness["not_run_gates"]
    pv = m._load_meta(run_id).get("pipeline_version", 0)
    (m._run_dir(run_id) / "charts").mkdir(exist_ok=True)
    for kind in m.gates.REQUIRED_CHARTS:
        (m._run_dir(run_id) / "charts" / f"pv{pv - 1}_{kind}.png").write_bytes(b"")
    assert (
        json.loads(m.check_readiness(run_id))["checks"]["evaluation_charts"]["status"]
        == "not_run"
    ), "charts of an earlier fit must not satisfy the gate"
    for kind in m.gates.REQUIRED_CHARTS:
        (m._run_dir(run_id) / "charts" / f"pv{pv}_{kind}.png").write_bytes(b"")
    readiness = json.loads(m.check_readiness(run_id))
    assert readiness["overall_status"] == "ready", (
        f"expected ready, got {readiness['overall_status']}: {readiness['failed_gates']} / {readiness['not_run_gates']}"
    )
    assert "✅ **Proceed to human review**" in m.generate_report(run_id)
    shutil.rmtree(m._run_dir(run_id))
    print(
        "  missing_evidence_is_not_a_pass: incomplete stays incomplete; a fully-checked run flips to ready"
    )


# --------------------------------------------------------------------------
# 3. An identifier that reaches the model blocks the run.
# --------------------------------------------------------------------------
def identifier_in_model_blocks() -> None:
    n = 400
    df = _frame(n)
    df["ticket_ref"] = [f"T{i // 2 + 1000}" for i in range(n)]
    path = _csv(df, "inmodel")
    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    m.detect_data_leakage(
        run_id
    )  # screened, flagged, and trained on anyway — the case under test
    m.train_model(run_id, "random_forest")

    readiness = json.loads(m.check_readiness(run_id))
    assert readiness["checks"]["no_identifier_in_model"]["status"] == "fail"
    assert readiness["overall_status"] == "blocked", (
        "an identifier feeding the model must block, not merely warn"
    )
    assert "ticket_ref" in readiness["checks"]["no_identifier_in_model"]["evidence"]

    # export must refuse a blocked run, and say why
    export = json.loads(m.export_model(run_id, out_path="/tmp/_gate_test.pkl"))
    assert "error" in export and "BLOCKED" in export["error"]
    assert "no_identifier_in_model" in export["failed_gates"]

    # the escape hatch works, but only with a written justification on the record
    assert "error" in json.loads(
        m.acknowledge_identifier_column(run_id, "ticket_ref", "   ")
    ), "the gate must not be waivable with an empty justification"
    m.acknowledge_identifier_column(
        run_id, "ticket_ref", "kept deliberately for this test"
    )
    readiness = json.loads(m.check_readiness(run_id))
    assert readiness["checks"]["no_identifier_in_model"]["status"] == "pass"
    assert readiness["overall_status"] != "blocked"

    # dropping it is the real fix
    m.apply_drop_columns(run_id, json.dumps(["ticket_ref"]))
    m.train_model(run_id, "random_forest")
    assert (
        json.loads(m.check_readiness(run_id))["checks"]["no_identifier_in_model"][
            "status"
        ]
        == "pass"
    )
    shutil.rmtree(m._run_dir(run_id))
    print(
        "  identifier_in_model_blocks: blocks the run, refuses export, waivable only in writing"
    )


# --------------------------------------------------------------------------
# 4. Explainability pollution — the signal that actually reached the human.
# --------------------------------------------------------------------------
def explainability_pollution_detected() -> None:
    from mcp_server.gates import _encoded_features_tracing_to

    train = pd.DataFrame(
        {
            "Ticket": ["A/5 21171", "PC 17599"] * 5,
            "PER_TO": np.linspace(0, 1, 10),
            "PER_TO_3": np.linspace(0, 1, 10),
            "label": [0, 1] * 5,
        }
    )
    # OneHotEncoder(verbose_feature_names_out=False) emits "{col}_{value}".
    features = ["Ticket_A/5 21171", "Ticket_PC 17599", "PER_TO_3", "PER_TO"]

    hits = _encoded_features_tracing_to({"Ticket"}, features, train)
    assert hits["Ticket"] == [
        "Ticket_A/5 21171",
        "Ticket_PC 17599",
    ], "must trace one-hot children back to the source column"

    # Adversarial: a flagged NUMERIC column must not swallow a same-prefixed
    # real feature. `PER_TO` and `PER_TO_3` are both genuine wangiri features —
    # a naive startswith() would report PER_TO_3 as polluted by PER_TO.
    hits = _encoded_features_tracing_to({"PER_TO"}, features, train)
    assert hits == {"PER_TO": ["PER_TO"]}, f"numeric prefix false positive: {hits}"
    print(
        "  explainability_pollution_detected: one-hot children traced, numeric prefix collision avoided"
    )


# --------------------------------------------------------------------------
# 5. A success threshold can only come from the record.
# --------------------------------------------------------------------------
def success_criteria_cannot_be_invented() -> None:
    path = _csv(_frame(), "success")
    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    m.train_model(run_id, "random_forest")

    # No threshold recorded -> the report must say so, not conjure one.
    m.record_business_context(
        run_id, "objective", "label definition", "no numeric bar was agreed"
    )
    readiness = json.loads(m.check_readiness(run_id))
    assert readiness["checks"]["success_criteria"]["status"] == "not_run"
    report = m.generate_report(run_id)
    assert "No measurable success threshold was recorded" in report
    assert "0.85" not in report.split("## Success Criteria")[1].split("##")[0], (
        "no threshold may appear from nowhere"
    )

    # A recorded threshold is compared against the measured value.
    m.record_business_context(
        run_id,
        "objective",
        "label definition",
        "must beat 0.99 ROC-AUC",
        success_metric="roc_auc",
        success_threshold="0.99",
    )
    check = json.loads(m.check_readiness(run_id))["checks"]["success_criteria"]
    assert check["status"] == "fail" and "MISSES" in check["evidence"]

    m.record_business_context(
        run_id,
        "objective",
        "label definition",
        "must beat 0.50 ROC-AUC",
        success_metric="roc_auc",
        success_threshold="0.50",
    )
    assert (
        json.loads(m.check_readiness(run_id))["checks"]["success_criteria"]["status"]
        == "pass"
    )
    assert "Recorded bar" in m.generate_report(run_id)

    # Garbage in is rejected at record time, not silently ignored at report time.
    assert "error" in json.loads(
        m.record_business_context(
            run_id,
            "a",
            "b",
            "c",
            success_metric="made_up_metric",
            success_threshold="0.5",
        )
    )
    assert "error" in json.loads(
        m.record_business_context(
            run_id, "a", "b", "c", success_metric="roc_auc", success_threshold="high"
        )
    )
    shutil.rmtree(m._run_dir(run_id))
    print(
        "  success_criteria_cannot_be_invented: absent bar reported as absent, recorded bar measured, junk rejected"
    )


# --------------------------------------------------------------------------
# 6. The execution ledger makes "the tool wasn't available" checkable.
# --------------------------------------------------------------------------
def execution_ledger_records_reality() -> None:
    path = _csv(_frame(), "ledger")
    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    m.train_model(run_id, "random_forest")
    m.analyze_segments(
        run_id, "no_such_column"
    )  # returns {"error": ...} without raising

    log = m._load_meta(run_id)["execution_log"]
    tools_called = [e["tool"] for e in log]
    assert "train_model" in tools_called and "analyze_segments" in tools_called
    failed = [e for e in log if not e["ok"]]
    assert any(e["tool"] == "analyze_segments" for e in failed), (
        "an error-returning call must be logged as not-ok"
    )
    assert all("seconds" in e and "at" in e for e in log)

    report = m.generate_report(run_id)
    assert "## Execution Log" in report and "train_model" in report
    assert "returned an error" in report, (
        "a failed call must be visible in the report, not swallowed"
    )
    shutil.rmtree(m._run_dir(run_id))
    print(
        "  execution_ledger_records_reality: every call logged, errors surfaced in the report"
    )


# --------------------------------------------------------------------------
# 7. Reflection cannot certify what the artifacts refute.
# --------------------------------------------------------------------------
def reflection_contradiction_detected() -> None:
    n = 400
    df = _frame(n)
    df["ticket_ref"] = [f"T{i // 2 + 1000}" for i in range(n)]
    path = _csv(df, "reflect")
    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    m.train_model(run_id, "random_forest")

    # The agent claims everything is clean. The artifacts say an identifier is
    # still in the model and fairness never ran.
    claim = {
        k: {"status": "clear", "evidence": "looks fine to me"}
        for k in m.diagnostics.REFLECTION_CHECKS
    }
    reflection = json.loads(m.record_reflection(run_id, checks=json.dumps(claim)))

    assert reflection["contradictions"], (
        "a 'clear' claim over a failed gate must be recorded as a contradiction"
    )
    assert any("no_identifier_in_model" in c for c in reflection["contradictions"])
    assert any("fairness_assessed" in c for c in reflection["contradictions"])

    readiness = json.loads(m.check_readiness(run_id))
    assert readiness["checks"]["reflection_recorded"]["status"] == "fail", (
        "a contradicted reflection must fail its gate"
    )
    assert "contradict" in readiness["checks"]["reflection_recorded"]["evidence"]
    shutil.rmtree(m._run_dir(run_id))
    print(
        "  reflection_contradiction_detected: self-assessment cross-checked against artifacts"
    )


# --------------------------------------------------------------------------
# 8. A refit invalidates everything that described the previous model.
# --------------------------------------------------------------------------
def refit_invalidates_stale_results() -> None:
    n = 400
    df = _frame(n)
    df["ticket_ref"] = [f"T{i // 2 + 1000}" for i in range(n)]
    path = _csv(df, "stale")
    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    m.detect_data_leakage(run_id)
    m.train_model(run_id, "random_forest")
    m.explain_model(run_id, method="shap")
    m.error_analysis(run_id)
    m.check_model_stability(run_id, n_repeats=2)
    m.analyze_segments(run_id, "channel")

    before = m._load_meta(run_id)
    assert (
        before["explain"]
        and before["error_analysis"]
        and before["stability"]
        and before["segment_analysis"]
    )

    # The standard remediation: drop the leaking column and retrain. Everything
    # describing the OLD fit must be cleared, not silently carried forward under
    # the new model's name.
    m.apply_drop_columns(run_id, json.dumps(["ticket_ref"]))
    m.detect_data_leakage(
        run_id
    )  # re-screen the changed feature set, as the ordering gate requires
    refit = json.loads(m.train_model(run_id, "random_forest"))
    assert set(refit["invalidated_by_refit"]) >= {
        "explain",
        "error_analysis",
        "stability",
    }

    meta = m._load_meta(run_id)
    for key in ("explain", "error_analysis", "stability", "segment_analysis"):
        assert meta.get(key) is None, (
            f"stale {key} survived a refit — the report would attribute it to the new model"
        )

    # Shipped in a report: analysis -> retrain -> analysis crashed with
    # "'NoneType' object does not support item assignment", because refit
    # nulled segment_analysis and setdefault() returns an existing None. The
    # report then printed the Segment section as "not run" for a run where it
    # HAD run. Re-analysing after a refit must just work.
    again = json.loads(m.analyze_segments(run_id, "channel"))
    assert "segments" in again, (
        f"REGRESSION: analyze_segments crashed after a refit: {again}"
    )
    assert set(m._load_meta(run_id)["segment_analysis"]) == {"channel"}

    # And the gate must now demand they be re-run rather than inheriting a pass.
    readiness = json.loads(m.check_readiness(run_id))
    for gate in ("explainability_run", "error_analysis", "stability_checked"):
        assert readiness["checks"][gate]["status"] == "not_run", (
            f"{gate} inherited the pre-refit result"
        )
    assert readiness["overall_status"] == "incomplete"

    report = m.generate_report(run_id)
    assert "ticket_ref_" not in report, (
        "REGRESSION: the retrained model's report still shows the old SHAP ranking"
    )
    assert "_not run yet — call explain_model first_" in report, (
        "the report must admit explainability is missing post-refit, not show the previous fit's"
    )
    shutil.rmtree(m._run_dir(run_id))
    print(
        "  refit_invalidates_stale_results: retrain clears the old fit's results and re-arms the gates"
    )


# --------------------------------------------------------------------------
# 9. The baseline gate must not be passable by a model that learned nothing.
# --------------------------------------------------------------------------
def baseline_gate_uses_imbalance_aware_metric() -> None:
    # 3% positive, and the features carry NO signal about the label — the best
    # a model can do is predict the majority class, scoring ~97% accuracy.
    # Against a stratified dummy's ~94%, an accuracy comparison would certify
    # this as "beats the baseline". PR-AUC is what exposes it.
    rng = np.random.default_rng(7)
    n = 600
    df = pd.DataFrame(
        {
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
            "label": (rng.random(n) < 0.03).astype(int),
        }
    )
    path = _csv(df, "nosignal")
    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    m.train_baseline(run_id)
    m.train_model(run_id, "logistic_regression")

    check = json.loads(m.check_readiness(run_id))["checks"]["baseline_beaten"]
    assert "pr_auc" in check["evidence"], (
        f"imbalanced run must compare on PR-AUC, got: {check['evidence']}"
    )
    assert check["status"] == "fail", (
        f"a model fit on pure noise must not pass the baseline gate: {check}"
    )

    # A balanced run keeps comparing on accuracy — PR-AUC is the imbalanced
    # branch, not a blanket replacement.
    balanced = _csv(_frame(), "balanced_baseline")
    b_run = json.loads(m.prepare_dataset(balanced, "label"))["run_id"]
    m.train_baseline(b_run)
    m.train_model(b_run, "random_forest")
    b_check = json.loads(m.check_readiness(b_run))["checks"]["baseline_beaten"]
    assert "accuracy" in b_check["evidence"] and b_check["status"] == "pass", b_check

    shutil.rmtree(m._run_dir(run_id))
    shutil.rmtree(m._run_dir(b_run))
    print(
        "  baseline_gate_uses_imbalance_aware_metric: PR-AUC on imbalanced targets, accuracy on balanced ones"
    )


# --------------------------------------------------------------------------
# 10. Decisions are made on validation, never on the test fold.
# --------------------------------------------------------------------------
def decisions_never_read_the_test_fold() -> None:
    """The threshold, the calibration verdict and the feature-drop ranking
    were all chosen on the test rows, then reported on them — optimistic by
    construction. Decisive check: flip EVERY test label; no decision moves."""
    run_id = json.loads(m.prepare_dataset(_csv(_frame(seed=3), "decide"), "label"))[
        "run_id"
    ]
    m.detect_data_leakage(run_id)
    m.train_model(run_id, "random_forest")
    before = (
        json.loads(m.check_calibration(run_id)),
        json.loads(m.tune_threshold(run_id)),
        json.loads(m.propose_feature_selection(run_id, bottom_k=2)),
    )
    assert "out-of-fold" in before[1]["selected_on"]["validation"]
    assert set(before[1]["chosen"]["ci"]) == {
        "precision",
        "recall",
        "f1",
    }, "the shipped numbers need intervals"

    test_path = m._run_dir(run_id) / "test.csv"
    test = pd.read_csv(test_path)
    test["label"] = 1 - test["label"]
    test.to_csv(test_path, index=False)
    after = (
        json.loads(m.check_calibration(run_id)),
        json.loads(m.tune_threshold(run_id)),
        json.loads(m.propose_feature_selection(run_id, bottom_k=2)),
    )

    assert (
        after[0]["expected_calibration_error"]
        == before[0]["expected_calibration_error"]
    ), "calibration read test"
    assert (
        after[0]["test_expected_calibration_error"]
        != before[0]["test_expected_calibration_error"]
    )
    assert after[1]["chosen"]["threshold"] == before[1]["chosen"]["threshold"], (
        "the threshold was chosen on test"
    )
    assert after[2]["drop_candidates"] == before[2]["drop_candidates"], (
        "feature selection read test"
    )
    shutil.rmtree(m._run_dir(run_id))
    print(
        "  decisions_never_read_the_test_fold: flipping every test label moves no decision"
    )


# --------------------------------------------------------------------------
# 11. The success verdict is judged on the interval, not the point.
# --------------------------------------------------------------------------
def success_verdict_uses_the_interval() -> None:
    """ROC-AUC 0.842 was reported as MISSING 0.85 on a 179-row fold whose
    95% interval was [0.76, 0.90]. That is not a miss; it is unknown."""

    def verdict(point, ci):
        meta = {
            "business_understanding": {
                "success_target": {
                    "metric": "roc_auc",
                    "threshold": 0.85,
                    "direction": ">=",
                }
            },
            "tuned_metrics": {"auc": {"roc_auc": point}, "ci": {"roc_auc": ci}},
        }
        return m.gates._evaluate_success_criteria(meta)

    assert verdict(0.842, [0.76, 0.90])["status"] == "not_run"
    assert "INCONCLUSIVE" in verdict(0.842, [0.76, 0.90])["evidence"]
    assert verdict(0.93, [0.88, 0.97])["status"] == "pass"
    assert verdict(0.70, [0.62, 0.79])["status"] == "fail"
    assert verdict(0.86, [0.80, 0.91])["status"] == "not_run", (
        "a point ABOVE the bar is not a pass either"
    )
    print(
        "  success_verdict_uses_the_interval: pass / fail / inconclusive decided by the 95% CI"
    )


# --------------------------------------------------------------------------
# 12. A column that nearly IS the label blocks; exact duplicates never split.
# --------------------------------------------------------------------------
def near_perfect_predictor_and_duplicates() -> None:
    df = _frame(seed=5)
    # A categorical copy of the label: invisible to the old numeric-only
    # correlation screen, and exactly what a post-outcome status field is.
    df["case_status"] = np.where(df["label"] == 1, "escalated", "closed")
    df = pd.concat([df, df.head(25)], ignore_index=True)  # 25 exact duplicate rows
    prep = json.loads(m.prepare_dataset(_csv(df, "leak"), "label"))
    run_id = prep["run_id"]
    assert prep["duplicates_dropped"] == 25
    train, test, _ = m._load_split(run_id)
    assert not len(train.merge(test, how="inner")), (
        "an exact duplicate row straddles the split"
    )

    leakage = json.loads(m.detect_data_leakage(run_id))
    assert "case_status" in leakage["near_perfect_predictors"], leakage[
        "single_column_auc_top"
    ]
    m.train_model(run_id, "random_forest")
    gate = json.loads(m.check_readiness(run_id))["checks"]["leakage_screened"]
    assert gate["status"] == "fail" and "case_status" in gate["evidence"]

    m.acknowledge_identifier_column(
        run_id, "case_status", "test: pretend a domain expert vouched for it"
    )
    assert (
        json.loads(m.check_readiness(run_id))["checks"]["leakage_screened"]["status"]
        == "pass"
    )
    shutil.rmtree(m._run_dir(run_id))
    print(
        "  near_perfect_predictor_and_duplicates: label-copy blocks until acknowledged; duplicates dropped pre-split"
    )


# --------------------------------------------------------------------------
# 13. The mechanical middle of a run is one deterministic call.
# --------------------------------------------------------------------------
def standard_diagnostics_is_one_ordered_call() -> None:
    run_id = json.loads(m.prepare_dataset(_csv(_frame(seed=7), "suite"), "label"))[
        "run_id"
    ]
    m.detect_data_leakage(run_id)
    m.train_model(run_id, "random_forest")
    result = json.loads(m.run_standard_diagnostics(run_id))
    errors = {k: v["error"] for k, v in result["steps"].items() if v.get("error")}
    assert not errors, errors
    order = [
        e["tool"]
        for e in m._load_meta(run_id)["execution_log"]
        if e["tool"] != "run_standard_diagnostics"
    ]
    expected = [
        "train_baseline",
        "check_calibration",
        "check_label_rule",
        "explain_model",
        "error_analysis",
        "check_model_stability",
        "check_readiness",
    ]
    first_seen = [
        order.index(t) for t in expected
    ]  # calibration may repeat after calibrate_model
    assert first_seen == sorted(first_seen), order
    for gate in (
        "baseline_beaten",
        "calibration_checked",
        "explainability_run",
        "error_analysis",
        "stability_checked",
    ):
        assert (
            json.loads(m.check_readiness(run_id))["checks"][gate]["status"] != "not_run"
        ), gate
    assert "tune_threshold" in result["remaining"][0]
    shutil.rmtree(m._run_dir(run_id))
    print(
        "  standard_diagnostics_is_one_ordered_call: every mechanical gate armed by one call, in order"
    )


def main() -> None:
    identifier_ratio_gap()
    missing_evidence_is_not_a_pass()
    identifier_in_model_blocks()
    explainability_pollution_detected()
    success_criteria_cannot_be_invented()
    execution_ledger_records_reality()
    reflection_contradiction_detected()
    refit_invalidates_stale_results()
    baseline_gate_uses_imbalance_aware_metric()
    decisions_never_read_the_test_fold()
    success_verdict_uses_the_interval()
    near_perfect_predictor_and_duplicates()
    standard_diagnostics_is_one_ordered_call()
    print("all gate checks passed")


if __name__ == "__main__":
    main()
