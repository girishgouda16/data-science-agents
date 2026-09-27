"""Production-scale end-to-end validation: 3.5M rows, 2.5% positive.

SLOW (~5-6 minutes) and requires `data/wangiri_final.csv`, so it is NOT part
of the fast suite — run it deliberately:

    python test_production_wangiri.py

The fixture suites (test_mcp_server.py, test_gates.py) prove the logic. This
proves the logic survives contact with a real file, which is a different
question: fixtures are small, clean, balanced and have tidy column names, and
every one of those four properties is one the gates could be accidentally
depending on.

Wangiri is the right adversary because it carries two *distinct* identifier
hazards at once, and the agent has to handle them differently:

  1. `FROM` is a hashed caller id — 2.28M distinct values, 0.649 uniqueness
     ratio. It clears the near-unique bar entirely, so only the absolute
     high-cardinality rule catches it. As a model feature it is useless.
  2. the same caller RECURS across the 1/3/7-day lag rows, so a plain
     per-row split puts the same entity in train and test. That is group
     leakage — a separate failure from the column being useless, and it is
     not fixed by dropping the column, only by splitting on it.

Plus the whole run sits at a 2.5% positive rate, where accuracy (97.5% for a
model that does nothing) and ROC-AUC are both actively misleading and PR-AUC
is the only headline worth printing.
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
import sys
import time
from pathlib import Path

import mcp_server as m

CSV = Path(__file__).resolve().parents[2] / "data" / "wangiri_final.csv"
TARGET = "Label"
_t0 = time.time()


def step(name: str) -> None:
    print(f"[{time.time() - _t0:7.1f}s] {name}", flush=True)


def main() -> None:
    if not CSV.exists():
        print(f"SKIP — {CSV} not present")
        return
    path = str(CSV)

    step("inspect_target")
    info = json.loads(m.inspect_target(path, TARGET))
    assert info["is_binary"], info
    assert info["class_pct"]["1"] < 5, (
        "fixture assumption: this is a severely imbalanced file"
    )

    step("propose_group_column — the ENTITY-leakage hazard")
    groups = json.loads(m.propose_group_column(path, TARGET))["group_column_candidates"]
    assert "FROM" in groups, "the recurring caller id must be offered as a group column"

    step("prepare_dataset (grouped split — no caller in both folds)")
    prep = json.loads(m.prepare_dataset(path, TARGET, group_column="FROM"))
    run_id = prep["run_id"]
    print(f"          train={prep['train_shape']} test={prep['test_shape']}")

    step("detect_data_leakage — the COLUMN hazard")
    leakage = json.loads(m.detect_data_leakage(run_id))
    assert "FROM" in leakage["strong_identifier_columns"], leakage

    step("check_readiness BEFORE the fix — must BLOCK on the identifier")
    pre = json.loads(m.check_readiness(run_id))
    assert pre["checks"]["no_identifier_in_model"]["status"] == "fail"
    assert pre["overall_status"] == "blocked", pre["overall_status"]

    step("record_business_context — telecom domain + a MEASURABLE bar")
    m.record_business_context(
        run_id,
        business_objective="Flag Wangiri (one-ring) fraud callers for blocking before they generate "
        "revenue-share callbacks.",
        target_definition="Label=1 means confirmed Wangiri fraud in this window; one row is one caller.",
        success_criteria="PR-AUC primary at a 2.5% positive rate — ROC-AUC is optimistic under this much "
        "imbalance. Bar PR-AUC >= 0.50, well above the 0.025 no-skill line.",
        domain="telecom",
        success_metric="pr_auc",
        success_threshold="0.50",
        assumptions=json.dumps(
            [
                "No business objective supplied with the file; framed from the telecom domain reference.",
                "Grouped split on FROM taken without asking — measurable, not a judgment call.",
            ]
        ),
    )

    step("apply_drop_columns([FROM]) + re-screen the changed feature set")
    m.apply_drop_columns(run_id, json.dumps(["FROM"]))
    m.detect_data_leakage(run_id)

    step("check_imbalance")
    imb = json.loads(m.check_imbalance(run_id))
    assert imb["is_imbalanced"], imb

    step("train_baseline + compare_models")
    m.train_baseline(run_id)
    cmp = json.loads(m.compare_models(run_id))
    print(
        f"          metric={cmp['cv_metric']} ranked={[(r['model'], r['cv_score_mean']) for r in cmp['ranked']]}"
    )
    # The whole point of the adaptive metric: at a 2.5% positive rate ranking by
    # f1_weighted scored every candidate 0.971-0.982 — indistinguishable from
    # each other and from a model that predicts nothing but the majority class.
    assert cmp["cv_metric"] == "average_precision", (
        "an imbalanced binary target must be ranked by PR-AUC, not f1_weighted"
    )

    step(f"train_model({cmp['recommended']})")
    trained = json.loads(m.train_model(run_id, cmp["recommended"]))
    assert trained["cv_metric"] == "average_precision"

    step(
        "run_standard_diagnostics — baseline, calibration, label rule, SHAP, errors, stability in one call"
    )
    suite = json.loads(m.run_standard_diagnostics(run_id))
    assert not {k: v["error"] for k, v in suite["steps"].items() if v.get("error")}, (
        suite["steps"]
    )
    print(
        f"          label rule: {suite['steps']['check_label_rule'].get('interpretation', '')[:110]}"
    )

    step("tune_threshold — chosen on out-of-fold predictions, measured once on test")
    op = json.loads(m.tune_threshold(run_id))
    assert "out-of-fold" in op["selected_on"]["validation"], op["selected_on"]

    step("feature engineering — declared, with a reason on the record")
    m.declare_feature_engineering_not_applicable(
        run_id,
        "The file is already per-caller call-pattern aggregates; the domain features are the columns.",
    )

    step("fairness — declared not applicable, with a reason on the record")
    m.declare_fairness_not_applicable(
        run_id,
        "Schema is call-volume aggregates keyed by a hashed caller id — no demographic or "
        "geographic attribute, and no proxy for one.",
    )

    step("record_reflection")
    reflection = json.loads(
        m.record_reflection(
            run_id,
            checks=json.dumps(
                {
                    k: {
                        "status": "clear",
                        "evidence": "verified against this run's artifacts",
                    }
                    for k in m.diagnostics.REFLECTION_CHECKS
                    if k != "fairness_assessment_status"
                }
                | {
                    "fairness_assessment_status": {
                        "status": "not_applicable",
                        "evidence": "no protected attribute in schema",
                    }
                }
            ),
        )
    )
    assert not reflection["contradictions"], reflection["contradictions"]

    step(
        "evaluation charts — stand-ins for the visualization agent's files for this fit"
    )
    pv = m._load_meta(run_id)["pipeline_version"]
    (m._run_dir(run_id) / "charts").mkdir(exist_ok=True)
    for kind in m.gates.REQUIRED_CHARTS:
        (m._run_dir(run_id) / "charts" / f"pv{pv}_{kind}.png").write_bytes(b"")

    step("check_readiness FINAL")
    final = json.loads(m.check_readiness(run_id))
    for name, c in final["checks"].items():
        print(f"            {c['status']:15s} {name}")
    assert final["overall_status"] == "ready", (
        final["failed_gates"],
        final["not_run_gates"],
    )

    step("generate_report")
    report = m.generate_report(run_id)
    metrics = (
        m._load_meta(run_id)["tuned_metrics"]
        or m._load_meta(run_id)["baseline_metrics"]
    )
    roc, pr = metrics["auc"]["roc_auc"], metrics["auc"]["pr_auc"]

    # The headline must lead with PR-AUC and must actively warn that accuracy
    # and ROC-AUC flatter this target, rather than printing the nice number.
    assert f"PR-AUC {pr}" in report and "the metric that matters" in report
    assert "Read accuracy with care" in report
    assert "FROM" not in report.split("## Explainability")[1].split("##")[0], (
        "the dropped identifier must not appear among the model's drivers"
    )

    print(f"\n{'=' * 68}")
    print(f"run_id     {run_id}")
    print(
        f"readiness  {final['overall_status']} ({final['gates_passed']}/{final['gates_total']})"
    )
    print(f"ROC-AUC    {roc}   <- flattered by the 2.5% positive rate")
    print(f"PR-AUC     {pr}   <- the honest headline")
    print(f"elapsed    {time.time() - _t0:.1f}s")
    print(f"{'=' * 68}")
    print("production validation passed")


if __name__ == "__main__":
    sys.exit(main())
