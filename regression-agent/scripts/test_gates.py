"""Proves the readiness gates actually gate — that each one fails on the
failure it names, and that a gate which never ran is never mistaken for a
gate that passed.

The fixture carries two real planted faults: an identifier column
(`customer_id`) that would be one-hot expanded into pure memorization, and a
transformed copy of the target (`spend_in_cents`) that a model would happily
"predict" the target from. A run holding either must not be exportable.

`usage` is the control: it drives the target (0.985 rank correlation) and is a
genuine feature, so a leakage rule that flags it is broken.

No MCP transport and no LLM — calls the tool functions directly.

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
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import mcp_server as m


def _fixture() -> str:
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "customer_id": [
                f"C{i:05d}" for i in range(n)
            ],  # identifier — must be caught
            "region": rng.choice(list("abcd"), n),
            "usage": rng.normal(50, 10, n),  # genuine driver — must NOT be caught
        }
    )
    df["spend"] = 3 * df["usage"] + rng.normal(0, 5, n)
    df["spend_in_cents"] = df["spend"] * 100  # a copy of the target — must be caught
    path = str(Path(tempfile.mkdtemp()) / "gate_fixture.csv")
    df.to_csv(path, index=False)
    return path


def test_gates():
    run = json.loads(m.prepare_dataset(_fixture(), "spend"))["run_id"]
    readiness = lambda: json.loads(m.check_readiness(run))  # noqa: E731

    # Before anything runs: the identifier is already a failure (the check
    # reads the training fold, not a tool's output), and every gate whose
    # evidence nobody produced is not_run — which is never a pass.
    start = readiness()
    assert start["overall_status"] == "blocked", start["overall_status"]
    assert start["failed_gates"] == ["no_identifier_in_model"], start["failed_gates"]
    assert {"business_context", "baseline_beaten", "success_criteria"} <= set(
        start["not_run_gates"]
    )
    assert all(start["checks"][g]["status"] != "pass" for g in start["not_run_gates"])

    # A success metric this agent cannot measure is rejected when it is SET,
    # not silently ignored when the report is written.
    rejected = json.loads(
        m.record_business_context(
            run,
            "telecom",
            "forecast spend",
            "monthly spend in EUR",
            success_metric="auc",
            success_threshold=0.9,
        )
    )
    assert "error" in rejected, rejected

    m.record_business_context(
        run,
        "telecom",
        "forecast monthly spend",
        "monthly spend in EUR",
        success_metric="r2",
        success_threshold=0.95,
    )
    m.detect_data_leakage(run)
    m.train_model(run, "random_forest")
    for step in (
        m.train_baseline,
        m.check_residuals,
        m.check_model_stability,
        m.explain_model,
    ):
        step(run)

    # Both planted faults are FAILURES, with the evidence naming the column.
    dirty = readiness()
    assert dirty["overall_status"] == "blocked", dirty
    assert {"leakage_screened", "no_identifier_in_model"} <= set(
        dirty["failed_gates"]
    ), dirty["failed_gates"]
    assert "spend_in_cents" in dirty["checks"]["leakage_screened"]["evidence"]
    assert "usage" not in dirty["checks"]["leakage_screened"]["evidence"], (
        "flagged a genuine driver as leakage"
    )

    # ...and a blocked run cannot be exported.
    blocked_export = Path(tempfile.mkdtemp()) / "should_not_exist.pkl"
    refused = json.loads(m.export_model(run, str(blocked_export)))
    assert "error" in refused and refused["failed_gates"], refused
    assert not blocked_export.exists()

    # Drop the faults, re-screen, retrain: the gates clear.
    m.apply_drop_columns(run, json.dumps(["spend_in_cents", "customer_id"]))
    m.detect_data_leakage(run)
    m.train_model(run, "random_forest")
    for step in (
        m.train_baseline,
        m.check_residuals,
        m.check_model_stability,
        m.explain_model,
    ):
        step(run)

    clean = readiness()
    assert not clean["failed_gates"], clean["failed_gates"]
    assert clean["overall_status"] == "ready", (
        clean["overall_status"],
        clean["not_run_gates"],
    )
    assert clean["gates_passed"] == clean["gates_total"] == len(m.REQUIRED_GATES)
    # The bar is the RECORDED one, judged either way — never invented.
    assert clean["checks"]["success_criteria"]["status"] in ("pass", "fail")
    assert "0.95" in clean["checks"]["success_criteria"]["evidence"]

    # The ledger recorded what actually ran.
    meta = json.loads((m._run_dir(run) / "meta.json").read_text())
    tools = [entry["tool"] for entry in meta["execution_log"]]
    assert {"train_model", "detect_data_leakage", "train_baseline"} <= set(tools), tools

    # A screen that predates a change to the feature set is stale, and says so
    # — the failure no other gate can see.
    m.apply_drop_columns(run, json.dumps(["region"]))
    stale = readiness()["checks"]["evidence_ordering"]
    assert stale["status"] == "fail" and "apply_drop_columns" in stale["evidence"], (
        stale
    )


def test_no_success_bar_is_reported_not_invented():
    run = json.loads(m.prepare_dataset(_fixture(), "spend"))["run_id"]
    m.record_business_context(run, "telecom", "forecast spend", "monthly spend in EUR")
    criteria = json.loads(m.check_readiness(run))["checks"]["success_criteria"]
    assert criteria["status"] == "not_run"
    assert "must NOT be invented" in criteria["evidence"]


if __name__ == "__main__":
    test_gates()
    test_no_success_bar_is_reported_not_invented()
    print("gates OK")
