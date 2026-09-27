"""Cross-agent integration test: the handoffs, not the agents.

Every agent has its own unit tests and they all pass in isolation. The bugs
that actually shipped lived in the SEAMS — a chart that plotted a different
class from the report that embedded it, a tuned threshold that serving
dropped on the floor, a drift baseline that got swept a week after
deployment. None of those are visible from inside one agent.

So this drives one dataset through the whole chain and asserts the claims
that cross a boundary:

    classification -> visualization   the chart agrees with the report
    classification -> serving         the reviewed model is the shipped model
    classification -> explain         the reason cites the shipped threshold
    classification -> drift           the baseline outlives the run directory

The dataset is deliberately WORD-LABELLED ({"fraud", "legit"}), because
"legit" sorts above "fraud" and that is the exact shape that silently
inverted every positive-class number in this repo. Labels like {0, 1} would
pass all of this while proving nothing.

No network and no LLM: the MCP tool functions are called directly. That
makes this a test of the tools and the contracts between them, not of any
model's ability to choose them.

Run:  python -m pytest scripts/test_swarm_e2e.py -q
      (needs the agentic-ml env — every agent's deps in one interpreter)
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

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
POSITIVE, NEGATIVE = "fraud", "legit"  # "legit" > "fraud" — the whole point


def _load(name: str, relpath: str):
    """Each agent ships its own scripts/mcp_server/ package AND its own
    pipeline_transformers.py — deliberately duplicated, since agents are
    independently deployable and never import each other.

    In one interpreter that collides: whichever agent loads first wins the
    bare name `pipeline_transformers`, and the next agent silently gets the
    wrong copy (classification's has no ForecastModel, serving's does). So
    each module is loaded with its own directory at the front of sys.path
    and the shared name evicted from the module cache first."""
    package = ROOT / relpath  # an agent's scripts/mcp_server/ package
    agent_dir = str(package.parent)
    sys.modules.pop("pipeline_transformers", None)
    sys.path.insert(0, agent_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            name, package / "__init__.py", submodule_search_locations=[str(package)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = (
            module  # before exec: its `from .core import ...` resolve through it
        )
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(agent_dir)
        sys.modules.pop("pipeline_transformers", None)


@pytest.fixture(scope="module")
def agents():
    loaded = {
        "visualization": _load("viz_ms", "visualization-agent/scripts/mcp_server"),
        "serving": _load("serving_ms", "serving-agent/scripts/mcp_server"),
        "explain": _load("explain_ms", "explain-agent/scripts/mcp_server"),
        "drift": _load("drift_ms", "drift-agent/scripts/mcp_server"),
    }
    # classification is a package (mcp_server/), not a single file — imported
    # last so its own pipeline_transformers is the one left resolvable for the
    # unpickling its tools do.
    sys.path.insert(0, str(ROOT / "classification-agent/scripts"))
    sys.modules.pop("pipeline_transformers", None)
    import mcp_server as classification

    return {
        "classification": classification,
        **loaded,
    }


@pytest.fixture(scope="module")
def wangiri_csv(tmp_path_factory):
    """Wangiri-shaped and deliberately HARD: the fraud signal overlaps the
    normal population. A perfectly separable toy passes every assertion here
    while hiding the behaviour that matters — on separable data the tuned
    threshold lands at the top of the score distribution and every margin is
    zero."""
    tmp = tmp_path_factory.mktemp("swarm")
    rng = np.random.default_rng(11)
    n = 4000
    fraud = rng.random(n) < 0.05
    df = pd.DataFrame(
        {
            "calls_premium_24h": rng.poisson(np.where(fraud, 4.0, 2.2)),
            "avg_call_sec": rng.normal(np.where(fraud, 55, 80), 40).clip(1),
            "distinct_callees": rng.poisson(np.where(fraud, 14, 9)),
            "tenure_days": rng.integers(1, 2000, n),
            "plan": rng.choice(["prepaid", "postpaid"], n),
            "label": np.where(fraud, POSITIVE, NEGATIVE),
        }
    )
    path = tmp / "wangiri.csv"
    df.to_csv(path, index=False)
    return tmp, path


@pytest.fixture(scope="module")
def trained(agents, wangiri_csv):
    """A full classification run, ending in an export."""
    tmp, path = wangiri_csv
    c = agents["classification"]

    run = json.loads(c.prepare_dataset(str(path), "label"))
    json.loads(
        c.record_business_context(
            run["run_id"],
            "flag wangiri fraud for blocking",
            f"label={POSITIVE} means confirmed wangiri",
            "recall >= 0.80 on the fraud class",
            domain="telecom",
            label_provenance="investigated",
            success_metric="recall_positive",
            success_threshold="0.80",
        )
    )
    run_id = run["run_id"]

    json.loads(c.detect_data_leakage(run_id))
    json.loads(c.train_baseline(run_id))
    json.loads(c.train_model(run_id, "random_forest"))
    calibration = json.loads(c.check_calibration(run_id))
    if calibration.get("miscalibration_warning"):
        json.loads(c.calibrate_model(run_id, "isotonic"))
        calibration = json.loads(c.check_calibration(run_id))
    json.loads(c.explain_model(run_id, method="shap"))
    threshold = json.loads(c.tune_threshold(run_id, target_recall=0.8))

    pkl = str(tmp / "wangiri_model.pkl")
    export = json.loads(c.export_model(run_id, pkl, force=True))

    # An unlabelled production file — what actually arrives at serving.
    production = pd.read_csv(path).drop(columns=["label"]).sample(400, random_state=3)
    production_path = tmp / "today.csv"
    production.to_csv(production_path, index=False)

    yield {
        "run_id": run_id,
        "prepare": run,
        "pkl": pkl,
        "export": export,
        "threshold": threshold["chosen"]["threshold"],
        "tune": threshold,
        "calibration": calibration,
        "production": str(production_path),
        "tmp": tmp,
        "source": str(path),
    }
    # serving appends every scored batch to the repo's data/predictions/<run_id>.csv
    predictions_log = json.loads(Path(export["monitoring_profile"]).read_text())[
        "predictions_log"
    ]
    Path(predictions_log).unlink(missing_ok=True)


# ───────────────────────────────────────────── the positive class, everywhere


def test_positive_class_is_the_event_not_the_higher_sorting_label(trained, agents):
    """`legit` sorts above `fraud`. Every agent must still mean `fraud`."""
    assert trained["prepare"]["positive_label"] == POSITIVE
    assert trained["prepare"]["positive_label_inferred"] is True
    assert trained["calibration"]["positive_label"] == POSITIVE
    assert trained["tune"]["positive_label"] == POSITIVE

    profile = json.loads(Path(trained["export"]["monitoring_profile"]).read_text())
    assert profile["positive_label"] == POSITIVE


def test_chart_and_report_describe_the_same_class_and_the_same_model(trained, agents):
    """The failure this catches: a report whose text described fraud and
    whose embedded ROC curve described legit. Both looked fine alone."""
    c, viz = agents["classification"], agents["visualization"]
    run_id = trained["run_id"]

    roc = json.loads(viz.plot_roc_curve(run_id, "roc.png"))
    assert roc["auc"]["positive_label"] == POSITIVE
    assert (
        json.loads(viz.plot_pr_curve(run_id, "pr.png"))["ap"]["positive_label"]
        == POSITIVE
    )

    _, _, meta = c._load_split(run_id)
    reported = (meta.get("tuned_metrics") or meta["baseline_metrics"])["auc"]["roc_auc"]
    assert abs(reported - roc["auc"]["positive_class"]) < 0.02, (
        f"chart AUC {roc['auc']['positive_class']} disagrees with reported {reported}"
    )

    # Charts must not collide across runs: a report links a URL, so a later
    # run's "roc.png" must not overwrite the image under it.
    assert run_id in roc["out_path"]


def test_feature_importance_chart_comes_from_the_artifact(trained, agents):
    viz = agents["visualization"]
    from_run = json.loads(
        viz.plot_feature_importance("fi.png", run_id=trained["run_id"])
    )
    assert "run artifact" in from_run["source"]

    # The caller-supplied path still exists, but says what it is.
    supplied = json.loads(
        viz.plot_feature_importance("fi2.png", importances='[["invented", 9.9]]')
    )
    assert "NOT verified" in supplied["source"]


def test_review_charts_show_the_shipped_decision_and_satisfy_the_gate(trained, agents):
    """The reviewer used to sign off a confusion matrix drawn at 0.5 while
    serving shipped the tuned cutoff — two decision rules under one name. And
    charts that never reached the run could be neither audited nor gated."""
    import joblib
    from sklearn.metrics import confusion_matrix

    c, viz = agents["classification"], agents["visualization"]
    run_id = trained["run_id"]

    cm = json.loads(viz.plot_confusion_matrix(run_id, "cm.png"))
    assert abs(cm["threshold"] - trained["threshold"]) < 1e-9, (
        "confusion matrix drawn at a cutoff serving doesn't use"
    )
    viz.plot_roc_curve(run_id, "roc.png")
    viz.plot_pr_curve(run_id, "pr.png")

    gate = json.loads(c.check_readiness(run_id))["checks"]["evaluation_charts"]
    assert gate["status"] == "pass", gate

    # Decisive: the report's matrix is the shipped threshold applied by hand.
    _, test, meta = c._load_split(run_id)
    pipeline = joblib.load(c._run_dir(run_id) / "pipeline.pkl")
    proba = pipeline.predict_proba(test.drop(columns=["label"]))[
        :, list(pipeline.classes_).index(POSITIVE)
    ]
    shipped = np.where(proba >= trained["threshold"], POSITIVE, NEGATIVE)
    expected = confusion_matrix(
        test["label"], shipped, labels=sorted(test["label"].unique())
    ).tolist()
    report = c.generate_report(run_id)
    assert "at the shipped threshold" in report and str(expected) in report
    assert f"](charts/pv{meta['pipeline_version']}_confusion_matrix.png)" in report, (
        "report must embed the run's chart"
    )


# ─────────────────────────────────── the reviewed model is the shipped model


def test_every_prediction_names_its_model_and_lands_in_the_log(trained, agents):
    """A prediction nobody can trace to a model version can't be audited,
    rolled back or monitored — and drift needs what production actually saw."""
    serving = agents["serving"]
    first = json.loads(serving.predict(trained["pkl"], trained["production"]))
    assert first["model"]["run_id"] == trained["run_id"]
    assert (pd.read_csv(first["out_path"])["model_run_id"] == trained["run_id"]).all()

    log = Path(first["predictions_log"])
    before = len(pd.read_csv(log))
    serving.predict(trained["pkl"], trained["production"])
    logged = pd.read_csv(log)
    assert len(logged) == before + 400, "each batch must append, not overwrite"
    assert (logged["model_run_id"] == trained["run_id"]).all() and logged[
        "scored_at"
    ].notna().all()

    # That log is the `current` side of drift when no file is given.
    profile_path = trained["export"]["monitoring_profile"]
    assert json.loads(Path(profile_path).read_text())["predictions_log"] == str(log)
    result = json.loads(agents["drift"].detect_drift(profile_path=profile_path))
    assert "error" not in result, result


def test_serving_applies_the_tuned_threshold_not_sklearns_default(trained, agents):
    """A model signed off at recall 0.80 must not ship at 0.5."""
    serving = agents["serving"]

    inspected = json.loads(serving.inspect_model(trained["pkl"]))
    assert inspected["operating_point"] is not None, (
        "serving cannot apply a threshold it cannot see"
    )
    assert "exported threshold" in inspected["decision_rule"]

    result = json.loads(
        serving.predict(trained["pkl"], trained["production"], review_threshold=0.05)
    )
    rule = result["decision_rule"]
    assert abs(rule["threshold"] - trained["threshold"]) < 1e-9
    assert rule["positive_class"] == POSITIVE

    # Decisive: serving's labels must equal thresholding the probabilities by
    # hand. Anything else means it used a different rule than it reported.
    out = pd.read_csv(result["out_path"])
    manual = np.where(out["positive_proba"] >= trained["threshold"], POSITIVE, NEGATIVE)
    assert (out["prediction"].values == manual).all()

    # The review queue bands the boundary rather than filtering max-class
    # confidence, which on a 5%-prevalence target queues the wrong rows.
    assert "decision threshold" in result["review_queue"]["basis"]


# ──────────────────────────────────────────────── why was THIS one flagged


def test_explanation_cites_the_shipped_threshold_and_checks_itself(trained, agents):
    explain, serving = agents["explain"], agents["serving"]

    inspected = json.loads(explain.inspect_explainability(trained["pkl"]))
    assert inspected["explainable"] is True
    assert inspected["reference_path"], (
        "no background distribution -> no defensible explanation"
    )

    scored = pd.read_csv(
        json.loads(serving.predict(trained["pkl"], trained["production"]))["out_path"]
    )
    flagged = scored.index[scored["prediction"] == POSITIVE].tolist()
    assert flagged, "no row was flagged — nothing to explain"

    result = json.loads(
        explain.explain_prediction(
            trained["pkl"], trained["production"], row=int(flagged[0])
        )
    )
    assert "error" not in result, result.get("error")
    assert result["positive_class"] == POSITIVE
    assert abs(result["threshold"] - trained["threshold"]) < 1e-9, (
        "an explanation anchored to 0.5 answers a question nobody asked"
    )
    assert result["decision"] == POSITIVE
    assert result["margin_over_threshold"] >= 0
    assert result["contributions"], (
        "a flagged row with no contributions is not an explanation"
    )

    # The two self-checks must be present and must not be silently false.
    assert result["additivity_ok"] is not False, (
        "contributions do not reconstruct this model's score"
    )
    assert "explanation_stability" in result


def test_counterfactual_never_ships_without_its_limitations(trained, agents):
    explain, serving = agents["explain"], agents["serving"]
    scored = pd.read_csv(
        json.loads(serving.predict(trained["pkl"], trained["production"]))["out_path"]
    )
    flagged = scored.index[scored["prediction"] == POSITIVE].tolist()

    result = json.loads(
        explain.explain_counterfactual(
            trained["pkl"], trained["production"], row=int(flagged[0])
        )
    )
    assert "error" not in result, result.get("error")
    # Association-not-causation, median-targeted, greedy, correlation-blind.
    # These go to people affected by decisions; dropping them is the harm.
    assert len(result["limitations"]) >= 4
    assert any("causation" in limit for limit in result["limitations"])


# ────────────────────────────────────── monitoring outlives the run directory


def test_drift_baseline_survives_the_run_being_swept(trained, agents):
    """data/runs/<run_id>/ is disposable and gets swept on a retention timer.
    If that is the only copy of the training fold, drift monitoring becomes
    impossible about a week after deployment — silently, and exactly when it
    starts to matter."""
    drift = agents["drift"]
    profile_path = trained["export"]["monitoring_profile"]
    profile = json.loads(Path(profile_path).read_text())

    assert Path(profile["reference_path"]).exists(), (
        "the baseline must live beside the .pkl, not only in the run dir"
    )
    assert profile["key_columns"], (
        "without key columns every column votes equally on the verdict"
    )
    # SHAP ranks encoded features (plan_prepaid); drift compares raw columns
    # (plan). An encoded name matches nothing and silently stops counting.
    assert set(profile["key_columns"]) <= set(profile["feature_columns"]), profile[
        "key_columns"
    ]

    current = pd.read_csv(trained["source"]).drop(columns=["label"])
    current["avg_call_sec"] = current["avg_call_sec"] * 2.5  # a column the model uses
    current_path = trained["tmp"] / "today_drifted.csv"
    current.to_csv(current_path, index=False)

    result = json.loads(
        drift.detect_drift(current_path=str(current_path), profile_path=profile_path)
    )
    assert result["reference_path"] == profile["reference_path"]
    assert "avg_call_sec" in result["drifted_key_columns"]
    assert result["overall_severity"] in ("moderate", "severe")

    # It must never imply it checked the model. It compared two tables.
    assert "DATA drift check" in result["scope_caveat"]
    assert len(result["what_this_did_not_check"]) == 3


def test_drift_verdict_is_not_hijacked_by_a_column_the_model_ignores(trained, agents):
    """Max-severity over every column means one drifting unmodelled column
    produces a permanent 'retrain now', and an alarm that always fires is an
    alarm nobody reads."""
    drift = agents["drift"]
    profile_path = trained["export"]["monitoring_profile"]
    profile = json.loads(Path(profile_path).read_text())

    current = pd.read_csv(trained["source"]).drop(columns=["label"])
    current["ignored_column"] = 1.0  # not in the model at all
    reference = pd.read_csv(profile["reference_path"])
    reference["ignored_column"] = np.random.default_rng(0).normal(
        500, 100, len(reference)
    )
    ref_path = trained["tmp"] / "ref_with_extra.csv"
    reference.to_csv(ref_path, index=False)
    cur_path = trained["tmp"] / "cur_with_extra.csv"
    current.to_csv(cur_path, index=False)

    result = json.loads(
        drift.detect_drift(
            reference_path=str(ref_path),
            current_path=str(cur_path),
            key_columns=",".join(profile["key_columns"]),
            target_column=profile["target"],
        )
    )

    assert "ignored_column" in result["drifted_incidental_columns"], (
        "it drifted and must still be reported"
    )
    assert "ignored_column" not in result["drifted_key_columns"]
    assert result["overall_severity"] == "none", (
        "a column the model ignores must not drive the verdict"
    )


# ─────────────────────────────────────────────────────── the gate still bites


def test_readiness_gate_blocks_an_export_it_has_evidence_against(trained, agents):
    """The gate is the thing that makes the rest believable: it computes from
    artifacts and can veto the model regardless of what any agent claims."""
    c = agents["classification"]
    readiness = json.loads(c.check_readiness(trained["run_id"]))

    assert "calibration_checked" in readiness["checks"], (
        "a tuned threshold needs measured calibration behind it"
    )
    assert readiness["checks"]["calibration_checked"]["status"] in (
        "pass",
        "not_applicable",
    )

    # Absence of evidence is reported as absence, never as a pass.
    for name, check in readiness["checks"].items():
        assert check["status"] in ("pass", "fail", "not_run", "not_applicable"), name
        assert check["evidence"], f"{name} passed or failed without saying why"


# ───────────────────────────────────────────── the MLflow record is auditable


def test_mlflow_record_carries_lineage_and_row_scores(
    trained, agents, tmp_path, monkeypatch
):
    """A registry version must say which data bytes and which test rows it
    came from, whether the code that ran was committed, and carry the per-row
    scores that let two versions be differenced with an interval."""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    c = agents["classification"]
    logged = json.loads(c.log_run_to_mlflow(trained["run_id"]))
    assert "version" in logged["registered"], logged

    run = mlflow.tracking.MlflowClient(
        tracking_uri=f"file:{tmp_path / 'mlruns'}"
    ).get_run(logged["mlflow_run_id"])
    meta = c._load_meta(trained["run_id"])
    assert run.data.tags["data_sha256"] == meta["source_sha256"] != "unrecorded"
    assert run.data.tags["test_fingerprint"] == meta["test_sha256"]
    assert run.data.tags["git_dirty"].split(" ")[0] in ("true", "false", "unknown")
    assert (
        run.data.metrics["roc_auc_ci_low"]
        <= run.data.metrics["roc_auc"]
        <= run.data.metrics["roc_auc_ci_high"]
    )
    assert "cv_score" in run.data.metrics

    scores = pd.read_csv(
        mlflow.artifacts.download_artifacts(
            run_id=logged["mlflow_run_id"],
            artifact_path="test_scores.csv",
            dst_path=str(tmp_path / "dl"),
        )
    )
    _, test, _ = c._load_split(trained["run_id"])
    assert len(scores) == len(test) and set(scores["y_true"]) <= {0, 1}
