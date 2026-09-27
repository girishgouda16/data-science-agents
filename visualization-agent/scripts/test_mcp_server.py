"""Smoke test for the plotting tools — calls functions directly, checks each
PNG actually gets written with non-trivial size. Also proves the
cross-agent-safety fixes: an invalid/path-traversal run_id is rejected
before touching the filesystem, and a run_id belonging to a non-classifier
model (or a meta.json shaped like clustering/anomaly-agent's, not
classification's) gets a clean error instead of a crash or a garbage chart.

Run: python test_mcp_server/
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
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import mcp_server as m

REPO_ROOT = Path(__file__).parent.parent.parent
TITANIC = str(REPO_ROOT / "data" / "titanic.csv")

# Built from sklearn's bundled dataset rather than a data/iris.csv file —
# that file doesn't actually exist in this repo (the test was silently
# broken on FileNotFoundError before this fix). sklearn is already a
# dependency; no new file to keep in sync or forget to commit.
_iris = load_iris(as_frame=True).frame
_iris["species"] = _iris.pop("target").map(dict(enumerate(load_iris().target_names)))
_iris_fd, IRIS = tempfile.mkstemp(suffix=".csv")
_iris.to_csv(IRIS, index=False)


def _check(out_path):
    assert os.path.exists(out_path), f"{out_path} was not created"
    assert os.path.getsize(out_path) > 1000, (
        f"{out_path} looks empty ({os.path.getsize(out_path)} bytes)"
    )


def _new_run_dir() -> tuple[str, Path]:
    run_id = str(uuid.uuid4())
    run_dir = m.RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    return run_id, run_dir


def _make_run(target: str = "Survived") -> str:
    """Builds a minimal run in the shared RUNS_DIR — the same file contract
    classification-agent/scripts/mcp_server/'s prepare_dataset/train_model
    produce (run_id/{train.csv, test.csv, meta.json, pipeline.pkl}) — so
    plot_confusion_matrix/plot_roc_curve are exercised against the real
    cross-agent handoff, not a shortcut only this test understands. Bare
    classifier, not a real encode+classifier Pipeline — fine for
    predict/predict_proba-only tools, NOT for plot_shap_beeswarm (see
    _make_shap_run)."""
    df = pd.read_csv(TITANIC)[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    df["Sex"] = (df["Sex"] == "male").astype(int)
    train, test = train_test_split(
        df, test_size=0.2, stratify=df[target], random_state=42
    )

    run_id, run_dir = _new_run_dir()
    train.to_csv(run_dir / "train.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)

    clf = RandomForestClassifier(
        n_estimators=50, random_state=42, class_weight="balanced"
    )
    clf.fit(train.drop(columns=[target]), train[target])
    joblib.dump(clf, run_dir / "pipeline.pkl")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "target": target,
                "model": "random_forest",
                "best_params": {"classifier__n_estimators": 50},
            }
        )
    )
    return run_id


def _make_multiclass_run(target: str = "species") -> str:
    """3-class target (iris) — exercises the one-vs-rest branches of
    plot_roc_curve/plot_pr_curve that a binary run never touches."""
    df = pd.read_csv(IRIS)
    train, test = train_test_split(
        df, test_size=0.3, stratify=df[target], random_state=42
    )

    run_id, run_dir = _new_run_dir()
    train.to_csv(run_dir / "train.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)

    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(train.drop(columns=[target]), train[target])
    joblib.dump(clf, run_dir / "pipeline.pkl")
    (run_dir / "meta.json").write_text(
        json.dumps({"target": target, "model": "random_forest", "best_params": None})
    )
    return run_id


def _make_shap_run(target: str = "Survived") -> str:
    """A real encode+classifier sklearn Pipeline — not _make_run's bare
    classifier. plot_shap_beeswarm reads pipeline.named_steps["encode"]/
    ["classifier"] directly, so it needs the actual shape
    classification-agent's _build_pipeline produces."""
    df = pd.read_csv(TITANIC)[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    train, test = train_test_split(
        df, test_size=0.2, stratify=df[target], random_state=42
    )

    run_id, run_dir = _new_run_dir()
    train.to_csv(run_dir / "train.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)

    encoder = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Sex"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        [
            ("encode", encoder),
            ("classifier", RandomForestClassifier(n_estimators=30, random_state=42)),
        ]
    )
    pipeline.fit(train.drop(columns=[target]), train[target])
    joblib.dump(pipeline, run_dir / "pipeline.pkl")
    (run_dir / "meta.json").write_text(
        json.dumps({"target": target, "model": "random_forest", "best_params": None})
    )
    return run_id


def _make_regression_run(target: str = "Fare") -> str:
    """Same data/runs/<run_id>/ shape a REGRESSION run would leave behind —
    meta.json has "target" (regression is supervised, same as
    classification), but the model has no predict_proba. Proves
    plot_confusion_matrix/roc/pr/calibration reject it cleanly instead of
    KeyError-ing or rendering a nonsense chart from continuous predictions."""
    df = pd.read_csv(TITANIC)[["Pclass", "Age", "Fare"]].dropna()
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    run_id, run_dir = _new_run_dir()
    train.to_csv(run_dir / "train.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)

    reg = RandomForestRegressor(n_estimators=20, random_state=42)
    reg.fit(train.drop(columns=[target]), train[target])
    joblib.dump(reg, run_dir / "pipeline.pkl")
    (run_dir / "meta.json").write_text(
        json.dumps({"target": target, "model": "random_forest", "best_params": None})
    )
    return run_id


def _make_unsupervised_run() -> str:
    """Same data/runs/<run_id>/ shape a CLUSTERING/ANOMALY run would leave —
    meta.json has no "target" key at all (clustering has none; anomaly-agent
    uses "label_column" instead — see anomaly-agent/scripts/mcp_server/).
    The pipeline here still exposes predict_proba (PyOD detectors do), so
    this specifically exercises the "target" not in meta half of the guard,
    not just the predict_proba half."""
    df = pd.read_csv(TITANIC)[["Pclass", "Age", "Fare", "Survived"]].dropna()
    run_id, run_dir = _new_run_dir()
    df.to_csv(run_dir / "train.csv", index=False)
    df.to_csv(run_dir / "test.csv", index=False)

    clf = RandomForestClassifier(
        n_estimators=10, random_state=42
    )  # has predict_proba, on purpose
    clf.fit(df.drop(columns=["Survived"]), df["Survived"])
    joblib.dump(clf, run_dir / "pipeline.pkl")
    (run_dir / "meta.json").write_text(
        json.dumps({"algorithm": "iforest", "label_column": "Survived"})
    )
    return run_id


def main():
    tmp = tempfile.mkdtemp()
    made_runs = []

    r = json.loads(m.plot_histogram(IRIS, "sepal length (cm)", f"{tmp}/hist.png"))
    _check(r["out_path"])

    r = json.loads(
        m.plot_boxplot(IRIS, "petal length (cm)", f"{tmp}/box.png", by="species")
    )
    _check(r["out_path"])

    r = json.loads(
        m.plot_scatter(
            IRIS,
            "sepal length (cm)",
            "petal length (cm)",
            f"{tmp}/scatter.png",
            hue="species",
        )
    )
    _check(r["out_path"])

    r = json.loads(m.plot_correlation_heatmap(IRIS, f"{tmp}/corr.png"))
    _check(r["out_path"])

    r = json.loads(
        m.plot_class_distribution(TITANIC, "Survived", f"{tmp}/classdist.png")
    )
    _check(r["out_path"])
    assert set(r["counts"].keys()) == {"0", "1"}

    r = json.loads(m.plot_missingness(TITANIC, f"{tmp}/missing.png"))
    _check(r["out_path"])
    assert "Cabin" in r["missing_pct"]

    r = json.loads(
        m.plot_feature_importance(
            f"{tmp}/fi.png",
            importances=json.dumps([["age", 0.4], ["fare", 0.3], ["sex", 0.2]]),
        )
    )
    _check(r["out_path"])

    # ── Return shape — the fields the chat UI actually renders, not just
    # out_path. A BASE_URL/PORT regression wouldn't be caught by _check alone. ──
    assert r["url"].startswith(m.BASE_URL + "/")
    assert r["markdown"] == f"![Feature importance]({r['url']})"

    # ── run_id safety: invalid/path-traversal run_id rejected before any
    # filesystem access outside RUNS_DIR, same clean-error contract as a
    # legitimately-missing run. ─────────────────────────────────────────────
    for bad_run_id in ["does-not-exist", "../../etc/passwd", "/etc/passwd", "..", ""]:
        missing = json.loads(
            m.plot_confusion_matrix(bad_run_id, f"{tmp}/cm_missing.png")
        )
        assert "error" in missing, f"run_id={bad_run_id!r} must fail clearly, not crash"

    # ── run_id-based eval plots: reads the shared pipeline, never refits. ──
    run_id = _make_run()
    made_runs.append(run_id)
    r = json.loads(m.plot_confusion_matrix(run_id, f"{tmp}/cm.png"))
    _check(r["out_path"])
    assert r["model"] == "random_forest"
    assert r["tuned"] is True  # meta.json carries best_params — this must be reflected

    r = json.loads(m.plot_roc_curve(run_id, f"{tmp}/roc.png"))
    _check(r["out_path"])
    assert "positive_class" in r["auc"]

    r = json.loads(m.plot_pr_curve(run_id, f"{tmp}/pr.png"))
    _check(r["out_path"])
    assert "positive_class" in r["ap"]

    r = json.loads(m.plot_calibration_curve(run_id, f"{tmp}/calib.png"))
    _check(r["out_path"])
    assert r["n_test"] > 0

    # ── multiclass: one-vs-rest branch of roc/pr, never exercised by the
    # binary titanic run above. ─────────────────────────────────────────────
    mc_run_id = _make_multiclass_run()
    made_runs.append(mc_run_id)
    r = json.loads(m.plot_roc_curve(mc_run_id, f"{tmp}/roc_mc.png"))
    _check(r["out_path"])
    assert set(r["auc"].keys()) == {"setosa", "versicolor", "virginica"}

    r = json.loads(m.plot_pr_curve(mc_run_id, f"{tmp}/pr_mc.png"))
    _check(r["out_path"])
    assert set(r["ap"].keys()) == {"setosa", "versicolor", "virginica"}

    # calibration_curve is binary-only by contract — must reject a 3-class target.
    r = json.loads(m.plot_calibration_curve(mc_run_id, f"{tmp}/calib_mc.png"))
    assert "error" in r and "binary" in r["error"]

    # ── plot_shap_beeswarm: needs the real encode+classifier Pipeline shape
    # (_make_run's bare classifier would fail this for the wrong reason). ──
    shap_run_id = _make_shap_run()
    made_runs.append(shap_run_id)
    r = json.loads(
        m.plot_shap_beeswarm(shap_run_id, f"{tmp}/shap.png", background_size=50)
    )
    _check(r["out_path"])
    assert r["n_samples"] > 0

    # plot_shap_beeswarm against _make_run's bare classifier (no named_steps
    # at all) must fail cleanly, not KeyError — this is exactly the
    # "different pipeline shape" cross-agent risk.
    r = json.loads(m.plot_shap_beeswarm(run_id, f"{tmp}/shap_bad.png"))
    assert "error" in r

    # ── Cross-agent safety: a regression run (has "target", no
    # predict_proba) and an unsupervised run (no "target" at all) must both
    # be rejected cleanly by every classification-only tool, not crash or
    # silently render a nonsense chart. ─────────────────────────────────────
    reg_run_id = _make_regression_run()
    made_runs.append(reg_run_id)
    for tool, kwargs in [
        (m.plot_confusion_matrix, {}),
        (m.plot_roc_curve, {}),
        (m.plot_pr_curve, {}),
        (m.plot_calibration_curve, {}),
        (m.plot_shap_beeswarm, {}),
    ]:
        r = json.loads(tool(reg_run_id, f"{tmp}/reg_{tool.__name__}.png", **kwargs))
        assert "error" in r, f"{tool.__name__} must reject a regression run_id cleanly"

    unsup_run_id = _make_unsupervised_run()
    made_runs.append(unsup_run_id)
    for tool in [
        m.plot_confusion_matrix,
        m.plot_roc_curve,
        m.plot_pr_curve,
        m.plot_calibration_curve,
        m.plot_shap_beeswarm,
    ]:
        r = json.loads(tool(unsup_run_id, f"{tmp}/unsup_{tool.__name__}.png"))
        assert "error" in r, (
            f"{tool.__name__} must reject a clustering/anomaly-shaped run_id cleanly"
        )

    for rid in made_runs:
        shutil.rmtree(m.RUNS_DIR / rid)
    os.close(_iris_fd)
    Path(IRIS).unlink()
    print("all checks passed —", tmp)


if __name__ == "__main__":
    main()
