"""Integration checks for serving-agent's mcp_server/.

Builds one real artifact per shape this agent has to serve — a classification
bundle (imblearn Pipeline: SMOTE + LogisticRegression, so predict_proba's
"confidence" path is exercised through a real imblearn Pipeline, not just
plain sklearn), a regression bundle (no predict_proba/decision_function —
"prediction" only), an anomaly bundle (a real pyod IForest detector, so
unpickling actually needs pyod installed, same as production), a kmeans
clustering bundle, an hierarchical clustering bundle (nearest-centroid
path), a dbscan bundle (must be blocked), and a forecasting ForecastModel —
then proves predict()/inspect_model() dispatch correctly on every one of
them, plus every error path returns an error instead of crashing.

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
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pyod.models.iforest import IForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mcp_server as m
from pipeline_transformers import ForecastModel

from core import (
    toolguard,
)  # noqa: E402 — importable once mcp_server put the repo root on sys.path


def _export(obj, path) -> None:
    """What export_model does: write the model AND its export record — the
    tool guard loads no .pkl without one (a crafted pickle is code)."""
    joblib.dump(obj, path)
    toolguard.trust(path)


ARTIFACTS_DIR = Path(__file__).parent / "test_artifacts"


def _make_classification_bundle(path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    y = (X["f1"] + X["f2"] > 0).astype(int)
    pipeline = ImbPipeline(
        [
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", LogisticRegression()),
        ]
    )
    pipeline.fit(X, y)
    _export(
        {"pipeline": pipeline, "target": "label", "model": "logistic_regression"}, path
    )


def _make_regression_bundle(path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 150
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    y = 3 * X["f1"] - 2 * X["f2"] + rng.normal(0, 0.05, n)
    pipeline = Pipeline([("scale", StandardScaler()), ("reg", LinearRegression())])
    pipeline.fit(X, y)
    _export(
        {"pipeline": pipeline, "target": "price", "model": "linear_regression"}, path
    )


def _make_anomaly_bundle(path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 200
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("detector", IForest(contamination=0.1, random_state=42)),
        ]
    )
    pipeline.fit(X)
    _export(
        {"pipeline": pipeline, "algorithm": "iforest", "label_column": "label"}, path
    )


def _make_kmeans_bundle(path: Path) -> None:
    rng = np.random.default_rng(4)
    half = 60
    X = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 0.3, half), rng.normal(6, 0.3, half)]),
            "f2": np.concatenate([rng.normal(0, 0.3, half), rng.normal(6, 0.3, half)]),
        }
    )
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("cluster", KMeans(n_clusters=2, random_state=42, n_init=10)),
        ]
    )
    pipeline.fit(X)
    _export(
        {"pipeline": pipeline, "algorithm": "kmeans", "cluster_centroids": None}, path
    )


def _make_hierarchical_bundle(path: Path) -> tuple[float, float, float, float]:
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame({"f1": [0.0, 0.1, 6.0, 6.1], "f2": [0.0, -0.1, 6.0, 5.9]}))
    pipeline = Pipeline([("preprocess", scaler)])
    near_zero = scaler.transform(pd.DataFrame({"f1": [0.05], "f2": [-0.05]}))[0]
    near_six = scaler.transform(pd.DataFrame({"f1": [6.05], "f2": [5.95]}))[0]
    centroids = {"cluster_a": near_zero.tolist(), "cluster_b": near_six.tolist()}
    _export(
        {
            "pipeline": pipeline,
            "algorithm": "hierarchical",
            "cluster_centroids": centroids,
        },
        path,
    )
    return 0.0, 0.0, 6.0, 6.0


def _make_dbscan_bundle(path: Path) -> None:
    pipeline = Pipeline(
        [
            (
                "scale",
                StandardScaler().fit(
                    pd.DataFrame({"f1": [0.0, 1.0], "f2": [0.0, 1.0]})
                ),
            )
        ]
    )
    _export({"pipeline": pipeline, "algorithm": "dbscan"}, path)


def _make_forecast_bundle(path: Path) -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    target = pd.Series(np.tile([1.0, 2.0, 3.0], 10))
    fm = ForecastModel(
        model_type="naive_seasonal",
        date_column="date",
        target="y",
        exog_columns=[],
        seasonal_periods=3,
        lags=[],
        rolling_windows=[],
        freq="D",
        fitted=np.array([1.0, 2.0, 3.0]),
        encoder=None,
        history_dates=dates,
        history_target=target,
        history_exog=None,
    )
    _export(fm, path)


def main():
    if ARTIFACTS_DIR.exists():
        shutil.rmtree(ARTIFACTS_DIR)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # predict() appends every batch to a predictions log; keep the test's out
    # of the repo's real data/predictions/.
    m.core.PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

    classification_pkl = ARTIFACTS_DIR / "classification.pkl"
    regression_pkl = ARTIFACTS_DIR / "regression.pkl"
    anomaly_pkl = ARTIFACTS_DIR / "anomaly.pkl"
    kmeans_pkl = ARTIFACTS_DIR / "kmeans.pkl"
    hierarchical_pkl = ARTIFACTS_DIR / "hierarchical.pkl"
    dbscan_pkl = ARTIFACTS_DIR / "dbscan.pkl"
    forecast_pkl = ARTIFACTS_DIR / "forecast.pkl"

    _make_classification_bundle(classification_pkl)
    _make_regression_bundle(regression_pkl)
    _make_anomaly_bundle(anomaly_pkl)
    _make_kmeans_bundle(kmeans_pkl)
    ax0, ay0, ax1, ay1 = _make_hierarchical_bundle(hierarchical_pkl)
    _make_dbscan_bundle(dbscan_pkl)
    _make_forecast_bundle(forecast_pkl)

    # ── list_exported_models ─────────────────────────────────────────────
    listing = json.loads(m.list_exported_models(str(ARTIFACTS_DIR)))
    listed_paths = {Path(entry["path"]).name for entry in listing["models"]}
    assert {"classification.pkl", "regression.pkl", "forecast.pkl"} <= listed_paths

    missing_dir = json.loads(
        m.list_exported_models(str(ARTIFACTS_DIR / "does_not_exist"))
    )
    assert "error" in missing_dir

    # ── inspect_model ─────────────────────────────────────────────────────
    inspected_clf = json.loads(m.inspect_model(str(classification_pkl)))
    assert inspected_clf["artifact_kind"] == "sklearn_pipeline"
    assert inspected_clf["target_column"] == "label"
    assert inspected_clf["supports_confidence"] is True

    inspected_reg = json.loads(m.inspect_model(str(regression_pkl)))
    assert inspected_reg["supports_confidence"] is False
    assert inspected_reg["supports_score"] is False

    inspected_fm = json.loads(m.inspect_model(str(forecast_pkl)))
    assert inspected_fm["artifact_kind"] == "forecast_model"
    assert inspected_fm["model_type"] == "naive_seasonal"
    assert inspected_fm["history_rows"] == 30

    # ── classification predict: target column dropped if present ────────
    clf_data = ARTIFACTS_DIR / "clf_input.csv"
    pd.DataFrame({"f1": [2.0, -2.0], "f2": [2.0, -2.0], "label": [1, 0]}).to_csv(
        clf_data, index=False
    )
    clf_result = json.loads(m.predict(str(classification_pkl), data_path=str(clf_data)))
    assert clf_result["artifact_kind"] == "sklearn_pipeline"
    assert clf_result["n_rows"] == 2
    pred_df = pd.read_csv(clf_result["out_path"])
    assert "label" not in pred_df.columns, (
        "target column must be dropped before predicting, not treated as a feature"
    )
    assert "prediction" in pred_df.columns and "confidence" in pred_df.columns
    assert pred_df.loc[0, "prediction"] == 1 and pred_df.loc[1, "prediction"] == 0

    # ── regression predict: no confidence/score column, no prediction_counts ─
    reg_data = ARTIFACTS_DIR / "reg_input.csv"
    # 25 distinct continuous rows -> nunique > 20, so prediction_counts
    # must come back None (a value-counts table isn't meaningful here).
    pd.DataFrame({"f1": np.linspace(-2, 2, 25), "f2": np.linspace(-1, 1, 25)}).to_csv(
        reg_data, index=False
    )
    reg_result = json.loads(m.predict(str(regression_pkl), data_path=str(reg_data)))
    reg_pred_df = pd.read_csv(reg_result["out_path"])
    assert (
        "confidence" not in reg_pred_df.columns and "score" not in reg_pred_df.columns
    )
    assert reg_result["prediction_counts"] is None
    assert reg_result["prediction_summary"]["count"] == len(reg_pred_df), (
        "a regressor's output comes back summarised"
    )
    first = json.loads(
        m.predict(str(regression_pkl), data_path=str(reg_data), max_rows=3)
    )
    assert first["n_rows"] == 3 and first["prediction_summary"]["count"] == 3, (
        "max_rows scores only the first N"
    )

    # ── anomaly predict: label_column dropped, decision-based "confidence" ──
    anomaly_data = ARTIFACTS_DIR / "anomaly_input.csv"
    pd.DataFrame({"f1": [0.1, 9.0], "f2": [0.1, 9.0], "label": [0, 1]}).to_csv(
        anomaly_data, index=False
    )
    anomaly_result = json.loads(
        m.predict(str(anomaly_pkl), data_path=str(anomaly_data))
    )
    anomaly_pred_df = pd.read_csv(anomaly_result["out_path"])
    assert "label" not in anomaly_pred_df.columns
    assert "prediction" in anomaly_pred_df.columns
    assert anomaly_pred_df.loc[1, "prediction"] == 1, (
        "the far-outlier row must be flagged anomalous"
    )

    # ── kmeans predict: pipeline.predict path, low-cardinality counts ────
    kmeans_data = ARTIFACTS_DIR / "kmeans_input.csv"
    pd.DataFrame({"f1": [0.05, 6.05], "f2": [-0.05, 5.95]}).to_csv(
        kmeans_data, index=False
    )
    kmeans_result = json.loads(m.predict(str(kmeans_pkl), data_path=str(kmeans_data)))
    assert kmeans_result["prediction_counts"] is not None
    kmeans_pred_df = pd.read_csv(kmeans_result["out_path"])
    assert kmeans_pred_df.loc[0, "prediction"] != kmeans_pred_df.loc[1, "prediction"]

    # ── hierarchical predict: nearest-centroid path ─────────────────────
    hierarchical_data = ARTIFACTS_DIR / "hierarchical_input.csv"
    pd.DataFrame(
        {"f1": [ax0 + 0.02, ax1 - 0.02], "f2": [ay0 - 0.02, ay1 + 0.02]}
    ).to_csv(hierarchical_data, index=False)
    hierarchical_result = json.loads(
        m.predict(str(hierarchical_pkl), data_path=str(hierarchical_data))
    )
    agg_pred_df = pd.read_csv(hierarchical_result["out_path"])
    assert agg_pred_df.loc[0, "prediction"] == "cluster_a"
    assert agg_pred_df.loc[1, "prediction"] == "cluster_b"

    # ── dbscan predict: blocked outright ─────────────────────────────────
    dbscan_data = ARTIFACTS_DIR / "dbscan_input.csv"
    pd.DataFrame({"f1": [0.0], "f2": [0.0]}).to_csv(dbscan_data, index=False)
    dbscan_result = json.loads(m.predict(str(dbscan_pkl), data_path=str(dbscan_data)))
    assert "error" in dbscan_result and "DBSCAN" in dbscan_result["error"]

    # ── forecast predict: naive_seasonal cycles the fitted values ────────
    forecast_result = json.loads(m.predict(str(forecast_pkl), horizon=6))
    assert forecast_result["artifact_kind"] == "forecast_model"
    forecast_df = pd.read_csv(forecast_result["out_path"])
    assert list(forecast_df["forecast"]) == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]

    # future_exog_path row-count validation is checked before the model is
    # ever called, regardless of whether this model type uses exog.
    bad_exog = ARTIFACTS_DIR / "bad_exog.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(bad_exog, index=False)  # 2 rows, horizon=6 below
    exog_mismatch = json.loads(
        m.predict(str(forecast_pkl), horizon=6, future_exog_path=str(bad_exog))
    )
    assert (
        "error" in exog_mismatch
        and "expected exactly horizon=6" in exog_mismatch["error"]
    )

    zero_horizon = json.loads(m.predict(str(forecast_pkl), horizon=0))
    assert "error" in zero_horizon and "horizon must be > 0" in zero_horizon["error"]

    # ── error paths ───────────────────────────────────────────────────────
    missing_pkl = json.loads(
        m.predict(str(ARTIFACTS_DIR / "does_not_exist.pkl"), data_path=str(reg_data))
    )
    assert "error" in missing_pkl and "no exported model" in missing_pkl["error"]

    no_data_path = json.loads(m.predict(str(regression_pkl)))
    assert "error" in no_data_path and "data_path is required" in no_data_path["error"]

    missing_data_file = json.loads(
        m.predict(str(regression_pkl), data_path=str(ARTIFACTS_DIR / "nope.csv"))
    )
    assert "error" in missing_data_file and "no data file" in missing_data_file["error"]

    corrupt_pkl = ARTIFACTS_DIR / "corrupt.pkl"
    corrupt_pkl.write_bytes(b"not a real pickle")
    unrecorded = json.loads(m.predict(str(corrupt_pkl), data_path=str(reg_data)))
    assert "no export record" in unrecorded.get("error", ""), (
        "a .pkl nobody exported is never loaded"
    )
    toolguard.trust(
        corrupt_pkl
    )  # recorded but unreadable: the load itself must fail cleanly
    corrupt_result = json.loads(m.predict(str(corrupt_pkl), data_path=str(reg_data)))
    assert "error" in corrupt_result and "could not load" in corrupt_result["error"]

    unknown_shape_pkl = ARTIFACTS_DIR / "unknown_shape.pkl"
    _export({"not": "a recognized bundle"}, unknown_shape_pkl)
    unknown_result = json.loads(
        m.predict(str(unknown_shape_pkl), data_path=str(reg_data))
    )
    assert (
        "error" in unknown_result
        and "not a recognized serving-agent artifact shape" in unknown_result["error"]
    )
    unknown_inspect = json.loads(m.inspect_model(str(unknown_shape_pkl)))
    assert "error" in unknown_inspect

    print("all checks passed")
    shutil.rmtree(ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
