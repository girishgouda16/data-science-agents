"""MCP server exposing model-serving tools: turns a .pkl exported by any
other agent's export_model into predictions, without needing that agent's
own Python session. Two incompatible artifact shapes exist across this
repo (see pipeline_transformers.py's docstring) and this server dispatches
on which one a given file actually is:

  - a dict bundle with a "pipeline" key — classification/regression/
    clustering/anomaly-agent's fitted sklearn (or imblearn) Pipeline plus
    a few metadata fields. predict() calls pipeline.predict(X) directly,
    except clustering's "hierarchical" export (no native out-of-sample
    predict — nearest-centroid assignment, same as clustering-agent's own
    predict()) and "dbscan" (blocked, same reasoning as clustering-agent).
  - a bespoke ForecastModel object — forecasting-agent's artifact. predict()
    calls its own .forecast(horizon, future_exog) method.

Nothing here trains, retrains, or modifies a .pkl — it only loads one,
writes predictions to a new CSV alongside the input, and appends the same
rows (stamped with the model's run_id/version) to data/predictions/<run_id>.csv.

Run: python -m mcp_server.server  (cwd=scripts/; agent.py does this)

Shared kernel of the package: imports, constants, run storage, every
helper and the `mcp` instance. Tools live in one module per skill."""

import json

import os

from datetime import datetime, timezone

import sys
from pathlib import Path

import joblib

import numpy as np

import pandas as pd

from mcp.server.fastmcp import FastMCP

from pipeline_transformers import ForecastModel

mcp = FastMCP("serving-agent")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for core.*
from core import toolguard  # noqa: E402

toolguard.install(
    mcp
)  # every tool's paths/models checked before it runs (core/toolguard.py)

# Every ML agent's export_model tool suggests saving here by default (see
# core/config.py's artifacts_dir) — the shared "where trained models live"
# convention this agent exists to make useful.
# AGENTIC_ML_DATA_DIR relocates data/ (runs, artifacts, predictions) for every
# agent at once — the test suites set it so they never write into the real
# data/runs that inspect_runs reports to the orchestrator as "recent runs".
ARTIFACTS_DIR = (
    Path(
        os.environ.get("AGENTIC_ML_DATA_DIR")
        or Path(__file__).parent.parent.parent.parent / "data"
    )
    / "artifacts"
)

# Every scored batch is appended to predictions/<model run_id>.csv — what
# production actually saw, and the `current` side of a drift check.
PREDICTIONS_DIR = ARTIFACTS_DIR.parent / "predictions"


def _json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_artifact(pkl_path: str):
    """Returns (artifact, error_dict). Never raises: joblib.load can throw
    almost anything (missing class, corrupt file, wrong pickle protocol) —
    a bad path/file is a normal, expected input here, not a bug."""
    path = Path(pkl_path)
    if not path.exists():
        return None, {"error": f"no exported model at '{pkl_path}'"}
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, {
            "error": f"could not load '{pkl_path}': {type(exc).__name__}: {exc}"
        }


def _assign_nearest_centroids(X: np.ndarray, centroids: dict) -> np.ndarray:
    """Same nearest-centroid assignment clustering-agent's own predict() uses
    for its "hierarchical" export — AgglomerativeClustering has no native
    out-of-sample predict, so the exported centroids stand in for one."""
    if not centroids:
        raise ValueError("no cluster_centroids available for assignment")
    ordered_labels = list(centroids.keys())
    centroid_matrix = np.asarray(
        [centroids[label] for label in ordered_labels], dtype=float
    )
    distances = ((X[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
    return np.asarray([ordered_labels[i] for i in np.argmin(distances, axis=1)])


def _positive_index(bundle: dict, classes) -> int:
    """The predict_proba column holding P(positive class). sklearn sorts
    classes_, so it is 1 unless the exporting agent recorded a positive
    label that sorts first ({"churn","no_churn"} -> "churn" is column 0)."""
    labels = sorted(classes)
    pos = bundle.get("positive_label")
    if len(labels) != 2 or pos is None:
        return 1
    return 0 if str(labels[0]) == str(pos) else 1


def _readiness_fields(readiness) -> dict:
    """The gate verdict travels with every prediction: a model exported past a
    BLOCKED gate (force=true) serves identically to a ready one otherwise."""
    fields = {"readiness_at_export": readiness}
    if readiness in ("blocked", "incomplete"):
        fields["warning"] = (
            f"this model was exported from a {readiness.upper()} run — "
            + (
                "it failed at least one readiness gate and was forced through. Relay this with the predictions."
                if readiness == "blocked"
                else "some readiness checks never ran, so nothing verified it. Relay this with the predictions."
            )
        )
    return fields


def _predict_sklearn(
    bundle: dict,
    data_path: str,
    out_path: str,
    review_threshold: float = 0.0,
    pkl_path: str = "",
    max_rows: int = 0,
) -> str:
    if not data_path:
        return json.dumps(
            {
                "error": "data_path is required for this artifact — it's an sklearn-Pipeline bundle, call predict(pkl_path, data_path=<csv>)"
            }
        )
    data_file = Path(data_path)
    if not data_file.exists():
        return json.dumps({"error": f"no data file at '{data_path}'"})

    algorithm = bundle.get("algorithm")
    if algorithm == "dbscan":
        return json.dumps(
            {
                "error": "DBSCAN exports do not support predict() on brand-new data; use the clustering agent's explain_model/analysis on the original run instead"
            }
        )

    pipeline = bundle["pipeline"]
    target_column = bundle.get("target") or bundle.get("label_column")
    X = pd.read_csv(data_file)
    if max_rows > 0:
        X = X.head(max_rows)
    if target_column and target_column in X.columns:
        X = X.drop(columns=[target_column])

    expected = getattr(pipeline, "feature_names_in_", None)
    if expected is not None:
        missing = [c for c in expected if c not in X.columns]
        if missing:
            return json.dumps(
                {
                    "error": f"data_path is missing column(s) required by this model: {missing}"
                }
            )

    # The exporting agent may have TUNED a decision threshold (see
    # classification-agent's tune_threshold) and shipped it in the bundle.
    # Ignoring it and calling pipeline.predict() means serving 0.5 — a
    # DIFFERENT model from the one that was reviewed, gated and reported.
    # A model signed off at recall 0.90 must not quietly ship at whatever
    # recall 0.50 happens to give.
    operating_point = bundle.get("operating_point") or {}
    classes = list(getattr(pipeline, "classes_", []))
    use_threshold = (
        algorithm != "hierarchical"
        and operating_point.get("threshold") is not None
        and hasattr(pipeline, "predict_proba")
        and len(classes) == 2
    )

    positive_proba = None
    try:
        if algorithm == "hierarchical":
            X_transformed = pipeline.named_steps["preprocess"].transform(X)
            predictions = _assign_nearest_centroids(
                np.asarray(X_transformed), bundle.get("cluster_centroids") or {}
            )
        elif use_threshold:
            pos_idx = _positive_index(bundle, classes)
            positive_proba = np.asarray(pipeline.predict_proba(X))[:, pos_idx]
            positive, negative = classes[pos_idx], classes[1 - pos_idx]
            predictions = np.where(
                positive_proba >= float(operating_point["threshold"]),
                positive,
                negative,
            )
        else:
            predictions = pipeline.predict(X)
    except Exception as exc:
        return json.dumps({"error": f"prediction failed: {type(exc).__name__}: {exc}"})

    out = X.copy()
    out["prediction"] = predictions
    if positive_proba is not None:
        out["positive_proba"] = np.round(positive_proba, 4)
    if algorithm != "hierarchical" and hasattr(pipeline, "predict_proba"):
        out["confidence"] = np.round(
            np.asarray(pipeline.predict_proba(X)).max(axis=1), 4
        )
    elif algorithm != "hierarchical" and hasattr(pipeline, "decision_function"):
        out["score"] = np.round(
            np.asarray(pipeline.decision_function(X), dtype=float), 6
        )

    review_file = None
    review_count = 0
    review_basis = None
    if review_threshold > 0:
        if "confidence" not in out.columns:
            return json.dumps(
                {
                    "error": "review_threshold requires a model with predict_proba (confidence); this pipeline doesn't support it"
                }
            )
        if positive_proba is not None:
            # A review queue for a classifier means "rows whose CLASS DECISION
            # is not safe to act on unseen", and that is a statement about the
            # decision boundary the model actually ships with.
            # predict_proba().max() is confidence in whichever class won, which
            # on an imbalanced target is a different question with a different
            # answer: at 1% fraud prevalence almost every flagged fraud sits
            # below 0.6 max-confidence while millions of confident negatives sit
            # above it, so a 0.6 "review queue" enqueues the whole negative
            # class and misses the alerts anyone actually wants checked.
            # Band the operating point instead: flag rows within
            # review_threshold of the cutoff, in probability units.
            cutoff = float(operating_point["threshold"])
            needs_review = np.abs(positive_proba - cutoff) <= review_threshold
            review_basis = (
                f"|P(positive) - {round(cutoff, 4)}| <= {review_threshold} — rows close enough to this model's "
                "decision threshold that the class assignment could go either way"
            )
        else:
            needs_review = out["confidence"] < review_threshold
            review_basis = (
                f"predict_proba().max() < {review_threshold} — this model ships no tuned operating point, so "
                "there is no decision boundary to band around; max-class confidence is the fallback and it is a "
                "weak proxy on an imbalanced target"
            )
        out["needs_review"] = needs_review
        review_count = int(needs_review.sum())
        if review_count:
            review_file = data_file.with_name(f"{data_file.stem}_review_queue.csv")
            out[needs_review].to_csv(review_file, index=False)

    # Every row says which model produced it — a prediction you can't trace
    # back to a model version can't be audited, rolled back, or monitored.
    model_run_id = bundle.get("run_id")
    model_version = (bundle.get("registry") or {}).get("version")
    out["model_run_id"] = model_run_id
    out["model_version"] = model_version
    out_file = (
        Path(out_path)
        if out_path
        else data_file.with_name(f"{data_file.stem}_predictions.csv")
    )
    out.to_csv(out_file, index=False)

    # Append to the model's predictions log. Aligned to the existing header,
    # so a batch with extra/missing raw columns can't shift the columns.
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = (
        PREDICTIONS_DIR
        / f"{model_run_id or Path(pkl_path).stem or 'unknown_model'}.csv"
    )
    batch = out.assign(
        scored_at=datetime.now(timezone.utc).isoformat(), source_file=str(data_file)
    )
    if log_path.exists():
        batch = batch.reindex(columns=pd.read_csv(log_path, nrows=0).columns)
    batch.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    prediction_series = pd.Series(predictions)
    continuous = pd.api.types.is_float_dtype(
        prediction_series
    )  # a regressor's output, not labels
    prediction_counts = (
        {str(k): int(v) for k, v in prediction_series.value_counts().items()}
        if not continuous and prediction_series.nunique() <= 20
        else None
    )
    response = {
        "artifact_kind": "sklearn_pipeline",
        "algorithm": algorithm or bundle.get("model"),
        "n_rows": len(out),
        "out_path": str(out_file),
        "model": {
            "run_id": model_run_id,
            "registry": bundle.get("registry"),
            "pkl_path": pkl_path,
        },
        "predictions_log": str(log_path),
        "prediction_counts": prediction_counts,
        # Numeric predictions summarised here, so nobody has to read the CSV back to answer "what came out?".
        "prediction_summary": (
            prediction_series.describe().round(4).to_dict() if continuous else None
        ),
        "preview": out.head(5).to_dict(orient="records"),
    }
    if "confidence" in out.columns:
        response["confidence_note"] = (
            "confidence is predict_proba().max(), a raw model probability — not calibrated "
            "unless the exporting agent explicitly calibrated it (most default RF/XGB paths don't)."
        )
    if review_threshold > 0:
        response["review_queue"] = {
            "band": review_threshold,
            "basis": review_basis,
            "flagged_rows": review_count,
            "review_file": str(review_file) if review_file else None,
            "note": "route these to a human before acting on the prediction.",
        }
    response.update(_readiness_fields(bundle.get("readiness_at_export")))
    # State the decision rule that produced these predictions, every time.
    # "Which threshold shipped" is not an optional detail — it is the
    # difference between the reviewed model and a different one.
    response["decision_rule"] = (
        {
            "threshold": round(float(operating_point["threshold"]), 4),
            "applied_to": "P(positive class)",
            "positive_class": (
                str(bundle.get("positive_label"))
                if bundle.get("positive_label") is not None
                else "highest-sorting class label"
            ),
            "chosen_by": operating_point.get("chosen_by"),
            "source": "the exporting agent's tuned operating point, shipped in this bundle",
        }
        if use_threshold
        else (
            {
                "threshold": 0.5,
                "applied_to": "sklearn's default argmax over predict_proba",
                "source": "no tuned operating point in this bundle — the exporting agent never called tune_threshold, "
                "or exported before doing so. On an imbalanced target 0.5 is rarely the right cutoff.",
            }
            if algorithm != "hierarchical" and hasattr(pipeline, "predict_proba")
            else None
        )
    )
    return json.dumps(response, default=_json_default)


def _predict_forecast(
    fm: ForecastModel, pkl_path: str, horizon: int, future_exog_path: str, out_path: str
) -> str:
    if horizon <= 0:
        return json.dumps(
            {
                "error": "horizon must be > 0 for a forecasting model, e.g. predict(pkl_path, horizon=30)"
            }
        )

    future_exog = None
    if future_exog_path:
        future_exog_file = Path(future_exog_path)
        if not future_exog_file.exists():
            return json.dumps({"error": f"no data file at '{future_exog_path}'"})
        future_exog = pd.read_csv(future_exog_file)
        if len(future_exog) != horizon:
            return json.dumps(
                {
                    "error": f"future_exog_path has {len(future_exog)} row(s), expected exactly horizon={horizon}"
                }
            )

    forecast = fm.forecast(horizon, future_exog)
    out_file = (
        Path(out_path)
        if out_path
        else Path(pkl_path).with_name(f"{Path(pkl_path).stem}_forecast.csv")
    )
    forecast.to_csv(out_file, index=False)
    # pd.Timestamp has no .item() (unlike numpy scalars), so _json_default
    # can't coerce it — stringify up front instead of fighting json.dumps.
    forecast[fm.date_column] = forecast[fm.date_column].astype(str)
    return json.dumps(
        {
            "artifact_kind": "forecast_model",
            "model_type": fm.model_type,
            "horizon": horizon,
            "out_path": str(out_file),
            # Exports older than the stamp have neither attribute — None says "unknown", not "fine".
            "model": {
                "run_id": getattr(fm, "run_id", None),
                "registry": getattr(fm, "registry", None),
                "pkl_path": pkl_path,
            },
            **_readiness_fields(getattr(fm, "readiness_at_export", None)),
            "preview": forecast.head(5).to_dict(orient="records"),
        },
        default=_json_default,
    )
