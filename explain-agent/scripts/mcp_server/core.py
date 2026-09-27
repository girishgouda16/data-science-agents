"""MCP server exposing model-explanation tools: why does this model behave
the way it does, and why did it decide THIS case the way it did.

Two questions, and the split between them is the reason this agent exists.

  GLOBAL — "what drives this model?" — is a property of a MODEL. It is
  computed on a held-out test fold at training time, by the agent that built
  the model, and it is a gated step there (classification-agent's
  `explainability_run` / `explainability_clean`). This server can compute it
  on any exported .pkl for standalone use, but it is not where a training
  run's gated evidence should come from: that agent already has the run, the
  pipeline and the test fold in-process, and routing a required gate through
  a network hop only adds a way for it to fail.

  LOCAL — "why was THIS row flagged?" — is a property of a DECISION, and it
  has nowhere else to live. The row in question is almost never in a test
  fold: it is a production row scored last night from an exported .pkl,
  often long after the training run's directory was swept. There is no
  run_id and no training agent in the picture. The signature is
  (model artifact, arbitrary row) -> why, which is the same shape
  serving-agent has, which is why this sits beside it rather than inside a
  training agent.

Everything local explanation needs already travels with the export bundle:
the training fold as `<model>.reference.csv` (the SHAP background), plus
`positive_label` and `operating_point` in the bundle and the
`<model>.monitoring.json` profile. No new contract.

Stateless and read-only: nothing here trains, refits or modifies a .pkl.

Run: python -m mcp_server.server  (cwd=scripts/; agent.py does this)

Shared kernel of the package: imports, constants, run storage, every
helper and the `mcp` instance. Tools live in one module per skill."""

import json

import os

import sys
from pathlib import Path

import joblib

import numpy as np

import pandas as pd

import shap

from mcp.server.fastmcp import FastMCP

from sklearn.calibration import CalibratedClassifierCV

from sklearn.pipeline import Pipeline as SkPipeline

# Unused directly, but joblib.load()-ing another agent's bundle needs these
# importable in this process — pickle resolves classes by module name.
from pipeline_transformers import (
    ColumnDropper,
    ColumnFiller,
    ForecastModel,
)  # noqa: F401

mcp = FastMCP("explain-agent")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for core.*
from core import toolguard  # noqa: E402

toolguard.install(
    mcp
)  # every tool's paths/models checked before it runs (core/toolguard.py)

# AGENTIC_ML_DATA_DIR relocates data/ (runs, artifacts, predictions) for every
# agent at once — the test suites set it so they never write into the real
# data/runs that inspect_runs reports to the orchestrator as "recent runs".
_DATA_DIR = Path(
    os.environ.get("AGENTIC_ML_DATA_DIR")
    or Path(__file__).parent.parent.parent.parent / "data"
)

ARTIFACTS_DIR = _DATA_DIR / "artifacts"

RUNS_DIR = _DATA_DIR / "runs"

# Models with an exact, fast explainer. Anything else is refused by name
# rather than silently falling back to KernelExplainer, which is a sampling
# approximation that can take minutes per row and whose error nobody reports.
TREE_MODELS = ("random_forest", "xgboost", "gradient_boosting", "decision_tree")

LINEAR_MODELS = (
    "logistic_regression",
    "linear_regression",
    "ridge",
    "lasso",
    "elasticnet",
)

# SHAP contributions are additive in the model's raw output. If base value
# plus contributions doesn't reconstruct the score, the explanation is not
# describing this prediction and must not be presented as if it were.
ADDITIVITY_TOLERANCE = 1e-3

# ─────────────────────────────────────────────────────── loading & unwrapping


def _load_artifact(pkl_path: str):
    """Returns (bundle, error). Never raises — a bad path or an unreadable
    pickle is normal input here, not a bug."""
    path = Path(pkl_path)
    if not path.exists():
        return None, {"error": f"no exported model at '{pkl_path}'"}
    try:
        obj = joblib.load(path)
    except Exception as exc:
        return None, {
            "error": f"could not load '{pkl_path}': {type(exc).__name__}: {exc}"
        }
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj, None
    return None, {
        "error": f"'{pkl_path}' is not an explainable artifact — expected an sklearn-Pipeline bundle from "
        "classification or regression agent. Clustering, anomaly and forecasting artifacts have their "
        "own notions of explanation and are not handled here.",
    }


def _unwrap_calibrated(pipeline):
    """Reach the pipeline SHAP can actually explain, through a calibrator.

    `calibrate_model` wraps the whole fitted pipeline in
    CalibratedClassifierCV. SHAP values are additive in the UNCALIBRATED
    score — the calibrator is a monotonic map applied afterwards, and
    contributions are not additive in its output. Explaining the calibrated
    probability as though they were would produce numbers that look right,
    sum to the wrong thing, and silently misstate every contribution.

    So: explain the inner estimator, and say so. Returns
    (pipeline_to_explain, calibrated: bool)."""
    if isinstance(pipeline, CalibratedClassifierCV):
        fitted = getattr(pipeline, "calibrated_classifiers_", None)
        if not fitted:
            return pipeline, True
        inner = getattr(fitted[0], "estimator", None) or getattr(
            fitted[0], "base_estimator", None
        )
        return (inner, True) if inner is not None else (pipeline, True)
    return pipeline, False


def _final_step_name(pipeline) -> str | None:
    """classification-agent names it "classifier", regression-agent
    "regressor". Read it rather than accepting a task type as an argument:
    a caller who passes the wrong one gets a confidently wrong explanation,
    and the artifact already knows the answer."""
    steps = getattr(pipeline, "named_steps", None)
    if not steps:
        return None
    for name in ("classifier", "regressor"):
        if name in steps:
            return name
    return list(steps)[-1]


def _encode(pipeline, X: pd.DataFrame):
    """Run X through every preprocessing step, stopping before the estimator.

    "smote" is skipped explicitly: a sampler has fit_resample but no
    transform, so a plain Pipeline.transform() raises on it. It never runs at
    predict time anyway."""
    final = _final_step_name(pipeline)
    steps = [(n, s) for n, s in pipeline.steps if n not in ("smote", final)]
    encoded = SkPipeline(steps).transform(X) if steps else X.to_numpy()
    try:
        names = list(pipeline.named_steps["encode"].get_feature_names_out())
    except (KeyError, AttributeError):
        names = [f"f{i}" for i in range(np.asarray(encoded).shape[1])]
    return np.asarray(encoded), names


def _task_of(pipeline, bundle: dict) -> str:
    if _final_step_name(pipeline) == "regressor" or bundle.get("algorithm") in (
        "linear_regression",
        "ridge",
        "lasso",
    ):
        return "regression"
    return "classification" if hasattr(pipeline, "predict_proba") else "regression"


def _model_name(bundle: dict) -> str:
    return str(bundle.get("model") or bundle.get("algorithm") or "")


def _explainer_for(model_name: str, estimator, X_background):
    if any(t in model_name for t in TREE_MODELS):
        return shap.TreeExplainer(estimator), "TreeExplainer (exact)"
    if any(l in model_name for l in LINEAR_MODELS):
        return shap.LinearExplainer(estimator, X_background), "LinearExplainer (exact)"
    return None, None


def _positive_index(bundle: dict, classes) -> int:
    """predict_proba column for the class the model's own metrics refer to.
    sklearn sorts classes_, so this is 1 unless the recorded positive label
    sorts first — {"churn","no_churn"} puts "churn" in column 0."""
    labels = sorted(classes)
    pos = bundle.get("positive_label")
    if len(labels) != 2 or pos is None:
        return 1
    return 0 if str(labels[0]) == str(pos) else 1


def _reference_frame(pkl_path: str, bundle: dict) -> tuple[pd.DataFrame | None, str]:
    """The background distribution SHAP measures contributions AGAINST.

    A contribution is always "relative to what" — without a reference, a
    local explanation has no defined meaning. export_model writes the
    training fold beside the .pkl for exactly this (and for drift), so the
    honest reference is the data the model was actually fitted on."""
    candidates = [
        Path(pkl_path).with_suffix(".reference.csv"),
        (
            Path((bundle.get("monitoring") or {}).get("reference_path", ""))
            if bundle.get("monitoring")
            else None
        ),
    ]
    profile = Path(pkl_path).with_suffix(".monitoring.json")
    if profile.exists():
        try:
            candidates.insert(
                0, Path(json.loads(profile.read_text()).get("reference_path", ""))
            )
        except (json.JSONDecodeError, OSError):
            pass
    for path in candidates:
        if path and str(path) and path.exists():
            df = pd.read_csv(path)
            target = bundle.get("target") or bundle.get("label_column")
            if target and target in df.columns:
                df = df.drop(columns=[target])
            return df, str(path)
    return None, ""


def _shap_matrix(raw, pos_idx: int) -> np.ndarray:
    """Normalise SHAP's several output shapes to (n_rows, n_features) for the
    class being explained."""
    if isinstance(raw, list):
        return np.asarray(raw[pos_idx] if len(raw) > pos_idx else raw[-1])
    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[:, :, pos_idx] if arr.shape[2] > pos_idx else arr[:, :, -1]
    return arr


def _prepare(pkl_path: str, data_path: str = ""):
    """Shared front half of every tool: load, unwrap, encode, resolve the
    reference. Returns (ctx, error_dict)."""
    bundle, err = _load_artifact(pkl_path)
    if err:
        return None, err
    pipeline, calibrated = _unwrap_calibrated(bundle["pipeline"])
    model_name = _model_name(bundle)
    task = _task_of(pipeline, bundle)

    reference, reference_path = _reference_frame(pkl_path, bundle)
    if reference is None:
        return None, {
            "error": "no reference dataset for this model — SHAP contributions are measured against a background "
            f"distribution, and neither '{Path(pkl_path).with_suffix('.reference.csv')}' nor a "
            "monitoring profile pointing at one exists. Re-export the model with a current "
            "classification/regression agent (its export writes the training fold beside the .pkl), or "
            "there is no defensible baseline to explain against.",
        }

    data = None
    if data_path:
        file = Path(data_path)
        if not file.exists():
            return None, {"error": f"no data file at '{data_path}'"}
        data = pd.read_csv(file)
        target = bundle.get("target") or bundle.get("label_column")
        if target and target in data.columns:
            data = data.drop(columns=[target])

    return {
        "bundle": bundle,
        "pipeline": pipeline,
        "calibrated": calibrated,
        "model_name": model_name,
        "task": task,
        "reference": reference,
        "reference_path": reference_path,
        "data": data,
        "pkl_path": pkl_path,
    }, None


def _score(ctx, X: pd.DataFrame) -> np.ndarray:
    """The model's output on the scale being explained."""
    pipeline, bundle = ctx["pipeline"], ctx["bundle"]
    if ctx["task"] == "regression":
        return np.asarray(pipeline.predict(X), dtype=float)
    classes = list(getattr(pipeline, "classes_", []))
    return np.asarray(pipeline.predict_proba(X))[:, _positive_index(bundle, classes)]


def _decision(ctx, score: float) -> dict:
    """What the model DID with that score, in the terms it actually ships."""
    bundle = ctx["bundle"]
    if ctx["task"] == "regression":
        return {"predicted_value": round(float(score), 6)}
    op = bundle.get("operating_point") or {}
    threshold = float(op["threshold"]) if op.get("threshold") is not None else 0.5
    classes = sorted(getattr(ctx["pipeline"], "classes_", []))
    pos_idx = _positive_index(bundle, classes)
    positive = classes[pos_idx] if classes else 1
    negative = classes[1 - pos_idx] if len(classes) == 2 else None
    return {
        "positive_class": str(positive),
        "probability_of_positive": round(float(score), 6),
        "threshold": threshold,
        "threshold_source": (
            (f"the model's tuned operating point, chosen by {op.get('chosen_by')}")
            if op.get("threshold") is not None
            else "sklearn's default 0.5 — this model ships no tuned threshold"
        ),
        "decision": str(positive) if score >= threshold else str(negative),
        # The quantity that actually answers "why was it flagged": distance
        # to the cutoff this model ships with, not distance from 0.5.
        "margin_over_threshold": round(float(score) - threshold, 6),
    }


# ───────────────────────────────────────────────────────────────── helpers


def _stability(ctx, explainer, pipeline, X_enc, values, pos_idx, feature_names) -> dict:
    """Do near-identical rows get near-identical explanations?

    An attribution nobody sanity-checks is a number with a story attached.
    This takes the closest rows in the reference data, explains them, and
    reports how much the top-feature ranking agrees. Low agreement doesn't
    mean the explanation is wrong — it means this model's attributions are
    locally unstable, and a single row's explanation should not be quoted
    as if it were a property of the model."""
    try:
        X_ref, _ = _encode(pipeline, ctx["reference"].head(400))
        distances = np.linalg.norm(X_ref - X_enc[0], axis=1)
        neighbours = X_ref[np.argsort(distances)[:5]]
        raw = explainer.shap_values(neighbours)
        neighbour_values = _shap_matrix(raw, pos_idx)
        top = set(np.argsort(-np.abs(values))[:5])
        overlaps = [
            len(top & set(np.argsort(-np.abs(nv))[:5])) / 5 for nv in neighbour_values
        ]
        agreement = float(np.mean(overlaps))
        return {
            "explanation_stability": round(agreement, 3),
            "stability_note": (
                f"the 5 most similar rows in the reference data share {agreement:.0%} of this row's top-5 drivers"
                + (
                    ""
                    if agreement >= 0.6
                    else " — LOW. This model's local attributions shift between near-identical cases, so treat this "
                    "explanation as indicative, not as the reason. Do not quote it to the affected party."
                )
            ),
        }
    except Exception:
        return {
            "explanation_stability": None,
            "stability_note": "stability could not be computed for this model/reference combination",
        }


def _narrate(
    contributions: list[dict], decision: dict, positive_name: str, task: str
) -> str:
    """One plain sentence, built from the numbers — never a free-text
    summary a model could drift away from the values above it."""
    if not contributions:
        return "no features contributed measurably to this prediction"
    pushing = [c for c in contributions if c["contribution"] > 0][:3]
    pulling = [c for c in contributions if c["contribution"] < 0][:2]
    if task == "regression":
        head = f"predicted {decision.get('predicted_value')}"
    else:
        head = (
            f"scored {decision.get('probability_of_positive')} against a {decision.get('threshold')} threshold "
            f"-> {decision.get('decision')}"
        )
    parts = [head]
    if pushing:
        parts.append(
            "pushed toward "
            + str(positive_name)
            + " mainly by "
            + ", ".join(
                f"{c['feature']}={c['value']} (+{c['contribution']})" for c in pushing
            )
        )
    if pulling:
        parts.append(
            "pulled the other way by "
            + ", ".join(
                f"{c['feature']}={c['value']} ({c['contribution']})" for c in pulling
            )
        )
    return "; ".join(parts)


def _native(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
