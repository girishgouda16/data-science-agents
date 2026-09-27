"""MCP server exposing tabular-regression tools. prepare_dataset splits
train/test BEFORE any learned preprocessing, so every downstream step —
imputation, encoding, the model itself — fits only on the training fold.
Everything after that is keyed by run_id, not a file path: one fitted
pipeline.pkl is built once by train_model and overwritten in place by
tune_hyperparams, so explain_model/export_model always reflect whichever
model is *actually current* for that run — no argument-threading
hyperparameters through five tool calls and hoping the caller remembers to
pass them every time.

Mirrors classification-agent/scripts/mcp_server/'s structure; the parts
that don't apply to a continuous target (class imbalance, SMOTE, ROC/PR
threshold tuning) are simply absent rather than stubbed out.

Run: python -m mcp_server.server  (cwd=scripts/; agent.py does this)

Shared kernel of the package: imports, constants, run storage, every
helper and the `mcp` instance. Tools live in one module per skill."""

import json

import os

import shutil

import sys

import time

import uuid

from pathlib import Path

import joblib

import numpy as np

import optuna

import pandas as pd

import shap

from pipeline_transformers import ColumnDropper, ColumnFiller

from run_persistence import save_training_run

from sklearn.pipeline import Pipeline

from mcp.server.fastmcp import FastMCP

from scipy import stats

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

from sklearn.dummy import DummyRegressor

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from sklearn.impute import SimpleImputer

from sklearn.inspection import permutation_importance

from sklearn.linear_model import LinearRegression, PoissonRegressor, QuantileRegressor, Ridge

from sklearn.metrics import make_scorer, mean_absolute_error, mean_pinball_loss, mean_squared_error, r2_score

from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    cross_val_score,
    train_test_split,
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor

sys.path.insert(
    0, str(Path(__file__).parent.resolve().parents[2])
)  # repo root, for core.*

from core import gates as gate_engine  # noqa: E402

from core import mlops as core_mlops  # noqa: E402

from core.data_quality import identifier_columns  # noqa: E402

from core import validation as shared_validation  # noqa: E402

from core.datasource import read_table  # noqa: E402,F401

mcp = FastMCP("regression-agent")
from core import toolguard  # noqa: E402

toolguard.install(
    mcp
)  # every tool's paths/models checked before it runs (core/toolguard.py)

optuna.logging.set_verbosity(
    optuna.logging.WARNING
)  # else every trial logs a line to stderr

# AGENTIC_ML_DATA_DIR relocates data/ for every agent at once (the test suites
# set it so they never write into the real data/runs).
RUNS_DIR = (
    Path(
        os.environ.get("AGENTIC_ML_DATA_DIR")
        or Path(__file__).parent.parent.parent.parent / "data"
    )
    / "runs"
)

RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ponytail: flat TTL swept opportunistically, not a scheduler — add a "pin"
# flag in meta.json if a run ever needs to outlive this on purpose.
RUN_RETENTION_DAYS = float(os.environ.get("RUN_RETENTION_DAYS", 7))


def _cleanup_old_runs(max_age_days: float = RUN_RETENTION_DAYS) -> None:
    """Runs are disposable working state (train/test CSVs + pipeline.pkl),
    not an archive — export_model is what persists a model long-term. Ages
    off meta.json's last write (not the run's creation time), so a run still
    being worked on never gets swept mid-session. Called from
    prepare_dataset, since every new run is a natural checkpoint to do this
    at — no separate cron/scheduler needed."""
    cutoff = time.time() - max_age_days * 86400
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        mtime = (
            meta_path.stat().st_mtime if meta_path.exists() else run_dir.stat().st_mtime
        )
        if mtime >= cutoff:
            continue
        # A run registered in MLflow is pinned: promoting its version exports
        # from this directory, so sweeping it strands a registry version.
        try:
            if json.loads(meta_path.read_text()).get("registry"):
                continue
        except (OSError, ValueError):
            pass
        shutil.rmtree(run_dir, ignore_errors=True)


MODELS = {
    "linear_regression": LinearRegression,
    # Linear with an L2 penalty: stays stable when features are correlated
    # (engineered windows of one quantity always are) — plain OLS does not.
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
    # Native missing values, histogram splits: the fast choice on millions of rows.
    "hist_gradient_boosting": HistGradientBoostingRegressor,
}
LINEAR_MODELS = ("linear_regression", "ridge")

# ponytail: small param grids for a fast first pass, widen if search quality matters.
# Keys are "regressor__"-prefixed — these tune the final Pipeline step, not a bare estimator.
PARAM_GRIDS = {
    "linear_regression": {},  # no hyperparameters worth searching — tune_hyperparams just refits it
    "random_forest": {
        "regressor__n_estimators": [100, 200, 400],
        "regressor__max_depth": [None, 5, 10, 20],
    },
    "xgboost": {
        "regressor__n_estimators": [100, 200, 400],
        "regressor__max_depth": [3, 5, 7],
        "regressor__learning_rate": [0.01, 0.1, 0.3],
    },
    "ridge": {"regressor__alpha": [0.1, 1.0, 10.0, 100.0]},
    "hist_gradient_boosting": {
        "regressor__learning_rate": [0.03, 0.1, 0.3],
        "regressor__max_leaf_nodes": [15, 31, 63],
        "regressor__l2_regularization": [0.0, 1.0],
    },
}


def param_grid(meta: dict, model: str) -> dict:
    """PARAM_GRIDS for this run's pipeline shape: with a transformed target
    the estimator sits one level deeper (TransformedTargetRegressor.regressor)."""
    grid = PARAM_GRIDS[model]
    if meta.get("target_transform") in TARGET_TRANSFORMS:
        return {k.replace("regressor__", "regressor__regressor__", 1): v for k, v in grid.items()}
    return grid


# Transforms a skewed, non-negative target (revenue, ARPU, claim amount,
# usage) can be modelled in. The model fits log1p(y); predictions and every
# metric come back in the original units via expm1.
TARGET_TRANSFORMS = {"log1p": (np.log1p, np.expm1)}


# What the model minimises. squared_error predicts the conditional MEAN;
# poisson also predicts the mean, through a log link, for counts and other
# non-negative targets — unbiased on totals, unlike a log1p target whose
# back-transform systematically under-predicts them; quantile predicts the
# q-th QUANTILE, for decisions whose errors cost asymmetrically (provision
# for P90 demand, not the average).
OBJECTIVES = ("squared_error", "poisson", "quantile")
OBJECTIVE_MODELS = {
    "squared_error": tuple(MODELS),
    "poisson": ("linear_regression", "ridge", "random_forest", "xgboost", "hist_gradient_boosting"),
    "quantile": ("linear_regression", "ridge", "xgboost", "hist_gradient_boosting"),
}


def objective_of(meta: dict | None) -> tuple[str, float]:
    meta = meta or {}
    return meta.get("objective") or "squared_error", float(meta.get("quantile") or 0.5)


def _make_model(model: str, meta: dict | None = None):
    objective, q = objective_of(meta)
    if model not in OBJECTIVE_MODELS[objective]:
        raise ValueError(f"{model} has no {objective} loss — use one of {list(OBJECTIVE_MODELS[objective])}")
    penalised = 0.0 if model == "linear_regression" else 1.0
    if objective == "poisson":
        if model in LINEAR_MODELS:
            return PoissonRegressor(alpha=penalised, max_iter=1000)
        if model == "random_forest":
            return RandomForestRegressor(criterion="poisson", random_state=42)
        if model == "xgboost":
            return XGBRegressor(objective="count:poisson", random_state=42)
        return HistGradientBoostingRegressor(loss="poisson", random_state=42)
    if objective == "quantile":
        if model in LINEAR_MODELS:
            return QuantileRegressor(quantile=q, alpha=penalised, solver="highs")
        if model == "xgboost":
            return XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, random_state=42)
        return HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=42)
    if model in ("xgboost", "random_forest", "hist_gradient_boosting"):
        return MODELS[model](random_state=42)
    return MODELS[model]()


def _cv_scoring(meta: dict | None):
    """(sklearn scorer, metric name, sign): R^2 for mean objectives; pinball
    loss for a quantile objective — R^2 scores distance from the MEAN, which
    a P90 model is deliberately not predicting. value = sign * sklearn score."""
    objective, q = objective_of(meta)
    if objective == "quantile":
        return make_scorer(mean_pinball_loss, alpha=q, greater_is_better=False), "pinball", -1
    return "r2", "r2", 1


def split_keys(meta: dict) -> list[str]:
    return shared_validation.split_keys(meta)


def _cv_sample(X, y, meta: dict | None = None):
    """Structure-keeping sample for CV / search only (core/validation.py)."""
    return shared_validation.cv_sample(X, y, meta, stratify=False)


def _validation_folds(X, y, meta: dict, n_splits: int = 5, seed: int = 42):
    """Folds that obey the run's split — forward-chained, grouped or shuffled
    (core/validation.py, shared with classification)."""
    return shared_validation.validation_folds(X, y, meta, n_splits, seed, stratify=False)


def _run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def _load_meta(run_id: str) -> dict:
    path = _run_dir(run_id) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Unknown run_id '{run_id}' — call prepare_dataset first."
        )
    return json.loads(path.read_text())


def _save_meta(run_id: str, meta: dict) -> None:
    (_run_dir(run_id) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


# Every @mcp.tool() call from here on appends to its run's execution ledger —
# what a readiness gate reads to tell a step that ran from a step that was
# only claimed. Installed before the tools below are defined, because it works
# by wrapping the decorator itself.
gate_engine.install_ledger(mcp, _run_dir, _load_meta, _save_meta)


def _load_split(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta = _load_meta(run_id)
    train = pd.read_csv(_run_dir(run_id) / "train.csv")
    test = pd.read_csv(_run_dir(run_id) / "test.csv")
    return train, test, meta


def _to_native(value):
    """numpy scalar -> native Python type, so it survives a json.dumps/loads
    round-trip as the same type."""
    return value.item() if hasattr(value, "item") else value


def _build_pipeline(meta: dict, model: str, X_train: pd.DataFrame) -> Pipeline:
    """The one place the full preprocess-then-regress pipeline is assembled
    — train_model and tune_hyperparams both go through this, so a tuned
    refit and a baseline fit are structurally the same shape."""
    steps = []
    # Split keys (entity id, time column) stay in the data for validation and
    # never reach the model: an id one-hot is a memorised-entity lookup, a raw
    # timestamp a trend no tree can extrapolate past the training period.
    dropped = [*(meta.get("dropped_columns") or []), *split_keys(meta)]
    if dropped:
        steps.append(("drop", ColumnDropper(dropped)))
    if meta.get("imputation"):
        steps.append(("impute", ColumnFiller(meta["imputation"])))

    remaining = [c for c in X_train.columns if c not in dropped]
    cat_cols = [c for c in remaining if not pd.api.types.is_numeric_dtype(X_train[c])]
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"), cat_cols)
    ]
    if model in LINEAR_MODELS:
        # Linear models need comparable scales and cannot take NaN; trees need
        # neither. Fitted inside the pipeline, on training rows only.
        num_cols = [c for c in remaining if c not in cat_cols]
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                              ("scale", StandardScaler())]), num_cols))
    steps.append(
        ("encode", ColumnTransformer(transformers, remainder="passthrough", verbose_feature_names_out=False))
    )
    estimator = _make_model(model, meta)
    transform = meta.get("target_transform")
    if transform in TARGET_TRANSFORMS:
        func, inverse = TARGET_TRANSFORMS[transform]
        estimator = TransformedTargetRegressor(regressor=estimator, func=func, inverse_func=inverse,
                                               check_inverse=False)
    steps.append(("regressor", estimator))
    return Pipeline(steps)


def _evaluate(pipeline: Pipeline, X_test, y_test, meta: dict | None = None) -> dict:
    """R^2, RMSE, MAE, median AE — and MAPE only when it means something: a
    target with values at or near zero makes percentage errors explode
    on rows nobody cares about, so it is withheld with the reason."""
    y_pred = pipeline.predict(X_test)
    y_true = np.asarray(y_test, dtype=float)
    out = {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "median_ae": round(float(np.median(np.abs(y_true - y_pred))), 4),
        # What a budget or capacity plan is set from: the predicted total
        # against the actual total. A log1p target's back-transform lands
        # below 1 by construction — the mean of exp is not exp of the mean.
        "sum_ratio": round(float(y_pred.sum() / y_true.sum()), 4) if y_true.sum() else None,
    }
    objective, q = objective_of(meta)
    if objective == "quantile":
        out["pinball_loss"] = round(float(mean_pinball_loss(y_true, y_pred, alpha=q)), 4)
        # The honest check of a quantile model: the share of actuals at or
        # below the prediction should be q.
        out["quantile_hit_rate"] = round(float(np.mean(y_true <= y_pred)), 4)
    floor = 0.01 * np.median(np.abs(y_true)) if len(y_true) else 0
    if len(y_true) and np.all(np.abs(y_true) > floor) and floor > 0:
        out["mape"] = round(float(np.mean(np.abs((y_true - y_pred) / y_true))), 4)
    else:
        out["mape"] = None
        out["mape_withheld"] = "target has values at or near zero — percentage error is not meaningful here"
    return out


# Results that describe one FITTED model. A refit clears them, so no report
# attributes the previous model's explanation, residuals, stability or
# intervals to the new one, and the readiness gate asks for them again.
MODEL_DERIVED_KEYS = ("explainability", "residuals", "stability", "prediction_interval", "error_analysis")


def _invalidate_model_derived(meta: dict) -> list:
    stale = [k for k in MODEL_DERIVED_KEYS if meta.get(k)]
    for k in stale:
        meta.pop(k, None)
    return stale


# ponytail: a flat threshold on CV-vs-test R^2 gap, not a learned overfitting
# detector — upgrade to a per-dataset-size threshold if this flags too many
# false positives on small test folds.
OVERFIT_GAP_THRESHOLD = 0.15


def _overfit_gate(cv_value: float, test_eval: dict, metric: str = "r2") -> tuple[float | None, bool]:
    """cv_to_test_gap on the metric the model was selected on. R^2: CV mean
    minus test R^2 (flag above OVERFIT_GAP_THRESHOLD). Pinball loss: the
    RELATIVE rise from CV to test (flag above 25%) — a loss has no fixed
    scale to subtract on. A large gap means CV was optimistic about the
    untouched test fold; export_model checks it."""
    if metric == "pinball":
        test_loss = test_eval.get("pinball_loss")
        if test_loss is None or not cv_value:
            return None, False
        gap = round((test_loss - cv_value) / abs(cv_value), 4)
        return gap, gap > 0.25
    test_r2 = test_eval.get("r2")
    if test_r2 is None:
        return None, False
    gap = round(cv_value - test_r2, 4)
    return gap, gap > OVERFIT_GAP_THRESHOLD


# ── Deterministic readiness layer ─────────────────────────────────────────────
# What is TRUE about a run, computed from its artifacts, independent of
# anything the agent says about it. The engine (statuses, the roll-up, the
# success-criteria rule, the ledger) is core/gates.py, shared with every other
# agent; what lives here is only what "finished and safe to believe" means for
# a continuous target.
#
# The cardinal rule, inherited from the engine: missing evidence is never
# passing evidence. A gate that never ran reports not_run, which is not a pass.

REQUIRED_GATES = (
    "business_context",
    "leakage_screened",
    "no_identifier_in_model",
    "baseline_beaten",
    "no_overfitting",
    "residuals_checked",
    "stability_checked",
    "explainability_run",
    "explainability_clean",
    "evidence_ordering",
    "success_criteria",
)

# Metrics a recorded success bar can point at, and which direction is better.
# Rejected at record time rather than silently ignored at report time.
SUCCESS_METRIC_DIRECTION = {"r2": "at_least", "rmse": "at_most", "mae": "at_most", "median_ae": "at_most",
                            "pinball_loss": "at_most"}

SUPPORTED_SUCCESS_METRICS = tuple(SUCCESS_METRIC_DIRECTION)

# |Spearman rho| above which a feature is treated as leaking the target.
# Deliberately this close to 1.0, because the correlation is RANK-based: a
# leaked column is a monotone transform of the target (other units, rounded,
# a post-outcome field derived from it), and every monotone transform
# preserves ranks exactly, so a copy scores ~1.0 however it was mangled. A
# genuinely strong driver does not: `spend = 3 * usage + noise` reaches 0.985
# and is a feature, not a leak. A lower bar here fails honest runs.
LEAKAGE_CORRELATION = 0.999

# Fraction of the target's own spread that the mean residual may drift by
# before the model is called biased.
RESIDUAL_BIAS_RATIO = 0.05

# |Spearman rho| between |residual| and prediction above which the error is
# treated as heteroscedastic — the model is reliable in one part of the range
# and not another, which a single RMSE hides.
HETEROSCEDASTICITY_RHO = 0.3

# Spread of R^2 across repeated CV splits above which a single reported score
# is one split's luck rather than a property of the model.
STABILITY_STD_CONCERN = 0.1


def _current_metrics(meta: dict) -> dict:
    return meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}


def measured_metric(meta: dict, metric: str):
    """One named metric out of this run's CURRENT held-out test results, or
    None when it genuinely isn't available — the caller reports that rather
    than substituting a different metric that happens to exist."""
    return _current_metrics(meta).get(metric)


def _explainability_gates(meta: dict, train) -> tuple:
    """explain_model ran, and what it found describes features rather than
    identifiers. The second half is the `Ticket` incident: an identifier
    reached the model, one-hot expanded, and dominated the importances."""
    explainability = meta.get("explainability")
    if not explainability:
        return (
            gate_engine.gate(
                gate_engine.NOT_RUN, "explain_model was never called for this run"
            ),
            gate_engine.gate(
                gate_engine.NOT_RUN,
                "no importances to screen — explain_model never ran",
            ),
        )

    ran = gate_engine.gate(
        gate_engine.PASS,
        f"explain_model ran ({explainability.get('method')}), top feature "
        f"'{(explainability.get('top_features') or [['none']])[0][0]}'",
    )

    features = [name for name, _ in explainability.get("top_features") or []]
    target = meta["target"]
    identifiers = {
        c
        for c, s in identifier_columns(train, target).items()
        if s["strength"] == "strong"
    }
    hits = {}
    for col in identifiers:
        expanded = col in train.columns and not pd.api.types.is_numeric_dtype(
            train[col]
        )
        matched = [
            f for f in features if f == col or (expanded and f.startswith(f"{col}_"))
        ]
        if matched:
            hits[col] = matched
    if hits:
        clean = gate_engine.gate(
            gate_engine.FAIL,
            "identifier-shaped columns are among the model's most important features: "
            + "; ".join(f"`{col}` (as {matched})" for col, matched in hits.items())
            + " — the model is memorizing rows, not learning the target",
        )
    else:
        clean = gate_engine.gate(
            gate_engine.PASS,
            "no identifier-shaped column appears among the top features",
        )
    return ran, clean


def compute_readiness(run_id: str) -> dict:
    """Every required gate's status for this run, computed from its artifacts."""
    meta = _load_meta(run_id)
    train, _, _ = _load_split(run_id)
    target = meta["target"]
    metrics = _current_metrics(meta)
    checks: dict = {}

    bu = meta.get("business_understanding") or {}
    checks["business_context"] = (
        gate_engine.gate(
            gate_engine.PASS,
            f"domain='{bu.get('domain') or 'not declared'}', objective and target definition recorded",
        )
        if bu.get("business_objective") and bu.get("target_definition")
        else gate_engine.gate(
            gate_engine.NOT_RUN,
            "record_business_context was not called — no business objective or target definition on file",
        )
    )

    leakage = meta.get("leakage")
    if not leakage:
        checks["leakage_screened"] = gate_engine.gate(
            gate_engine.NOT_RUN, "detect_data_leakage was never called for this run"
        )
    elif leakage.get("target_correlated_columns"):
        checks["leakage_screened"] = gate_engine.gate(
            gate_engine.FAIL,
            f"columns that give away the target are still in the training fold: {list(leakage['target_correlated_columns'])}"
            " — drop them (apply_drop_columns) and retrain",
        )
    else:
        checks["leakage_screened"] = gate_engine.gate(
            gate_engine.PASS, "detect_data_leakage ran; no target-copy column found"
        )

    acknowledged = set(meta.get("acknowledged_identifiers") or {})
    live = {
        c: s
        for c, s in identifier_columns(train, target).items()
        if s["strength"] == "strong"
    }
    unacknowledged = {c: s for c, s in live.items() if c not in acknowledged}
    if unacknowledged:
        checks["no_identifier_in_model"] = gate_engine.gate(
            gate_engine.FAIL,
            "identifier-shaped columns are still in the training fold and will be encoded as model features: "
            + "; ".join(f"`{c}` ({s['reason']})" for c, s in unacknowledged.items())
            + " — drop them (apply_drop_columns) and retrain, or record a justification via "
            "acknowledge_identifier_column if one is genuinely a feature here",
        )
    elif live:
        checks["no_identifier_in_model"] = gate_engine.gate(
            gate_engine.PASS,
            f"identifier-shaped columns kept with recorded justification: {sorted(live)}",
        )
    else:
        checks["no_identifier_in_model"] = gate_engine.gate(
            gate_engine.PASS, "no identifier-shaped column remains in the training fold"
        )

    baseline = meta.get("baseline_comparison")
    quantile_run = objective_of(meta)[0] == "quantile"
    key = "pinball_loss" if quantile_run else "r2"
    model_r2, base_r2 = measured_metric(meta, key), (baseline or {}).get(key)
    if baseline is None:
        checks["baseline_beaten"] = gate_engine.gate(
            gate_engine.NOT_RUN,
            "train_baseline was never called — no reference point for 'better than guessing'",
        )
    elif model_r2 is None:
        checks["baseline_beaten"] = gate_engine.gate(
            gate_engine.NOT_RUN, "no model has been trained for this run"
        )
    else:
        better = model_r2 < base_r2 if quantile_run else model_r2 > base_r2
        checks["baseline_beaten"] = gate_engine.gate(
            gate_engine.PASS if better else gate_engine.FAIL,
            f"model test {key} {model_r2} vs {baseline.get('kind', 'mean')}-prediction baseline {base_r2}"
            + (" (the strongest of the naive baselines)" if baseline.get("candidates") else ""),
        )

    if not metrics:
        checks["no_overfitting"] = gate_engine.gate(
            gate_engine.NOT_RUN, "no model has been trained for this run"
        )
    elif metrics.get("overfitting_warning"):
        checks["no_overfitting"] = gate_engine.gate(
            gate_engine.FAIL,
            f"CV-to-test gap on {metrics.get('cv_metric', 'r2')} is {metrics.get('cv_to_test_gap')} — above the limit",
        )
    else:
        checks["no_overfitting"] = gate_engine.gate(
            gate_engine.PASS,
            f"CV-to-test gap on {metrics.get('cv_metric', 'r2')} is {metrics.get('cv_to_test_gap')} — within the limit",
        )

    residuals = meta.get("residuals")
    if not residuals:
        checks["residuals_checked"] = gate_engine.gate(
            gate_engine.NOT_RUN,
            "check_residuals was never called — how the model is wrong is unknown",
        )
    elif residuals.get("biased") or residuals.get("heteroscedastic"):
        problems = []
        if residuals.get("biased"):
            problems.append(
                f"systematically biased (mean residual {residuals['mean_residual']}, "
                f"{residuals['bias_ratio']} of the target's own spread)"
            )
        if residuals.get("heteroscedastic"):
            problems.append(
                f"heteroscedastic (|residual| vs prediction rho {residuals['abs_residual_vs_prediction_rho']}) "
                "— accuracy depends on where in the range the prediction falls"
            )
        checks["residuals_checked"] = gate_engine.gate(
            gate_engine.FAIL, "residuals show the model is " + "; ".join(problems)
        )
    else:
        checks["residuals_checked"] = gate_engine.gate(
            gate_engine.PASS,
            "residuals are unbiased and evenly spread across the prediction range",
        )

    stability = meta.get("stability")
    if not stability:
        checks["stability_checked"] = gate_engine.gate(
            gate_engine.NOT_RUN,
            "check_model_stability was never called — the reported score may be one split's luck",
        )
    elif not stability.get("stable"):
        checks["stability_checked"] = gate_engine.gate(
            gate_engine.FAIL,
            f"R^2 swings {stability['r2_min']}–{stability['r2_max']} across {stability['splits']} splits "
            f"(std {stability['r2_std']}) — the single reported score is not reproducible",
        )
    else:
        checks["stability_checked"] = gate_engine.gate(
            gate_engine.PASS,
            f"R^2 {stability['r2_mean']} ± {stability['r2_std']} across {stability['splits']} splits",
        )

    checks["explainability_run"], checks["explainability_clean"] = (
        _explainability_gates(meta, train)
    )

    checks["evidence_ordering"] = gate_engine.check_screen_ordering(
        meta,
        "detect_data_leakage",
        ("apply_drop_columns", "apply_imputation", "apply_datetime_features", "apply_custom_feature",
         "apply_entity_features", "apply_peer_features", "apply_features"),
    )
    checks["success_criteria"] = gate_engine.evaluate_success_criteria(
        meta, measured_metric, SUPPORTED_SUCCESS_METRICS
    )

    return gate_engine.readiness(run_id, checks, REQUIRED_GATES)
