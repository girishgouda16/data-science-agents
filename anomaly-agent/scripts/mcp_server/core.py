"""MCP server exposing tabular-anomaly-detection tools. prepare_dataset
splits train/test BEFORE any learned preprocessing, so every downstream step
— imputation, encoding, scaling, the detector itself — fits only on the
training fold. Everything after that is keyed by run_id, not a file path:
one fitted pipeline.pkl is built once by train_model, so explain_model /
export_model / predict always reflect whichever detector is actually current
for that run.

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

import pandas as pd

from mcp.server.fastmcp import FastMCP

from pipeline_transformers import ColumnDropper, ColumnFiller

from pyod.models.ecod import ECOD

from pyod.models.iforest import IForest

from pyod.models.knn import KNN

from pyod.models.lof import LOF

from pyod.models.ocsvm import OCSVM

from run_persistence import save_training_run

from sklearn.compose import ColumnTransformer

from scipy.stats import norm

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.insert(
    0, str(Path(__file__).parent.resolve().parents[2])
)  # repo root, for core.*

from core import mlops as core_mlops  # noqa: E402
from core.datasource import read_table  # noqa: E402,F401

mcp = FastMCP("anomaly-agent")
from core import toolguard  # noqa: E402

toolguard.install(
    mcp
)  # every tool's paths/models checked before it runs (core/toolguard.py)

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

ALGORITHMS = {
    "iforest": IForest,
    "knn": KNN,
    "lof": LOF,
    "ecod": ECOD,
    "ocsvm": OCSVM,
}
# Neighbour / kernel detectors cost O(n^2): fitted on a sample of this many
# rows (every row is still scored). IForest and ECOD fit on everything.
QUADRATIC = {"knn", "lof", "ocsvm"}
FIT_SAMPLE_ROWS = 20_000
# Non-negative numerics this right-skewed (usage, spend, counts) are logged
# before scaling, or every heavy user reads as an anomaly.
SKEW_LOG_THRESHOLD = 1.0
# Without labels the only evidence a flagged list is real is that it comes
# back: refit on 80% subsamples, compare the top alerts on the same rows by
# Jaccard overlap. Below this the list is mostly an artefact of the sample.
STABLE_JACCARD = 0.5
STABILITY_RESAMPLES = 5
STABILITY_MAX_ROWS = 5_000
# With labels: the review queue (top `contamination` share by score) must be
# this many times richer in anomalies than the base rate to be worth working.
MIN_LIFT_AT_BUDGET = 2.0

MIN_CONTAMINATION = 0.0001

MAX_CONTAMINATION = 0.5

EXPORT_MIN_ROC_AUC = 0.55

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


def _load_split(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta = _load_meta(run_id)
    train = pd.read_csv(_run_dir(run_id) / "train.csv")
    test = pd.read_csv(_run_dir(run_id) / "test.csv")
    return train, test, meta


def _to_native(value):
    return value.item() if hasattr(value, "item") else value


def _json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return _to_native(value)


def _feature_frame(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    label_column = meta.get("label_column")
    if label_column and label_column in df.columns:
        return df.drop(columns=[label_column])
    return df.copy()


def _label_classes(series: pd.Series) -> tuple[object, object]:
    counts = series.value_counts()
    anomaly_label = counts.idxmin()
    normal_label = counts.idxmax()
    return normal_label, anomaly_label


def _label_to_binary(series: pd.Series, meta: dict) -> pd.Series:
    return (series == meta["anomaly_label"]).astype(int)


def _clamp_contamination(value: float) -> float:
    value = float(value)
    return min(MAX_CONTAMINATION, max(MIN_CONTAMINATION, value))


def _make_detector(algorithm: str, contamination: float):
    kwargs = {"contamination": _clamp_contamination(contamination)}
    if algorithm == "iforest":
        kwargs["random_state"] = 42
    return ALGORITHMS[algorithm](**kwargs)


def split_keys(meta: dict) -> list:
    """The entity and time columns the split was made on: kept in the data
    (features are built from them), never fed to the detector."""
    return [c for c in (meta.get("group_column"), meta.get("time_column")) if c]


def detector_inputs(X: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """The raw columns the detector sees — what per-row reasons are about."""
    out = set(meta.get("dropped_columns") or []) | set(split_keys(meta))
    return X[[c for c in X.columns if c not in out]]


def _build_pipeline(
    meta: dict, algorithm: str, contamination: float, X_train: pd.DataFrame
) -> Pipeline:
    """drop (dropped columns, split keys) -> impute -> encode -> detector.
    Distances are only as sensible as the space they are measured in:
      - right-skewed non-negative numerics (usage, spend, counts) are logged
        before standardising, or every heavy user reads as an anomaly
      - categoricals are one-hot and NOT scaled: a standardised 0.1% dummy
        is worth 30 sd, and every row of a rare but normal category would
        be flagged by the distance detectors"""
    dropped = [*(meta.get("dropped_columns") or []), *split_keys(meta)]
    steps = []
    if dropped:
        steps.append(("drop", ColumnDropper(dropped)))
    if meta.get("imputation"):
        steps.append(("impute", ColumnFiller(meta["imputation"])))
    remaining = [c for c in X_train.columns if c not in dropped]
    cat_cols = [c for c in remaining if not pd.api.types.is_numeric_dtype(X_train[c])
                or pd.api.types.is_bool_dtype(X_train[c])]
    num_cols = [c for c in remaining if c not in cat_cols]
    log_cols = [c for c in num_cols
                if X_train[c].notna().any() and X_train[c].min() >= 0 and X_train[c].skew() > SKEW_LOG_THRESHOLD]
    plain_cols = [c for c in num_cols if c not in log_cols]
    transformers = []
    if log_cols:
        transformers.append(("log", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler())]), log_cols))
    if plain_cols:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                              ("scale", StandardScaler())]), plain_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                                  drop="if_binary"), cat_cols))
    steps.append(("encode", ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)))
    steps.append(("detector", _make_detector(algorithm, contamination)))
    return Pipeline(steps)


def log_scaled_columns(pipeline: Pipeline) -> list:
    return [c for name, _, cols in pipeline.named_steps["encode"].transformers_ if name == "log" for c in cols]


def _score_vector(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipeline.decision_function(X), dtype=float)  # PyOD: higher = more anomalous


def _evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, budget_share: float) -> dict:
    """Held-out detection quality. At a 1% base rate ROC-AUC flatters: 0.9
    can still mean most alerts are false. What a review team lives with is
    the QUEUE — the top `budget_share` of rows by score: its precision, the
    share of all anomalies it catches, and its lift over the base rate —
    plus average precision (PR-AUC) as the threshold-free summary."""
    y = np.asarray(y_test).astype(int)
    y_pred = pipeline.predict(X_test)
    scores = _score_vector(pipeline, X_test)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    prevalence = float(y.mean())
    k = max(1, int(round(budget_share * len(y))))
    queue = np.argsort(-scores, kind="stable")[:k]
    precision_at = float(y[queue].mean())
    result = {
        "classification_report": report,
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "predicted_anomaly_rate": round(float(np.mean(y_pred)), 4),
        "test_label_prevalence": round(prevalence, 4),
        "alerts_at_budget": k,
        "precision_at_budget": round(precision_at, 4),
        "recall_at_budget": round(float(y[queue].sum() / max(y.sum(), 1)), 4),
        "lift_at_budget": round(precision_at / prevalence, 2) if prevalence else None,
    }
    try:
        result["roc_auc"] = round(float(roc_auc_score(y, scores)), 4)
        result["average_precision"] = round(float(average_precision_score(y, scores)), 4)
    except ValueError:
        result["roc_auc"] = result["average_precision"] = None
    return result


def alert_stability(algorithm: str, contamination: float, Xt_fit: np.ndarray, Xt_eval: np.ndarray,
                    eval_scores: np.ndarray, seed: int = 42) -> dict:
    """Do the same rows come back as the top alerts? Refit the detector on
    80% subsamples of the fit rows (a STABILITY_MAX_ROWS sample) and compare
    each refit's top-k on the evaluation rows with the full detector's, by
    Jaccard overlap. k = contamination x rows, at least 5."""
    rng = np.random.default_rng(seed)
    base = Xt_fit[rng.choice(len(Xt_fit), size=min(len(Xt_fit), STABILITY_MAX_ROWS), replace=False)]
    k = min(len(Xt_eval), max(5, int(round(contamination * len(Xt_eval)))))
    full_top = set(np.argsort(-eval_scores, kind="stable")[:k].tolist())
    overlaps = []
    for _ in range(STABILITY_RESAMPLES):
        sub = base[rng.choice(len(base), size=max(2, int(0.8 * len(base))), replace=False)]
        refit = _make_detector(algorithm, contamination).fit(sub)
        top = set(np.argsort(-np.asarray(refit.decision_function(Xt_eval)), kind="stable")[:k].tolist())
        overlaps.append(len(top & full_top) / len(top | full_top))
    return {"top_alert_jaccard_mean": round(float(np.mean(overlaps)), 4),
            "top_alert_jaccard_min": round(float(np.min(overlaps)), 4),
            "alerts_compared": k, "resamples": len(overlaps)}


def rare_category_warnings(X_train: pd.DataFrame, X_alerts: pd.DataFrame, rare: float = 0.05) -> list:
    """Tree and tail detectors (iforest, ecod) isolate a rare category in one
    split, so a rare but perfectly normal plan or device model can fill the
    queue. Flags a categorical whose rare levels (< 5% of training rows)
    make up 30%+ of the alerts at 3x+ their base rate."""
    warnings = []
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]) and not pd.api.types.is_bool_dtype(X_train[col]):
            continue
        shares = X_train[col].astype(str).value_counts(normalize=True)
        in_alerts = float((X_alerts[col].astype(str).map(shares).fillna(0) < rare).mean())
        base = float((X_train[col].astype(str).map(shares) < rare).mean())
        if in_alerts >= 0.3 and in_alerts > 3 * max(base, 1e-9):
            warnings.append(f"{in_alerts:.0%} of the top alerts are rows with a rare `{col}` value (vs {base:.1%} of "
                            f"rows) — the detector may be flagging rarity, not behaviour. If those values are normal "
                            f"business, drop `{col}` from the detector or use it as a peer key "
                            "(apply_peer_features), or compare knn / lof, which weigh a category by distance")
    return warnings


def reason_reference(X: pd.DataFrame, max_categories: int = 1000) -> dict:
    """What 'typical' means per column, from the training fold: median and a
    robust scale (IQR / 1.349, std when the IQR is zero) for numerics, value
    shares for categoricals."""
    numeric, categorical = {}, {}
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            s = s.dropna().astype(float)
            if s.empty:
                continue
            q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
            scale = (q3 - q1) / 1.349 or (float(s.std()) if len(s) > 1 else 0.0) or 1.0
            numeric[col] = [float(med), float(scale)]
        else:
            shares = s.astype(str).value_counts(normalize=True)
            categorical[col] = {"shares": shares.head(max_categories).round(6).to_dict(),
                                "tail_share": float(shares.iloc[max_categories]) if len(shares) > max_categories else 0.0}
    return {"numeric": numeric, "categorical": categorical}


def row_reasons(X: pd.DataFrame, ref: dict, top: int = 3, min_z: float = 2.0) -> list:
    """Why each row is unusual, in words a reviewer can check: its most
    extreme columns against the training fold — numerics in robust standard
    deviations from the median, categories by how rare they were (converted
    to the same z scale, so the two rank together)."""
    parts = {}
    for col, (med, scale) in ref["numeric"].items():
        if col in X.columns:
            parts[col] = (pd.to_numeric(X[col], errors="coerce") - med) / scale
    for col, cat in ref["categorical"].items():
        if col in X.columns:
            share = X[col].astype(str).map(cat["shares"]).fillna(cat["tail_share"])
            parts[col] = pd.Series(norm.isf(share.clip(lower=1e-5).to_numpy() / 2), index=X.index)
    if not parts:
        return [""] * len(X)
    Z = pd.DataFrame(parts, index=X.index)
    out = []
    for idx, row in Z.abs().iterrows():
        texts = []
        for col in row[row >= min_z].sort_values(ascending=False).index[:top]:
            value = X.at[idx, col]
            if col in ref["numeric"]:
                texts.append(f"{col}={float(value):.4g} ({Z.at[idx, col]:+.1f} robust sd vs typical "
                             f"{ref['numeric'][col][0]:.4g})")
            else:
                share = ref["categorical"][col]["shares"].get(str(value), 0.0)
                texts.append(f"{col}={value} ({f'{share:.2%} of training rows' if share else 'never seen in training'})")
        out.append("; ".join(texts) or "no single column is extreme: an unusual combination")
    return out


def _score_distribution(scores: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=float)
    return {
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "p05": round(float(np.percentile(scores, 5)), 4),
        "p25": round(float(np.percentile(scores, 25)), 4),
        "p50": round(float(np.percentile(scores, 50)), 4),
        "p75": round(float(np.percentile(scores, 75)), 4),
        "p95": round(float(np.percentile(scores, 95)), 4),
    }


def _current_metrics(meta: dict) -> dict:
    return meta.get("metrics") or {}


def _looks_id_like(series: pd.Series, n_rows: int) -> bool:
    if series.nunique(dropna=True) < n_rows * 0.98:
        return False
    if "id" in series.name.lower():
        return True
    if pd.api.types.is_integer_dtype(series):
        return True
    return series.dtype == object


# Every tool call on a run is recorded in its execution ledger (core/gates.py),
# so the readiness gates judge what ACTUALLY ran, in what order. Installed here,
# before any tool module decorates its tools.
from core import gates as _gate_engine  # noqa: E402

_gate_engine.install_ledger(mcp, _run_dir, _load_meta, _save_meta)
