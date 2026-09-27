"""MCP server exposing tabular-clustering tools. prepare_dataset splits a
fit fold and holdout fold BEFORE any learned preprocessing, so every
downstream step — imputation, encoding, scaling, the clusterer itself —
follows the same split-before-fit discipline as the classification agent.
Everything after that is keyed by run_id, not a file path.

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

from pipeline_transformers import ColumnDropper, ColumnFiller, KMedoids

from run_persistence import save_training_run

from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture

from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler

sys.path.insert(
    0, str(Path(__file__).parent.resolve().parents[2])
)  # repo root, for core.*

from core import mlops as core_mlops  # noqa: E402

mcp = FastMCP("clustering-agent")
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

# ponytail: flat TTL swept opportunistically, not a scheduler — add a "pin"
# flag in meta.json if a run ever needs to outlive this on purpose.
RUN_RETENTION_DAYS = float(os.environ.get("RUN_RETENTION_DAYS", 7))


def _cleanup_old_runs(max_age_days: float = RUN_RETENTION_DAYS) -> None:
    """Runs are disposable working state (fit/holdout CSVs + pipeline.pkl),
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


ALGORITHMS = {"kmeans", "minibatch_kmeans", "kmedoids", "gmm", "hierarchical", "dbscan", "hdbscan"}
# Algorithms that place a NEW row in an existing cluster themselves; the rest
# go through nearest-centroid assignment (density-based ones not at all).
PREDICTS = {"kmeans", "minibatch_kmeans", "kmedoids", "gmm"}
# Clustered in the Gower space under Manhattan distance (mixed numeric +
# categorical data); everything else in the standardised Euclidean space.
GOWER = {"kmedoids"}
DENSITY_BASED = {"dbscan", "hdbscan"}
# Agglomerative clustering holds an n x n structure: fine for tens of
# thousands of rows, a memory failure at telco scale.
HIERARCHICAL_MAX_ROWS = 20_000

OVERFIT_GAP_THRESHOLD = 0.15  # kept for older runs' reports; stability (ARI) is the check now
# Segments that do not come back when the data is resampled are artefacts of
# one sample. Adjusted Rand index between the full-fit labels and refits on
# 80% subsamples: >= 0.75 stable, 0.5-0.75 partly, < 0.5 dissolves (after
# Hennig's clusterboot bands, on the ARI scale).
STABLE_ARI = 0.7
STABILITY_RESAMPLES = 10
STABILITY_MAX_ROWS = 3_000
# Silhouette is O(n^2); a stratification-free 5,000-row sample estimates it
# to ~0.01 and keeps every call interactive.
METRIC_SAMPLE_ROWS = 5_000
# Non-negative numerics this right-skewed (usage, spend, counts) are logged
# before scaling, or a handful of heavy users become their own segments.
SKEW_LOG_THRESHOLD = 1.0


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
    fit = pd.read_csv(_run_dir(run_id) / "train.csv")
    holdout = pd.read_csv(_run_dir(run_id) / "test.csv")
    return fit, holdout, meta


def _to_native(value):
    return value.item() if hasattr(value, "item") else value


def _round_or_none(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def _label_counts(labels) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(_to_native(v)): int(c) for v, c in zip(values, counts)}


def _cluster_count(labels) -> int:
    unique = set(np.unique(labels).tolist())
    return len(unique - {-1}) if -1 in unique else len(unique)


def distance_metric(algorithm: str) -> str:
    return "manhattan" if algorithm in GOWER else "euclidean"


def _safe_cluster_metrics(X, labels, metric: str = "euclidean") -> dict:
    """Size, noise and separation of one labelling. Separation metrics are
    computed on non-noise points only (a noise 'cluster' is not a segment),
    silhouette on a METRIC_SAMPLE_ROWS sample."""
    labels = np.asarray(labels)
    noise = labels == -1
    metrics = {
        "n_clusters_found": int(_cluster_count(labels)),
        "cluster_size_distribution": _label_counts(labels),
        "noise_points": int(noise.sum()),
        "noise_share": round(float(noise.mean()), 4),
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
    }
    X_kept, kept = np.asarray(X)[~noise], labels[~noise]
    n_kept = len(np.unique(kept))
    if n_kept < 2 or n_kept >= len(kept):
        return metrics
    metrics["silhouette_score"] = _round_or_none(
        silhouette_score(X_kept, kept, metric=metric, sample_size=min(len(kept), METRIC_SAMPLE_ROWS),
                         random_state=42))
    metrics["davies_bouldin_score"] = _round_or_none(davies_bouldin_score(X_kept, kept))
    metrics["calinski_harabasz_score"] = _round_or_none(calinski_harabasz_score(X_kept, kept))
    return metrics


def _make_clusterer(algorithm: str, *, n_clusters: int = 4, eps: float = 0.5, min_samples: int = 5,
                    linkage: str = "ward", min_cluster_size: int = 15, covariance_type: str = "full"):
    if algorithm == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    if algorithm == "kmedoids":
        return KMedoids(n_clusters=n_clusters)
    if algorithm == "minibatch_kmeans":
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=4096)
    if algorithm == "gmm":
        return GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, random_state=42, n_init=2)
    if algorithm == "hierarchical":
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    if algorithm == "dbscan":
        return DBSCAN(eps=eps, min_samples=min_samples)
    if algorithm == "hdbscan":
        return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    raise ValueError(f"unknown algorithm '{algorithm}', use one of {sorted(ALGORITHMS)}")


def _algorithm_params(algorithm: str, *, n_clusters: int, eps: float, min_samples: int, linkage: str,
                      min_cluster_size: int = 15, covariance_type: str = "full") -> dict:
    if algorithm == "dbscan":
        return {"eps": eps, "min_samples": min_samples}
    if algorithm == "hdbscan":
        return {"min_cluster_size": min_cluster_size, "min_samples": min_samples}
    if algorithm == "hierarchical":
        return {"n_clusters": n_clusters, "linkage": linkage}
    if algorithm == "gmm":
        return {"n_clusters": n_clusters, "covariance_type": covariance_type}
    return {"n_clusters": n_clusters}


def fit_labels(clusterer, X) -> np.ndarray:
    """Labels of the rows a clusterer was just fitted on (GMM has no labels_)."""
    labels = getattr(clusterer, "labels_", None)
    return np.asarray(labels if labels is not None else clusterer.predict(X))


def _build_preprocessor(meta: dict, X: pd.DataFrame, gower: bool = False) -> Pipeline:
    """Distances are only as sensible as the space they are measured in:
      - numerics are imputed and standardised; right-skewed non-negative
        ones (usage, spend, counts) are logged first, or a few heavy users
        become their own segments
      - categoricals are one-hot and NOT scaled: a standardised rare dummy
        gets a huge magnitude and the rarest categories would define the
        clusters
      - profile columns (outcomes, protected attributes) are kept out of the
        distance entirely — segments are described by them, not built on them
    gower=True builds the Gower space for mixed data: numerics min-max
    scaled to [0, 1], every category one-hot and weighted 0.5, so a
    Manhattan distance counts a category mismatch as 1 and a numeric gap as
    its share of the range — Gower distance times the number of variables."""
    steps = []
    dropped = [*(meta.get("dropped_columns") or []), *(meta.get("profile_columns") or [])]
    if dropped:
        steps.append(("drop", ColumnDropper(dropped)))
    if meta.get("imputation"):
        steps.append(("impute", ColumnFiller(meta["imputation"])))
    remaining = [c for c in X.columns if c not in dropped]
    cat_cols = [c for c in remaining if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in remaining if c not in cat_cols]
    log_cols = [c for c in num_cols
                if X[c].notna().any() and X[c].min() >= 0 and X[c].skew() > SKEW_LOG_THRESHOLD]
    plain_cols = [c for c in num_cols if c not in log_cols]
    scaler = MinMaxScaler if gower else StandardScaler
    transformers = []
    if log_cols:
        transformers.append(("log", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", scaler())]), log_cols))
    if plain_cols:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                              ("scale", scaler())]), plain_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                                  drop=None if gower else "if_binary"), cat_cols))
    weights = {"cat": 0.5} if gower and cat_cols else None
    steps.append(("encode", ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False,
                                              transformer_weights=weights)))
    return Pipeline(steps)


def log_scaled_columns(pipeline) -> list:
    encode = pipeline.named_steps["preprocess"].named_steps["encode"]
    return [c for name, _, cols in encode.transformers_ if name == "log" for c in cols]


def _build_pipeline(meta: dict, algorithm: str, X_fit: pd.DataFrame, **params) -> Pipeline:
    return Pipeline([("preprocess", _build_preprocessor(meta, X_fit, gower=algorithm in GOWER)),
                     ("clusterer", _make_clusterer(algorithm, **params))])


def _nearest_centroids(X: np.ndarray, labels) -> dict[str, list[float]]:
    centroids = {}
    labels_array = np.asarray(labels)
    for label in sorted(np.unique(labels_array).tolist()):
        members = X[labels_array == label]
        centroids[str(_to_native(label))] = np.asarray(members).mean(axis=0).tolist()
    return centroids


def _assign_nearest_centroids(
    X: np.ndarray, centroids: dict[str, list[float]]
) -> np.ndarray:
    if not centroids:
        raise ValueError("no centroids available for assignment")
    ordered_labels = list(centroids.keys())
    centroid_matrix = np.asarray(
        [centroids[label] for label in ordered_labels], dtype=float
    )
    distances = ((X[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
    return np.asarray([ordered_labels[i] for i in np.argmin(distances, axis=1)])


def assign_segments(pipeline, algorithm: str, centroids: dict, X: pd.DataFrame) -> np.ndarray:
    """Segment of each row under a fitted pipeline: the clusterer's own
    predict, or nearest centroid (hierarchical). Not for density-based."""
    if algorithm in PREDICTS:
        return np.asarray(pipeline.predict(X))
    return _assign_nearest_centroids(np.asarray(pipeline.named_steps["preprocess"].transform(X)), centroids)


def bootstrap_stability(X: np.ndarray, algorithm: str, params: dict, labels, seed: int = 42) -> dict:
    """Do the segments come back? Refit the same algorithm on 80% subsamples
    and compare each refit's labels with the full fit's on the rows both
    kept (noise excluded) by adjusted Rand index — invariant to how clusters
    are numbered. Runs on a STABILITY_MAX_ROWS sample."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    base = rng.choice(len(X), size=min(len(X), STABILITY_MAX_ROWS), replace=False)
    X_s, l_s = np.asarray(X)[base], labels[base]
    scores = []
    for _ in range(STABILITY_RESAMPLES):
        sub = rng.choice(len(X_s), size=int(0.8 * len(X_s)), replace=False)
        refit = _make_clusterer(algorithm, **params).fit(X_s[sub])
        refit_labels = fit_labels(refit, X_s[sub])
        keep = (l_s[sub] != -1) & (refit_labels != -1)
        if keep.sum() >= 10:
            scores.append(adjusted_rand_score(l_s[sub][keep], refit_labels[keep]))
    if not scores:
        return {"stability_ari_mean": None, "resamples": 0, "rows_used": int(len(X_s))}
    return {"stability_ari_mean": round(float(np.mean(scores)), 4),
            "stability_ari_min": round(float(np.min(scores)), 4),
            "resamples": len(scores), "rows_used": int(len(X_s))}


def _current_metrics(meta: dict) -> dict:
    return meta.get("training_metrics") or {}


def _validate_feature_matrix(X: np.ndarray) -> None:
    if len(X) < 2:
        raise ValueError("need at least 2 rows after preprocessing")
    if np.isnan(X).any():
        raise ValueError(
            "preprocessed matrix still contains NaN — apply_imputation first"
        )


# Every tool call on a run is recorded in its execution ledger (core/gates.py),
# so the readiness gates judge what ACTUALLY ran, in what order. Installed here,
# before any tool module decorates its tools.
from core import gates as _gate_engine  # noqa: E402

_gate_engine.install_ledger(mcp, _run_dir, _load_meta, _save_meta)
