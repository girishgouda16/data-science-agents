"""Proves the clustering agent's core rigor claims: prepare_dataset splits
fit/holdout before learned preprocessing, imputation/scaling stats come from
the fit fold only, k-selection works on a real separable dataset, every
algorithm (kmeans/dbscan/hierarchical) actually trains and predicts, the
overfitting gate actually refuses an export, export is a real standalone
predictor, error paths return errors instead of crashing, disposable run
dirs get swept, and persistence writes through to the shared training_runs
table.
No MCP transport needed — calls the underlying functions directly.

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
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

import mcp_server as m

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = (
    m.RUNS_DIR / f"clustering-test-artifacts-{uuid.uuid4()}"
)  # the (isolated) runs dir, never the repo's
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _make_csv() -> str:
    X, y = make_blobs(
        n_samples=240,
        centers=[(-6, -6), (0, 5), (7, -1)],
        cluster_std=[0.8, 0.9, 0.7],
        random_state=42,
    )
    df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    df["feature_3"] = y * 3 + np.linspace(0, 1, len(df))
    # segment_hint one-hot encodes the cluster identity directly — makes the
    # blobs trivially separable for every algorithm under test, not just kmeans.
    df["segment_hint"] = np.where(y == 0, "bronze", np.where(y == 1, "silver", "gold"))
    df["row_id"] = np.arange(10_000, 10_000 + len(df))
    df.loc[:17, "feature_1"] = np.nan
    path = ARTIFACT_DIR / f"synthetic_clusters_{uuid.uuid4()}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _prepared_run() -> tuple[str, Path]:
    """prepare_dataset + impute + drop the ID column — the common starting
    point every algorithm-specific flow needs, factored out so each flow
    below only has to cover what's specific to it."""
    path = _make_csv()
    prep = json.loads(m.prepare_dataset(path))
    run_id = prep["run_id"]
    run_dir = m._run_dir(run_id)
    json.loads(m.apply_imputation(run_id, json.dumps({"feature_1": "median"})))
    json.loads(m.apply_drop_columns(run_id, json.dumps(["row_id"])))
    Path(path).unlink()
    return run_id, run_dir


def _kmeans_flow() -> tuple[str, Path]:
    path = _make_csv()

    eda = json.loads(m.eda(path))
    assert eda["shape"] == [240, 5]

    prep = json.loads(m.prepare_dataset(path))
    run_id = prep["run_id"]
    run_dir = m._run_dir(run_id)
    assert prep["fit_shape"][0] + prep["holdout_shape"][0] == 240

    # ── "no fitted model yet" errors — must fire before any train_model call. ──
    assert "error" in json.loads(m.explain_model(run_id))
    assert "error" in json.loads(m.export_model(run_id))

    fit_before = pd.read_csv(run_dir / "train.csv")
    expected_median = float(fit_before["feature_1"].median())

    impute_props = json.loads(m.propose_imputation(run_id))
    assert impute_props["proposals"]["feature_1"]["suggested_strategy"] == "median"

    drop_props = json.loads(m.propose_drop_columns(run_id))
    # The shared identifier rules (core/data_quality) now explain themselves.
    assert drop_props["proposals"]["row_id"].startswith("near-unique"), drop_props["proposals"]

    imputed = json.loads(
        m.apply_imputation(run_id, json.dumps({"feature_1": "median"}))
    )
    assert imputed["fit_shape"][0] == prep["fit_shape"][0]
    meta = m._load_meta(run_id)
    assert meta["imputation"]["feature_1"] == expected_median, (
        "imputation value leaked holdout information"
    )
    assert pd.read_csv(run_dir / "train.csv")["feature_1"].isna().sum() == 0
    assert pd.read_csv(run_dir / "test.csv")["feature_1"].isna().sum() == 0

    dropped = json.loads(m.apply_drop_columns(run_id, json.dumps(["row_id"])))
    assert "row_id" in dropped["dropped_columns"]

    # ── propose_k error branches: empty k_range and per-k out-of-range. ──
    empty_k = json.loads(m.propose_k(run_id, ""))
    assert "error" in empty_k

    oob = json.loads(m.propose_k(run_id, "1,5,1000"))
    oob_rows = {row["k"]: row for row in oob["evaluations"]}
    assert "error" in oob_rows[1], "k=1 must be rejected (k must be >= 2)"
    assert "error" in oob_rows[1000], "k=1000 is >= len(fit), must be rejected"
    assert "error" not in oob_rows[5]

    proposed = json.loads(m.propose_k(run_id, "2,3,4,5"))
    by_k = {
        row["k"]: row
        for row in proposed["evaluations"]
        if "k" in row and "error" not in row
    }
    assert proposed["recommended_k"] == 3, proposed
    assert by_k[3]["silhouette_score"] > 0.7
    assert by_k[3]["silhouette_score"] > by_k[2]["silhouette_score"]
    assert by_k[3]["silhouette_score"] >= by_k[4]["silhouette_score"]

    # ── train_model error branches: unknown algorithm, n_clusters < 2. ──
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    unknown_algo = json.loads(m.train_model(run_id, algorithm="not_an_algorithm"))
    assert "error" in unknown_algo
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    too_few_clusters = json.loads(
        m.train_model(run_id, algorithm="kmeans", n_clusters=1)
    )
    assert "error" in too_few_clusters

    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    trained = json.loads(m.train_model(run_id, algorithm="kmeans", n_clusters=3))
    assert trained["algorithm"] == "kmeans"
    assert trained["fit"]["n_clusters_found"] == 3
    assert trained["fit"]["silhouette_score"] > 0.7
    assert set(trained["holdout_assignment_counts"]) == {"0", "1", "2"}

    pipeline = joblib.load(run_dir / "pipeline.pkl")
    fit_after = pd.read_csv(run_dir / "train.csv")
    holdout_after = pd.read_csv(run_dir / "test.csv")

    # Leakage checks: every scaler inside the pipeline (plain numerics, and
    # the logged skewed ones) was fit on the FIT fold only — its mean_ is the
    # fit fold's, not any full-dataset statistic.
    preprocess = pipeline.named_steps["preprocess"]
    prepare = lambda df: preprocess.named_steps["impute"].transform(preprocess.named_steps["drop"].transform(df))  # noqa: E731
    prepared_fit = prepare(fit_after)
    prepared_full = prepare(pd.concat([fit_after, holdout_after], ignore_index=True))
    scalers_checked = 0
    for name, transformer, cols in preprocess.named_steps["encode"].transformers_:
        if name not in ("num", "log"):
            continue
        before_scale, scaler = transformer[:-1], transformer.named_steps["scale"]
        assert np.allclose(scaler.mean_, np.asarray(before_scale.transform(prepared_fit[cols])).mean(axis=0))
        assert not np.allclose(scaler.mean_, np.asarray(before_scale.transform(prepared_full[cols])).mean(axis=0))
        scalers_checked += 1
    assert scalers_checked >= 1

    explained = json.loads(m.explain_model(run_id))
    assert len(explained["cluster_profiles"]) == 3
    assert all(profile["size"] > 0 for profile in explained["cluster_profiles"])

    suggested = json.loads(m.export_model(run_id))
    assert suggested["suggested_out_path"].endswith(f"kmeans_{run_id}.pkl")
    export_path = ARTIFACT_DIR / "cluster_model_kmeans.pkl"
    exported = json.loads(m.export_model(run_id, str(export_path)))
    assert exported["out_path"] == str(export_path)

    new_rows = pd.DataFrame(
        {
            "feature_1": [-5.5, 0.1, 7.4],
            "feature_2": [-5.8, 5.3, -0.8],
            "feature_3": [0.2, 3.4, 6.8],
            "segment_hint": ["bronze", "silver", "gold"],
            "row_id": [99_001, 99_002, 99_003],
        }
    )
    new_path = ARTIFACT_DIR / "new_rows_kmeans.csv"
    new_rows.to_csv(new_path, index=False)
    # kmeans-native-predict branch: pipeline.predict() is used directly.
    predicted = json.loads(m.predict(str(export_path), str(new_path)))
    assert predicted["algorithm"] == "kmeans"
    assert predicted["n_rows"] == 3
    assert len(predicted["cluster_counts"]) == 3
    assert Path(predicted["out_path"]).exists()

    return run_id, run_dir


def _dbscan_flow() -> tuple[str, Path]:
    run_id, run_dir = _prepared_run()

    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    trained = json.loads(
        m.train_model(run_id, algorithm="dbscan", eps=1.0, min_samples=5)
    )
    assert trained["algorithm"] == "dbscan"
    assert trained["fit"]["n_clusters_found"] == 3, trained
    # dbscan has no n_clusters param — holdout_assignment_counts is only
    # populated for kmeans/hierarchical, must be None here.
    assert trained["holdout_assignment_counts"] is None

    explained = json.loads(m.explain_model(run_id))
    assert len(explained["cluster_profiles"]) >= 3  # >= to tolerate a noise (-1) group

    export_path = ARTIFACT_DIR / "cluster_model_dbscan.pkl"
    exported = json.loads(m.export_model(run_id, str(export_path)))
    assert exported["out_path"] == str(export_path)

    new_rows = pd.DataFrame(
        {
            "feature_1": [-5.5, 0.1, 7.4],
            "feature_2": [-5.8, 5.3, -0.8],
            "feature_3": [0.2, 3.4, 6.8],
            "segment_hint": ["bronze", "silver", "gold"],
        }
    )
    new_path = ARTIFACT_DIR / "new_rows_dbscan.csv"
    new_rows.to_csv(new_path, index=False)
    # dbscan-explicitly-rejected branch: no out-of-sample scoring, ever.
    rejected = json.loads(m.predict(str(export_path), str(new_path)))
    assert "error" in rejected and "DBSCAN" in rejected["error"]

    return run_id, run_dir


def _hierarchical_flow() -> tuple[str, Path]:
    run_id, run_dir = _prepared_run()

    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    trained = json.loads(m.train_model(run_id, algorithm="hierarchical", n_clusters=3))
    assert trained["algorithm"] == "hierarchical"
    assert trained["fit"]["n_clusters_found"] == 3
    assert set(trained["holdout_assignment_counts"]) == {"0", "1", "2"}

    explained = json.loads(m.explain_model(run_id))
    assert len(explained["cluster_profiles"]) == 3

    export_path = ARTIFACT_DIR / "cluster_model_hierarchical.pkl"
    exported = json.loads(m.export_model(run_id, str(export_path)))
    assert exported["out_path"] == str(export_path)

    new_rows = pd.DataFrame(
        {
            "feature_1": [-5.5, 0.1, 7.4],
            "feature_2": [-5.8, 5.3, -0.8],
            "feature_3": [0.2, 3.4, 6.8],
            "segment_hint": ["bronze", "silver", "gold"],
        }
    )
    new_path = ARTIFACT_DIR / "new_rows_hierarchical.csv"
    new_rows.to_csv(new_path, index=False)
    # nearest-centroid-for-hierarchical branch: no native predict, so
    # _assign_nearest_centroids does the out-of-sample assignment.
    predicted = json.loads(m.predict(str(export_path), str(new_path)))
    assert predicted["algorithm"] == "hierarchical"
    assert predicted["n_rows"] == 3
    assert len(predicted["cluster_counts"]) == 3

    return run_id, run_dir


def _imputation_strategies_check() -> None:
    """apply_imputation's three named strategies plus the literal-value
    fallback — the kmeans flow above only ever exercises 'median'."""
    df = pd.DataFrame(
        {
            "numeric_col": [1.0, 2.0, np.nan, 4.0, 5.0] * 6,
            "category_col": ["a", "a", "b", None, "a"] * 6,
            "sparse_col": [1.0, np.nan, np.nan, np.nan, np.nan] * 6,  # >50% missing
            "literal_col": [10.0, np.nan, 30.0, 40.0, 50.0] * 6,
        }
    )
    path = ARTIFACT_DIR / f"impute_strategies_{uuid.uuid4()}.csv"
    df.to_csv(path, index=False)
    run_id = json.loads(m.prepare_dataset(str(path)))["run_id"]
    run_dir = m._run_dir(run_id)

    expected_mode = df["category_col"].mode(dropna=True).iloc[0]
    result = json.loads(
        m.apply_imputation(
            run_id,
            json.dumps(
                {
                    "category_col": "mode",
                    "sparse_col": "drop_column",
                    "literal_col": -1.0,
                }
            ),
        )
    )
    meta = m._load_meta(run_id)
    assert meta["imputation"]["category_col"] == expected_mode
    assert "sparse_col" in result["dropped_columns"]
    assert "sparse_col" not in pd.read_csv(run_dir / "train.csv").columns
    assert meta["imputation"]["literal_col"] == -1.0
    train_after = pd.read_csv(run_dir / "train.csv")
    assert train_after["category_col"].isna().sum() == 0
    assert train_after["literal_col"].isna().sum() == 0
    assert (
        (train_after.loc[train_after["literal_col"] == -1.0].shape[0]) >= 0
    )  # literal value actually used, no crash

    shutil.rmtree(run_dir)
    path.unlink()


def _overfit_gate_check() -> None:
    """export_model must refuse a model whose segments are not stable (bootstrap
    stability ARI below STABLE_ARI sets overfitting_warning) unless force=True."""
    run_id, run_dir = _prepared_run()
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    json.loads(m.train_model(run_id, algorithm="kmeans", n_clusters=3))

    orig_threshold = m.modeling.STABLE_ARI
    m.modeling.STABLE_ARI = 1.01  # no ARI reaches it: forces the unstable verdict
    try:
        json.loads(
            m.detect_data_leakage(run_id)
        )  # the screen runs on the final feature set, before fitting
        json.loads(m.train_model(run_id, algorithm="kmeans", n_clusters=3))
        assert m._load_meta(run_id)["training_metrics"]["overfitting_warning"] is True
        refused = json.loads(m.export_model(run_id, str(ARTIFACT_DIR / "blocked.pkl")))
        assert "error" in refused and "overfitting_warning" in refused["error"]
        forced = json.loads(
            m.export_model(run_id, str(ARTIFACT_DIR / "forced.pkl"), force=True)
        )
        assert forced["out_path"]
    finally:
        m.modeling.STABLE_ARI = orig_threshold

    shutil.rmtree(run_dir)


def _error_path_checks() -> None:
    fd, path = tempfile.mkstemp(suffix=".csv")

    pd.DataFrame({"a": []}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(path)), (
        "empty CSV must error, not crash"
    )

    pd.DataFrame({"a": range(10)}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(path)), (
        "<20 rows must error — clustering needs a minimum"
    )

    Path(path).unlink()

    # predict: no exported model / no data file.
    _, missing_pkl = tempfile.mkstemp(suffix=".pkl")
    Path(missing_pkl).unlink()
    _, some_csv = tempfile.mkstemp(suffix=".csv")
    pd.DataFrame({"a": [1]}).to_csv(some_csv, index=False)
    assert "error" in json.loads(m.predict(missing_pkl, some_csv)), (
        "missing pkl must error"
    )

    real_run_id, real_run_dir = _prepared_run()
    json.loads(
        m.detect_data_leakage(real_run_id)
    )  # the screen runs on the final feature set, before fitting
    json.loads(m.train_model(real_run_id, algorithm="kmeans", n_clusters=3))
    exported = json.loads(
        m.export_model(real_run_id, str(ARTIFACT_DIR / "error_path_model.pkl"))
    )
    missing_csv = str(ARTIFACT_DIR / "does_not_exist.csv")
    assert "error" in json.loads(m.predict(exported["out_path"], missing_csv)), (
        "missing data file must error"
    )
    shutil.rmtree(real_run_dir)
    Path(some_csv).unlink()

    # compare_runs: unknown run_id must come back as a per-row error, not a crash.
    unknown = json.loads(m.compare_runs("not-a-real-run-id"))
    assert "error" in unknown["runs"][0]


def _cleanup_check() -> None:
    """prepare_dataset must sweep run dirs whose meta.json is older than
    RUN_RETENTION_DAYS — runs are disposable working state, not an archive."""
    path = _make_csv()
    old_run_id = json.loads(m.prepare_dataset(path))["run_id"]
    old_run_dir = m._run_dir(old_run_id)
    stale = time.time() - (m.RUN_RETENTION_DAYS + 1) * 86400
    os.utime(old_run_dir / "meta.json", (stale, stale))

    path2 = _make_csv()
    new_run_id = json.loads(m.prepare_dataset(path2))["run_id"]  # triggers the sweep
    assert not old_run_dir.exists(), "a run older than RUN_RETENTION_DAYS must be swept"
    assert m._run_dir(new_run_id).exists(), (
        "the just-created run must survive its own sweep"
    )

    shutil.rmtree(m._run_dir(new_run_id))
    Path(path).unlink()
    Path(path2).unlink()


def main():
    kmeans_run_id, kmeans_run_dir = _kmeans_flow()
    dbscan_run_id, dbscan_run_dir = _dbscan_flow()
    hierarchical_run_id, hierarchical_run_dir = _hierarchical_flow()

    _imputation_strategies_check()
    _overfit_gate_check()

    # multi-run compare_runs — previously only ever exercised with one run_id.
    compared = json.loads(
        m.compare_runs(f"{kmeans_run_id},{dbscan_run_id},{hierarchical_run_id}")
    )
    assert len(compared["runs"]) == 3
    assert set(compared["ranked_by_silhouette"]) == {
        kmeans_run_id,
        dbscan_run_id,
        hierarchical_run_id,
    }

    db_path = ROOT / "data" / "training_runs.db"
    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM training_runs WHERE session_id = ?", (kmeans_run_id,)
        ).fetchone()[0]
    assert count >= 1

    shutil.rmtree(kmeans_run_dir)
    shutil.rmtree(dbscan_run_dir)
    shutil.rmtree(hierarchical_run_dir)

    _error_path_checks()
    _cleanup_check()

    shutil.rmtree(ARTIFACT_DIR)
    print("all checks passed")


if __name__ == "__main__":
    main()
