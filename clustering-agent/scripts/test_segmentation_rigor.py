"""Clustering rigour: the distance space (skewed numerics logged, dummies not
scaled, outcomes and protected attributes kept out), stability as the proof
that segments are real, algorithms beyond kmeans, and descriptions that say
what distinguishes a segment and whether it matters.
No pytest fixtures — CI also runs this file as a plain script.

Run: python test_segmentation_rigor.py
"""
import os as _os
import tempfile as _tempfile

_os.environ.setdefault("AGENTIC_ML_DATA_DIR", _tempfile.mkdtemp(prefix="agentic-ml-test-"))
_os.environ.setdefault("MLFLOW_TRACKING_URI", "file:" + _tempfile.mkdtemp(prefix="agentic-ml-test-mlruns-"))

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

import mcp_server as m


def _segments(n=1500, seed=0):
    """Three behavioural segments in two numeric features, plus: a lognormal
    usage column (skewed), a rare category unrelated to the segments, an
    outcome that differs by segment, and a protected attribute."""
    rng = np.random.default_rng(seed)
    X, truth = make_blobs(n, centers=[[0, 0], [6, 6], [0, 8]], cluster_std=1.0, random_state=seed)
    return pd.DataFrame({
        "calls_per_day": X[:, 0], "data_share": X[:, 1],
        "usage_mb": rng.lognormal(3 + 0.4 * truth, 1.0),
        "rare_flag": np.where(rng.random(n) < 0.02, "yes", "no"),
        "churned": (rng.random(n) < np.array([0.05, 0.35, 0.15])[truth]).astype(int),
        "age": rng.integers(18, 80, n),
        "truth": truth,
    })


def _write(tmp_path, df, name="d.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _prepared(tmp_path, df, name="d.csv"):
    """`truth` rides along as a profile column: in the data, never in the distance."""
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, df, name), test_size=0.2))["run_id"]
    m.set_profile_columns(run_id, "churned,age,truth")
    return run_id


def test_space_logs_skew_keeps_rare_dummies_small_and_profiles_stay_out(tmp_path):
    df = _segments()
    run_id = _prepared(tmp_path, df)
    trained = json.loads(m.train_model(run_id, "kmeans", n_clusters=3))
    assert "error" not in trained, trained
    assert "usage_mb" in trained["log_scaled_columns"]
    assert set(trained["profile_columns_excluded"]) == {"churned", "age", "truth"}
    pipeline = joblib.load(m._run_dir(run_id) / "pipeline.pkl")
    features = set(pipeline.named_steps["preprocess"].named_steps["encode"].get_feature_names_out())
    assert not features & {"churned", "age", "truth"}, features
    fit, _, _ = m._load_split(run_id)
    labels = np.load(m._run_dir(run_id) / "fit_labels.npy")
    # the rare category must not have become a segment of its own
    rare = fit["rare_flag"] == "yes"
    assert len(set(labels[rare])) > 1, "the 2% rare category was pulled into its own cluster"
    assert trained["stability"]["stability_ari_mean"] >= 0.9, trained["stability"]
    assert m.compute_readiness(run_id)["checks"]["holdout_stability"]["status"] == "pass"


def test_segments_are_recovered_described_and_judged_on_the_outcome(tmp_path):
    df = _segments()
    run_id = _prepared(tmp_path, df)
    m.train_model(run_id, "kmeans", n_clusters=3)
    fit, _, _ = m._load_split(run_id)
    labels = np.load(m._run_dir(run_id) / "fit_labels.npy")
    assert adjusted_rand_score(fit["truth"], labels) > 0.95
    explained = json.loads(m.explain_model(run_id))
    assert explained["outcomes"]["churned"]["eta_squared"] > 0.05, explained["outcomes"]
    assert explained["rules"]["accuracy"] > 0.9
    for profile in explained["cluster_profiles"]:
        assert profile["distinguishing"], profile
        assert profile["distinguishing"][0]["feature"] in ("calls_per_day", "data_share", "usage_mb")


def test_propose_k_recommends_the_stable_k_and_noise_is_not_stable(tmp_path):
    run_id = _prepared(tmp_path, _segments())
    proposed = json.loads(m.propose_k(run_id, "2,3,4,5"))
    assert proposed["recommended_k"] == 3 and 3 in proposed["stable_ks"], proposed["stable_ks"]
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(0, 1, (1500, 6)), columns=[f"f{i}" for i in range(6)])
    noise_run = json.loads(m.prepare_dataset(_write(tmp_path, noise, "noise.csv")))["run_id"]
    trained = json.loads(m.train_model(noise_run, "kmeans", n_clusters=6))
    assert trained["overfitting_warning"] is True and trained["stability"]["stability_ari_mean"] < 0.7, trained
    assert m.compute_readiness(noise_run)["checks"]["holdout_stability"]["status"] == "fail"


def test_gmm_hdbscan_and_the_hierarchical_size_guard(tmp_path):
    mixed = _prepared(tmp_path, _segments(), "mixed.csv")
    collapsed = json.loads(m.train_model(mixed, "gmm", n_clusters=3))
    assert collapsed["model_warnings"] and collapsed["overfitting_warning"] is True, collapsed
    run_id = _prepared(tmp_path, _segments().drop(columns=["rare_flag"]), "numeric.csv")
    gmm = json.loads(m.train_model(run_id, "gmm", n_clusters=3))
    assert not gmm["model_warnings"] and gmm["holdout_ari"] > 0.9, gmm
    assert json.loads(m.explain_model(run_id))["cluster_profiles"]
    hdb = json.loads(m.train_model(run_id, "hdbscan", min_cluster_size=50))
    assert "error" not in hdb and hdb["fit"]["n_clusters_found"] >= 2, hdb
    assert hdb["holdout_assignment_counts"] is None  # density clusters don't score new rows
    big = pd.DataFrame(make_blobs(26_000, centers=3, n_features=2, random_state=0)[0], columns=["a", "b"])
    big_run = json.loads(m.prepare_dataset(_write(tmp_path, big, "big.csv")))["run_id"]
    assert "quadratic" in json.loads(m.train_model(big_run, "hierarchical", n_clusters=3))["error"]


def _mixed(n=1500, seed=3):
    """Segments carried by a category, a flag and one numeric, plus three
    noise numerics: standardised, the noise outweighs the categories."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 3, n)
    plans = np.array(["prepaid", "postpaid", "business"])
    return pd.DataFrame({
        "plan": np.where(rng.random(n) < 0.85, plans[truth], rng.choice(plans, n)),
        "roaming": np.where(rng.random(n) < np.array([0.05, 0.2, 0.85])[truth], "yes", "no"),
        "calls": rng.normal(np.array([10, 18, 26])[truth], 6),
        **{f"noise_{i}": rng.normal(0, 1, n) for i in range(3)},
        "truth": truth,
    })


def test_gower_kmedoids_segments_mixed_data_and_scores_new_rows(tmp_path):
    df = _mixed()
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, df), test_size=0.2))["run_id"]
    m.set_profile_columns(run_id, "truth")
    fit, holdout, _ = m._load_split(run_id)
    recovered = {}
    for algorithm in ("kmeans", "kmedoids"):
        trained = json.loads(m.train_model(run_id, algorithm, n_clusters=3))
        assert "error" not in trained, trained
        recovered[algorithm] = adjusted_rand_score(fit["truth"], np.load(m._run_dir(run_id) / "fit_labels.npy"))
    assert recovered["kmedoids"] > 0.65 and recovered["kmedoids"] > recovered["kmeans"] + 0.2, recovered
    assert trained["stability"]["stability_ari_mean"] >= 0.9 and not trained["overfitting_warning"], trained
    pipeline = joblib.load(m._run_dir(run_id) / "pipeline.pkl")
    encode = pipeline.named_steps["preprocess"].named_steps["encode"]
    assert encode.transformer_weights == {"cat": 0.5}  # a category mismatch counts 1, like a full numeric range
    assigned = pipeline.predict(holdout)  # medoids are real rows: new rows go to the nearest one
    assert adjusted_rand_score(holdout["truth"], assigned) > 0.65
    proposed = json.loads(m.propose_k(run_id, "2,3,4", algorithm="kmedoids"))
    assert proposed["algorithm"] == "kmedoids" and 3 in proposed["stable_ks"], proposed
    assert "density" in json.loads(m.propose_k(run_id, "2,3", algorithm="hdbscan"))["error"]


def _panel(tmp_path, subscribers=300, seed=4):
    """Three months of usage per subscriber in three well-separated segments.
    Between January and February a fifth of segment 0 moves to segment 2;
    nobody moves between February and March."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [8, 0], [0, 8]])
    home = rng.integers(0, 3, subscribers)
    movers = (home == 0) & (rng.random(subscribers) < 0.2)
    rows = []
    for month, shifted in (("2026-01", False), ("2026-02", True), ("2026-03", True)):
        segment = np.where(movers & shifted, 2, home)
        xy = centres[segment] + rng.normal(0, 0.7, (subscribers, 2))
        rows += [{"sub": f"S{i}", "month": month, "voice": xy[i, 0], "data": xy[i, 1]} for i in range(subscribers)]
    return _write(tmp_path, pd.DataFrame(rows), "panel.csv"), int(movers.sum())


def test_segment_migration_finds_the_planted_move_and_when_it_happened(tmp_path):
    path, n_movers = _panel(tmp_path)
    leaky = json.loads(m.prepare_dataset(path, test_size=0.2))["run_id"]
    m.set_profile_columns(leaky, "sub")
    m.train_model(leaky, "kmeans", n_clusters=3)
    assert "distance" in json.loads(m.segment_migration(leaky, "sub", "month"))["error"]  # month was one-hot

    run_id = json.loads(m.prepare_dataset(path, test_size=0.2))["run_id"]
    m.set_profile_columns(run_id, "sub,month")
    m.train_model(run_id, "kmeans", n_clusters=3)
    moved = json.loads(m.segment_migration(run_id, "sub", "month"))
    assert "error" not in moved, moved
    assert moved["transitions"] == 600 and moved["entities"] == 300
    top = moved["top_moves"][0]
    assert abs(top["count"] - n_movers) <= 3, (top, n_movers)
    assert moved["stay_rate"] > 0.9 and moved["stay_rate_by_segment"][top["from"]] < 0.95
    first, second = moved["by_period"]
    assert (first["from_period"], second["to_period"]) == ("2026-01", "2026-03")
    assert abs(first["moved_share"] - n_movers / 300) < 0.01 and second["moved_share"] < 0.01, moved["by_period"]
    assert m._load_meta(run_id)["migration"]["transitions"] == 600


def test_intake_and_behaviour_features_reach_clustering(tmp_path):
    events = []
    rng = np.random.default_rng(2)
    for s in range(60):
        t, machine = pd.Timestamp("2026-01-01"), s % 2 == 0
        for _ in range(40):
            t += pd.Timedelta(seconds=300 if machine else float(rng.exponential(3000)))
            events.append({"sim": f"S{s}", "ts": t, "bytes": 200 if machine else float(rng.gamma(2, 500))})
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(events).to_parquet(events_path, index=False)
    loaded = json.loads(m.load_dataset(str(events_path), name="events"))
    built = json.loads(m.aggregate_events(loaded["path"], "sim", "ts", json.dumps([
        {"name": "gap_cv", "column": "_gap", "agg": "cv"},
        {"name": "bytes_cv", "column": "bytes", "agg": "cv"}])))
    assert "error" not in built, built
    run_id = json.loads(m.prepare_dataset(built["out_path"], test_size=0.2))["run_id"]
    m.apply_drop_columns(run_id, "sim")
    trained = json.loads(m.train_model(run_id, "kmeans", n_clusters=2))
    assert trained["fit"]["silhouette_score"] > 0.6, trained["fit"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with _tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"ok {name}")
    print("all segmentation rigour checks passed")
