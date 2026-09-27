"""Anomaly rigour: the distance space (a rare but normal category is not an
anomaly by itself), the alert share set from review capacity, the review
queue measured the way a team lives with it (precision / recall / lift at
the budget, average precision), novelty detection on known-normal rows,
stability as the evidence without labels, a reason for every alert, a
temporal split, and behaviour features from event rows.
No pytest fixtures — CI also runs this file as a plain script.

Run: python test_detection_rigor.py
"""
import os as _os
import tempfile as _tempfile

_os.environ.setdefault("AGENTIC_ML_DATA_DIR", _tempfile.mkdtemp(prefix="agentic-ml-test-"))
_os.environ.setdefault("MLFLOW_TRACKING_URI", "file:" + _tempfile.mkdtemp(prefix="agentic-ml-test-mlruns-"))

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mcp_server as m
import mcp_server.diagnostics as diagnostics


def _write(tmp_path, df, name="d.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _calls(n=3000, seed=0):
    """Subscribers' daily usage: heavy-tailed volumes, a 2% plan that is rare
    but normal, and 1% labelled fraud lines calling premium numbers."""
    rng = np.random.default_rng(seed)
    fraud = (rng.random(n) < 0.01).astype(int)
    return pd.DataFrame({
        "minutes": rng.lognormal(3, 1, n),
        "data_mb": rng.lognormal(5, 1.2, n),
        "premium_calls": np.where(fraud == 1, rng.poisson(40, n), rng.poisson(0.2, n)),
        "plan": np.where(rng.random(n) < 0.02, "satellite", rng.choice(["prepaid", "postpaid"], n)),
        "fraud": fraud,
    })


def test_rare_normal_category_and_heavy_users_are_not_the_alerts(tmp_path):
    df = _calls()
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, df), label_column="fraud"))["run_id"]
    trained = json.loads(m.train_model(run_id, "knn", contamination=0.02))
    assert "error" not in trained, trained
    assert {"minutes", "data_mb"} <= set(trained["log_scaled_columns"])
    assert trained["precision_at_budget"] >= 0.4 and trained["recall_at_budget"] >= 0.8, trained
    _, test, _ = m._load_split(run_id)
    queue = pd.read_csv(json.loads(m.explain_model(run_id, top_n=int(trained["alerts_at_budget"])))["review_queue_path"])
    satellite_share = (test.iloc[queue["row"]]["plan"] == "satellite").mean()
    assert satellite_share < 0.2, f"{satellite_share:.0%} of alerts are the rare-but-normal plan"


def test_review_capacity_sets_the_alert_share_and_caps_recall(tmp_path):
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, _calls()), label_column="fraud"))["run_id"]
    proposed = json.loads(m.propose_contamination(run_id, review_capacity=5, rows_per_period=1000))
    assert proposed["basis"] == "review_capacity" and proposed["recommended_contamination"] == 0.005
    assert 0.3 < proposed["recall_ceiling"] < 0.7 and "at most" in proposed["warning"], proposed
    unlabelled = pd.DataFrame(np.random.default_rng(1).normal(0, 1, (500, 3)), columns=list("abc"))
    guess = json.loads(m.propose_contamination(json.loads(m.prepare_dataset(_write(tmp_path, unlabelled, "u.csv")))["run_id"]))
    assert guess["basis"] == "default_guess" and "review" in guess["reasoning"]


def test_queue_metrics_novelty_training_and_the_quality_gate(tmp_path):
    rng = np.random.default_rng(2)
    n = 3000
    y = (rng.random(n) < 0.08).astype(int)
    X = rng.normal(0, 1, (n, 5))
    X[y == 1] = rng.normal(3.5, 0.3, (int(y.sum()), 5))  # a dense 8% cluster masks itself when trained on
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]).assign(label=y)
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, df), label_column="label"))["run_id"]
    everything = json.loads(m.train_model(run_id, "iforest", 0.08, train_on="all"))
    novelty = json.loads(m.train_model(run_id, "iforest", 0.08))  # auto = known-normal rows with labels
    assert novelty["trained_on"].startswith("normal rows only") and everything["trained_on"] == "all training rows"
    assert novelty["average_precision"] > everything["average_precision"] + 0.05, (novelty, everything)
    for key in ("precision_at_budget", "recall_at_budget", "lift_at_budget", "alerts_at_budget", "test_label_prevalence"):
        assert novelty[key] is not None, key
    assert abs(novelty["holdout_alert_rate"] - 0.08) < 0.05  # threshold set on the whole fold, not the normals
    m.propose_contamination(run_id)
    assert m.compute_readiness(run_id)["checks"]["detection_quality"]["status"] == "pass"
    ranked = json.loads(m.compare_detectors(run_id, "iforest,ecod,knn"))
    assert ranked["recommended"] in ("iforest", "ecod", "knn") and ranked["ranked"][0]["average_precision"] >= 0.9
    assert json.loads(m.train_model(run_id, "iforest", 0.08, train_on="bogus"))["error"]


def test_stability_is_the_evidence_without_labels(tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 1, (3000, 6)), columns=[f"f{i}" for i in range(6)])
    df.iloc[:60, :3] += 6
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, df)))["run_id"]
    trained = json.loads(m.train_model(run_id, "iforest", 0.02))
    assert trained["stability"]["top_alert_jaccard_mean"] >= 0.9, trained["stability"]
    checks = m.compute_readiness(run_id)["checks"]
    assert checks["alerts_stable"]["status"] == "pass" and checks["detection_quality"]["status"] == "not_applicable"
    noise = pd.DataFrame(rng.normal(0, 1, (3000, 6)), columns=[f"f{i}" for i in range(6)])
    noise_run = json.loads(m.prepare_dataset(_write(tmp_path, noise, "noise.csv")))["run_id"]
    shaky = json.loads(m.train_model(noise_run, "iforest", 0.02))
    assert shaky["stability"]["top_alert_jaccard_mean"] < diagnostics.STABLE_JACCARD, shaky["stability"]
    assert m.compute_readiness(noise_run)["checks"]["alerts_stable"]["status"] == "fail"
    agreed = json.loads(m.compare_detectors(run_id, "iforest,ecod"))
    assert agreed["ranking_rule"].startswith("stability") and agreed["ranked"][0]["agreement_with_consensus"] > 0.5


def test_every_alert_carries_its_reasons(tmp_path):
    df = _calls(seed=4)
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, df), label_column="fraud"))["run_id"]
    rarity = json.loads(m.train_model(run_id, "iforest", 0.01))  # isolates the rare plan in one split
    assert any("`plan`" in w for w in rarity["model_warnings"]), rarity["model_warnings"]
    ranked = json.loads(m.compare_detectors(run_id, "iforest,knn", 0.01))
    assert ranked["recommended"] == "knn", ranked  # the labels expose it
    assert not json.loads(m.train_model(run_id, "knn", 0.01))["model_warnings"]
    queue = json.loads(m.explain_model(run_id, top_n=10))["review_queue"]
    assert queue[0]["known_anomaly"] == 1 and "premium_calls" in queue[0]["reasons"], queue[0]
    ref = m._load_meta(run_id)["reason_reference"]
    new = pd.DataFrame({"minutes": [20.0], "data_mb": [150.0], "premium_calls": [0], "plan": ["roaming_iot"]})
    assert "never seen in training" in m.row_reasons(new, ref)[0]


def test_temporal_split_and_behaviour_features_from_events(tmp_path):
    rng = np.random.default_rng(5)
    events = []
    for s in range(200):
        machine = s % 20 == 0  # 5% SIM-box-like lines: short calls at a fixed rhythm
        t = pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=int(rng.integers(0, 600)))
        for _ in range(40):
            t += pd.Timedelta(seconds=120 if machine else float(rng.exponential(4000)))
            events.append({"sim": f"S{s:03d}", "ts": t, "secs": 15 if machine else float(rng.gamma(2, 60)),
                           "simbox": int(machine)})
    path = tmp_path / "cdr.parquet"
    pd.DataFrame(events).to_parquet(path, index=False)
    built = json.loads(m.aggregate_events(str(path), "sim", "ts", json.dumps([
        {"name": "gap_cv", "column": "_gap", "agg": "cv"},
        {"name": "secs_mean", "column": "secs", "agg": "mean"},
        {"name": "simbox", "column": "simbox", "agg": "max"}])))
    assert "error" not in built, built
    run_id = json.loads(m.prepare_dataset(built["out_path"], label_column="simbox", test_size=0.5))["run_id"]
    m.apply_drop_columns(run_id, "sim")
    trained = json.loads(m.train_model(run_id, "iforest", 0.05))
    assert trained["recall_at_budget"] >= 0.8 and trained["lift_at_budget"] >= 5, trained

    timed = pd.DataFrame(events).assign(ts=lambda d: d["ts"].astype(str))
    t_run = json.loads(m.prepare_dataset(_write(tmp_path, timed, "timed.csv"), time_column="ts", group_column="sim"))
    train, test, meta = m._load_split(t_run["run_id"])
    assert t_run["split_type"] == "temporal" and pd.to_datetime(train["ts"]).max() <= pd.to_datetime(test["ts"]).min()
    dropped = json.loads(m.apply_drop_columns(t_run["run_id"], "sim,ts"))
    assert dropped["kept_split_keys"] == ["sim", "ts"]
    pipeline_inputs = m.detector_inputs(train, meta).columns
    assert "sim" not in pipeline_inputs and "ts" not in pipeline_inputs


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with _tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"ok {name}")
    print("all detection rigour checks passed")
