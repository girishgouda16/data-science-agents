"""An exported model must score rows shaped like the ORIGINAL file, and the
screens that block export (fairness, overfitting, calibration) must not fire
on noise. Each test here fails if one of those regresses.

Run: python -m pytest test_export_and_screens.py
"""

import os as _os
import tempfile as _tempfile

_os.environ.setdefault("AGENTIC_ML_DATA_DIR", _tempfile.mkdtemp(prefix="agentic-ml-test-"))
_os.environ.setdefault("MLFLOW_TRACKING_URI", "file:" + _tempfile.mkdtemp(prefix="agentic-ml-test-mlruns-"))

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import mcp_server as m  # noqa: E402


def test_exported_model_rebuilds_engineered_features_from_raw_rows():
    """apply_features, apply_custom_feature, datetime decomposition and
    frequency encoding all rewrite train/test; the export must replay them on
    raw data instead of failing with `columns are missing`."""
    rng = np.random.default_rng(3)
    n = 600
    df = pd.DataFrame({
        "a": rng.normal(10, 2, n),
        "b": rng.normal(5, 1, n),
        "cell": rng.choice([f"c{i}" for i in range(40)], n),
        "seen": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), "D"),
    })
    df["label"] = np.where(df["a"] / df["b"] + rng.normal(0, 0.3, n) > 2.1, "yes", "no")
    work = Path(_tempfile.mkdtemp())
    raw = work / "raw.csv"
    df.to_csv(raw, index=False)

    run_id = json.loads(m.prepare_dataset(str(raw), "label"))["run_id"]
    ratio = next(k for k in json.loads(m.propose_features(run_id))["candidates"] if "_div_" in k)
    assert "error" not in json.loads(m.apply_features(run_id, ratio))
    assert "error" not in json.loads(m.apply_custom_feature(run_id, "a_minus_b", "a - b", "spread"))
    assert "error" not in json.loads(m.apply_datetime_features(run_id, "seen"))
    assert "error" not in json.loads(m.apply_frequency_encoding(run_id, "cell"))
    trained = json.loads(m.train_model(run_id, "random_forest"))
    assert "error" not in trained, trained

    out = work / "model.pkl"
    exported = json.loads(m.export_model(run_id, str(out), force=True))
    assert "error" not in exported, exported
    new = work / "new.csv"
    df.drop(columns=["label"]).head(50).to_csv(new, index=False)
    scored = json.loads(m.predict(str(out), str(new)))
    assert "error" not in scored and scored["n_rows"] == 50, scored

    # And the pipeline's own evaluation is unchanged: on the already-engineered
    # test fold the replay step is a no-op.
    _, test, meta = m._load_split(run_id)
    import joblib

    pipe = joblib.load(m._run_dir(run_id) / "pipeline.pkl")
    X = test.drop(columns=["label"])
    assert pipe.named_steps["features"].transform(X)[X.columns].equals(X)


def test_history_features_are_named_as_upstream_columns():
    from pipeline_transformers import FeatureRecipes

    step = FeatureRecipes([{"kind": "history", "outputs": ["calls_roll_mean_7"]}])
    try:
        step.transform(pd.DataFrame({"x": [1]}))
        raise AssertionError("a missing history feature must not be silently skipped")
    except ValueError as e:
        assert "calls_roll_mean_7" in str(e) and "upstream" in str(e)
    assert step.transform(pd.DataFrame({"calls_roll_mean_7": [1.0]})).shape == (1, 1)


def test_fairness_needs_a_gap_bigger_than_chance():
    """Rates of 0.27-0.36 across four groups of 150 clear a 0.8 disparate-
    impact ratio (0.75) yet are within chance. Not flagged. A 0.15-vs-0.45
    split on the same sizes is."""
    from mcp_server.diagnostics import _chi2_p

    noise = [(40, 110), (45, 105), (50, 100), (54, 96)]
    assert min(r[0] for r in noise) / max(r[0] for r in noise) < 0.8
    assert _chi2_p(noise) > 0.05
    real = [(22, 128), (68, 82)]
    assert _chi2_p(real) < 0.001
    assert _chi2_p([(0, 150), (0, 140)]) is None  # nobody selected — nothing to test


def test_overfit_gate_reads_the_interval():
    from mcp_server.core import _overfit_gate

    ev = {"auc": {"pr_auc": 0.30}, "ci": {"pr_auc": [0.20, 0.42]}}
    assert _overfit_gate(0.40, ev, "average_precision") == (0.10, False)  # inside the interval: noise
    assert _overfit_gate(0.50, ev, "average_precision") == (0.20, True)  # above it: real
    tight = {"auc": {"pr_auc": 0.30}, "ci": {"pr_auc": [0.295, 0.305]}}
    assert _overfit_gate(0.33, tight, "average_precision")[1] is False  # below the floor
    no_ci = {"classification_report": {"weighted avg": {"f1-score": 0.6}}}
    assert _overfit_gate(0.8, no_ci, "f1_weighted") == (0.2, True)  # flat fallback


def test_ece_noise_floor_covers_a_perfect_model_on_small_data():
    from mcp_server.core import ECE_CONCERN
    from mcp_server.diagnostics import _ece, _ece_noise_floor

    rng = np.random.default_rng(0)
    proba = rng.beta(1, 2, 700)
    y = (rng.random(700) < proba).astype(int)  # perfectly calibrated by construction
    assert _ece(y, proba, 10)[0] <= max(ECE_CONCERN, _ece_noise_floor(proba, 10))
    assert _ece_noise_floor(rng.beta(1, 2, 50_000), 10) < ECE_CONCERN  # big data: 0.05 is the bar


def test_parquet_input_keeps_parquet_storage_end_to_end():
    """Parquet in: the run's folds, the exported reference snapshot and the
    predictions are Parquet; the chain still runs on Parquet's own types
    (datetime, category, nullable ints, leading-zero codes). CSV in: CSV."""
    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame({
        "zip": [f"{z:05d}" for z in rng.integers(0, 900, n)],  # leading zeros a CSV would lose
        "plan": pd.Categorical(rng.choice(["pre", "post"], n)),
        "calls": pd.array(np.where(rng.random(n) < 0.1, None, rng.integers(0, 20, n)), dtype="Int64"),
        "joined": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 500, n), "D"),
        "amount": rng.normal(50, 10, n),
    })
    df["label"] = np.where(df["amount"] + rng.normal(0, 5, n) > 55, "churn", "stay")
    work = Path(_tempfile.mkdtemp())
    src = work / "data.parquet"
    df.to_parquet(src, index=False)

    run_id = json.loads(m.prepare_dataset(str(src), "label"))["run_id"]
    run = m._run_dir(run_id)
    assert (run / "train.parquet").exists() and not (run / "train.csv").exists()
    train, _, meta = m._load_split(run_id)
    assert meta["storage_format"] == "parquet" and train["zip"].str.len().eq(5).all()

    assert "error" not in json.loads(m.apply_datetime_features(run_id, "joined"))
    imp = json.loads(m.propose_imputation(run_id))["proposals"]
    assert "error" not in json.loads(m.apply_imputation(run_id, json.dumps({c: p["suggested_strategy"] for c, p in imp.items()})))
    assert (run / "test.parquet").exists() and not (run / "test.csv").exists()
    assert "error" not in json.loads(m.train_model(run_id, "random_forest"))

    out = work / "model.pkl"
    exported = json.loads(m.export_model(run_id, str(out), force=True))
    assert exported["monitoring_reference"].endswith(".reference.parquet"), exported
    new = work / "new.parquet"
    df.drop(columns=["label"]).head(30).to_parquet(new, index=False)
    scored = json.loads(m.predict(str(out), str(new)))
    assert scored["out_path"].endswith("_predictions.parquet") and scored["n_rows"] == 30, scored
    assert len(pd.read_parquet(scored["out_path"])) == 30

    csv_src = work / "data.csv"
    df.to_csv(csv_src, index=False)
    csv_run = json.loads(m.prepare_dataset(str(csv_src), "label"))["run_id"]
    assert (m._run_dir(csv_run) / "train.csv").exists()
    assert m._load_meta(csv_run)["storage_format"] == "csv"
