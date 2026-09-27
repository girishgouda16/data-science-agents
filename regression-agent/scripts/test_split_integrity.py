"""Regression rigour: the split (random / grouped / forward in time, label
maturity) is respected by every CV; split keys never reach the model; a
skewed target can be modelled in log space with results in original units;
prediction intervals are calibrated out-of-fold and measured on test; the
shared feature and intake tools work here too.
No pytest fixtures — CI also runs this file as a plain script.

Run: python test_split_integrity.py
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

import mcp_server as m


def _subscriber_months(n_subs=60, n_months=18, seed=0):
    """Subscriber x month: next month's spend follows the subscriber's own
    level with noise — right-skewed, non-negative, like real ARPU."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subs):
        level = rng.lognormal(3, 0.8)
        for month in range(n_months):
            usage = level * rng.lognormal(0, 0.3)
            rows.append({"msisdn": f"S{s:03d}", "month": month, "usage": round(usage, 3),
                         "plan": rng.choice(["basic", "plus", "max"]),
                         "spend_next": round(usage * 1.4 + rng.gamma(1, 2), 3)})
    return pd.DataFrame(rows)


def _write(tmp_path, df, name="d.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_forward_and_grouped_splits_hold_in_every_cv_and_keys_never_reach_the_model(tmp_path):
    frame = _subscriber_months()
    run = json.loads(m.prepare_dataset(_write(tmp_path, frame), "spend_next", group_column="msisdn",
                                       time_column="month", immature_after="15"))
    assert run["split_type"] == "temporal" and run["immature_rows_dropped"] == 60 * 2, run
    train, test, meta = m._load_split(run["run_id"])
    assert train["month"].max() <= test["month"].min()
    folds, scheme = m.core._validation_folds(train.drop(columns=["spend_next"]), train["spend_next"], meta)
    assert "forward-chained" in scheme
    for fit, val in folds:
        assert train["month"].iloc[fit].max() <= train["month"].iloc[val].min()

    trained = json.loads(m.train_model(run["run_id"], "ridge"))
    assert "error" not in trained and "forward-chained" in trained["cv_scheme"], trained
    features = set(joblib.load(m._run_dir(run["run_id"]) / "pipeline.pkl").named_steps["encode"].get_feature_names_out())
    assert not any(f == "month" or f.startswith("msisdn") for f in features), features
    assert "msisdn" not in json.loads(m.propose_drop_columns(run["run_id"]))["proposals"]
    kept = json.loads(m.apply_drop_columns(run["run_id"], "msisdn"))
    assert kept["kept_split_keys"] == ["msisdn"]
    assert "msisdn" not in json.loads(m.detect_data_leakage(run["run_id"]))["identifier_columns"]

    grouped = json.loads(m.prepare_dataset(_write(tmp_path, frame, "g.csv"), "spend_next", group_column="msisdn"))
    gtrain, gtest, _ = m._load_split(grouped["run_id"])
    assert not set(gtrain["msisdn"]) & set(gtest["msisdn"])


def test_log_target_fits_in_log_space_and_reports_in_original_units(tmp_path):
    run = json.loads(m.prepare_dataset(_write(tmp_path, _subscriber_months()), "spend_next"))
    assert run["target_skew"] > 1, run["target_skew"]
    out = json.loads(m.apply_target_transform(run["run_id"], "log1p"))
    assert out["skew_transformed"] < out["skew_raw"]
    for model in ("hist_gradient_boosting", "ridge"):
        trained = json.loads(m.train_model(run["run_id"], model))
        assert "error" not in trained and trained["target_transform"] == "log1p", trained
    tuned = json.loads(m.tune_hyperparams(run["run_id"], "ridge", n_trials=2))
    assert "error" not in tuned, tuned
    pipeline = joblib.load(m._run_dir(run["run_id"]) / "pipeline.pkl")
    _, test, _ = m._load_split(run["run_id"])
    preds = pipeline.predict(test.drop(columns=["spend_next"]))
    assert preds.min() > 0 and abs(np.median(preds) - test["spend_next"].median()) < test["spend_next"].median()
    negative = _subscriber_months().assign(spend_next=lambda d: d["spend_next"] - 50)
    neg_run = json.loads(m.prepare_dataset(_write(tmp_path, negative, "neg.csv"), "spend_next"))["run_id"]
    assert "non-negative" in json.loads(m.apply_target_transform(neg_run, "log1p"))["error"]


def test_prediction_intervals_are_calibrated_out_of_fold_and_travel_with_the_export(tmp_path):
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, _subscriber_months(n_subs=120)), "spend_next"))["run_id"]
    m.train_model(run_id, "random_forest")
    m.explain_model(run_id)
    interval = json.loads(m.calibrate_intervals(run_id, coverage=0.9))
    assert "error" not in interval and interval["test_coverage"] >= 0.8, interval
    retrained = json.loads(m.train_model(run_id, "hist_gradient_boosting"))
    assert {"explainability", "prediction_interval"} <= set(retrained["invalidated_by_refit"]), retrained
    json.loads(m.calibrate_intervals(run_id, coverage=0.9))
    meta = m._load_meta(run_id)
    meta["business_understanding"] = {"objective": "test"}  # export's readiness check only needs not-blocked
    m._save_meta(run_id, meta)
    pkl = str(tmp_path / "model.pkl")
    exported = json.loads(m.export_model(run_id, out_path=pkl, force=True))
    assert "error" not in exported, exported
    new = _write(tmp_path, _subscriber_months(n_subs=5, seed=9).drop(columns=["spend_next"]), "new.csv")
    predicted = pd.read_csv(json.loads(m.predict(pkl, new))["out_path"])
    assert (predicted["prediction_lower"] < predicted["prediction"]).all()
    assert (predicted["prediction"] < predicted["prediction_upper"]).all()


def test_shared_feature_and_intake_tools_work_for_a_continuous_target(tmp_path):
    frame = _subscriber_months()
    parquet = tmp_path / "subs.parquet"
    frame.to_parquet(parquet, index=False)
    loaded = json.loads(m.load_dataset(str(parquet), name="subs"))
    run_id = json.loads(m.prepare_dataset(loaded["path"], "spend_next", group_column="msisdn",
                                          time_column="month"))["run_id"]
    built = json.loads(m.apply_entity_features(run_id, json.dumps([
        {"name": "usage_lag1", "op": "lag", "column": "usage"},
        {"name": "usage_vs_own_3m", "op": "vs_roll_mean", "column": "usage", "n": 3}])))
    assert "error" not in built, built
    profile = json.loads(m.profile_features(run_id, "usage,usage_lag1,plan"))
    assert profile["signal_metric"] == "spearman"
    assert profile["features"]["usage"]["spearman_alone"] > 0.8
    assert set(profile["features"]["usage"]["median_by_group"]) == {"q1_lowest", "q2", "q3", "q4_highest"}
    assert profile["features"]["plan"]["flag"] == "weak_alone"



def _usage_counts(n=900, seed=3):
    """A count-like, zero-heavy, skewed target (calls per day, claims per
    year): what poisson is for, and where log1p under-predicts totals."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    plan = rng.choice(["basic", "plus", "max"], n)
    rate = np.exp(0.8 + 0.9 * x1 + 0.4 * x2 + np.where(plan == "max", 0.8, 0))
    return pd.DataFrame({"x1": x1, "x2": x2, "plan": plan, "calls": rng.poisson(rate)})


def test_poisson_predicts_totals_without_the_log1p_shortfall(tmp_path):
    path = _write(tmp_path, _usage_counts())
    ratios = {}
    for mode in ("log1p", "poisson"):
        run_id = json.loads(m.prepare_dataset(path, "calls"))["run_id"]
        if mode == "log1p":
            m.apply_target_transform(run_id, "log1p")
            assert "already models the log" in json.loads(m.set_objective(run_id, "poisson"))["error"]
        else:
            assert json.loads(m.set_objective(run_id, "poisson"))["cv_metric"] == "r2"
            assert "twice" in json.loads(m.apply_target_transform(run_id, "log1p"))["error"]
        out = json.loads(m.train_model(run_id, "hist_gradient_boosting"))
        assert "error" not in out, out
        ratios[mode] = out["sum_ratio"]
    assert ratios["log1p"] < ratios["poisson"] and abs(ratios["poisson"] - 1) < 0.1, ratios
    negative = _usage_counts().assign(calls=lambda d: d["calls"] - 5)
    neg_run = json.loads(m.prepare_dataset(_write(tmp_path, negative, "neg.csv"), "calls"))["run_id"]
    assert "non-negative" in json.loads(m.set_objective(neg_run, "poisson"))["error"]


def test_quantile_objective_is_scored_on_pinball_and_hits_its_quantile(tmp_path):
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, _usage_counts(n=1500)), "calls"))["run_id"]
    m.set_objective(run_id, "quantile", quantile=0.9)
    compared = json.loads(m.compare_models(run_id, "random_forest,ridge,hist_gradient_boosting"))
    assert compared["cv_metric"] == "pinball" and compared["skipped_no_such_loss"] == ["random_forest"], compared
    assert "error" in json.loads(m.train_model(run_id, "random_forest"))
    out = json.loads(m.train_model(run_id, "hist_gradient_boosting"))
    assert out["cv_metric"] == "pinball" and 0.8 <= out["quantile_hit_rate"] <= 0.97, out
    m.train_baseline(run_id)
    gate = m.compute_readiness(run_id)["checks"]["baseline_beaten"]
    assert gate["status"] == "pass" and "pinball" in gate["evidence"], gate


def test_persistence_is_the_bar_on_entity_period_data(tmp_path):
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, _subscriber_months()), "spend_next",
                                          group_column="msisdn", time_column="month"))["run_id"]
    base = json.loads(m.train_baseline(run_id))
    assert base["strongest"] == "persistence", base
    assert base["baselines"]["persistence"]["r2"] > base["baselines"]["mean"]["r2"]
    assert m._load_meta(run_id)["baseline_comparison"]["kind"] == "persistence"


def test_error_analysis_finds_the_segment_the_model_misses_out_of_fold(tmp_path):
    rng = np.random.default_rng(4)
    n = 1200
    region = rng.choice(["north", "south", "east", "west"], n)
    x = rng.normal(0, 1, n)
    y = 10 + 3 * x + rng.normal(0, np.where(region == "east", 6.0, 0.5), n)  # east is unpredictable
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, pd.DataFrame({"x": x, "region": region, "y": y})),
                                          "y"))["run_id"]
    m.train_model(run_id, "random_forest")
    out = json.loads(m.analyze_errors(run_id))
    assert "out-of-fold" in out["measured_on"], out
    assert out["worst_segments"][0]["segment"] == "region = east", out["worst_segments"][:3]
    assert out["worst_segments"][0]["mae_vs_overall"] > 1.5
    retrained = json.loads(m.train_model(run_id, "ridge"))
    assert "error_analysis" in retrained["invalidated_by_refit"]

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with _tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"ok {name}")
    print("all regression split-integrity checks passed")
