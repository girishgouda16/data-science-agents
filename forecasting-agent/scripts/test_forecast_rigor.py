"""Forecasting rigour: hourly telecom traffic (daily AND weekly cycles, hour
of day as a feature), evaluation at the business horizon, MASE / WAPE /
bias, a calibrated P10-P90 interval that widens with the horizon, gaps and
duplicate timestamps handled as decisions, one series or the total of a
long table, and intake from parquet.
No pytest fixtures — CI also runs this file as a plain script.

Run: python test_forecast_rigor.py
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


def _traffic(weeks=8, seed=0):
    """Hourly cell traffic: a daily cycle, quieter weekends, noise, never < 0."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-05", periods=weeks * 168, freq="h")
    daily = 50 + 40 * np.clip(np.sin((ts.hour.to_numpy() - 6) / 24 * 2 * np.pi), -0.8, None)
    weekly = np.where(ts.dayofweek.to_numpy() >= 5, 0.6, 1.0)
    return pd.DataFrame({"ts": ts, "gb": (daily * weekly + rng.normal(0, 4, len(ts))).clip(min=0).round(2)})


def test_hourly_traffic_at_the_weekly_horizon_with_an_interval(tmp_path):
    path = tmp_path / "traffic.parquet"
    _traffic().to_parquet(path, index=False)
    prep = json.loads(m.prepare_dataset(str(path), "ts", "gb", horizon=168))
    assert "error" not in prep, prep
    assert prep["horizon"] == 168 and prep["test_shape"][0] == 168
    assert {24, 168} <= set(prep["lags"]) and set(prep["seasonality_autocorrelation"]) == {"24", "168"}
    assert prep["seasonal_periods"] == 168 and prep["seasonal_periods_basis"] == "strongest autocorrelation"
    run_id = prep["run_id"]
    base = json.loads(m.train_baseline(run_id))
    trained = json.loads(m.train_model(run_id, "xgboost"))
    assert trained["horizon"] == 168 and trained["backtest_origins"] >= 3, trained
    assert base["wape"] < 15, base  # the weekly-season naive is a real baseline, not a straw man
    assert trained["mase"] < 1 and trained["wape"] < base["wape"], (trained, base)
    assert 0.6 <= trained["interval_coverage"] <= 0.98, trained["interval_coverage"]
    fm = joblib.load(m._run_dir(run_id) / "model.pkl")
    assert "cal_hour" in fm.fitted.feature_names_in_
    widths = np.asarray(fm.interval["upper"]) - np.asarray(fm.interval["lower"])
    assert (np.diff(widths) >= -1e-9).all(), "the band must widen with the horizon"
    ahead = fm.forecast(200)  # beyond the calibrated horizon: still banded, never below zero
    assert {"lower", "upper"} <= set(ahead.columns) and (ahead["lower"] >= 0).all()
    assert (ahead["upper"] >= ahead["forecast"]).all() and (ahead["lower"] <= ahead["forecast"]).all()
    m.record_business_context(run_id, "plan cell capacity", "hourly GB", "beat the naive forecast",
                              success_metric="wape", success_threshold=30)
    assert m.compute_readiness(run_id)["checks"]["baseline_beaten"]["status"] == "pass"


def test_duplicates_and_gaps_are_decisions_not_accidents(tmp_path):
    rng = np.random.default_rng(1)
    days = pd.date_range("2026-01-01", periods=120, freq="D")
    rows = [{"day": d, "channel": c, "calls": int(rng.poisson(100 if c == "app" else 40))}
            for d in days for c in ("app", "ivr")]
    df = pd.DataFrame(rows)
    df = df[~df["day"].isin(days[[20, 21, 50, 51, 52]])]  # a five-day outage in the feed
    path = tmp_path / "calls.csv"
    df.drop(columns="channel").to_csv(path, index=False)
    refused = json.loads(m.prepare_dataset(str(path), "day", "calls"))
    assert "aggregate" in refused["error"], refused
    prep = json.loads(m.prepare_dataset(str(path), "day", "calls", aggregate="sum", horizon=14))
    assert prep["freq"] == "D" and "5 missing period" in prep["gaps_inserted"], prep
    run_id = prep["run_id"]
    assert "fill" in json.loads(m.train_model(run_id, "exponential_smoothing"))["error"]
    proposal = json.loads(m.propose_imputation(run_id))["proposals"]["calls"]
    assert proposal["suggested_strategy"] == "interpolate", proposal  # no zero days: a level, not sparse counts
    m.apply_imputation(run_id, json.dumps({"calls": "interpolate"}))
    trained = json.loads(m.train_model(run_id, "exponential_smoothing"))
    assert "error" not in trained and trained["horizon"] == 14, trained
    report_lines = dict(m.diagnostics._report_sections(m._load_meta(run_id)))["Forecast setup"]
    assert any("5 missing period(s) inserted" in line and "interpolate" in line for line in report_lines)


def test_one_series_or_the_total_of_a_long_table(tmp_path):
    rng = np.random.default_rng(2)
    days = pd.date_range("2026-01-01", periods=90, freq="D")
    long = pd.DataFrame([{"day": d, "cell": c, "gb": float(rng.normal(10 * (i + 1), 1))}
                         for d in days for i, c in enumerate(("A", "B", "C"))])
    path = tmp_path / "cells.csv"
    long.to_csv(path, index=False)
    assert "series_value" in json.loads(m.prepare_dataset(str(path), "day", "gb", series_column="cell"))["error"]
    one = json.loads(m.prepare_dataset(str(path), "day", "gb", series_column="cell", series_value="B"))
    assert one["series"] == "cell = B" and abs(one["target_summary"]["mean"] - 20) < 1, one
    total = json.loads(m.prepare_dataset(str(path), "day", "gb", series_column="cell", aggregate="sum"))
    assert "across 3 cell" in total["series"] and abs(total["target_summary"]["mean"] - 60) < 2, total


def _cells(n_cells=30, days=150, seed=6):
    """Daily traffic of many cells: one weekly shape, very different sizes,
    a few gaps, one cell switched off early and one switched on late."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-05", periods=days, freq="D")
    shape = np.where(dates.dayofweek.to_numpy() >= 5, 0.6, 1.0)
    rows = []
    for c in range(n_cells):
        size = float(rng.lognormal(3.5, 1.0))
        y = (size * shape * (1 + 0.002 * np.arange(days)) * rng.normal(1, 0.08, days)).clip(min=0)
        frame = pd.DataFrame({"day": dates, "cell": f"C{c:02d}", "gb": y.round(3)})
        if c == 0:
            frame = frame.iloc[:90]  # decommissioned before the held-out weeks
        if c == 1:
            frame = frame.iloc[-30:]  # switched on a month ago: too short to backtest
        if c in (2, 3):
            frame = frame.drop(frame.index[[40, 41, 77]])  # missing reports
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_many_series_one_global_model(tmp_path):
    path = tmp_path / "cells.parquet"
    _cells().to_parquet(path, index=False)
    prep = json.loads(m.prepare_dataset(str(path), "day", "gb", series_column="cell", each_series=True, horizon=14))
    assert "error" not in prep, prep
    assert prep["series"] == 28 and prep["series_left_out"] == {"stopped_before_held_out": 1, "too_short": 1}
    assert prep["seasonal_periods"] == 7 and "6 missing period" in prep["gaps_inserted"], prep
    run_id = prep["run_id"]
    assert "fill" in json.loads(m.train_model(run_id, "xgboost"))["error"]
    filled = json.loads(m.apply_imputation(run_id, json.dumps({"gb": "interpolate"})))
    assert filled["within"] == "cell" and filled["still_missing"] == 0, filled
    naive = json.loads(m.train_baseline(run_id))
    trained = json.loads(m.train_model(run_id, "xgboost"))
    assert trained["series"] == 28 and trained["wape"] < naive["wape"], (trained["wape"], naive["wape"])
    assert trained["series_beating_naive_share"] >= 0.6 and trained["mase"] < 1, trained
    assert 0.6 <= trained["interval_coverage"] <= 0.99, trained["interval_coverage"]
    assert m.compute_readiness(run_id)["checks"]["baseline_beaten"]["status"] == "pass"
    explained = json.loads(m.explain_model(run_id))
    top = [f for f, _ in explained["feature_importance"][:4]]
    assert {"lag_7", "cal_dayofweek", "cal_is_weekend", "roll_7_mean"} & set(top), top
    ets = json.loads(m.train_model(run_id, "exponential_smoothing"))
    assert ets["series"] == 28 and ets["wape"] < naive["wape"], ets
    out = tmp_path / "cells.pkl"
    m.record_business_context(run_id, "plan cell capacity", "daily GB per cell", "beat naive")
    exported = json.loads(m.export_model(run_id, out_path=str(out), force=True))
    assert "error" not in exported, exported
    ahead = pd.read_csv(json.loads(m.predict(str(out), 14))["out_path"])
    assert len(ahead) == 28 * 14 and ahead["cell"].nunique() == 28 and {"lower", "upper"} <= set(ahead.columns)
    assert (ahead["lower"] >= 0).all() and pd.to_datetime(ahead["day"]).min() > pd.Timestamp("2026-06-03")


def test_scale_free_metrics_and_mape_withheld_near_zero():
    got = m._rmse_mae_mape(np.array([0.0, 10, 20]), np.array([1.0, 12, 18]), scale=2.0)
    assert got["mape"] is None and got["wape"] == round(5 / 30 * 100, 4) and got["mase"] == round(5 / 3 / 2, 4)
    assert got["bias_pct"] == round(1 / 30 * 100, 4)
    # one lucky held-out window does not decide the baseline gate: every window counts
    pooled = m.diagnostics.pooled_rmse({"rmse": 11.56, "backtest_rmse_mean": 20.94, "backtest_origins": 4})
    assert pooled == round(float(np.sqrt((4 * 20.94**2 + 11.56**2) / 5)), 4)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if fn.__code__.co_argcount:
                with _tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"ok {name}")
    print("all forecast rigour checks passed")
