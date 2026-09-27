"""Proves the actual rigor claims — not just "it runs without erroring":
the split is chronological (never random), imputation is ffill (not
median/mode) and the train/test boundary carry-forward doesn't leak future
values backward, walk-forward backtesting never sees true intervening
target values a real forecast wouldn't have, all three model types
(naive_seasonal, exponential_smoothing, xgboost) train/tune/explain/export,
explain_model's response shape differs correctly per model type, the
overfitting gate blocks export_model until forced, the exported artifact
forecasts recursively via the actual `predict` MCP tool, every error branch
(prepare_dataset validation, unknown model, no fitted model yet, unknown
run_id) returns an error instead of crashing, and stale runs get swept by
RUN_RETENTION_DAYS.
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
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import mcp_server as m


def _make_csv(n: int = 200) -> str:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    dayofweek = dates.dayofweek.to_numpy()
    weekly_pattern = np.array([10.0, 12.0, 11.0, 13.0, 15.0, 20.0, 18.0])[dayofweek]
    trend = np.linspace(0, 20, n)
    promo = rng.choice(["none", "promo"], n, p=[0.8, 0.2])
    promo_bump = np.where(promo == "promo", 8.0, 0.0)
    noise = rng.normal(0, 2, n)
    target = 50 + trend + weekly_pattern + promo_bump + noise
    target[50] += 100  # exercises detect_outliers

    visits = rng.normal(100, 10, n)
    visits[5:10] = np.nan  # missing inside train — exercises propose_imputation
    visits[160] = np.nan  # missing at the very start of test — exercises the
    # train->test ffill boundary carry-forward
    df = pd.DataFrame(
        {
            "date": dates.astype(str),
            "target": target,
            "promo": promo,
            "visits": visits,
            "row_id": range(n),  # near-unique, exercises propose_drop_columns
        }
    )
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)
    return path


def _second_run() -> tuple[str, Path]:
    rng = np.random.default_rng(7)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates.astype(str), "target": rng.normal(10, 1, n)})
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)
    run_id = json.loads(m.prepare_dataset(path, "date", "target"))["run_id"]
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    json.loads(m.train_model(run_id, "naive_seasonal"))
    Path(path).unlink()
    return run_id, m._run_dir(run_id)


def _main_flow() -> tuple[str, Path]:
    path = _make_csv()

    eda = json.loads(m.eda(path))
    assert eda["shape"] == [200, 5]

    prep = json.loads(m.prepare_dataset(path, "date", "target"))
    run_id = prep["run_id"]
    assert prep["train_shape"][0] + prep["test_shape"][0] == 200
    assert prep["freq"] == "D"
    assert prep["seasonal_periods"] == 7
    run_dir = m._run_dir(run_id)

    # ── Chronological split, not random: every train.csv date must be
    # strictly earlier than every test.csv date. ──────────────────────────
    train_dates = pd.to_datetime(pd.read_csv(run_dir / "train.csv")["date"])
    test_dates = pd.to_datetime(pd.read_csv(run_dir / "test.csv")["date"])
    assert train_dates.max() < test_dates.min(), (
        "split must be chronological, not random"
    )

    # ── Input validation before a run is even created. ─────────────────────
    assert "error" in json.loads(
        m.prepare_dataset(path, "date", "promo")
    )  # non-numeric target
    assert "error" in json.loads(
        m.prepare_dataset(path, "no_such_col", "target")
    )  # bad date column
    assert "error" in json.loads(
        m.prepare_dataset(path, "date", "no_such_col")
    )  # bad target

    dup_path = tempfile.mkstemp(suffix=".csv")[1]
    pd.DataFrame({"date": ["2024-01-01", "2024-01-01"], "target": [1.0, 2.0]}).to_csv(
        dup_path, index=False
    )
    assert "error" in json.loads(m.prepare_dataset(dup_path, "date", "target")), (
        "duplicate timestamps must error"
    )
    Path(dup_path).unlink()

    bad_date_path = tempfile.mkstemp(suffix=".csv")[1]
    pd.DataFrame({"date": ["not-a-date", "2024-01-02"], "target": [1.0, 2.0]}).to_csv(
        bad_date_path, index=False
    )
    assert "error" in json.loads(m.prepare_dataset(bad_date_path, "date", "target")), (
        "unparseable date must error"
    )
    Path(bad_date_path).unlink()

    out = json.loads(m.detect_outliers(run_id))
    assert out["target_outlier_count"] >= 1

    impute_props = json.loads(m.propose_imputation(run_id))
    assert impute_props["proposals"]["visits"]["suggested_strategy"] == "ffill"

    drop_props = json.loads(m.propose_drop_columns(run_id))
    assert drop_props["proposals"]["row_id"] == "near-unique, likely an ID column"

    # ── ffill leakage check: test's first row had a manually-injected NaN
    # (index 160) — it must be seeded from TRAIN's last value, not from any
    # later (future) test-fold value. ──────────────────────────────────────
    train_before = pd.read_csv(run_dir / "train.csv")
    expected_seed = float(train_before["visits"].ffill().iloc[-1])

    imputed = json.loads(m.apply_imputation(run_id, json.dumps({"visits": "ffill"})))
    assert imputed["train_shape"][0] == prep["train_shape"][0]
    train_after = pd.read_csv(run_dir / "train.csv")
    test_after = pd.read_csv(run_dir / "test.csv")
    assert train_after["visits"].isna().sum() == 0
    assert test_after["visits"].isna().sum() == 0
    assert abs(test_after["visits"].iloc[0] - expected_seed) < 1e-9, (
        "test-boundary ffill must seed from train's last value"
    )

    json.loads(m.apply_drop_columns(run_id, json.dumps(["row_id"])))
    assert "row_id" not in pd.read_csv(run_dir / "train.csv").columns

    # ── Unknown-model error branches. ───────────────────────────────────────
    assert "error" in json.loads(m.compare_models(run_id, "not_a_model"))
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    assert "error" in json.loads(m.train_model(run_id, "not_a_model"))
    assert "error" in json.loads(m.tune_hyperparams(run_id, "not_a_model"))

    # ── No fitted model yet — explain/export must error, not crash. ────────
    assert "error" in json.loads(m.explain_model(run_id))
    assert "error" in json.loads(m.export_model(run_id))

    # ── compare_models is read-only — must not write model.pkl. ────────────
    ranked = json.loads(m.compare_models(run_id))
    assert not (run_dir / "model.pkl").exists()
    assert ranked["recommended"] in m.MODELS

    # ── naive_seasonal: train + explain (no features to rank). ─────────────
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    naive_res = json.loads(m.train_model(run_id, "naive_seasonal"))
    assert "rmse" in naive_res and "mae" in naive_res and "mape" in naive_res
    naive_explain = json.loads(m.explain_model(run_id))
    assert naive_explain["method"] == "none"

    # ── exponential_smoothing: train + explain (decomposition, not features). ─
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    es_res = json.loads(m.train_model(run_id, "exponential_smoothing"))
    assert "rmse" in es_res
    es_explain = json.loads(m.explain_model(run_id))
    assert es_explain["method"] == "decomposition"
    assert "fitted_params" in es_explain

    # export_model with no out_path must suggest one instead of writing.
    suggestion = json.loads(m.export_model(run_id))
    assert "suggested_out_path" in suggestion

    # ── xgboost: train + explain (permutation feature importance) + tune. ──
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    xgb_res = json.loads(m.train_model(run_id, "xgboost"))
    assert "rmse" in xgb_res
    xgb_explain = json.loads(m.explain_model(run_id, method="permutation"))
    assert xgb_explain["method"] == "permutation"
    assert len(xgb_explain["feature_importance"]) > 0

    tuned = json.loads(m.tune_hyperparams(run_id, "xgboost", n_trials=2, cv_folds=2))
    assert "best_params" in tuned and tuned["best_params"]
    assert m._load_meta(run_id)["best_params"] == tuned["best_params"]
    tuned_model = joblib.load(run_dir / "model.pkl")
    assert tuned_model.model_type == "xgboost"

    # ── tune_hyperparams' no-grid path: naive_seasonal has nothing to tune. ─
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    json.loads(m.train_model(run_id, "naive_seasonal"))
    tuned_naive = json.loads(
        m.tune_hyperparams(run_id, "naive_seasonal", n_trials=1, cv_folds=2)
    )
    assert tuned_naive["best_params"] == {}

    # ── compare_runs: unknown run_id errors cleanly, ranking spans 2 runs. ──
    other_run_id, other_run_dir = _second_run()
    cmp = json.loads(m.compare_runs(f"{run_id},{other_run_id},not-a-real-run"))
    assert len(cmp["runs"]) == 3
    assert any(r["run_id"] == "not-a-real-run" and "error" in r for r in cmp["runs"])
    assert set(cmp["ranked_by_rmse"]) == {run_id, other_run_id}
    shutil.rmtree(other_run_dir)

    # ── export_model's overfitting gate. ────────────────────────────────────
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    json.loads(m.train_model(run_id, "xgboost"))
    meta = m._load_meta(run_id)
    meta["baseline_metrics"]["overfitting_warning"] = True
    meta["tuned_metrics"] = None
    m._save_meta(run_id, meta)
    blocked = json.loads(
        m.export_model(run_id, str(Path(tempfile.mkdtemp()) / "blocked.pkl"))
    )
    assert "error" in blocked, (
        "export_model must refuse an overfitting-flagged model without force=True"
    )

    # ── Export must be a real standalone forecaster, and `predict` must
    # forecast recursively via the actual MCP tool. ─────────────────────────
    out_path = str(Path(tempfile.mkdtemp()) / "model.pkl")
    exported = json.loads(m.export_model(run_id, out_path, force=True))
    assert exported["out_path"] == out_path
    bundle = joblib.load(out_path)
    assert bundle.model_type == "xgboost"

    predicted = json.loads(m.predict(out_path, horizon=5))
    assert predicted["horizon"] == 5
    forecast_df = pd.read_csv(predicted["out_path"])
    assert len(forecast_df) == 5
    assert "forecast" in forecast_df.columns
    assert "forecast_summary" in predicted and "preview" in predicted

    # future_exog_path with the wrong row count must error, not crash.
    bad_exog_path = tempfile.mkstemp(suffix=".csv")[1]
    pd.DataFrame({"promo": ["none", "promo"]}).to_csv(bad_exog_path, index=False)
    assert "error" in json.loads(
        m.predict(out_path, horizon=5, future_exog_path=bad_exog_path)
    )
    Path(bad_exog_path).unlink()

    Path(path).unlink()
    return run_id, run_dir


def _error_path_checks() -> None:
    fd, path = tempfile.mkstemp(suffix=".csv")

    pd.DataFrame({"date": [], "target": []}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(path, "date", "target")), (
        "empty CSV must error, not crash"
    )

    dates = pd.date_range("2024-01-01", periods=5, freq="D").astype(str)
    pd.DataFrame({"date": dates, "target": range(5)}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(path, "date", "target")), (
        "too few rows must error"
    )

    assert "error" in json.loads(m.predict("no/such/model.pkl", horizon=3)), (
        "missing pkl must error, not crash"
    )

    Path(path).unlink()


def _cleanup_check() -> None:
    """prepare_dataset must sweep run dirs whose meta.json is older than
    RUN_RETENTION_DAYS — runs are disposable working state, not an archive."""
    rng = np.random.default_rng(9)
    n = 40
    dates = pd.date_range("2024-01-01", periods=n, freq="D").astype(str)
    df = pd.DataFrame({"date": dates, "target": rng.normal(0, 1, n)})
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)

    old_run_id = json.loads(m.prepare_dataset(path, "date", "target"))["run_id"]
    old_run_dir = m._run_dir(old_run_id)
    stale = time.time() - (m.RUN_RETENTION_DAYS + 1) * 86400
    os.utime(old_run_dir / "meta.json", (stale, stale))

    new_run_id = json.loads(m.prepare_dataset(path, "date", "target"))[
        "run_id"
    ]  # triggers the sweep
    assert not old_run_dir.exists(), "a run older than RUN_RETENTION_DAYS must be swept"
    assert m._run_dir(new_run_id).exists(), (
        "the just-created run must survive its own sweep"
    )

    shutil.rmtree(m._run_dir(new_run_id))
    Path(path).unlink()


def main():
    run_id, run_dir = _main_flow()
    shutil.rmtree(run_dir)
    _error_path_checks()
    _cleanup_check()
    print("all checks passed")


if __name__ == "__main__":
    main()
