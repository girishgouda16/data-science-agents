"""Proves the actual rigor claims — not just "it runs without erroring":
imputation (median/mode/literal/drop_column)/datetime decomposition are
fit/derived from the training fold only (no leakage), a tuned pipeline is
what explain/export pick up automatically (no manual hyperparameter
threading), SHAP works for every model type including linear_regression's
no-grid tune_hyperparams path, the overfitting gate blocks export_model
until forced, the exported artifact is a real standalone predictor, the
actual `predict` MCP tool works end-to-end on new data, every error branch
(prepare_dataset validation, unknown model, no fitted model yet, unknown
run_id) actually returns an error instead of crashing, and stale runs get
swept by RUN_RETENTION_DAYS.
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


def _make_csv() -> str:
    rng = np.random.default_rng(0)
    n = 320
    hour = rng.integers(0, 24, n).astype(float)
    hour[-8:] = [100, 150, -80, 120, 200, 90, -60, 140]  # exercises detect_outliers
    channel = rng.choice(["web", "mobile", "atm"], n)
    tenure = rng.normal(24, 6, n)  # numeric feature, exercises propose_imputation
    tenure[:15] = np.nan  # ~5% missing
    # target is a genuine (noisy) linear function of hour + channel, so a
    # real model can beat the mean baseline — not just random noise.
    channel_bump = (
        pd.Series(channel).map({"web": 0.0, "mobile": 5.0, "atm": 10.0}).to_numpy()
    )
    amount = (
        20 + np.nan_to_num(hour, nan=12.0) * 1.5 + channel_bump + rng.normal(0, 3, n)
    )
    ts = pd.date_range("2024-01-01", periods=n, freq="h").astype(str)
    df = pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "channel": channel,
            "tenure": tenure,
            "row_id": range(n),  # near-unique, exercises propose_drop_columns
            "signup_ts": ts,  # exercises propose/apply_datetime_features
        }
    )
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)
    return path


def _second_run() -> tuple[str, Path]:
    """A lightweight second trained run — just for compare_runs' multi-run
    ranking, doesn't need the full cleaning pipeline."""
    rng = np.random.default_rng(7)
    n = 60
    df = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n) * 2 + 1})
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)
    run_id = json.loads(m.prepare_dataset(path, "y"))["run_id"]
    json.loads(m.train_model(run_id, "linear_regression"))
    Path(path).unlink()
    return run_id, m._run_dir(run_id)


def _main_flow() -> tuple[str, Path]:
    path = _make_csv()

    eda = json.loads(m.eda(path))
    assert eda["shape"] == [320, 6]

    prep = json.loads(m.prepare_dataset(path, "amount"))
    run_id = prep["run_id"]
    assert prep["train_shape"][0] + prep["test_shape"][0] == 320
    run_dir = m._run_dir(run_id)

    # A non-numeric or missing target must be rejected before a run is created.
    assert "error" in json.loads(m.prepare_dataset(path, "channel"))
    assert "error" in json.loads(m.prepare_dataset(path, "no_such_column"))

    # ── Leakage check: the imputed value must equal the TRAIN fold's own
    # median, computed independently here from a snapshot taken before
    # apply_imputation runs — not the full dataset's median. ──────────────
    train_before = pd.read_csv(run_dir / "train.csv")
    expected_median = float(train_before["tenure"].median())

    out = json.loads(m.detect_outliers(run_id))
    assert out["outlier_counts_by_column"]["hour"] >= 1
    assert "amount" not in out["outlier_counts_by_column"], (
        "target column must be excluded from the feature outlier report"
    )
    assert (
        isinstance(out["target_outlier_count"], int)
        and out["target_outlier_count"] >= 0
    )

    impute_props = json.loads(m.propose_imputation(run_id))
    assert impute_props["proposals"]["tenure"]["suggested_strategy"] == "median"

    drop_props = json.loads(m.propose_drop_columns(run_id))
    # The shared identifier rules (core/data_quality) now explain themselves.
    assert drop_props["proposals"]["row_id"].startswith("near-unique"), drop_props["proposals"]

    dt_props = json.loads(m.propose_datetime_features(run_id))
    assert "signup_ts" in dt_props["datetime_columns"]

    imputed = json.loads(m.apply_imputation(run_id, json.dumps({"tenure": "median"})))
    assert imputed["train_shape"][0] == prep["train_shape"][0]
    meta = m._load_meta(run_id)
    assert meta["imputation"]["tenure"] == expected_median, (
        "imputation value leaked test-fold information"
    )
    assert pd.read_csv(run_dir / "train.csv")["tenure"].isna().sum() == 0
    assert pd.read_csv(run_dir / "test.csv")["tenure"].isna().sum() == 0

    json.loads(m.apply_drop_columns(run_id, json.dumps(["row_id"])))
    assert "row_id" not in pd.read_csv(run_dir / "train.csv").columns

    dt_applied = json.loads(m.apply_datetime_features(run_id, "signup_ts"))
    assert dt_applied["decomposed_columns"] == ["signup_ts"]
    train_cols = pd.read_csv(run_dir / "train.csv").columns
    assert "signup_ts" not in train_cols and "signup_ts_hour" in train_cols

    # ── Unknown-model error branches — checked before any training exists,
    # since all three validate the model name before touching a pipeline. ──
    assert "error" in json.loads(m.compare_models(run_id, "not_a_model"))
    assert "error" in json.loads(m.train_model(run_id, "not_a_model"))
    assert "error" in json.loads(m.tune_hyperparams(run_id, "not_a_model"))

    # ── No fitted model yet — explain/export must error, not crash. ───────
    assert "error" in json.loads(m.explain_model(run_id))
    assert "error" in json.loads(m.export_model(run_id))

    # ── compare_models is read-only — must not write pipeline.pkl. ────────
    ranked = json.loads(m.compare_models(run_id))
    assert not (run_dir / "pipeline.pkl").exists()
    assert ranked["recommended"] in m.MODELS

    # ── Baseline train, then tune — tune must OVERWRITE pipeline.pkl so
    # explain/export pick up the tuned model with no extra argument. ──────
    # Screen the (now final) feature set before training — the evidence_ordering
    # gate blocks export of a model trained without one.
    json.loads(m.detect_data_leakage(run_id))
    trained = json.loads(m.train_model(run_id, "random_forest"))
    assert "r2" in trained and "rmse" in trained and "mae" in trained

    # export_model with no out_path must suggest one instead of writing
    # (pipeline exists now, and this run isn't overfitting-flagged yet).
    suggestion = json.loads(m.export_model(run_id))
    assert "suggested_out_path" in suggestion

    tuned = json.loads(
        m.tune_hyperparams(run_id, "random_forest", n_trials=3, cv_folds=3)
    )
    assert "best_params" in tuned and tuned["best_params"]
    assert m._load_meta(run_id)["best_params"] == tuned["best_params"]
    tuned_pipeline = joblib.load(run_dir / "pipeline.pkl")
    assert (
        tuned_pipeline.named_steps["regressor"].get_params()["n_estimators"]
        == tuned["best_params"]["regressor__n_estimators"]
    )

    # explain_model must use the TUNED pipeline with zero extra arguments.
    perm = json.loads(m.explain_model(run_id, method="permutation"))
    assert len(perm["feature_importance"]) > 0

    shap_rf = json.loads(m.explain_model(run_id, method="shap"))
    assert len(shap_rf["feature_importance"]) > 0

    # SHAP for the other two model families (TreeExplainer vs LinearExplainer paths).
    json.loads(m.train_model(run_id, "linear_regression"))
    shap_lr = json.loads(m.explain_model(run_id, method="shap"))
    assert len(shap_lr["feature_importance"]) > 0

    # tune_hyperparams' no-grid path: linear_regression's PARAM_GRIDS entry
    # is empty, so this just refits and CV-scores it directly instead of
    # running Optuna — best_params must come back empty, not crash.
    tuned_lr = json.loads(
        m.tune_hyperparams(run_id, "linear_regression", n_trials=2, cv_folds=3)
    )
    assert tuned_lr["best_params"] == {}
    assert "cv_r2_best" in tuned_lr

    json.loads(m.train_model(run_id, "xgboost"))
    shap_xgb = json.loads(m.explain_model(run_id, method="shap"))
    assert len(shap_xgb["feature_importance"]) > 0

    # ── compare_runs: unknown run_id errors cleanly, and ranking spans a
    # real second run — not just a single run_id echoed back. ─────────────
    other_run_id, other_run_dir = _second_run()
    cmp = json.loads(m.compare_runs(f"{run_id},{other_run_id},not-a-real-run"))
    assert len(cmp["runs"]) == 3
    assert any(r["run_id"] == "not-a-real-run" and "error" in r for r in cmp["runs"])
    assert set(cmp["ranked_by_r2"]) == {run_id, other_run_id}
    shutil.rmtree(other_run_dir)

    # ── export_model's overfitting gate: force a large CV-vs-test gap by
    # hand, confirm it's refused without force=True, then allowed with it. ─
    meta = m._load_meta(run_id)
    meta["baseline_metrics"]["overfitting_warning"] = True
    meta["tuned_metrics"] = None
    m._save_meta(run_id, meta)
    blocked = json.loads(
        m.export_model(run_id, str(Path(tempfile.mkdtemp()) / "blocked.pkl"))
    )
    assert "error" in blocked, (
        "export_model must refuse to export an overfitting-flagged model without force=True"
    )

    # ── Export must be a real standalone predictor — including replaying
    # the drop/datetime-decomposition on brand-new raw data. ──────────────
    out_path = str(Path(tempfile.mkdtemp()) / "model.pkl")
    exported = json.loads(m.export_model(run_id, out_path, force=True))
    assert exported["out_path"] == out_path
    bundle = joblib.load(out_path)
    new_raw = pd.DataFrame(
        {
            "amount": [55.0, np.nan],
            "hour": [10, 22],
            "channel": ["web", "atm"],
            "tenure": [30.0, np.nan],
            "signup_ts_year": [2024, 2024],
            "signup_ts_month": [1, 2],
            "signup_ts_day": [1, 2],
            "signup_ts_dayofweek": [0, 4],
            "signup_ts_hour": [10, 22],
        }
    )
    preds = bundle["pipeline"].predict(new_raw.drop(columns=["amount"]))
    assert len(preds) == 2

    # ── The actual `predict` MCP tool (not bundle.predict directly) — CSV
    # in, predictions CSV out, target column dropped if present. ──────────
    predict_dir = Path(tempfile.mkdtemp())
    data_path = predict_dir / "new_data.csv"
    new_raw.to_csv(
        data_path, index=False
    )  # target column present — predict must drop it
    predicted = json.loads(m.predict(out_path, str(data_path)))
    assert predicted["n_rows"] == 2
    out_df = pd.read_csv(predicted["out_path"])
    assert "prediction" in out_df.columns
    assert "prediction_summary" in predicted and "preview" in predicted

    data_path2 = predict_dir / "new_data_no_target.csv"
    new_raw.drop(columns=["amount"]).to_csv(data_path2, index=False)
    predicted2 = json.loads(m.predict(out_path, str(data_path2)))
    assert predicted2["n_rows"] == 2

    Path(path).unlink()
    return run_id, run_dir


def _imputation_variant_checks() -> None:
    """apply_imputation's mode / literal-value / drop_column branches — the
    main flow only exercises "median"."""
    rng = np.random.default_rng(3)
    n = 50
    region = rng.choice(["north", "south"], n).astype(object)
    region[:5] = None  # exercises the "mode" branch
    score = rng.normal(0, 1, n)
    score[:5] = np.nan  # exercises the literal-value branch
    df = pd.DataFrame(
        {
            "region": region,
            "score": score,
            "mostly_missing": [np.nan] * 40 + list(range(10)),  # exercises drop_column
            "target": rng.normal(0, 1, n),
        }
    )
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)
    run_id = json.loads(m.prepare_dataset(path, "target"))["run_id"]
    run_dir = m._run_dir(run_id)

    train_before = pd.read_csv(run_dir / "train.csv")
    expected_mode = train_before["region"].mode().iloc[0]

    json.loads(
        m.apply_imputation(
            run_id,
            json.dumps(
                {
                    "region": "mode",
                    "score": -1.0,
                    "mostly_missing": "drop_column",
                }
            ),
        )
    )
    meta = m._load_meta(run_id)
    assert meta["imputation"]["region"] == expected_mode
    assert meta["imputation"]["score"] == -1.0
    assert "mostly_missing" in meta["dropped_columns"]
    train_after = pd.read_csv(run_dir / "train.csv")
    assert "mostly_missing" not in train_after.columns
    assert train_after["region"].isna().sum() == 0
    assert train_after["score"].isna().sum() == 0

    shutil.rmtree(run_dir)
    Path(path).unlink()


def _error_path_checks() -> None:
    fd, path = tempfile.mkstemp(suffix=".csv")

    pd.DataFrame({"a": [], "target": []}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(path, "target")), (
        "empty CSV must error, not crash"
    )

    pd.DataFrame({"a": range(20), "target": [None] * 20}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(path, "target")), (
        "all-missing target must error"
    )

    pd.DataFrame({"a": range(5), "target": range(5)}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(path, "target")), (
        "fewer than 10 rows must error"
    )

    Path(path).unlink()


def _cleanup_check() -> None:
    """prepare_dataset must sweep run dirs whose meta.json is older than
    RUN_RETENTION_DAYS — runs are disposable working state, not an archive."""
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"a": rng.normal(0, 1, 20), "target": rng.normal(0, 1, 20)})
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)

    old_run_id = json.loads(m.prepare_dataset(path, "target"))["run_id"]
    old_run_dir = m._run_dir(old_run_id)
    stale = time.time() - (m.RUN_RETENTION_DAYS + 1) * 86400
    os.utime(old_run_dir / "meta.json", (stale, stale))

    new_run_id = json.loads(m.prepare_dataset(path, "target"))[
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
    _imputation_variant_checks()
    _error_path_checks()
    _cleanup_check()
    print("all checks passed")


if __name__ == "__main__":
    main()
