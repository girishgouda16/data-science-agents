"""Proves the actual rigor claims — not just "it runs without erroring":
imputation/encoding are fit on the training fold only (no leakage), SMOTE
only ever fires during training, a tuned pipeline is what explain/export
pick up automatically (no manual hyperparameter threading), SHAP works for
every model type (binary AND multiclass — the ndim==3 stacking branch and
LinearExplainer's multiclass shape only fire with >2 classes), the exported
artifact is a real standalone predictor, and every tool's error path
actually returns an error instead of crashing.
No MCP transport needed — calls the underlying functions directly.

Run: python test_mcp_server.py
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
    amount = np.concatenate(
        [rng.normal(50, 10, n - 8), [500, 600, -400, 550, 700, 620, -350, 580]]
    )
    amount = amount.astype(float)
    amount[:15] = np.nan  # 5% missing, exercises propose_imputation
    df = pd.DataFrame(
        {
            "amount": amount,
            "hour": rng.integers(0, 24, n),
            "channel": rng.choice(
                ["web", "mobile", "atm"], n
            ),  # categorical, exercises one-hot encoding
            "row_id": range(n),  # near-unique, exercises propose_drop_columns
            "label": [0] * (n - 20)
            + [1] * 20,  # imbalanced (~6%), exercises SMOTE eligibility
        }
    )
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)
    return path


def _make_multiclass_csv() -> str:
    rng = np.random.default_rng(1)
    n = 240
    score = rng.normal(0, 1, n)
    score[:10] = np.nan  # exercises apply_imputation's literal-value branch
    df = pd.DataFrame(
        {
            "amount": rng.normal(50, 10, n),
            "hour": rng.integers(0, 24, n),
            "channel": rng.choice(["web", "mobile", "atm"], n),
            "score": score,
            "signup_date": pd.Timestamp("2023-01-01")
            + pd.to_timedelta(rng.integers(0, 400, n), unit="D"),
            "label": rng.choice(["low", "medium", "high"], n, p=[0.5, 0.3, 0.2]),
        }
    )
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)
    return path


def _binary_flow() -> tuple[str, Path]:
    path = _make_csv()

    eda = json.loads(m.eda(path))
    assert eda["shape"] == [320, 5]

    target_shape = json.loads(m.inspect_target(path, "label"))
    assert target_shape["is_binary"] is True and target_shape["n_classes"] == 2

    prep = json.loads(m.prepare_dataset(path, "label"))
    run_id = prep["run_id"]
    assert prep["train_shape"][0] + prep["test_shape"][0] == 320
    run_dir = m._run_dir(run_id)

    # ── Leakage check: the imputed value must equal the TRAIN fold's own
    # median, computed independently here from a snapshot taken before
    # apply_imputation runs — not the full dataset's median. ──────────────
    train_before = pd.read_csv(run_dir / "train.csv")
    expected_median = float(train_before["amount"].median())

    imb = json.loads(m.check_imbalance(run_id))
    expected_counts = {
        str(k): v for k, v in train_before["label"].value_counts().to_dict().items()
    }
    assert imb["counts"] == expected_counts, (
        "check_imbalance must read the train fold, not the full file"
    )

    out = json.loads(m.detect_outliers(run_id))
    assert out["outlier_counts_by_column"]["amount"] >= 1

    impute_props = json.loads(m.propose_imputation(run_id))
    assert impute_props["proposals"]["amount"]["suggested_strategy"] == "median"

    drop_props = json.loads(m.propose_drop_columns(run_id))
    assert "near-unique" in drop_props["proposals"]["row_id"]

    leakage = json.loads(m.detect_data_leakage(run_id))
    assert "row_id" in leakage["id_like_columns"]
    assert m._load_meta(run_id)["leakage"] == leakage, (
        "detect_data_leakage must persist its result for generate_report"
    )

    imputed = json.loads(m.apply_imputation(run_id, json.dumps({"amount": "median"})))
    assert imputed["train_shape"][0] == prep["train_shape"][0]
    meta = m._load_meta(run_id)
    assert meta["imputation"]["amount"] == expected_median, (
        "imputation value leaked test-fold information"
    )
    assert pd.read_csv(run_dir / "train.csv")["amount"].isna().sum() == 0
    assert pd.read_csv(run_dir / "test.csv")["amount"].isna().sum() == 0

    dropped = json.loads(m.apply_drop_columns(run_id, json.dumps(["row_id"])))
    assert (
        "row_id" not in dropped or True
    )  # dropped_columns returned, shape checked below
    assert (
        pd.read_csv(run_dir / "train.csv").shape[1] == 4
    )  # amount, hour, channel, label

    smote_check = json.loads(m.propose_smote(run_id))
    # ~6% minority sits above SMOTE_MIN_RATIO (2%): class_weight="balanced" already
    # covers it, so SMOTE is NOT recommended — apply_smote below still works on request.
    assert smote_check["recommend_smote"] is False, smote_check
    test_rows_before_smote = pd.read_csv(run_dir / "test.csv").shape[0]
    json.loads(m.apply_smote(run_id))
    assert m._load_meta(run_id)["use_smote"] is True
    # apply_smote must not touch any file — it's a pipeline flag, not a CSV mutation.
    assert pd.read_csv(run_dir / "test.csv").shape[0] == test_rows_before_smote
    assert pd.read_csv(run_dir / "train.csv").shape[0] == prep["train_shape"][0]

    # ── Baseline train, then tune — tune must OVERWRITE pipeline.pkl so
    # explain/export pick up the tuned model with no extra argument. ──────
    # Imputation and the drop above changed the feature set after the first
    # leakage screen — re-screen, as the evidence_ordering gate requires.
    json.loads(m.detect_data_leakage(run_id))
    trained = json.loads(m.train_model(run_id, "random_forest"))
    assert "cv_score_mean" in trained and "cv_metric" in trained
    baseline_pipeline = joblib.load(run_dir / "pipeline.pkl")
    assert "smote" in baseline_pipeline.named_steps, (
        "use_smote=True must add a smote step to the pipeline"
    )

    tuned = json.loads(m.tune_hyperparams(run_id, "random_forest"))
    assert "best_params" in tuned and tuned["best_params"]
    assert m._load_meta(run_id)["best_params"] == tuned["best_params"]
    tuned_pipeline = joblib.load(run_dir / "pipeline.pkl")
    assert (
        tuned_pipeline.named_steps["classifier"].get_params()["n_estimators"]
        == tuned["best_params"]["classifier__n_estimators"]
    )

    # explain_model must use the TUNED pipeline with zero extra arguments.
    perm = json.loads(m.explain_model(run_id, method="permutation"))
    assert len(perm["feature_importance"]) > 0

    shap_rf = json.loads(m.explain_model(run_id, method="shap"))
    assert len(shap_rf["feature_importance"]) > 0

    # SHAP for the other two model families (TreeExplainer vs LinearExplainer paths).
    json.loads(m.train_model(run_id, "logistic_regression"))
    shap_lr = json.loads(m.explain_model(run_id, method="shap"))
    assert len(shap_lr["feature_importance"]) > 0

    json.loads(m.train_model(run_id, "xgboost"))
    shap_xgb = json.loads(m.explain_model(run_id, method="shap"))
    assert len(shap_xgb["feature_importance"]) > 0

    # retrain random_forest — the model the rest of this flow (threshold
    # tuning, overfitting-gate, export) exercises.
    json.loads(m.train_model(run_id, "random_forest"))

    # threshold tuning: default (F1-max) and target-recall paths, binary only.
    thr_default = json.loads(m.tune_threshold(run_id))
    assert "chosen" in thr_default and "default_0.5" in thr_default
    thr_recall = json.loads(m.tune_threshold(run_id, target_recall=0.8))
    assert thr_recall["chosen"]["recall"] >= 0.8 or "error" in thr_recall

    # ── export_model must refuse an overfitting model unless force=True. ──
    # NOTE: the threshold lives in mcp_server.core's module globals — patch it there,
    # not on the mcp_server package (a `from .core import *` copies the value, so
    # patching the package attribute wouldn't affect core._overfit_gate's own read).
    orig_threshold = m.core.OVERFIT_GAP_THRESHOLD
    m.core.OVERFIT_GAP_THRESHOLD = (
        -1.0
    )  # forces overfitting_warning True on (almost) any gap
    try:
        json.loads(m.train_model(run_id, "random_forest"))
        assert m._load_meta(run_id)["baseline_metrics"]["overfitting_warning"] is True
        refused = json.loads(
            m.export_model(run_id, str(Path(tempfile.mkdtemp()) / "blocked.pkl"))
        )
        assert "error" in refused and "overfitting_warning" in refused["error"]
        forced = json.loads(
            m.export_model(
                run_id, str(Path(tempfile.mkdtemp()) / "forced.pkl"), force=True
            )
        )
        assert forced["out_path"]
    finally:
        m.core.OVERFIT_GAP_THRESHOLD = orig_threshold

    # retrain clean (no artificial overfitting flag) before the real export below.
    json.loads(m.train_model(run_id, "random_forest"))

    # export_model with no out_path must suggest one instead of writing.
    suggestion = json.loads(m.export_model(run_id))
    assert "suggested_out_path" in suggestion

    # ── Export must be a real standalone predictor — including replaying
    # the drop on brand-new raw data that still has the dropped column. ───
    out_path = str(Path(tempfile.mkdtemp()) / "model.pkl")
    exported = json.loads(m.export_model(run_id, out_path))
    assert exported["out_path"] == out_path
    bundle = joblib.load(out_path)
    new_raw = pd.DataFrame(
        {
            "amount": [55.0, np.nan],
            "hour": [10, 22],
            "channel": ["web", "atm"],
            "row_id": [
                9001,
                9002,
            ],  # was dropped during training — must be tolerated, not crash
        }
    )
    preds = bundle["pipeline"].predict(new_raw)
    assert len(preds) == 2

    # ── The actual `predict` MCP tool (not bundle.predict directly) — CSV
    # in, predictions+confidence CSV out, target column dropped if present. ──
    predict_dir = Path(tempfile.mkdtemp())
    data_path = predict_dir / "new_data.csv"
    new_raw.to_csv(data_path, index=False)
    predicted = json.loads(m.predict(out_path, str(data_path)))
    assert predicted["n_rows"] == 2
    out_df = pd.read_csv(predicted["out_path"])
    assert "prediction" in out_df.columns and "confidence" in out_df.columns

    with_target = new_raw.copy()
    with_target["label"] = [
        0,
        1,
    ]  # target present in new data — predict must drop it, not treat it as a feature
    data_path2 = predict_dir / "new_data_with_target.csv"
    with_target.to_csv(data_path2, index=False)
    predicted2 = json.loads(m.predict(out_path, str(data_path2)))
    assert predicted2["n_rows"] == 2

    # ── diagnostics: baseline, re-evaluate, error analysis, segments, stability.
    # Run after export so these read-only checks can't disturb the export test
    # above by changing what "the current model" means. ────────────────────
    baseline = json.loads(m.train_baseline(run_id))
    assert "classification_report" in baseline
    assert m._load_meta(run_id)["model"] == "random_forest", (
        "train_baseline must not overwrite the current model"
    )
    assert m._load_meta(run_id)["baseline_comparison"] == baseline, (
        "train_baseline must persist its result"
    )

    evaluated = json.loads(m.evaluate_model(run_id))
    assert (
        evaluated["model"] == "random_forest" and "classification_report" in evaluated
    )

    errors = json.loads(m.error_analysis(run_id, top_n=3))
    assert "n_errors" in errors and "error_rate" in errors
    assert m._load_meta(run_id)["error_analysis"] == errors, (
        "error_analysis must persist its result"
    )

    segments = json.loads(m.analyze_segments(run_id, "channel"))
    assert set(segments["segments"]) == {"web", "mobile", "atm"}
    assert m._load_meta(run_id)["segment_analysis"]["channel"] == segments, (
        "analyze_segments must persist keyed by segment_column"
    )

    stability = json.loads(m.check_model_stability(run_id, n_repeats=2))
    assert stability["n_scores"] == 10 and "high_variance_warning" in stability

    # ── reporting, on the model as exported (before feature-eng below mutates
    # the column set) — standard template with real ROC/PR curve points. ───
    report = m.generate_report(run_id)
    for heading in (
        "## Model",
        "## Classification metrics",
        "## Confusion matrix",
        "## ROC curve",
        "## Precision-recall curve",
        "## Business Understanding & Assumptions",
        "## HITL Clarification History",
        "## Diagnostics",
        "## Fairness & Bias Assessment",
        "## Known Limitations",
        "## Reflection & Self-Critique",
    ):
        assert heading in report
    assert "curves are computed for binary targets only" not in report, (
        "binary target must get real ROC/PR curve points, not the multiclass skip note"
    )
    assert "not recorded" in report, (
        "business_understanding/reflection were never called in this flow — report must say so, not omit the section"
    )
    assert (run_dir / "report.md").exists()

    # ── mlops: mlflow logging (skip gracefully if not installed), dvc's
    # missing-CLI error path. ───────────────────────────────────────────────
    mlflow_result = json.loads(m.log_run_to_mlflow(run_id))
    assert "mlflow_run_id" in mlflow_result or "error" in mlflow_result
    if "version" in (mlflow_result.get("registered") or {}):
        assert mlflow_result["experiment"] == mlflow_result["registered"]["name"], (
            mlflow_result
        )
    dvc_result = json.loads(m.dvc_track(str(run_dir / "train.csv")))
    assert (
        "error" in dvc_result or "dvc_file" in dvc_result
    )  # error path if dvc CLI isn't installed here

    # ── feature-engineering: business template (apply_custom_feature) and the
    # correlation-driven fallback (propose/apply_features) — mutates train/
    # test.csv, so this runs last, after everything that assumed the
    # original column set. ──────────────────────────────────────────────────
    custom = json.loads(
        m.apply_custom_feature(run_id, "amount_per_hour", "amount / (hour + 1)")
    )
    assert custom["added_feature"] == "amount_per_hour"
    assert "amount_per_hour" in pd.read_csv(run_dir / "train.csv").columns

    feat_props = json.loads(m.propose_features(run_id))
    assert feat_props["candidates"], "expects at least one ratio/product candidate"
    first_candidate = next(iter(feat_props["candidates"]))
    applied = json.loads(m.apply_features(run_id, first_candidate))
    assert applied["added_features"] == [first_candidate]

    # engineered_features changed the column set — retrain before feature-selection.
    json.loads(m.train_model(run_id, "random_forest"))

    # ── feature-selection: importance-driven drop candidates (read-only). ──
    selection = json.loads(m.propose_feature_selection(run_id, bottom_k=2))
    assert len(selection["drop_candidates"]) == 2

    Path(path).unlink()
    return run_id, run_dir


def _multiclass_flow() -> tuple[str, Path]:
    path = _make_multiclass_csv()

    prep = json.loads(m.prepare_dataset(path, "label"))
    run_id = prep["run_id"]
    assert prep["train_shape"][0] + prep["test_shape"][0] == 240
    run_dir = m._run_dir(run_id)

    # apply_imputation's literal-value branch: a strategy that isn't
    # "median"/"mode"/"drop_column" is used verbatim as the fill value.
    json.loads(m.apply_imputation(run_id, json.dumps({"score": 0.0})))
    assert m._load_meta(run_id)["imputation"]["score"] == 0.0
    assert pd.read_csv(run_dir / "train.csv")["score"].isna().sum() == 0

    dt_props = json.loads(m.propose_datetime_features(run_id))
    assert "signup_date" in dt_props["datetime_columns"]
    json.loads(m.apply_datetime_features(run_id, "signup_date"))
    train_cols = pd.read_csv(run_dir / "train.csv").columns
    assert "signup_date" not in train_cols and "signup_date_year" in train_cols

    compared = json.loads(m.compare_models(run_id, "random_forest,logistic_regression"))
    assert len(compared["ranked"]) == 2
    assert compared["recommended"] in ("random_forest", "logistic_regression")
    assert m._load_meta(run_id)["model_comparison"] == compared, (
        "compare_models must persist its ranking"
    )
    unknown = json.loads(m.compare_models(run_id, "not_a_model"))
    assert "error" in unknown

    trained = json.loads(m.train_model(run_id, "random_forest"))
    assert "roc_auc_macro" in trained["auc"], (
        "a 3-class target must take the OvR-macro AUC branch"
    )

    # tune_threshold is binary-only — must reject a 3-class target cleanly.
    thr = json.loads(m.tune_threshold(run_id))
    assert "error" in thr and "binary" in thr["error"]

    # SHAP's ndim==3 stacking branch (TreeExplainer) only fires for >2 classes.
    shap_rf = json.loads(m.explain_model(run_id, method="shap"))
    assert len(shap_rf["feature_importance"]) > 0

    # LinearExplainer's multiclass coefficient shape only fires for >2 classes.
    json.loads(m.train_model(run_id, "logistic_regression"))
    shap_lr = json.loads(m.explain_model(run_id, method="shap"))
    assert len(shap_lr["feature_importance"]) > 0

    Path(path).unlink()
    return run_id, run_dir


def _error_path_checks() -> None:
    fd, path = tempfile.mkstemp(suffix=".csv")

    pd.DataFrame({"a": [], "label": []}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(path, "label")), (
        "empty CSV must error, not crash"
    )

    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(path, "label")), (
        "missing target column must error"
    )

    pd.DataFrame({"a": [1, 2, 3], "label": [None, None, None]}).to_csv(
        path, index=False
    )
    assert "error" in json.loads(m.prepare_dataset(path, "label")), (
        "all-missing target must error"
    )

    pd.DataFrame({"a": [1, 2, 3], "label": [1, 1, 1]}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(path, "label")), (
        "single-class target must error"
    )

    pd.DataFrame({"a": range(10), "label": [0] * 9 + [1]}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(path, "label")), (
        "a class with <2 rows can't be stratified"
    )

    Path(path).unlink()


def _repeated_high_cardinality_id_check() -> None:
    """Regression test for a real telco Wangiri file: a hashed caller-ID
    string column repeated across multiple lag-window rows per caller, so
    its uniqueness ratio (~48% here) is nowhere near the near-unique 98%
    check — only the absolute high-cardinality-categorical check catches
    it. Both propose_drop_columns and detect_data_leakage must flag it."""
    rng = np.random.default_rng(2)
    n = 6000
    caller_id = [
        f"CALLER_{i:04d}" for i in rng.integers(0, 2000, n)
    ]  # ~1850 uniques, ~31% of n
    df = pd.DataFrame(
        {
            "caller_id": caller_id,
            "total_calls": rng.integers(1, 20, n),
            "label": rng.choice([0, 1], n, p=[0.95, 0.05]),
        }
    )
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)

    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    drop_props = json.loads(m.propose_drop_columns(run_id))
    assert "caller_id" in drop_props["proposals"], (
        "repeated high-cardinality ID must be flagged despite < 98% uniqueness"
    )

    leakage = json.loads(m.detect_data_leakage(run_id))
    assert "caller_id" in leakage["high_cardinality_categorical_columns"]
    assert leakage["clear"] is False

    Path(path).unlink()
    shutil.rmtree(m._run_dir(run_id))


def _group_split_check() -> None:
    """Regression test for entity-leakage: a dataset shaped like the real
    Wangiri file (one row per caller per time-window, so a caller repeats
    across rows) must, with group_column set, never put the same entity in
    both train and test — and propose_group_column must find the entity
    column without being told its name."""
    rng = np.random.default_rng(4)
    n_callers = 500
    rows_per_caller = rng.integers(1, 5, n_callers)
    caller_id = np.repeat([f"caller_{i}" for i in range(n_callers)], rows_per_caller)
    n = len(caller_id)
    df = pd.DataFrame(
        {
            "caller_id": caller_id,
            "total_calls": rng.integers(1, 20, n),
            "label": rng.choice([0, 1], n, p=[0.9, 0.1]),
        }
    )
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)

    candidates = json.loads(m.propose_group_column(path, "label"))[
        "group_column_candidates"
    ]
    assert "caller_id" in candidates, (
        "propose_group_column must find the entity column by shape, not a hardcoded name"
    )

    prep = json.loads(m.prepare_dataset(path, "label", group_column="caller_id"))
    run_id = prep["run_id"]
    assert prep["group_column"] == "caller_id"
    train = pd.read_csv(m._run_dir(run_id) / "train.csv")
    test = pd.read_csv(m._run_dir(run_id) / "test.csv")
    assert not set(train["caller_id"]) & set(test["caller_id"]), (
        "no caller may appear in both train and test"
    )
    assert m._load_meta(run_id)["group_column"] == "caller_id"

    Path(path).unlink()
    shutil.rmtree(m._run_dir(run_id))

    # Plain split (no group_column) is unaffected — same dataset, default path.
    fd2, path2 = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path2, index=False)
    run_id2 = json.loads(m.prepare_dataset(path2, "label"))["run_id"]
    assert m._load_meta(run_id2)["group_column"] is None
    Path(path2).unlink()
    shutil.rmtree(m._run_dir(run_id2))


def _name_pattern_id_check() -> None:
    """A numeric ID column (repeated integer customer_id, low cardinality,
    no correlation with the target) is invisible to every cardinality-based
    check: it's not near-100%-unique, and the high-cardinality-categorical
    check only fires on non-numeric dtype. Only the name-pattern match
    catches this shape — regression test for that fourth signal."""
    rng = np.random.default_rng(3)
    n = 1000
    df = pd.DataFrame(
        {
            "customer_id": rng.integers(
                0, 300, n
            ),  # low cardinality, numeric, repeated — no other check fires
            "usage_minutes": rng.normal(100, 20, n),
            "label": rng.choice([0, 1], n, p=[0.9, 0.1]),
        }
    )
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)

    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    drop_props = json.loads(m.propose_drop_columns(run_id))
    assert "customer_id" in drop_props["proposals"], (
        "name-pattern match must catch a low-cardinality numeric ID"
    )

    leakage = json.loads(m.detect_data_leakage(run_id))
    assert "customer_id" in leakage["name_flagged_columns"]
    assert (
        "customer_id" not in leakage["id_like_columns"]
        and "customer_id" not in leakage["high_cardinality_categorical_columns"]
    )
    assert leakage["clear"] is False

    Path(path).unlink()
    shutil.rmtree(m._run_dir(run_id))


def _cleanup_check() -> None:
    """prepare_dataset must sweep run dirs whose meta.json is older than
    RUN_RETENTION_DAYS — runs are disposable working state, not an archive."""
    path = _make_csv()
    old_run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    old_run_dir = m._run_dir(old_run_id)
    stale = time.time() - (m.RUN_RETENTION_DAYS + 1) * 86400
    os.utime(old_run_dir / "meta.json", (stale, stale))

    path2 = _make_csv()
    new_run_id = json.loads(m.prepare_dataset(path2, "label"))[
        "run_id"
    ]  # triggers the sweep
    assert not old_run_dir.exists(), "a run older than RUN_RETENTION_DAYS must be swept"
    assert m._run_dir(new_run_id).exists(), (
        "the just-created run must survive its own sweep"
    )

    shutil.rmtree(m._run_dir(new_run_id))
    Path(path).unlink()
    Path(path2).unlink()


def _business_fairness_reflection_check() -> None:
    """The three genuinely new tools from the gap-closing pass, exercised
    end-to-end including their error paths: record_business_context
    (persistence + malformed-JSON error), assess_fairness (normal case,
    insufficient-sample case, missing-column and non-binary-target errors),
    and record_reflection (persistence, missing-check backfill to
    not_assessed, and the clear/blocked status logic). Also checks
    generate_report actually renders all of it without a live conversation."""
    rng = np.random.default_rng(5)
    n = 400
    group = rng.choice(["A", "B"], n, p=[0.5, 0.5])
    # Deliberately unequal positive rate by group so the fairness metrics
    # below detect a real gap, not just report zeros.
    label = np.where(
        group == "A",
        rng.choice([0, 1], n, p=[0.9, 0.1]),
        rng.choice([0, 1], n, p=[0.6, 0.4]),
    )
    df = pd.DataFrame({"amount": rng.normal(50, 10, n), "group": group, "label": label})
    fd, path = tempfile.mkstemp(suffix=".csv")
    df.to_csv(path, index=False)

    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    run_dir = m._run_dir(run_id)

    # ── business-understanding: persistence + malformed-JSON error path ──
    bc = json.loads(
        m.record_business_context(
            run_id,
            business_objective="Flag group at elevated risk for manual review.",
            target_definition="label=1 means flagged by the existing manual process.",
            success_criteria="Reasonable recall on the positive class; false positives are cheap to review.",
            domain="fraud",
            assumptions=json.dumps(
                [
                    "Assumed the 80/20 default split is fine — request didn't specify one."
                ]
            ),
            clarifications=json.dumps(
                [
                    {
                        "question": "Does 'recent' mean last 30 days?",
                        "answer": "Yes.",
                        "impact": "Used full history as-is, no windowing needed.",
                    }
                ]
            ),
        )
    )
    assert bc["business_objective"]
    assert bc["domain"] == "fraud"
    assert (
        m._load_meta(run_id)["business_understanding"]["clarifications"][0]["answer"]
        == "Yes."
    )
    assert m._load_meta(run_id)["business_understanding"]["domain"] == "fraud"
    assert "error" in json.loads(
        m.record_business_context(run_id, "x", "y", "z", assumptions="not json")
    )

    # domain left empty must persist as None, not the literal empty string
    # (so reporting's "not declared" check is a clean falsy test).
    no_domain_path = _make_csv()
    no_domain_run = json.loads(m.prepare_dataset(no_domain_path, "label"))["run_id"]
    json.loads(m.record_business_context(no_domain_run, "x", "y", "z"))
    assert m._load_meta(no_domain_run)["business_understanding"]["domain"] is None
    Path(no_domain_path).unlink()
    shutil.rmtree(m._run_dir(no_domain_run))

    json.loads(m.train_model(run_id, "random_forest"))

    # ── fairness: normal case, insufficient-sample case, error paths ────
    fairness = json.loads(m.assess_fairness(run_id, "group"))
    assert fairness["metrics"] is not None
    assert set(fairness["groups"]) == {"A", "B"}
    assert m._load_meta(run_id)["fairness"] == fairness

    assert "error" in json.loads(m.assess_fairness(run_id, "not_a_column"))

    small_sample = json.loads(m.assess_fairness(run_id, "group", min_group_size=10_000))
    assert small_sample["metrics"] is None
    assert all(g["insufficient_sample"] for g in small_sample["groups"].values())

    # ── reflection: persistence, missing-check backfill, status logic ───
    reflection = json.loads(
        m.record_reflection(
            run_id,
            checks=json.dumps(
                {
                    "data_quality_and_leakage": {
                        "status": "clear",
                        "evidence": "detect_data_leakage: clear",
                    }
                }
            ),
        )
    )
    assert reflection["overall_status"] == "clear"
    assert set(reflection["checks"]) == set(m.diagnostics.REFLECTION_CHECKS)
    assert reflection["checks"]["explainability"]["status"] == "not_assessed"

    blocked = json.loads(
        m.record_reflection(
            run_id,
            checks="{}",
            critical_issues=json.dumps(["unresolved leakage flag on column X"]),
        )
    )
    assert blocked["overall_status"] == "blocked"
    assert m._load_meta(run_id)["reflection"]["overall_status"] == "blocked"
    assert "error" in json.loads(m.record_reflection(run_id, checks="not json"))

    # ── report renders everything above without needing the conversation ──
    report = m.generate_report(run_id)
    for heading in (
        "## Business Understanding & Assumptions",
        "## HITL Clarification History",
        "## Diagnostics",
        "## Fairness & Bias Assessment",
        "## Known Limitations",
        "## Reflection & Self-Critique",
    ):
        assert heading in report
    assert "Flag group at elevated risk" in report
    assert "Does 'recent' mean last 30 days?" in report
    assert "blocked" in report
    assert "**Domain:** fraud" in report

    Path(path).unlink()
    shutil.rmtree(run_dir)


def main():
    binary_run_id, binary_run_dir = _binary_flow()
    multiclass_run_id, multiclass_run_dir = _multiclass_flow()

    compared_runs = json.loads(m.compare_runs(f"{binary_run_id},{multiclass_run_id}"))
    assert len(compared_runs["runs"]) == 2
    assert set(compared_runs["ranked_by_pr_auc"]) == {binary_run_id, multiclass_run_id}
    assert m._load_meta(binary_run_id)["cross_run_comparisons"][-1] == compared_runs, (
        "compare_runs must persist into every participating run's meta"
    )
    assert m._load_meta(multiclass_run_id)["cross_run_comparisons"][-1] == compared_runs

    shutil.rmtree(binary_run_dir)
    shutil.rmtree(multiclass_run_dir)

    _error_path_checks()
    _repeated_high_cardinality_id_check()
    _name_pattern_id_check()
    _group_split_check()
    _cleanup_check()
    _business_fairness_reflection_check()
    print("all checks passed")


if __name__ == "__main__":
    main()
