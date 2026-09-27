"""Integration checks for anomaly-agent's mcp_server/.

Proves the actual rigor claims: imputation/scaling are fit on the training
fold only, labels are never passed into the unsupervised detector's fit,
labeled and unlabeled flows both work, all 4 ALGORITHMS train and at least
one non-default algorithm round-trips through export+predict, every
propose/apply and quality-gate branch is exercised, and every tool's error
path actually returns an error instead of crashing.

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
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from sklearn.pipeline import Pipeline as SkPipeline

import mcp_server as m

ARTIFACTS_DIR = Path(__file__).parent / "test_artifacts"


class SpyIForest(IForest):
    """Records what fit() received — labels must never reach it."""

    def fit(self, X, y=None):
        self.fit_received_y = y
        return super().fit(X, y=y)


def _make_csv(path: Path, *, include_label: bool) -> str:
    rng = np.random.default_rng(7)
    normal_n = 380
    anomaly_n = 20
    total = normal_n + anomaly_n

    normal = pd.DataFrame(
        {
            "f1": rng.normal(0.0, 1.0, normal_n),
            "f2": rng.normal(0.5, 1.2, normal_n),
            "f3": rng.normal(-0.2, 0.8, normal_n),
            "channel": rng.choice(
                ["web", "mobile", "api"], normal_n, p=[0.5, 0.35, 0.15]
            ),
            "row_id": np.arange(normal_n),
        }
    )
    anomaly = pd.DataFrame(
        {
            "f1": rng.normal(7.5, 0.6, anomaly_n),
            "f2": rng.normal(8.0, 0.7, anomaly_n),
            "f3": rng.normal(6.8, 0.5, anomaly_n),
            "channel": ["api"] * anomaly_n,
            "row_id": np.arange(normal_n, total),
        }
    )
    df = pd.concat([normal, anomaly], ignore_index=True)
    df.loc[df.index[:24], "f2"] = np.nan
    if include_label:
        df["label"] = [0] * normal_n + [1] * anomaly_n
    df.to_csv(path, index=False)
    return str(path)


def _cleanup_run(run_id: str) -> None:
    run_dir = m._run_dir(run_id)
    if run_dir.exists():
        shutil.rmtree(run_dir)


def main():
    if ARTIFACTS_DIR.exists():
        shutil.rmtree(ARTIFACTS_DIR)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    labeled_path = ARTIFACTS_DIR / "labeled.csv"
    unlabeled_path = ARTIFACTS_DIR / "unlabeled.csv"
    predict_path = ARTIFACTS_DIR / "predict_input.csv"
    export_path = ARTIFACTS_DIR / "anomaly_model.pkl"
    run_ids_path = ARTIFACTS_DIR / "last_run_ids.json"
    run_ids = []

    _make_csv(labeled_path, include_label=True)
    _make_csv(unlabeled_path, include_label=False)

    eda = json.loads(m.eda(str(labeled_path)))
    assert eda["shape"] == [400, 6]

    prep = json.loads(m.prepare_dataset(str(labeled_path), label_column="label"))
    run_id = prep["run_id"]
    run_ids.append(run_id)
    assert prep["evaluation_available"] is True
    assert prep["train_shape"][0] + prep["test_shape"][0] == 400

    run_dir = m._run_dir(run_id)
    train_before = pd.read_csv(run_dir / "train.csv")
    expected_f2_median = float(train_before["f2"].median())

    impute_props = json.loads(m.propose_imputation(run_id))
    assert impute_props["proposals"]["f2"]["suggested_strategy"] == "median"

    dropped_props = json.loads(m.propose_drop_columns(run_id))
    assert dropped_props["proposals"]["row_id"] == "near-unique, likely an ID column"
    assert "f1" not in dropped_props["proposals"]

    imputed = json.loads(m.apply_imputation(run_id, json.dumps({"f2": "median"})))
    assert imputed["imputed_columns"] == ["f2"]
    meta_after_impute = m._load_meta(run_id)
    assert meta_after_impute["imputation"]["f2"] == expected_f2_median
    assert pd.read_csv(run_dir / "train.csv")["f2"].isna().sum() == 0
    assert pd.read_csv(run_dir / "test.csv")["f2"].isna().sum() == 0

    dropped = json.loads(m.apply_drop_columns(run_id, json.dumps(["row_id"])))
    assert dropped["dropped_columns"] == ["row_id"]
    assert "row_id" not in pd.read_csv(run_dir / "train.csv").columns

    contam = json.loads(m.propose_contamination(run_id))
    assert 0.03 <= contam["recommended_contamination"] <= 0.07

    original_iforest = m.ALGORITHMS["iforest"]

    m.ALGORITHMS["iforest"] = SpyIForest
    try:
        json.loads(
            m.detect_data_leakage(run_id)
        )  # the screen runs on the final feature set, before fitting
        trained_iforest = json.loads(
            m.train_model(
                run_id,
                algorithm="iforest",
                contamination=contam["recommended_contamination"],
            )
        )
    finally:
        m.ALGORITHMS["iforest"] = original_iforest

    assert trained_iforest["roc_auc"] >= 0.95
    assert trained_iforest["precision"] >= 0.8
    assert trained_iforest["recall"] >= 0.75

    pipeline = joblib.load(run_dir / "pipeline.pkl")
    detector = pipeline.named_steps["detector"]
    assert type(detector).__name__ == "SpyIForest" and hasattr(detector, "fit_received_y"), "spy not used"
    assert detector.fit_received_y is None, "labels must not be passed into detector.fit()"

    train_after = pd.read_csv(run_dir / "train.csv")
    test_after = pd.read_csv(run_dir / "test.csv")
    X_train = m._feature_frame(train_after, meta_after_impute)
    X_all = m._feature_frame(
        pd.concat([train_after, test_after], ignore_index=True), meta_after_impute
    )
    # Every scaler in the pipeline (plain and logged numerics) is fitted on
    # the fit rows only — known-normal training rows, the default with labels.
    def upstream(X):  # drop / impute: stateless column steps before the encoder
        for name, step in pipeline.steps[:-2]:
            X = step.transform(X)
        return X

    fit_rows = X_train[m._label_to_binary(train_after["label"], meta_after_impute).to_numpy() == 0]
    prepared_fit, prepared_all = upstream(fit_rows), upstream(X_all)
    scalers_checked = 0
    for name, transformer, cols in pipeline.named_steps["encode"].transformers_:
        if name not in ("num", "log"):
            continue
        before_scale, scaler = transformer[:-1], transformer.named_steps["scale"]
        assert np.allclose(scaler.mean_, np.asarray(before_scale.transform(prepared_fit[cols])).mean(axis=0))
        assert not np.allclose(scaler.mean_, np.asarray(before_scale.transform(prepared_all[cols])).mean(axis=0))
        scalers_checked += 1
    assert scalers_checked >= 1

    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    trained_ecod = json.loads(
        m.train_model(
            run_id, algorithm="ecod", contamination=contam["recommended_contamination"]
        )
    )
    assert trained_ecod["roc_auc"] >= 0.95

    # ── All 4 ALGORITHMS must train, not just iforest/ecod. ────────────────
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    trained_knn = json.loads(
        m.train_model(
            run_id, algorithm="knn", contamination=contam["recommended_contamination"]
        )
    )
    assert trained_knn["roc_auc"] >= 0.65

    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    trained_ocsvm = json.loads(
        m.train_model(
            run_id, algorithm="ocsvm", contamination=contam["recommended_contamination"]
        )
    )
    assert trained_ocsvm["roc_auc"] is not None and trained_ocsvm["roc_auc"] >= 0.7

    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    unknown_algo = json.loads(m.train_model(run_id, algorithm="not_an_algo"))
    assert "error" in unknown_algo and "not_an_algo" in unknown_algo["error"]

    # ── Non-default algorithm (knn) round-trips through export+predict too
    # — the bundle records whichever algorithm actually trained last. ──────
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    json.loads(
        m.train_model(
            run_id, algorithm="knn", contamination=contam["recommended_contamination"]
        )
    )
    knn_export_path = ARTIFACTS_DIR / "knn_model.pkl"
    knn_export = json.loads(m.export_model(run_id, out_path=str(knn_export_path)))
    assert knn_export["algorithm"] == "knn"
    knn_bundle = joblib.load(knn_export_path)
    assert knn_bundle["algorithm"] == "knn"
    knn_predict_path = ARTIFACTS_DIR / "knn_predict_input.csv"
    pd.DataFrame(
        {
            "f1": [0.1, 8.1, -0.3],
            "f2": [0.2, 7.9, 0.4],
            "f3": [0.0, 7.2, -0.5],
            "channel": ["web", "api", "mobile"],
        }
    ).to_csv(knn_predict_path, index=False)
    knn_preds = json.loads(m.predict(str(knn_export_path), str(knn_predict_path)))
    assert knn_preds["algorithm"] == "knn"
    assert knn_preds["n_rows"] == 3

    # retrain ecod so the rest of this flow (compare_runs/explain/export below)
    # sees the same current model the pre-existing assertions expect.
    json.loads(
        m.detect_data_leakage(run_id)
    )  # the screen runs on the final feature set, before fitting
    trained_ecod = json.loads(
        m.train_model(
            run_id, algorithm="ecod", contamination=contam["recommended_contamination"]
        )
    )
    assert trained_ecod["roc_auc"] >= 0.95

    compared = json.loads(m.compare_runs(run_id))
    assert compared["ranked"] == [run_id]

    explained = json.loads(m.explain_model(run_id))
    assert explained["method"] == "score_correlation"
    assert len(explained["feature_importance"]) > 0

    # ── export_model's quality gate — force it by raising the bar above
    # what even this well-separated dataset can score. ─────────────────────
    orig_export_min = m.modeling.EXPORT_MIN_ROC_AUC
    m.modeling.EXPORT_MIN_ROC_AUC = 1.1
    try:
        gated = json.loads(m.export_model(run_id, str(ARTIFACTS_DIR / "gated.pkl")))
        assert "error" in gated and "roc_auc" in gated["error"]
        forced = json.loads(
            m.export_model(run_id, str(ARTIFACTS_DIR / "gated_forced.pkl"), force=True)
        )
        assert forced["out_path"]
    finally:
        m.modeling.EXPORT_MIN_ROC_AUC = orig_export_min

    # export_model with no out_path must suggest one instead of writing.
    suggestion = json.loads(m.export_model(run_id))
    assert "suggested_out_path" in suggestion

    export_info = json.loads(m.export_model(run_id, out_path=str(export_path)))
    assert export_info["out_path"] == str(export_path)
    bundle = joblib.load(export_path)
    assert bundle["algorithm"] == "ecod"

    pd.DataFrame(
        {
            "f1": [0.1, 8.1, -0.3],
            "f2": [0.2, np.nan, 0.4],
            "f3": [0.0, 7.2, -0.5],
            "channel": ["web", "api", "mobile"],
            "row_id": [9001, 9002, 9003],
            "label": [0, 1, 0],
        }
    ).to_csv(predict_path, index=False)
    preds = json.loads(m.predict(str(export_path), str(predict_path)))
    assert preds["n_rows"] == 3
    pred_df = pd.read_csv(preds["out_path"])
    assert ["anomaly", "anomaly_score", "anomaly_reasons"] == list(pred_df.columns[-3:])
    assert pred_df.loc[1, "anomaly"] == 1
    assert "robust sd" in pred_df.loc[1, "anomaly_reasons"], pred_df.loc[1, "anomaly_reasons"]

    missing_pkl = json.loads(
        m.predict(str(ARTIFACTS_DIR / "does_not_exist.pkl"), str(predict_path))
    )
    assert "error" in missing_pkl and "no exported model" in missing_pkl["error"]

    missing_data = json.loads(
        m.predict(str(export_path), str(ARTIFACTS_DIR / "does_not_exist.csv"))
    )
    assert "error" in missing_data and "no data file" in missing_data["error"]

    unsup_prep = json.loads(m.prepare_dataset(str(unlabeled_path), label_column=""))
    unsup_run_id = unsup_prep["run_id"]
    run_ids.append(unsup_run_id)
    assert unsup_prep["evaluation_available"] is False

    # "no fitted model" error branches — before train_model has run for this run_id.
    no_model_explain = json.loads(m.explain_model(unsup_run_id))
    assert (
        "error" in no_model_explain and "no fitted model" in no_model_explain["error"]
    )
    no_model_export = json.loads(m.export_model(unsup_run_id))
    assert "error" in no_model_export and "no fitted model" in no_model_export["error"]

    unsup_impute = json.loads(m.propose_imputation(unsup_run_id))
    assert unsup_impute["proposals"]["f2"]["suggested_strategy"] == "median"
    json.loads(m.apply_imputation(unsup_run_id, json.dumps({"f2": "median"})))
    json.loads(m.apply_drop_columns(unsup_run_id, json.dumps(["row_id"])))
    unsup_contam = json.loads(m.propose_contamination(unsup_run_id))
    assert unsup_contam["recommended_contamination"] == 0.05
    json.loads(
        m.detect_data_leakage(unsup_run_id)
    )  # the screen runs on the final feature set, before fitting
    unsup_train = json.loads(
        m.train_model(unsup_run_id, algorithm="iforest", contamination=0.05)
    )
    assert "roc_auc" not in unsup_train
    assert unsup_train["anomaly_rate_match"] >= 0.9

    run_ids_path.write_text(json.dumps(run_ids))
    for rid in run_ids:
        _cleanup_run(rid)
    for path in [
        labeled_path,
        unlabeled_path,
        predict_path,
        export_path,
        Path(preds["out_path"]),
    ]:
        if path.exists():
            path.unlink()

    _imputation_edge_cases()
    _compare_runs_checks()
    _prepare_dataset_error_checks()
    _cleanup_check()
    print("all checks passed")


def _imputation_edge_cases() -> None:
    """apply_imputation's mode / literal-value / drop_column branches — the
    main flow above only exercises "median"."""
    path = ARTIFACTS_DIR / "imputation_edge.csv"
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0],
            "cat": ["a", "a", "b", "a", None, "a", "b", "a", "a", "b"],
            "junk": [1.0] * 10,
        }
    )
    df.to_csv(path, index=False)
    prep = json.loads(m.prepare_dataset(str(path), label_column=""))
    run_id = prep["run_id"]
    run_dir = m._run_dir(run_id)
    train_before = pd.read_csv(run_dir / "train.csv")
    expected_mode = m._to_native(train_before["cat"].mode(dropna=True).iloc[0])

    result = json.loads(
        m.apply_imputation(
            run_id, json.dumps({"cat": "mode", "num": -1.0, "junk": "drop_column"})
        )
    )
    assert set(result["imputed_columns"]) == {"cat", "num"}
    assert result["dropped_columns"] == ["junk"]

    meta = m._load_meta(run_id)
    assert meta["imputation"]["cat"] == expected_mode
    assert meta["imputation"]["num"] == -1.0, (
        "a non-strategy value must be used as a literal fill"
    )

    train_after = pd.read_csv(run_dir / "train.csv")
    assert "junk" not in train_after.columns
    assert train_after["cat"].isna().sum() == 0
    assert train_after["num"].isna().sum() == 0

    shutil.rmtree(run_dir)
    path.unlink()


def _compare_runs_checks() -> None:
    """Multi-run ranking (roc_auc desc for labeled runs, anomaly_rate_gap
    asc for unlabeled ones) and the unknown-run_id error row — the main
    flow above only ever compares a single run_id against itself."""
    rng = np.random.default_rng(3)

    def _labeled_df(separation: float) -> pd.DataFrame:
        n_normal, n_anom = 190, 10
        normal = pd.DataFrame(
            {"x": rng.normal(0, 1, n_normal), "y": rng.normal(0, 1, n_normal)}
        )
        anomaly = pd.DataFrame(
            {
                "x": rng.normal(separation, 1, n_anom),
                "y": rng.normal(separation, 1, n_anom),
            }
        )
        df = pd.concat([normal, anomaly], ignore_index=True)
        df["label"] = [0] * n_normal + [1] * n_anom
        return df

    strong_path = ARTIFACTS_DIR / "compare_strong.csv"
    weak_path = ARTIFACTS_DIR / "compare_weak.csv"
    _labeled_df(8.0).to_csv(strong_path, index=False)
    _labeled_df(0.3).to_csv(weak_path, index=False)

    strong_run = json.loads(m.prepare_dataset(str(strong_path), label_column="label"))[
        "run_id"
    ]
    weak_run = json.loads(m.prepare_dataset(str(weak_path), label_column="label"))[
        "run_id"
    ]
    json.loads(
        m.detect_data_leakage(strong_run)
    )  # the screen runs on the final feature set, before fitting
    json.loads(m.train_model(strong_run, algorithm="iforest", contamination=0.05))
    json.loads(
        m.detect_data_leakage(weak_run)
    )  # the screen runs on the final feature set, before fitting
    json.loads(m.train_model(weak_run, algorithm="iforest", contamination=0.05))

    ranked_labeled = json.loads(
        m.compare_runs(f"{strong_run},{weak_run},not-a-real-run")
    )
    assert ranked_labeled["ranked"][0] == strong_run, (
        "the well-separated run must rank first by roc_auc"
    )
    assert ranked_labeled["ranked"][1] == weak_run
    error_rows = [r for r in ranked_labeled["runs"] if r["run_id"] == "not-a-real-run"]
    assert len(error_rows) == 1 and "error" in error_rows[0]

    # Unlabeled ranking sorts by anomaly_rate_gap ascending. IForest's own
    # threshold tracks its requested contamination almost by construction,
    # so set the persisted gap directly — this is testing compare_runs'
    # sort, not train_model's gap computation (real_train's anomaly_rate_match
    # assertion above already covers that).
    unlabeled_path = ARTIFACTS_DIR / "compare_unlabeled.csv"
    _labeled_df(8.0).drop(columns=["label"]).to_csv(unlabeled_path, index=False)
    close_run = json.loads(m.prepare_dataset(str(unlabeled_path), label_column=""))[
        "run_id"
    ]
    far_run = json.loads(m.prepare_dataset(str(unlabeled_path), label_column=""))[
        "run_id"
    ]
    json.loads(
        m.detect_data_leakage(close_run)
    )  # the screen runs on the final feature set, before fitting
    json.loads(m.train_model(close_run, algorithm="iforest", contamination=0.05))
    json.loads(
        m.detect_data_leakage(far_run)
    )  # the screen runs on the final feature set, before fitting
    json.loads(m.train_model(far_run, algorithm="iforest", contamination=0.05))

    meta_close = m._load_meta(close_run)
    meta_close["metrics"]["anomaly_rate_gap"] = 0.01
    m._save_meta(close_run, meta_close)
    meta_far = m._load_meta(far_run)
    meta_far["metrics"]["anomaly_rate_gap"] = 0.30
    m._save_meta(far_run, meta_far)

    ranked_unlabeled = json.loads(m.compare_runs(f"{far_run},{close_run}"))
    assert ranked_unlabeled["ranked"] == [
        close_run,
        far_run,
    ], "smaller anomaly_rate_gap must rank first"

    for run_id in (strong_run, weak_run, close_run, far_run):
        shutil.rmtree(m._run_dir(run_id))
    for path in (strong_path, weak_path, unlabeled_path):
        path.unlink()


def _prepare_dataset_error_checks() -> None:
    path = ARTIFACTS_DIR / "prepare_dataset_errors.csv"

    pd.DataFrame({"a": [], "label": []}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(str(path), label_column="label")), (
        "empty CSV must error, not crash"
    )

    pd.DataFrame({"a": range(10), "label": [0] * 5 + [1] * 5}).to_csv(path, index=False)
    assert "error" in json.loads(
        m.prepare_dataset(str(path), label_column="label", test_size=1.5)
    ), "test_size > 1 must error"
    assert "error" in json.loads(
        m.prepare_dataset(str(path), label_column="label", test_size=0)
    ), "test_size = 0 must error"

    assert "error" in json.loads(
        m.prepare_dataset(str(path), label_column="not_a_column")
    ), "missing label column must error"

    pd.DataFrame({"a": range(10), "label": [None] * 10}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(str(path), label_column="label")), (
        "all-missing label column must error"
    )

    pd.DataFrame({"a": range(10), "label": [0, 1, None, 0, 1, 0, 1, 0, 1, 0]}).to_csv(
        path, index=False
    )
    assert "error" in json.loads(m.prepare_dataset(str(path), label_column="label")), (
        "partial-missing label column must error"
    )

    pd.DataFrame({"a": range(9), "label": [0, 1, 2] * 3}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(str(path), label_column="label")), (
        "non-binary label column must error"
    )

    pd.DataFrame({"a": range(10), "label": [0] * 9 + [1]}).to_csv(path, index=False)
    assert "error" in json.loads(m.prepare_dataset(str(path), label_column="label")), (
        "a minority class with <2 rows can't be stratified"
    )

    path.unlink()


def _cleanup_check() -> None:
    """prepare_dataset must sweep run dirs whose meta.json is older than
    RUN_RETENTION_DAYS — runs are disposable working state, not an archive."""
    path = ARTIFACTS_DIR / "cleanup_check.csv"
    _make_csv(path, include_label=False)
    old_run_id = json.loads(m.prepare_dataset(str(path), label_column=""))["run_id"]
    old_run_dir = m._run_dir(old_run_id)
    stale = time.time() - (m.RUN_RETENTION_DAYS + 1) * 86400
    os.utime(old_run_dir / "meta.json", (stale, stale))

    new_run_id = json.loads(m.prepare_dataset(str(path), label_column=""))[
        "run_id"
    ]  # triggers the sweep
    assert not old_run_dir.exists(), "a run older than RUN_RETENTION_DAYS must be swept"
    assert m._run_dir(new_run_id).exists(), (
        "the just-created run must survive its own sweep"
    )

    shutil.rmtree(m._run_dir(new_run_id))
    path.unlink()


if __name__ == "__main__":
    main()
