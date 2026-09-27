"""Train, tune, explain, threshold, compare, export, predict — everything
that reads or writes this run's CURRENT fitted pipeline.pkl. train_model
creates it; tune_hyperparams overwrites it in place; every other tool here
reads back whichever one is current with no hyperparameters threaded
through by hand. See the `modeling` skill."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from core import runtime, toolguard
from core.datasource import read_table
import shap
from run_persistence import save_training_run
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.inspection import permutation_importance

from . import core
from .core import (
    CV_SAMPLE_MAX_ROWS,
    MODELS,
    PARAM_GRIDS,
    RUNS_DIR,
    _bootstrap_cis,
    _build_pipeline,
    _cv_sample,
    _cv_scoring,
    _evaluate,
    _load_meta,
    _load_split,
    _overfit_gate,
    _run_dir,
    _save_meta,
    _invalidate_model_derived,
    _to_native,
    _validation_folds,
    _validation_proba,
    positive_index,
    positive_label,
    mcp,
)
from .gates import compute_readiness

optuna.logging.set_verbosity(
    optuna.logging.WARNING
)  # else every trial logs a line to stderr


@mcp.tool()
def compare_models(run_id: str, models: str = "") -> str:
    """Cross-validates candidate models (default: all of MODELS —
    logistic_regression, random_forest, xgboost) on this run's training fold
    with the SAME pipeline shape train_model uses, and ranks them by 5-fold
    CV that respects the run's split (forward-chained on a temporal run,
    grouped on an entity run, stratified otherwise — see cv_scheme) — see relative fit before
    committing to one. Doesn't touch pipeline.pkl or this run's current
    model — call train_model(run_id, the winner) yourself to make it
    current — but the ranking IS persisted to meta["model_comparison"], so
    generate_report can show which model was picked over which alternatives
    and why (the CV ranking), not just the one that got trained. models:
    comma-separated subset of {list(MODELS)}, empty = all. Above
    CV_SAMPLE_MAX_ROWS training rows, the CV search runs on a stratified
    sample (never the final model, since compare_models doesn't fit one) —
    see cv_computed_on_n_rows in the result."""
    train, _, meta = _load_split(run_id)
    target = meta["target"]
    X_train, y_train = train.drop(columns=[target]), train[target]
    candidates = [m.strip() for m in models.split(",") if m.strip()] or list(MODELS)
    unknown = [m for m in candidates if m not in MODELS]
    if unknown:
        return json.dumps(
            {"error": f"unknown model(s) {unknown}, use any of {list(MODELS)}"}
        )

    X_cv, y_cv, cv_n_rows = _cv_sample(X_train, y_train, meta)
    scoring = _cv_scoring(y_train)
    cv, cv_scheme = _validation_folds(X_cv, y_cv, meta)
    ranked = []
    for model in candidates:
        pipeline = _build_pipeline(meta, model, X_cv, y_cv)
        scores = cross_val_score(pipeline, X_cv, y_cv, cv=cv, scoring=scoring)
        ranked.append(
            {
                "model": model,
                "cv_metric": scoring,
                "cv_score_mean": round(float(scores.mean()), 4),
                "cv_score_std": round(float(scores.std()), 4),
            }
        )
    ranked.sort(key=lambda r: -r["cv_score_mean"])
    result = {
        "run_id": run_id,
        "cv_metric": scoring,
        "ranked": ranked,
        "recommended": ranked[0]["model"],
        "cv_computed_on_n_rows": cv_n_rows,
        "cv_scheme": cv_scheme,
    }
    meta["model_comparison"] = result
    _save_meta(run_id, meta)
    return json.dumps(result)


@mcp.tool()
def train_model(run_id: str, model: str = "random_forest") -> str:
    """Fits model as one Pipeline (drop -> impute -> one-hot encode -> [SMOTE
    if enabled] -> classifier) on the run's FULL training fold — the CV score
    used to detect overfitting is computed on a stratified sample above
    CV_SAMPLE_MAX_ROWS rows (see cv_computed_on_n_rows in the result), but
    the saved model always sees every training row. Reports 5-fold CV that
    respects the run's split (mean±std, and `cv_scheme` — forward-chained,
    grouped or stratified: a CV that ignored the split would let an entity
    or the future into its own validation rows) and held-out
    test metrics (precision/recall/F1 + ROC-AUC/PR-AUC). Saves the fitted
    pipeline as this run's current model — tune_hyperparams, explain_model,
    export_model, and the visualization agent's eval plots all read it back
    from here."""
    if model not in MODELS:
        return json.dumps(
            {"error": f"unknown model '{model}', use one of {list(MODELS)}"}
        )
    train, test, meta = _load_split(run_id)
    target = meta["target"]
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    X_cv, y_cv, cv_n_rows = _cv_sample(X_train, y_train, meta)
    cv, cv_scheme = _validation_folds(X_cv, y_cv, meta)
    scoring = _cv_scoring(y_train)
    cv_scores = cross_val_score(
        _build_pipeline(meta, model, X_cv, y_cv), X_cv, y_cv, cv=cv, scoring=scoring
    )

    pipeline = _build_pipeline(meta, model, X_train, y_train)
    pipeline.fit(X_train, y_train)
    test_eval = _evaluate(pipeline, X_test, y_test, meta)
    cv_mean = round(float(cv_scores.mean()), 4)
    cv_to_test_gap, overfitting_warning = _overfit_gate(cv_mean, test_eval, scoring)

    artifact_path = str(_run_dir(run_id) / "pipeline.pkl")
    joblib.dump(pipeline, artifact_path)
    meta["model"] = model
    meta["best_params"] = None
    meta["tuning_trials"] = None
    meta["baseline_metrics"] = {
        "cv_metric": scoring,
        "cv_score_mean": cv_mean,
        "cv_score_std": round(float(cv_scores.std()), 4),
        "cv_computed_on_n_rows": cv_n_rows,
        "cv_scheme": cv_scheme,
        "cv_to_test_gap": cv_to_test_gap,
        "overfitting_warning": overfitting_warning,
        **test_eval,
    }
    meta["tuned_metrics"] = None
    meta["pipeline_version"] = meta.get("pipeline_version", 0) + 1
    stale = _invalidate_model_derived(meta)
    _save_meta(run_id, meta)
    save_training_run(
        run_id,
        target_column=target,
        best_model=model,
        metric=scoring,
        best_score=cv_mean,
        test_auc=(test_eval.get("auc") or {}).get(
            "roc_auc", (test_eval.get("auc") or {}).get("roc_auc_macro")
        ),
        cv_to_test_gap=cv_to_test_gap,
        overfitting_warning=overfitting_warning,
        best_params=None,
        artifact_path=artifact_path,
    )
    return json.dumps(
        {
            "run_id": run_id,
            "model": model,
            **meta["baseline_metrics"],
            **({"invalidated_by_refit": stale} if stale else {}),
        }
    )


@mcp.tool()
def tune_hyperparams(
    run_id: str, model: str = "random_forest", n_trials: int = 8, cv_folds: int = 5
) -> str:
    """Optuna TPE search over the SAME pipeline shape as train_model — TPE
    picks each next candidate from the trials so far instead of sampling
    blind. Defaults (5-fold, 8 trials) are a fast first pass, not an
    exhaustive search; raise n_trials/cv_folds when the metric actually
    matters and you can afford the extra compute (each trial refits
    cv_folds models). Refits the best candidate on the full training fold
    and OVERWRITES this run's pipeline — every tool that reads "the current
    model" for this run_id (explain_model, export_model, the visualization
    agent's plots) picks up the tuned version automatically. Nothing needs
    best_params handed to it manually."""
    if model not in MODELS:
        return json.dumps(
            {"error": f"unknown model '{model}', use one of {list(MODELS)}"}
        )
    train, test, meta = _load_split(run_id)
    target = meta["target"]
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    # The search itself runs on a capped sample (each of n_trials * cv_folds
    # fits would otherwise repeat train_model's per-fit cost n_trials times
    # over) — only the final refit below touches every training row.
    X_cv, y_cv, cv_n_rows = _cv_sample(X_train, y_train, meta)
    scoring = _cv_scoring(y_train)
    cv, cv_scheme = _validation_folds(X_cv, y_cv, meta, n_splits=cv_folds)

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: trial.suggest_categorical(name, choices)
            for name, choices in PARAM_GRIDS[model].items()
        }
        pipeline = _build_pipeline(meta, model, X_cv, y_cv)
        pipeline.set_params(**params)
        return cross_val_score(pipeline, X_cv, y_cv, cv=cv, scoring=scoring).mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)

    best = _build_pipeline(meta, model, X_train, y_train)
    best.set_params(**study.best_params)
    best.fit(X_train, y_train)
    test_eval = _evaluate(best, X_test, y_test, meta)
    cv_best = round(float(study.best_value), 4)
    cv_to_test_gap, overfitting_warning = _overfit_gate(cv_best, test_eval, scoring)

    artifact_path = str(_run_dir(run_id) / "pipeline.pkl")
    joblib.dump(best, artifact_path)
    meta["model"] = model
    meta["best_params"] = study.best_params
    # Every trial, not just the winner — what was searched is part of the
    # record (log_run_to_mlflow ships it as tuning_trials.json).
    meta["tuning_trials"] = [
        {"trial": t.number, "cv_score": t.value, "params": t.params}
        for t in study.trials
    ]
    meta["pipeline_version"] = meta.get("pipeline_version", 0) + 1
    meta["tuned_metrics"] = {
        "cv_metric": scoring,
        "cv_score_best": cv_best,
        "cv_computed_on_n_rows": cv_n_rows,
        "cv_scheme": cv_scheme,
        "cv_to_test_gap": cv_to_test_gap,
        "overfitting_warning": overfitting_warning,
        **test_eval,
    }
    stale = _invalidate_model_derived(meta)
    _save_meta(run_id, meta)
    save_training_run(
        run_id,
        target_column=target,
        best_model=model,
        metric=scoring,
        best_score=cv_best,
        test_auc=(test_eval.get("auc") or {}).get(
            "roc_auc", (test_eval.get("auc") or {}).get("roc_auc_macro")
        ),
        cv_to_test_gap=cv_to_test_gap,
        overfitting_warning=overfitting_warning,
        best_params=study.best_params,
        artifact_path=artifact_path,
    )

    baseline_auc = ((meta.get("baseline_metrics") or {}).get("auc") or {}).get(
        "roc_auc"
    )
    tuned_auc = (meta["tuned_metrics"].get("auc") or {}).get("roc_auc")
    improvement = (
        round(tuned_auc - baseline_auc, 4)
        if baseline_auc is not None and tuned_auc is not None
        else None
    )
    return json.dumps(
        {
            "run_id": run_id,
            "model": model,
            "best_params": study.best_params,
            "roc_auc_improvement_vs_baseline": improvement,
            **meta["tuned_metrics"],
            **({"invalidated_by_refit": stale} if stale else {}),
        }
    )


@mcp.tool()
def calibrate_model(run_id: str, method: str = "isotonic", cv_folds: int = 5) -> str:
    """Wraps this run's CURRENT model in a calibrator fitted by cross-
    validation on the training fold, and makes the calibrated version the
    run's model. Use when check_calibration reports a miscalibration
    warning.

    method: "isotonic" (non-parametric, flexible, needs a few thousand rows
    to avoid overfitting the calibration curve itself) or "sigmoid"
    (Platt scaling — a two-parameter fit, the right choice on small data or
    when the distortion is a simple systematic squash). Defaults to
    isotonic; the result reports which was used and warns if the training
    fold is small enough that sigmoid was the better call.

    What this does and does not change: calibration is a monotonic
    remapping of the scores, so ROC-AUC (pure ranking) is essentially
    unchanged, while Brier/ECE improve and every probability-dependent
    decision — thresholds, expected-cost arithmetic, "we alert on anything
    above 0.9" — starts meaning what it says. PR-AUC can move slightly
    because isotonic regression's ties reorder equal-score rows.

    Refits the underlying pipeline inside the calibrator, so the whole
    thing stays a single fitted object that predicts from raw rows. This
    OVERWRITES pipeline.pkl and bumps pipeline_version, which invalidates
    every model-derived result (SHAP, fairness, the operating point) — so
    re-run explain_model/assess_fairness/tune_threshold afterwards. That
    ordering is enforced by the readiness gate, not left to memory:
    calibrating after tuning a threshold and keeping the old threshold is
    exactly the "evidence describes a model that no longer exists" failure
    check_readiness fails a run for."""
    if method not in ("isotonic", "sigmoid"):
        return json.dumps(
            {"error": f"method must be 'isotonic' or 'sigmoid', got '{method}'"}
        )
    train, test, meta = _load_split(run_id)
    model = meta.get("model")
    if not model:
        return json.dumps(
            {"error": "no trained model for this run yet — call train_model first"}
        )
    target = meta["target"]
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]
    if y_train.nunique() != 2:
        return json.dumps({"error": "calibrate_model supports binary targets only"})

    before = meta.get("calibration") or {}
    base = _build_pipeline(meta, model, X_train, y_train)
    if meta.get("best_params"):
        base.set_params(**meta["best_params"])

    # cv=cv_folds refits the base pipeline per fold and fits the calibrator on
    # each held-out part — the calibration map never sees rows the model it is
    # calibrating was fitted on, which is the whole point. Test fold untouched.
    calibrated = CalibratedClassifierCV(base, method=method, cv=cv_folds)
    calibrated.fit(X_train, y_train)
    test_eval = _evaluate(calibrated, X_test, y_test, meta)

    artifact_path = str(_run_dir(run_id) / "pipeline.pkl")
    joblib.dump(calibrated, artifact_path)
    meta["calibration_method"] = method
    meta["pipeline_version"] = meta.get("pipeline_version", 0) + 1
    metrics_key = "tuned_metrics" if meta.get("tuned_metrics") else "baseline_metrics"
    previous = dict(meta.get(metrics_key) or {})
    meta[metrics_key] = {**previous, **test_eval, "calibrated_with": method}
    stale = _invalidate_model_derived(meta)
    _save_meta(run_id, meta)

    small_fold_note = (
        "training fold is under 1000 rows — isotonic regression can overfit the calibration curve at this size; "
        "consider method='sigmoid' and compare ECE"
        if method == "isotonic" and len(X_train) < 1000
        else None
    )
    return json.dumps(
        {
            "run_id": run_id,
            "model": model,
            "calibration_method": method,
            "cv_folds": cv_folds,
            "ece_before": before.get("expected_calibration_error"),
            "brier_before": before.get("brier_score"),
            "next_step": "call check_calibration again to measure the new ECE/Brier, then re-run explain_model / "
            "assess_fairness / tune_threshold — this refit invalidated them",
            **test_eval,
            **({"invalidated_by_refit": stale} if stale else {}),
            **({"warning": small_fold_note} if small_fold_note else {}),
        }
    )


@mcp.tool()
def explain_model(
    run_id: str, method: str = "permutation", background_size: int = 200
) -> str:
    """Explains this run's CURRENT model (tuned if tune_hyperparams ran,
    baseline otherwise — always whatever train_model/tune_hyperparams last
    wrote). method: "permutation" (default — cheap, model-agnostic, computed
    on the held-out test fold so it reflects generalization, not
    memorization) or "shap" (exact TreeExplainer for random_forest/xgboost,
    LinearExplainer for logistic_regression — use for regulated domains that
    need per-prediction-shaped reasoning, not just a global ranking).
    background_size caps how many test rows SHAP runs on (stratified by
    target, so a rare positive class like fraud stays represented instead
    of being averaged away) — this is both the LinearExplainer background
    distribution and the rows explained for the global ranking; raise it
    for a more exact global average, lower it when the test fold is huge
    and a representative sample is enough. Ignored above the test fold size
    (no sampling needed if there's nothing to cut down)."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    pipeline = joblib.load(pipeline_path)
    _, test, meta = _load_split(run_id)
    X_test, y_test = test.drop(columns=[meta["target"]]), test[meta["target"]]

    if method == "shap":
        pos_label = _to_native(positive_label(meta, y_test))
        # Auto-detect background_size when the caller left it at the default (200).
        # At very low positive rates the default yields < 1 expected positive row in
        # the background — SHAP's reference distribution becomes almost entirely the
        # majority class, which distorts every importance value.
        # Strategy: ensure at least MIN_POSITIVE_BACKGROUND positive rows are present.
        # We compute the minimum sample size that achieves that, then clamp to the
        # full test fold (no sampling needed if the fold is already small enough).
        MIN_POSITIVE_BACKGROUND = 10  # floor: below this SHAP values are unreliable
        positive_rate = float((y_test == pos_label).mean())
        if positive_rate > 0:
            min_size_for_floor = int(np.ceil(MIN_POSITIVE_BACKGROUND / positive_rate))
        else:
            min_size_for_floor = len(X_test)  # degenerate — use everything
        # Only override the caller's value when it is the unchanged default AND the
        # auto-detected size is larger; never silently shrink a deliberately large
        # background_size the caller passed in.
        if background_size == 200 and min_size_for_floor > background_size:
            background_size = min(min_size_for_floor, len(X_test))

        if 0 < background_size < len(X_test):
            X_test, _, y_test, _ = train_test_split(
                X_test,
                y_test,
                train_size=background_size,
                stratify=y_test,
                random_state=42,
            )
        from sklearn.pipeline import Pipeline as SkPipeline

        # calibrate_model wraps the whole fitted pipeline in CalibratedClassifierCV,
        # which has no .steps at all. SHAP values are additive in the UNCALIBRATED
        # score anyway (the calibrator is a monotonic map applied after), so explain
        # the inner pipeline it wraps rather than erroring on a missing attribute.
        calibrated = isinstance(pipeline, CalibratedClassifierCV)
        explain_pipeline = pipeline
        if calibrated:
            fitted = getattr(pipeline, "calibrated_classifiers_", None)
            inner = None
            if fitted:
                inner = getattr(fitted[0], "estimator", None) or getattr(
                    fitted[0], "base_estimator", None
                )
            if inner is None:
                return json.dumps(
                    {
                        "error": "model is calibrated but its inner pipeline could not be recovered for SHAP"
                    }
                )
            explain_pipeline = inner

        encoder_steps = [
            (name, step)
            for name, step in explain_pipeline.steps
            if name not in ("smote", "classifier")
        ]
        X_encoded = SkPipeline(encoder_steps).transform(X_test)
        feature_names = explain_pipeline.named_steps["encode"].get_feature_names_out()
        clf = explain_pipeline.named_steps["classifier"]
        explainer = (
            shap.TreeExplainer(clf)
            if meta["model"] in ("random_forest", "xgboost", "hist_gradient_boosting")
            else shap.LinearExplainer(clf, X_encoded)
        )
        try:
            shap_values = explainer.shap_values(X_encoded)
        except Exception as e:
            if "additivity" not in str(e).lower():
                raise
            shap_values = explainer.shap_values(X_encoded, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=-1)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[
                :, :, positive_index(meta, y=y_test)
            ]  # the run's positive class, not mean across classes
        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_signed = shap_values.mean(
            axis=0
        )  # sign only meaningful in aggregate/on average — a feature can push
        # individual rows both ways (see the beeswarm plot for the real per-row spread); this is a global tendency.
        ranked = sorted(
            zip(feature_names, mean_abs.round(4), mean_signed.round(4)),
            key=lambda kv: -kv[1],
        )[:10]
        explain_result = {
            "method": "shap",
            "model": meta["model"],
            "positive_class": pos_label,
            "feature_importance": [[f, float(a)] for f, a, _ in ranked],
            "direction": {
                f: ("increases" if s > 0 else "decreases") for f, _, s in ranked
            },
            **(
                {
                    "note": "model is calibrated — SHAP explains the inner uncalibrated pipeline; "
                    "contributions are additive in the pre-calibration score, not the reported probability"
                }
                if calibrated
                else {}
            ),
        }
        meta["explain"] = explain_result
        _save_meta(run_id, meta)
        return json.dumps(explain_result)

    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=5, random_state=42
    )
    importances = sorted(
        zip(X_test.columns, result.importances_mean.round(4)), key=lambda kv: -kv[1]
    )
    explain_result = {
        "method": "permutation",
        "model": meta["model"],
        "feature_importance": [[f, float(v)] for f, v in importances[:10]],
    }
    meta["explain"] = explain_result
    _save_meta(run_id, meta)
    return json.dumps(explain_result)


def _metrics_at_threshold(y_test, proba, threshold: float, pos_label=1) -> dict:
    """proba is P(pos_label), so the 0/1 prediction it thresholds into is
    "is this the positive class", and y_test is binarised the same way —
    scoring it against the raw labels would silently use sklearn's
    pos_label=1 default and measure the wrong class on a {"fraud","legit"}
    target."""
    y_pred = (proba >= threshold).astype(int)
    y_true = (pd.Series(y_test).values == pos_label).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        # Share of ROWS flagged — the workload this threshold creates, which
        # precision and recall between them never state outright.
        "alert_rate": round(float(y_pred.mean()), 6),
    }


@mcp.tool()
def tune_threshold(
    run_id: str,
    target_recall: float = 0.0,
    max_alert_rate: float = 0.0,
    fn_to_fp_cost_ratio: float = 0.0,
) -> str:
    """Picks a decision threshold for this run's CURRENT fitted model
    (binary targets only) and PERSISTS it as the run's operating point so
    that export_model, predict, generate_report and the success gate all use
    the same number.

    The threshold is CHOSEN on out-of-fold predictions for the training fold
    (see core._validation_proba — grouped/forward-chained when the run is),
    then MEASURED once on the held-out test fold, with bootstrap intervals.
    Choosing and measuring on the same test rows would report how well the
    cutoff fits those rows, not how the model will do. So a recall target can
    come in under target on test — that is the honest number, and it is the
    one reported.

    Exactly one mode may be set; with none, it maximizes F1.

    target_recall (0-1): best precision among thresholds that still reach at
      least this recall. The healthcare/fraud pattern "never miss more than
      X% of positives".
    max_alert_rate (0-1): best recall among thresholds that flag no more
      than this FRACTION OF ROWS. This is the capacity constraint a review
      queue actually has — "we can work 5,000 alerts a day out of 700,000
      rows" is max_alert_rate≈0.007 — and it is a different question from
      recall. A recall floor asks how much fraud you catch; an alert budget
      asks how much work you create. When the two disagree, the budget wins,
      because a model that emits more alerts than anyone can review has an
      effective recall of whatever gets reviewed.
    fn_to_fp_cost_ratio (>0): how many false positives one missed positive
      is worth. Chooses the threshold minimizing fn_to_fp_cost_ratio*FN + FP.
      Use when the business can state relative costs (a missed fraud costs
      ~40 analyst reviews → 40) rather than a recall target.

    Returns metrics at the chosen threshold and at the sklearn-default 0.5,
    so the tradeoff is visible rather than asserted."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    pipeline = joblib.load(pipeline_path)
    train, test, meta = _load_split(run_id)
    X_test, y_test = test.drop(columns=[meta["target"]]), test[meta["target"]]
    if not hasattr(pipeline, "predict_proba"):
        return json.dumps(
            {"error": f"{meta['model']} has no predict_proba, can't tune a threshold"}
        )
    if y_test.nunique() != 2:
        return json.dumps({"error": "threshold tuning needs a binary target"})

    # Every mode below reads precision/recall off the probability
    # distribution, and the cost mode does arithmetic on it. If nobody has
    # checked that those numbers are probabilities, the chosen threshold is a
    # number with no stated meaning — so say so rather than returning it bare.
    calibration = meta.get("calibration")
    calibration_caveat = None
    if not calibration:
        calibration_caveat = (
            "calibration was never measured for this model — run check_calibration. This threshold is a cutoff on "
            "a ranking score, not on a probability, and the precision/recall it promises rest on scores nobody checked."
        )
    elif calibration.get("miscalibration_warning"):
        calibration_caveat = (
            f"this model is MISCALIBRATED (ECE {calibration.get('expected_calibration_error')}) — run "
            "calibrate_model and re-tune; the cutoff below encodes a tradeoff the stated objective didn't ask for."
        )

    pos_label = positive_label(meta, y_test)
    test_proba = pipeline.predict_proba(X_test)[:, positive_index(meta, y=y_test)]
    # Everything from here to the choice of idx reads VALIDATION predictions.
    y_val, proba, validation = _validation_proba(
        pipeline, train.drop(columns=[meta["target"]]), train[meta["target"]], meta
    )
    precision, recall, thresholds = precision_recall_curve(
        y_val, proba, pos_label=pos_label
    )
    precision, recall = precision[:-1], recall[:-1]

    modes = [
        bool(target_recall > 0),
        bool(max_alert_rate > 0),
        bool(fn_to_fp_cost_ratio > 0),
    ]
    if sum(modes) > 1:
        return json.dumps(
            {
                "error": "set at most one of target_recall / max_alert_rate / fn_to_fp_cost_ratio — "
                "they are three different questions and would pick different thresholds"
            }
        )

    n_pos = int((y_val == pos_label).sum())
    if target_recall > 0:
        eligible = np.where(recall >= target_recall)[0]
        if len(eligible) == 0:
            return json.dumps(
                {
                    "error": f"no threshold reaches recall >= {target_recall} on the validation predictions"
                }
            )
        idx = eligible[np.argmax(precision[eligible])]
        chosen_by = f"target_recall>={target_recall}"
    elif max_alert_rate > 0:
        # Alert rate at threshold t is the share of ROWS flagged, which is what
        # a review queue is sized against — not the share of positives caught.
        alert_rate = np.array([float((proba >= t).mean()) for t in thresholds])
        eligible = np.where(alert_rate <= max_alert_rate)[0]
        if len(eligible) == 0:
            return json.dumps(
                {
                    "error": f"even the strictest threshold flags more than {max_alert_rate:.4%} of rows — "
                    "the model cannot meet this alert budget; either the budget or the model has to move"
                }
            )
        idx = int(eligible[np.argmax(recall[eligible])])
        chosen_by = f"max_alert_rate<={max_alert_rate}"
    elif fn_to_fp_cost_ratio > 0:
        tp = recall * n_pos
        fn = n_pos - tp
        fp = np.where(
            precision > 0, tp * (1 - precision) / np.maximum(precision, 1e-12), 0
        )
        idx = int(np.argmin(fn_to_fp_cost_ratio * fn + fp))
        chosen_by = f"min cost at {fn_to_fp_cost_ratio}:1 FN:FP"
    else:
        f1_curve = np.where(
            (precision + recall) > 0, 2 * precision * recall / (precision + recall), 0
        )
        idx = int(np.argmax(f1_curve))
        chosen_by = "max F1"

    threshold = float(thresholds[idx])
    selected_on = _metrics_at_threshold(y_val, proba, threshold, pos_label=pos_label)
    # The persisted numbers are the TEST fold's at that threshold — measured
    # once, never searched — with their bootstrap intervals.
    chosen = _metrics_at_threshold(y_test, test_proba, threshold, pos_label=pos_label)
    ci = _bootstrap_cis(y_test == pos_label, test_proba, test_proba >= threshold, True)
    chosen["ci"] = {k: ci[f"{k}_positive"] for k in ("precision", "recall", "f1")}

    # A threshold sitting at the very top of the score distribution is a
    # degenerate operating point: `proba >= max_score` only fires on rows
    # tied at the maximum. It looks like a normal threshold and reports
    # normal precision/recall on this test fold, but on new data — where
    # nothing happens to hit that exact value — it can alert on NOTHING and
    # give no indication that it has stopped working. Seen on a perfectly
    # separable fold, where every threshold meets the recall target and the
    # max-precision one is the highest available.
    max_proba = float(np.max(proba)) if len(proba) else 1.0
    degenerate = threshold >= max_proba
    if degenerate:
        chosen["degenerate_warning"] = (
            f"the chosen threshold ({threshold:.4f}) is at the top of the validation score distribution "
            f"(max score {max_proba:.4f}) — it only flags rows tied at the maximum. On this fold that is "
            f"{chosen['alert_rate']:.2%} of rows; on new data it can be zero, silently. Usually means the classes "
            "separate perfectly here, which is itself worth checking for leakage before shipping this threshold."
        )
    # Persisted, not just returned. An operating point that lives only in the
    # chat transcript is not the model anyone deploys: export/predict would
    # keep using sklearn's 0.5, so the reviewed model and the shipped model
    # would be different models with different precision.
    meta["operating_point"] = {
        **chosen,
        "chosen_by": chosen_by,
        "selected_on": {"validation": validation, **selected_on},
        "measured_on": f"held-out test fold ({len(y_test)} rows), once — intervals are 95% bootstrap",
        "default_0.5": _metrics_at_threshold(
            y_test, test_proba, 0.5, pos_label=pos_label
        ),
        # tune_hyperparams refits the pipeline in place; a threshold tuned
        # against the previous fit no longer describes this one.
        "pipeline_version": meta.get("pipeline_version", 0),
        "positive_label": _to_native(pos_label),
        # Persisted, not just mentioned once in chat: a threshold whose
        # probability basis was never checked must carry that caveat into the
        # report and into the exported bundle, not lose it at the first hop.
        "calibration_caveat": calibration_caveat,
    }
    _save_meta(run_id, meta)
    return json.dumps(
        {
            "run_id": run_id,
            "chosen": chosen,
            "chosen_by": chosen_by,
            "selected_on": meta["operating_point"]["selected_on"],
            "positive_label": str(pos_label),
            **(
                {"calibration_caveat": calibration_caveat} if calibration_caveat else {}
            ),
            "default_0.5": meta["operating_point"]["default_0.5"],
            "persisted": "this is now the run's operating point — export_model, predict, the report and the "
            "success gate all use it",
        }
    )


@mcp.tool()
def compare_runs(run_ids: str) -> str:
    """Side-by-side comparison of already-trained runs (comma-separated
    run_ids) — reads each run's saved meta.json, no retraining. Minimal
    cross-run tracking ("which of my last N runs was best") without a
    MLflow/W&B dependency. Each entry reports its CURRENT (tuned if
    tune_hyperparams ran, baseline otherwise) roc_auc/pr_auc, ranked by
    pr_auc — the metric the modeling skill's domain-notes recommend for
    imbalanced targets. The result is appended to every involved run's
    meta["cross_run_comparisons"] (not just returned), so a later
    generate_report(run_id) on any of them can show what it was compared
    against without needing this call replayed."""
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    rows = []
    for run_id in ids:
        try:
            meta = _load_meta(run_id)
        except FileNotFoundError as e:
            rows.append({"run_id": run_id, "error": str(e)})
            continue
        metrics = meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
        auc = metrics.get("auc") or {}
        rows.append(
            {
                "run_id": run_id,
                "model": meta.get("model"),
                "tuned": bool(meta.get("best_params")),
                "roc_auc": auc.get("roc_auc", auc.get("roc_auc_macro")),
                "pr_auc": auc.get("pr_auc", auc.get("pr_auc_macro")),
            }
        )
    ranked = sorted(
        (r for r in rows if r.get("pr_auc") is not None), key=lambda r: -r["pr_auc"]
    )
    result = {"runs": rows, "ranked_by_pr_auc": [r["run_id"] for r in ranked]}
    for run_id in ids:
        try:
            run_meta = _load_meta(run_id)
        except FileNotFoundError:
            continue
        run_meta.setdefault("cross_run_comparisons", []).append(result)
        _save_meta(run_id, run_meta)
    return json.dumps(result)


@mcp.tool()
def export_model(run_id: str, out_path: str = "", force: bool = False) -> str:
    """Saves this run's CURRENT fitted pipeline (tuned if tune_hyperparams
    ran) to out_path via joblib — a single self-contained sklearn Pipeline,
    so pipeline.predict(new_raw_df) works directly on data shaped like the
    original CSV (minus the target column); no manual re-encoding needed.
    Leave out_path empty to get a suggested path back instead of writing —
    call again with the path the user confirms (or their own path).

    Refuses to export a run the readiness gate has BLOCKED (a failed gate is
    positive evidence of a problem — an identifier column still feeding the
    model, a failed baseline comparison, a reflection contradicted by the
    artifacts) unless force=True. An `incomplete` run — one with gates that
    never ran — exports with a warning rather than a refusal, because missing
    evidence is a reason to look, not proof of a defect; the warning names
    exactly what is missing so it cannot pass silently. Ask the user before
    overriding either."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    meta = _load_meta(run_id)
    current_metrics = meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
    if current_metrics.get("overfitting_warning") and not force:
        return json.dumps(
            {
                "error": "overfitting_warning is set for this run's current model "
                f"(cv_to_test_gap={current_metrics.get('cv_to_test_gap')} > {core.OVERFIT_GAP_THRESHOLD}) "
                "— CV score doesn't reflect the held-out test fold. Ask the user before exporting; "
                "retry with force=true if they confirm.",
            }
        )

    readiness = compute_readiness(run_id)
    if readiness["overall_status"] == "blocked" and not force:
        return json.dumps(
            {
                "error": "readiness gate BLOCKED this run — exporting it would ship a model with known defects.",
                "failed_gates": {
                    g: readiness["checks"][g]["evidence"]
                    for g in readiness["failed_gates"]
                },
                "remedy": "fix the failed gate(s) and re-run, or retry with force=true if the user explicitly "
                "accepts shipping a blocked model.",
            }
        )
    export_warning = None
    if readiness["overall_status"] == "incomplete":
        export_warning = (
            "exported from an INCOMPLETE run — these required checks never ran: "
            f"{readiness['not_run_gates']}. The model may still be fine; nothing has verified that it is."
        )
    if not out_path:
        (RUNS_DIR.parent / "artifacts").mkdir(parents=True, exist_ok=True)
        suggested = str(
            RUNS_DIR.parent / "artifacts" / f"{meta['model']}_{run_id}.pkl"
        )  # data/artifacts: where serving looks
        return json.dumps(
            {
                "prompt": "Where should the .pkl be saved?",
                "suggested_out_path": suggested,
            }
        )
    pipeline = joblib.load(pipeline_path)
    # The operating point travels WITH the model. Without it the exported
    # artifact silently reverts to sklearn's 0.5, so a run that reported
    # precision 0.71 at a tuned threshold would deploy at precision 0.42 —
    # the reviewed model and the shipped model would not be the same model.
    operating_point = meta.get("operating_point")
    joblib.dump(
        {
            "pipeline": pipeline,
            "target": meta["target"],
            "model": meta["model"],
            "best_params": meta["best_params"],
            "run_id": run_id,
            "readiness_at_export": readiness["overall_status"],
            # The operating point is a cutoff on P(positive_label) — shipping the
            # threshold without the label it was tuned for is how predict() ends
            # up thresholding the other class.
            "positive_label": meta.get("positive_label"),
            "operating_point": operating_point,
            # Which registry version this is, so serving can stamp every
            # prediction with it — None if the run was never logged to MLflow.
            "registry": meta.get("registry"),
        },
        out_path,
    )

    # --- monitoring baseline -------------------------------------------------
    # data/runs/<run_id>/ is disposable working state and _cleanup_old_runs
    # deletes it after RUN_RETENTION_DAYS. train.csv is the ONLY reference
    # distribution a drift check has, so a model older than the retention
    # window would have nothing to be compared against — monitoring would
    # quietly become impossible a week after deployment, which is roughly
    # when it starts to matter.
    #
    # So the export carries its own baseline: the training fold beside the
    # .pkl, plus a plain-JSON profile (no pickle, no cross-agent import) that
    # tells the drift agent which columns actually feed the model, which is
    # the target, and which features the model leans on. Those last two turn
    # drift from "some column moved" into "something the model depends on
    # moved", which is the only version worth alerting on.
    train_fold, _, _ = _load_split(run_id)
    reference_path = str(Path(out_path).with_suffix(".reference.csv"))
    train_fold.to_csv(reference_path, index=False)
    explain = meta.get("explain") or {}
    feature_columns = [
        c
        for c in train_fold.columns
        if c != meta["target"] and c not in (meta.get("dropped_columns") or {})
    ]
    # The explainer ranks ENCODED features (Sex_male, Title_Mr); drift compares
    # RAW columns (Sex, Title). Map each back to the raw column it was encoded
    # from (longest matching prefix), or the model's top driver never counts.
    top_features = []
    for feature, _ in explain.get("feature_importance") or []:
        raw = max(
            (c for c in feature_columns if feature == c or feature.startswith(f"{c}_")),
            key=len,
            default=None,
        )
        if raw and raw not in top_features:
            top_features.append(raw)
    top_features = top_features[:10]
    profile_path = str(Path(out_path).with_suffix(".monitoring.json"))
    Path(profile_path).write_text(
        json.dumps(
            {
                "model_path": out_path,
                "run_id": run_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "reference_path": reference_path,
                "reference_rows": len(train_fold),
                "target": meta["target"],
                "positive_label": meta.get("positive_label"),
                "feature_columns": feature_columns,
                "dropped_columns": sorted(meta.get("dropped_columns") or {}),
                # Named by the explainer, so "key" means "this model actually uses it",
                # not "a human guessed it was important".
                "key_columns": top_features,
                "key_columns_source": explain.get("method")
                or "not available — explain_model was never run for this run",
                "operating_point": operating_point,
                # Where serving appends every batch this model scores — the `current`
                # side of a drift check (detect_drift falls back to it).
                "predictions_log": str(
                    RUNS_DIR.parent / "predictions" / f"{run_id}.csv"
                ),
            },
            indent=2,
            default=str,
        )
    )

    # Recorded on the run so "has this been exported, and to where" is
    # answerable from the artifact — the orchestrator reads this to decide
    # whether serving/monitoring is even a possible next step.
    meta["exported_path"] = out_path
    meta["export_readiness"] = readiness["overall_status"]
    meta["monitoring_profile_path"] = profile_path
    _save_meta(run_id, meta)
    threshold_note = (
        f"decision threshold {operating_point['threshold']} travels with this model "
        f"(chosen by {operating_point['chosen_by']}; precision {operating_point['precision']}, "
        f"recall {operating_point['recall']}, flags {operating_point['alert_rate']:.3%} of rows). "
        "predict() applies it automatically."
        if operating_point
        else "no tuned operating point — this model will predict at sklearn's default 0.5. On an imbalanced target "
        "that is rarely the right cutoff; call tune_threshold before exporting if the threshold matters."
    )
    return json.dumps(
        {
            "out_path": out_path,
            "model": meta["model"],
            "tuned": meta["best_params"] is not None,
            "readiness": readiness["overall_status"],
            "threshold": threshold_note,
            "monitoring_profile": profile_path,
            "monitoring_reference": reference_path,
            "monitoring_note": (
                "the training fold and a monitoring profile were written beside the .pkl — hand "
                f"'{profile_path}' to the drift agent (detect_drift(profile_path=...)) to compare production data "
                "against what this model was actually trained on. This survives the run directory being swept."
            ),
            **({"warning": export_warning} if export_warning else {}),
        }
    )


@mcp.tool()
def predict(pkl_path: str, data_path: str) -> str:
    """Loads a model exported by export_model and runs it on new data.
    data_path is a CSV shaped like the original training data (target
    column optional — dropped if present, since it isn't a feature). Writes
    one prediction (+ confidence, if the pipeline supports predict_proba)
    per row to a CSV next to data_path — full per-row predictions over the
    MCP JSON-RPC channel isn't viable for realistic row counts — and returns
    that path plus a class-distribution summary and small preview.

    If the exported bundle carries an operating point (tune_threshold was
    called before export), predictions use THAT threshold and the raw
    positive-class probability is written alongside them. Otherwise the
    sklearn default of 0.5 applies, and the result says so explicitly rather
    than leaving the reader to assume the tuned threshold was honoured."""

    if not Path(pkl_path).exists():
        return json.dumps(
            {"error": f"no exported model at '{pkl_path}' — call export_model first"}
        )
    data_file = Path(data_path)
    if not data_file.exists():
        return json.dumps({"error": f"no data file at '{data_path}'"})
    bundle = joblib.load(pkl_path)
    pipeline, target = bundle["pipeline"], bundle["target"]
    X = read_table(data_file)
    if target in X.columns:
        X = X.drop(columns=[target])
    out = X.copy()
    operating_point = bundle.get("operating_point")
    applied_threshold = None
    if (
        operating_point
        and hasattr(pipeline, "predict_proba")
        and len(getattr(pipeline, "classes_", [])) == 2
    ):
        # Apply the threshold the run actually chose, not sklearn's 0.5, and
        # apply it to the same class the threshold was tuned for — the run's
        # recorded positive label, carried in the export bundle. Thresholding
        # classes_[1] instead would flip the decision on any target whose
        # positive class isn't the sorted-max label.
        applied_threshold = float(operating_point["threshold"])
        pos_idx = positive_index(
            {"positive_label": bundle.get("positive_label")},
            classes=list(pipeline.classes_),
        )
        proba = pipeline.predict_proba(X)[:, pos_idx]
        positive, negative = pipeline.classes_[pos_idx], pipeline.classes_[1 - pos_idx]
        out["prediction"] = np.where(proba >= applied_threshold, positive, negative)
        out["positive_proba"] = proba.round(4)
    else:
        out["prediction"] = pipeline.predict(X)
    if hasattr(pipeline, "predict_proba") and "positive_proba" not in out:
        out["confidence"] = pipeline.predict_proba(X).max(axis=1).round(4)
    # Beside the input when the caller may write there; otherwise (a shared
    # dataset, with a user bound) in that user's predictions folder. The
    # tool guard skips outputs a tool names itself, so this one is checked here.
    stem = data_file.name.split(".")[0] or "data"
    out_file = data_file.with_name(f"{stem}_predictions.csv")
    if runtime.user():
        try:
            toolguard.check_output(str(out_file), "predict")
        except toolguard.Refused:
            out_file = toolguard.data_dir() / "predictions" / toolguard.user_folder(runtime.user()) / out_file.name
            toolguard.check_output(str(out_file), "predict")
    out_path = str(out_file)
    out.to_csv(out_path, index=False)
    result = {
        "model": bundle["model"],
        "n_rows": len(out),
        "out_path": out_path,
        "decision_threshold": (
            applied_threshold
            if applied_threshold is not None
            else "0.5 (sklearn default — this model carries no tuned operating point)"
        ),
        "class_counts": {
            str(k): int(v) for k, v in out["prediction"].value_counts().items()
        },
        "preview": out.head(5).to_dict(orient="records"),
    }
    return json.dumps(result, default=_to_native)


@mcp.tool()
def run_standard_diagnostics(run_id: str) -> str:
    """The judgment-free middle of an autonomous run, in its one correct
    order, as ONE call:

        train_baseline -> check_calibration (-> calibrate_model and re-check
        if miscalibrated) -> check_label_rule -> explain_model(shap) ->
        error_analysis -> check_model_stability -> check_readiness

    Call it after the model is trained/tuned. Each step has exactly one right
    way to run on the current model, so spending an LLM round per step only
    bought the chance to skip or reorder one — and a turn budget that ran out
    mid-pipeline. Calibration comes first because it refits the pipeline and
    would invalidate anything measured before it.

    What is left needs judgment and is returned as `remaining`: tune_threshold
    (which constraint), assess_fairness (which attribute) or
    declare_fairness_not_applicable, record_reflection, generate_report,
    log_run_to_mlflow — plus every gate still not_run."""
    from .diagnostics import (
        check_calibration,
        check_label_rule,
        check_model_stability,
        error_analysis,
        train_baseline,
    )
    from .gates import check_readiness

    steps = {}

    def run(name, fn, *args, **kwargs):
        try:
            out = json.loads(fn(*args, **kwargs))
        except Exception as exc:  # one failing step must not hide the others' results
            out = {"error": f"{type(exc).__name__}: {exc}"}
        # Scalars only — the per-bin/per-row detail is persisted on the run.
        steps[name] = {k: v for k, v in out.items() if not isinstance(v, (list, dict))}
        return out

    run("train_baseline", train_baseline, run_id)
    calibration = run("check_calibration", check_calibration, run_id)
    if calibration.get("miscalibration_warning"):
        # Isotonic overfits small folds (calibrate_model warns below 1000 rows).
        n_train = len(pd.read_csv(_run_dir(run_id) / "train.csv", usecols=[0]))
        run(
            "calibrate_model",
            calibrate_model,
            run_id,
            method="isotonic" if n_train >= 1000 else "sigmoid",
        )
        run("check_calibration_after", check_calibration, run_id)
    run("check_label_rule", check_label_rule, run_id)
    run("explain_model", explain_model, run_id, method="shap")
    run("error_analysis", error_analysis, run_id)
    run("check_model_stability", check_model_stability, run_id)
    readiness = run("check_readiness", check_readiness, run_id)
    return json.dumps(
        {
            "run_id": run_id,
            "steps": steps,
            "readiness": readiness.get("overall_status"),
            "failed_gates": readiness.get("failed_gates"),
            "not_run_gates": readiness.get("not_run_gates"),
            "remaining": [
                "tune_threshold — choose the mode that matches the business constraint (binary targets)",
                "assess_fairness(protected_attribute) or declare_fairness_not_applicable(reason)",
                "record_reflection",
                "generate_report",
                "log_run_to_mlflow",
            ],
        },
        default=str,
    )
