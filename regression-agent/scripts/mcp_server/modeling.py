"""regression agent — `modeling` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import (
    _build_pipeline,
    _cv_sample,
    _evaluate,
    _invalidate_model_derived,
    _load_meta,
    _load_split,
    _overfit_gate,
    _run_dir,
    _save_meta,
    _to_native,
    _cv_scoring,
    _validation_folds,
    OBJECTIVE_MODELS,
    OBJECTIVES,
    objective_of,
    param_grid,
)  # noqa: F401
from core import runtime, toolguard


@mcp.tool()
def compare_models(run_id: str, models: str = "") -> str:
    """Cross-validates candidate models (default: all of MODELS) on this
    run's training fold with the SAME pipeline shape train_model uses, and
    ranks them by 5-fold CV R^2 (mean/std) on folds that obey the run's
    split (forward-chained / grouped / shuffled — see cv_scheme).
    Read-only: doesn't touch pipeline.pkl or this run's current model.
    models: comma-separated subset of {list(MODELS)}, empty = all."""
    train, _, meta = _load_split(run_id)
    target = meta["target"]
    X_train, y_train = train.drop(columns=[target]), train[target]
    objective, _ = objective_of(meta)
    supported = list(OBJECTIVE_MODELS[objective])
    candidates = [m.strip() for m in models.split(",") if m.strip()] or supported
    unknown = [m for m in candidates if m not in MODELS]
    if unknown:
        return json.dumps(
            {"error": f"unknown model(s) {unknown}, use any of {list(MODELS)}"}
        )
    skipped = [m for m in candidates if m not in supported]
    candidates = [m for m in candidates if m in supported]

    X_cv, y_cv, cv_n_rows = _cv_sample(X_train, y_train, meta)
    cv, cv_scheme = _validation_folds(X_cv, y_cv, meta)
    scoring, metric, sign = _cv_scoring(meta)
    ranked = []
    for model in candidates:
        pipeline = _build_pipeline(meta, model, X_cv)
        scores = sign * cross_val_score(pipeline, X_cv, y_cv, cv=cv, scoring=scoring)
        ranked.append(
            {
                "model": model,
                f"cv_{metric}_mean": round(float(scores.mean()), 4),
                f"cv_{metric}_std": round(float(scores.std()), 4),
            }
        )
    ranked.sort(key=lambda r: -sign * r[f"cv_{metric}_mean"])
    return json.dumps(
        {"run_id": run_id, "objective": objective, "cv_metric": metric, "ranked": ranked,
         "recommended": ranked[0]["model"], "cv_scheme": cv_scheme, "cv_computed_on_n_rows": cv_n_rows,
         **({"skipped_no_such_loss": skipped} if skipped else {})}
    )



@mcp.tool()
def set_objective(run_id: str, objective: str = "squared_error", quantile: float = 0.5) -> str:
    """What the model minimises — decide it from the business decision, before
    comparing models:
      squared_error  the conditional MEAN (default)
      poisson        the mean, through a log link, for counts and other
                     non-negative targets (calls, claims, usage, spend with
                     many zeros). Predicts totals without bias — prefer it to
                     a log1p target when the predictions will be SUMMED into a
                     budget or a capacity plan (log1p's back-transform
                     under-predicts totals; sum_ratio shows by how much).
      quantile       the `quantile`-th quantile (0.9 = P90), for decisions
                     whose errors cost asymmetrically: provision for the
                     busy hour, not the average one. Scored on pinball loss;
                     quantile_hit_rate (share of actuals at or below the
                     prediction) should land near `quantile`.
    Changes which models apply (random_forest has no quantile loss) and the
    CV metric (pinball for quantile). Retrain after it."""
    if objective not in OBJECTIVES:
        return json.dumps({"error": f"objective must be one of {list(OBJECTIVES)}"})
    if objective == "quantile" and not 0 < quantile < 1:
        return json.dumps({"error": "quantile must be between 0 and 1"})
    train, _, meta = _load_split(run_id)
    y = train[meta["target"]]
    if objective == "poisson":
        if float(y.min()) < 0 or float(y.sum()) <= 0:
            return json.dumps({"error": "poisson needs a non-negative target that is not all zero"})
        if meta.get("target_transform"):
            return json.dumps({"error": "poisson already models the log of the mean — revert the target "
                                        "transform first (apply_target_transform(run_id, 'none'))"})
    meta["objective"] = None if objective == "squared_error" else objective
    meta["quantile"] = quantile if objective == "quantile" else None
    stale = _invalidate_model_derived(meta)
    _save_meta(run_id, meta)
    return json.dumps({
        "run_id": run_id,
        "objective": objective,
        **({"quantile": quantile} if objective == "quantile" else {}),
        "models_with_this_loss": list(OBJECTIVE_MODELS[objective]),
        "cv_metric": _cv_scoring(meta)[1],
        "next": "compare_models / train_model — earlier results describe the previous objective",
        **({"invalidated": stale} if stale else {}),
    })

@mcp.tool()
def train_model(run_id: str, model: str = "random_forest") -> str:
    """Fits model as one Pipeline (drop -> impute -> one-hot encode ->
    regressor) on the run's training fold. Reports 5-fold CV R^2 (mean±std,
    on train — the honest estimate of variance) and held-out test metrics
    (R^2, RMSE, MAE). Saves the fitted pipeline as this run's current model
    — tune_hyperparams, explain_model, export_model all read it back from
    here."""
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
    scoring, metric, sign = _cv_scoring(meta)
    try:
        cv_scores = sign * cross_val_score(_build_pipeline(meta, model, X_cv), X_cv, y_cv, cv=cv, scoring=scoring)
        pipeline = _build_pipeline(meta, model, X_train)
        pipeline.fit(X_train, y_train)
        test_eval = _evaluate(pipeline, X_test, y_test, meta)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    cv_mean = round(float(cv_scores.mean()), 4)
    cv_to_test_gap, overfitting_warning = _overfit_gate(cv_mean, test_eval, metric)

    artifact_path = str(_run_dir(run_id) / "pipeline.pkl")
    joblib.dump(pipeline, artifact_path)
    meta["model"] = model
    meta["best_params"] = None
    meta["baseline_metrics"] = {
        "cv_metric": metric,
        f"cv_{metric}_mean": cv_mean,
        f"cv_{metric}_std": round(float(cv_scores.std()), 4),
        "cv_scheme": cv_scheme,
        "cv_computed_on_n_rows": cv_n_rows,
        "objective": objective_of(meta)[0],
        "target_transform": meta.get("target_transform"),
        "cv_to_test_gap": cv_to_test_gap,
        "overfitting_warning": overfitting_warning,
        **test_eval,
    }
    meta["tuned_metrics"] = None
    stale = _invalidate_model_derived(meta)
    _save_meta(run_id, meta)
    save_training_run(
        run_id,
        target_column=target,
        best_model=model,
        metric="r2",
        best_score=cv_mean,
        test_auc=test_eval.get("r2"),
        cv_to_test_gap=cv_to_test_gap,
        overfitting_warning=overfitting_warning,
        best_params=None,
        artifact_path=artifact_path,
    )
    return json.dumps({"run_id": run_id, "model": model, **meta["baseline_metrics"],
                       **({"invalidated_by_refit": stale} if stale else {})})


@mcp.tool()
def tune_hyperparams(
    run_id: str, model: str = "random_forest", n_trials: int = 8, cv_folds: int = 5
) -> str:
    """Optuna TPE search over the SAME pipeline shape as train_model. No-op
    (just refits) for linear_regression, which has no hyperparameters worth
    searching. Defaults (5-fold, 8 trials) are a fast first pass; raise
    n_trials/cv_folds when the metric actually matters. Refits the best
    candidate on the full training fold and OVERWRITES this run's pipeline
    — explain_model/export_model pick up the tuned version automatically."""
    if model not in MODELS:
        return json.dumps(
            {"error": f"unknown model '{model}', use one of {list(MODELS)}"}
        )
    train, test, meta = _load_split(run_id)
    target = meta["target"]
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    if model not in OBJECTIVE_MODELS[objective_of(meta)[0]]:
        return json.dumps({"error": f"{model} has no {objective_of(meta)[0]} loss — use one of "
                                    f"{list(OBJECTIVE_MODELS[objective_of(meta)[0]])}"})
    X_cv, y_cv, cv_n_rows = _cv_sample(X_train, y_train, meta)
    cv, cv_scheme = _validation_folds(X_cv, y_cv, meta, n_splits=cv_folds)
    scoring, metric, sign = _cv_scoring(meta)
    grid = param_grid(meta, model)

    if grid:

        def objective(trial: optuna.Trial) -> float:
            params = {
                name: trial.suggest_categorical(name, choices)
                for name, choices in grid.items()
            }
            pipeline = _build_pipeline(meta, model, X_cv)
            pipeline.set_params(**params)
            return cross_val_score(pipeline, X_cv, y_cv, cv=cv, scoring=scoring).mean()

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)
        best_params, cv_best = study.best_params, round(float(sign * study.best_value), 4)
    else:
        best_params = {}
        cv_best = round(float(sign * cross_val_score(_build_pipeline(meta, model, X_cv), X_cv, y_cv, cv=cv,
                                                     scoring=scoring).mean()), 4)

    best = _build_pipeline(meta, model, X_train)
    best.set_params(**best_params)
    best.fit(X_train, y_train)
    test_eval = _evaluate(best, X_test, y_test, meta)
    cv_to_test_gap, overfitting_warning = _overfit_gate(cv_best, test_eval, metric)

    artifact_path = str(_run_dir(run_id) / "pipeline.pkl")
    joblib.dump(best, artifact_path)
    meta["model"] = model
    meta["best_params"] = best_params
    meta["tuned_metrics"] = {
        "cv_metric": metric,
        f"cv_{metric}_best": cv_best,
        "cv_scheme": cv_scheme,
        "objective": objective_of(meta)[0],
        "cv_computed_on_n_rows": cv_n_rows,
        "target_transform": meta.get("target_transform"),
        "cv_to_test_gap": cv_to_test_gap,
        "overfitting_warning": overfitting_warning,
        **test_eval,
    }
    _invalidate_model_derived(meta)
    _save_meta(run_id, meta)
    save_training_run(
        run_id,
        target_column=target,
        best_model=model,
        metric="r2",
        best_score=cv_best,
        test_auc=test_eval.get("r2"),
        cv_to_test_gap=cv_to_test_gap,
        overfitting_warning=overfitting_warning,
        best_params=best_params,
        artifact_path=artifact_path,
    )

    baseline_r2 = (meta.get("baseline_metrics") or {}).get("r2")
    tuned_r2 = meta["tuned_metrics"].get("r2")
    improvement = (
        round(tuned_r2 - baseline_r2, 4)
        if baseline_r2 is not None and tuned_r2 is not None
        else None
    )
    return json.dumps(
        {
            "run_id": run_id,
            "model": model,
            "best_params": best_params,
            "r2_improvement_vs_baseline": improvement,
            **meta["tuned_metrics"],
        }
    )


@mcp.tool()
def explain_model(
    run_id: str, method: str = "permutation", background_size: int = 200
) -> str:
    """Explains this run's CURRENT model (tuned if tune_hyperparams ran,
    baseline otherwise). method: "permutation" (default — cheap,
    model-agnostic, computed on the held-out test fold) or "shap" (exact
    TreeExplainer for random_forest/xgboost, LinearExplainer for
    linear_regression). background_size caps how many test rows SHAP runs
    on (plain random sample — no class to stratify by for a continuous
    target)."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    pipeline = joblib.load(pipeline_path)
    _, test, meta = _load_split(run_id)
    X_test, y_test = test.drop(columns=[meta["target"]]), test[meta["target"]]

    if method == "shap":
        if 0 < background_size < len(X_test):
            X_test, _ = train_test_split(
                X_test, train_size=background_size, random_state=42
            )

        encoder_steps = [
            (name, step) for name, step in pipeline.steps if name != "regressor"
        ]
        X_encoded = Pipeline(encoder_steps).transform(X_test)
        feature_names = pipeline.named_steps["encode"].get_feature_names_out()
        reg = pipeline.named_steps["regressor"]
        # A transformed target wraps the fitted estimator; its SHAP values are
        # in the transformed (e.g. log) units — rankings hold, magnitudes don't.
        units = "original"
        if hasattr(reg, "regressor_"):
            reg, units = reg.regressor_, meta.get("target_transform")
        explainer = (
            shap.TreeExplainer(reg)
            if meta["model"] in ("random_forest", "xgboost", "hist_gradient_boosting")
            else shap.LinearExplainer(reg, X_encoded)
        )
        shap_values = explainer.shap_values(X_encoded)
        importances = sorted(
            zip(feature_names, np.abs(shap_values).mean(axis=0).round(4)),
            key=lambda kv: -kv[1],
        )
        top = [[f, float(v)] for f, v in importances[:10]]
        meta["explainability"] = {"method": "shap", "top_features": top, "shap_units": units}
        _save_meta(run_id, meta)
        return json.dumps(
            {"method": "shap", "model": meta["model"], "feature_importance": top}
        )

    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=5, random_state=42, scoring=_cv_scoring(meta)[0]
    )
    importances = sorted(
        zip(X_test.columns, result.importances_mean.round(4)), key=lambda kv: -kv[1]
    )
    top = [[f, float(v)] for f, v in importances[:10]]
    meta["explainability"] = {"method": "permutation", "top_features": top}
    _save_meta(run_id, meta)
    return json.dumps(
        {"method": "permutation", "model": meta["model"], "feature_importance": top}
    )


@mcp.tool()
def compare_runs(run_ids: str) -> str:
    """Side-by-side comparison of already-trained runs (comma-separated
    run_ids) — reads each run's saved meta.json, no retraining. Each entry
    reports its CURRENT (tuned if tune_hyperparams ran, baseline otherwise)
    r2/rmse/mae, ranked by r2."""
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    rows = []
    for run_id in ids:
        try:
            meta = _load_meta(run_id)
        except FileNotFoundError as e:
            rows.append({"run_id": run_id, "error": str(e)})
            continue
        metrics = meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
        rows.append(
            {
                "run_id": run_id,
                "model": meta.get("model"),
                "tuned": bool(meta.get("best_params")),
                "r2": metrics.get("r2"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
            }
        )
    ranked = sorted(
        (r for r in rows if r.get("r2") is not None), key=lambda r: -r["r2"]
    )
    return json.dumps({"runs": rows, "ranked_by_r2": [r["run_id"] for r in ranked]})


@mcp.tool()
def export_model(run_id: str, out_path: str = "", force: bool = False) -> str:
    """Saves this run's CURRENT fitted pipeline (tuned if tune_hyperparams
    ran) to out_path via joblib — a single self-contained sklearn Pipeline,
    so pipeline.predict(new_raw_df) works directly on data shaped like the
    original CSV (minus the target column). Leave out_path empty to get a
    suggested path back instead of writing. Refuses to export a model with
    a large CV-vs-test-fold R^2 gap (overfitting_warning) unless
    force=True."""
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
                f"(cv_to_test_gap={current_metrics.get('cv_to_test_gap')} > {OVERFIT_GAP_THRESHOLD}) "
                "— CV score doesn't reflect the held-out test fold. Ask the user before exporting; "
                "retry with force=true if they confirm.",
            }
        )
    readiness = compute_readiness(run_id)
    if readiness["overall_status"] == "blocked" and not force:
        return json.dumps(
            {
                "error": "readiness gates FAILED for this run — a gate found positive evidence of a problem, "
                "not just missing evidence. Fix and re-run, or retry with force=true if the user "
                "explicitly accepts it.",
                "failed_gates": {
                    name: readiness["checks"][name]["evidence"]
                    for name in readiness["failed_gates"]
                },
            }
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
    joblib.dump(
        {
            "pipeline": pipeline,
            "target": meta["target"],
            "model": meta["model"],
            "best_params": meta["best_params"],
            # Which model this is — serving stamps every prediction with it.
            "run_id": run_id,
            "registry": meta.get("registry"),
            "readiness_at_export": readiness["overall_status"],
            "target_transform": meta.get("target_transform"),
            "prediction_interval": meta.get("prediction_interval"),
        },
        out_path,
    )
    # The drift baseline travels with the model: the run dir is swept.
    train, _, _ = _load_split(run_id)
    profile = core_mlops.write_monitoring_profile(
        out_path,
        run_id,
        train,
        meta["target"],
        meta.get("dropped_columns"),
        (meta.get("explainability") or {}).get("top_features"),
    )
    return json.dumps(
        {
            "out_path": out_path,
            "model": meta["model"],
            "tuned": meta["best_params"] is not None,
            "monitoring_profile": profile,
        }
    )


@mcp.tool()
def predict(pkl_path: str, data_path: str) -> str:
    """Loads a model exported by export_model and runs it on new data.
    data_path is a CSV shaped like the original training data (target
    column optional — dropped if present). Writes one prediction per row to
    a CSV next to data_path, and returns that path plus a summary and small
    preview."""
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
    out["prediction"] = pipeline.predict(X)
    interval = bundle.get("prediction_interval")
    if interval:
        # Split-conformal half-width from out-of-fold residuals (calibrate_intervals).
        out["prediction_lower"] = out["prediction"] - interval["half_width"]
        out["prediction_upper"] = out["prediction"] + interval["half_width"]
    stem = data_file.name.split(".")[0] or "data"
    out_file = data_file.with_name(f"{stem}_predictions.csv")
    if runtime.user():  # beside the input when allowed, else the user's predictions folder
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
        "prediction_summary": out["prediction"].describe().round(4).to_dict(),
        "preview": out.head(5).to_dict(orient="records"),
    }
    return json.dumps(result, default=_to_native)
