"""forecasting agent — `modeling` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .diagnostics import compute_readiness  # noqa: F401
from .core import (
    _backtest,
    _fit_and_score,
    _panel_features,
    _exog_columns,
    _fit_forecast_model,
    _load_meta,
    _load_split,
    _overfit_gate,
    _rmse_mae_mape,
    _run_dir,
    _save_meta,
    _to_native,
)  # noqa: F401


@mcp.tool()
def compare_models(run_id: str, models: str = "") -> str:
    """Walk-forward backtests candidate models (default: all of MODELS) on
    this run's training fold and ranks by backtest RMSE (mean/std,
    lower better). Read-only — writes no artifact. models: comma-separated
    subset of MODELS, empty = all."""
    train, _, meta = _load_split(run_id)
    candidates = [m.strip() for m in models.split(",") if m.strip()] or list(MODELS)
    unknown = [m for m in candidates if m not in MODELS]
    if unknown:
        return json.dumps({"error": f"unknown model(s) {unknown}, use any of {MODELS}"})

    ranked = []
    for model_type in candidates:
        bt = _backtest(train, meta, model_type, {})
        ranked.append({"model": model_type, "backtest_rmse_mean": bt["rmse_mean"],
                       "backtest_rmse_std": bt["rmse_std"], "origins": bt["origins"], "horizon": bt["horizon"]})
    scored = [r for r in ranked if r["backtest_rmse_mean"] is not None]
    scored.sort(key=lambda r: r["backtest_rmse_mean"])
    unscored = [r for r in ranked if r["backtest_rmse_mean"] is None]
    return json.dumps(
        {
            "run_id": run_id,
            "ranked": scored + unscored,
            "recommended": scored[0]["model"] if scored else None,
        }
    )


def _target_gaps(train: pd.DataFrame, test: pd.DataFrame, meta: dict) -> str | None:
    empty = int(train[meta["target"]].isna().sum() + test[meta["target"]].isna().sum())
    if empty:
        return (f"the target is empty in {empty} period(s) (gaps prepare_dataset inserted, or missing values) — "
                "decide how to fill them first: propose_imputation / apply_imputation (0 for counts that truly "
                "were zero, interpolate for levels, ffill for short gaps)")
    return None


def _save_current(run_id: str, meta: dict, fm, forecast: pd.DataFrame, model: str, params, metrics: dict,
                  key: str) -> None:
    artifact_path = str(_run_dir(run_id) / "model.pkl")
    joblib.dump(fm, artifact_path)
    forecast.to_csv(_run_dir(run_id) / "test_forecast.csv", index=False)  # visualization's plot_forecast reads it
    meta["model"] = model
    meta["best_params"] = params
    meta[key] = metrics
    if key == "baseline_metrics":
        meta["tuned_metrics"] = None
    meta["interval"] = fm.interval  # carried to the exported refit
    _save_meta(run_id, meta)
    save_training_run(
        run_id,
        target_column=meta["target"],
        best_model=model,
        metric="rmse",
        best_score=metrics["backtest_rmse_mean"] if metrics["backtest_rmse_mean"] is not None else metrics["rmse"],
        test_auc=metrics["rmse"],
        cv_to_test_gap=metrics["cv_to_test_gap"],
        overfitting_warning=metrics["overfitting_warning"],
        best_params=params,
        artifact_path=artifact_path,
    )


@mcp.tool()
def train_model(run_id: str, model: str = "xgboost") -> str:
    """Fits `model` on the run's full training fold and makes it the current
    model. Reports a rolling-origin backtest at the business horizon (RMSE
    mean±std over the last origins, each forecast recursively from history
    only) AND the held-out period forecast exactly as predict will: RMSE,
    MAE, MASE (< 1 beats the seasonal naive), WAPE, bias, MAPE when no actual
    is near zero, and the P10-P90 interval calibrated on the backtest errors
    with its coverage of the held-out actuals (should be near 80%)."""
    if model not in MODELS:
        return json.dumps({"error": f"unknown model '{model}', use one of {MODELS}"})
    train, test, meta = _load_split(run_id)
    gaps = _target_gaps(train, test, meta)
    if gaps:
        return json.dumps({"error": gaps})
    fm, metrics, forecast = _fit_and_score(train, test, meta, model, {})
    _save_current(run_id, meta, fm, forecast, model, None, metrics, "baseline_metrics")
    return json.dumps({"run_id": run_id, "model": model, **metrics})


@mcp.tool()
def tune_hyperparams(
    run_id: str, model: str = "xgboost", n_trials: int = 8, cv_folds: int = 5
) -> str:
    """Optuna TPE search, scored by the rolling-origin backtest RMSE at the
    business horizon (lower better). No-op (just re-scores) for
    naive_seasonal, which has no hyperparameters. Refits the best candidate
    on the full training fold and OVERWRITES this run's model —
    explain_model/export_model pick up the tuned version automatically."""
    if model not in MODELS:
        return json.dumps({"error": f"unknown model '{model}', use one of {MODELS}"})
    train, test, meta = _load_split(run_id)
    gaps = _target_gaps(train, test, meta)
    if gaps:
        return json.dumps({"error": gaps})
    grid = PARAM_GRIDS[model]
    best_params = {}
    if grid:

        def objective(trial: optuna.Trial) -> float:
            params = {name: trial.suggest_categorical(name, choices) for name, choices in grid.items()}
            mean = _backtest(train, meta, model, params, n_splits=cv_folds)["rmse_mean"]
            return mean if mean is not None else float("inf")

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params

    fm, metrics, forecast = _fit_and_score(train, test, meta, model, best_params, n_splits=cv_folds)
    _save_current(run_id, meta, fm, forecast, model, best_params, metrics, "tuned_metrics")
    baseline_rmse = (meta.get("baseline_metrics") or {}).get("rmse")
    improvement = round(baseline_rmse - metrics["rmse"], 4) if baseline_rmse is not None else None
    return json.dumps({"run_id": run_id, "model": model, "best_params": best_params,
                       "rmse_improvement_vs_baseline": improvement, **metrics})


def _explain_panel(run_id: str, fm, meta: dict) -> str:
    """Multi-series: Holt-Winters — the level, trend and season of the five
    largest series; xgboost — permutation importance on the held-out
    one-step features of every series (lags, calendar, series size)."""
    train, test, _ = _load_split(run_id)
    if fm.model_type == "exponential_smoothing":
        largest = sorted(fm.scales, key=fm.scales.get, reverse=True)[:5]
        per = {str(s): {k: (round(float(v), 4) if np.isscalar(v) else None) for k, v in fm.fitted[s].params.items()
                        if k in ("smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend")}
               for s in largest if hasattr(fm.fitted[s], "params")}
        return json.dumps({"method": "decomposition", "model": fm.model_type, "series": len(fm.scales),
                           "largest_series_params": per})
    both = pd.concat([train, test], ignore_index=True).sort_values([meta["series_column"], meta["date_column"]])
    both[meta["date_column"]] = pd.to_datetime(both[meta["date_column"]])
    X, y = _panel_features(both.reset_index(drop=True), meta, fm.scales, fm.codes)
    in_test = both.reset_index(drop=True)[meta["date_column"]] >= pd.to_datetime(test[meta["date_column"]]).min()
    keep = in_test & X.notna().all(axis=1) & y.notna()
    X_test, y_test = X.loc[keep, list(fm.fitted.feature_names_in_)], y[keep].to_numpy()
    base = float(np.sqrt(np.mean((fm.fitted.predict(X_test) - y_test) ** 2)))
    rng = np.random.default_rng(42)
    importances = []
    for feat in X_test.columns:
        shuffled = X_test.copy()
        shuffled[feat] = rng.permutation(shuffled[feat].to_numpy())
        importances.append((feat, round(float(np.sqrt(np.mean((fm.fitted.predict(shuffled) - y_test) ** 2))) - base, 4)))
    importances.sort(key=lambda kv: -kv[1])
    meta["explainability"] = {"method": "permutation", "top_features": [[f, float(v)] for f, v in importances[:10]]}
    _save_meta(run_id, meta)
    return json.dumps({"method": "permutation", "model": fm.model_type, "units": "error in each series' own scale",
                       "baseline_scaled_rmse": round(base, 4), "feature_importance": importances[:10]})


@mcp.tool()
def explain_model(run_id: str, method: str = "permutation") -> str:
    """Explains this run's CURRENT model (tuned if tune_hyperparams ran,
    baseline otherwise). What "explanation" means depends on the model
    type — there's no universal feature-importance concept across a naive
    baseline, an exponential-smoothing decomposition, and a lag-feature
    regressor:
      - "xgboost": permutation importance on the lag/rolling/calendar/exog
        features, computed against the held-out test period's actual
        forecast error.
      - "exponential_smoothing": the fitted trend/seasonal/level components
        instead of a feature ranking — there are no input features, the
        model IS the decomposition.
      - "naive_seasonal": returns a message — a lookup table has nothing to
        explain beyond "last cycle's value", which is already the whole
        model.
    """
    model_path = _run_dir(run_id) / "model.pkl"
    if not model_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    fm: ForecastModel = joblib.load(model_path)
    _, test, meta = _load_split(run_id)

    if fm.model_type == "naive_seasonal":
        return json.dumps(
            {
                "method": "none",
                "model": fm.model_type,
                "message": "naive_seasonal has no features to rank — it repeats the last seasonal cycle directly.",
            }
        )

    if isinstance(fm, PanelForecastModel):
        return _explain_panel(run_id, fm, meta)

    if fm.model_type == "exponential_smoothing":
        params = fm.fitted.params
        components = {
            k: (float(v) if np.isscalar(v) else None) for k, v in params.items()
        }
        return json.dumps(
            {
                "method": "decomposition",
                "model": fm.model_type,
                "fitted_params": components,
            }
        )

    # xgboost: permutation importance, scored against one-step predictions on
    # the held-out test period built with the SAME build_features() training
    # used. Reordering fm.lags (a list) is a no-op — forecast() looks feature
    # columns up by name/value, not position — so every importance came back
    # 0.0. Perturbing the column *values* themselves fixes that, and the same
    # mechanism now covers exog/one-hot columns too instead of skipping them.
    exog_cols = _exog_columns(meta)
    full_dates = pd.concat(
        [fm.history_dates, test[meta["date_column"]].reset_index(drop=True)],
        ignore_index=True,
    )
    full_target = pd.concat(
        [fm.history_target, test[meta["target"]].reset_index(drop=True)],
        ignore_index=True,
    )
    full_exog = None
    if exog_cols:
        full_exog = pd.concat(
            [fm.history_exog, test[exog_cols].reset_index(drop=True)], ignore_index=True
        )
    feats = build_features(
        full_dates, full_target, full_exog, fm.lags, fm.rolling_windows
    )
    if fm.encoder is not None:
        feats = fm._encode_exog(feats)
    test_feats = feats.iloc[-len(test) :].reset_index(drop=True)
    X_test = test_feats[fm.fitted.feature_names_in_]
    y_test = test[meta["target"]].to_numpy()

    baseline_pred = fm.fitted.predict(X_test)
    baseline_rmse = float(mean_squared_error(y_test, baseline_pred) ** 0.5)

    importances = []
    rng = np.random.default_rng(42)
    for feat in fm.fitted.feature_names_in_:
        perturbed_X = X_test.copy()
        perturbed_X[feat] = rng.permutation(perturbed_X[feat].to_numpy())
        pred = fm.fitted.predict(perturbed_X)
        rmse = float(mean_squared_error(y_test, pred) ** 0.5)
        importances.append((feat, round(rmse - baseline_rmse, 4)))
    importances.sort(key=lambda kv: -kv[1])
    meta["explainability"] = {
        "method": "permutation",
        "top_features": [[f, float(v)] for f, v in importances[:10]],
    }
    _save_meta(run_id, meta)  # visualization's plot_feature_importance(run_id) reads it
    return json.dumps(
        {
            "method": "permutation",
            "model": fm.model_type,
            "baseline_rmse": round(baseline_rmse, 4),
            "feature_importance": importances[:10],
        }
    )


@mcp.tool()
def compare_runs(run_ids: str) -> str:
    """Side-by-side comparison of already-trained runs (comma-separated),
    reading each run's saved meta.json — no retraining. Ranked by test RMSE
    (lower better)."""
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
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mase": metrics.get("mase"),
                "wape": metrics.get("wape"),
                "interval_coverage": metrics.get("interval_coverage"),
                "mape": metrics.get("mape"),
            }
        )
    ranked = sorted(
        (r for r in rows if r.get("rmse") is not None), key=lambda r: r["rmse"]
    )
    return json.dumps({"runs": rows, "ranked_by_rmse": [r["run_id"] for r in ranked]})


@mcp.tool()
def export_model(run_id: str, out_path: str = "", force: bool = False) -> str:
    """Saves this run's CURRENT fitted ForecastModel to out_path via joblib.
    Leave out_path empty to get a suggested path back instead of writing.
    Refuses to export when the backtest-vs-test-period RMSE gap is large
    (overfitting_warning) unless force=True."""
    model_path = _run_dir(run_id) / "model.pkl"
    if not model_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    meta = _load_meta(run_id)
    current = meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
    if current.get("overfitting_warning") and not force:
        return json.dumps(
            {
                "error": f"overfitting_warning is set (cv_to_test_gap={current.get('cv_to_test_gap')} > {OVERFIT_GAP_THRESHOLD}) "
                "— backtest RMSE doesn't reflect the held-out test period. Ask the user before exporting; retry with force=true if they confirm.",
            }
        )
    # A run whose own evidence found a problem does not ship (core/gates.py).
    readiness = compute_readiness(run_id)
    if readiness["overall_status"] == "blocked" and not force:
        return json.dumps(
            {
                "error": "readiness gate BLOCKED this run — exporting it would ship a model with known defects.",
                "failed_gates": {
                    g: readiness["checks"][g]["evidence"]
                    for g in readiness["failed_gates"]
                },
                "remedy": "fix the failed gate(s) and re-run, or retry with force=true if the user explicitly accepts it.",
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
    # The evaluated model was fit on the training period only, so its history
    # ends where the held-out test period starts. Shipped as-is, "forecast the
    # next 30 days" forecasts 30 days that already happened. Refit the SAME
    # model and settings on the full history, so forecasts start after the
    # last observed date; the evaluation metrics describe the train-only fit.
    train, test, _ = _load_split(run_id)
    fm = _fit_forecast_model(
        pd.concat([train, test], ignore_index=True),
        meta,
        meta["model"],
        meta.get("best_params") or {},
    )
    # Which model this is and what its gates said — serving stamps every forecast with them.
    fm.interval = meta.get("interval")  # calibrated on the train-only backtest
    fm.run_id, fm.registry, fm.readiness_at_export = (
        run_id,
        meta.get("registry"),
        readiness["overall_status"],
    )
    joblib.dump(fm, out_path)
    return json.dumps(
        {
            "out_path": out_path,
            "model": meta["model"],
            "tuned": meta["best_params"] is not None,
            "history_ends_at": str(pd.to_datetime(fm.history_dates.iloc[-1])),
            "readiness": readiness["overall_status"],
            "refit": "refit on train + test period with the evaluated settings — forecasts start after the last observed "
            "date; the reported backtest/test metrics are from the train-only fit",
        }
    )


@mcp.tool()
def predict(
    pkl_path: str, horizon: int, future_exog_path: str = "", out_path: str = ""
) -> str:
    """Forecasts `horizon` steps beyond wherever training ended, using a
    model exported by export_model — with the P10-P90 interval (`lower`,
    `upper`) when the model was calibrated. future_exog_path (optional): a CSV with
    exactly `horizon` rows for this run's exogenous columns, in order —
    if omitted, each exogenous column's last known value is carried forward
    flat (an approximation: it forecasts the target, not the regressors).
    Writes date + forecast to out_path (default: alongside pkl_path)."""
    if not Path(pkl_path).exists():
        return json.dumps(
            {"error": f"no exported model at '{pkl_path}' — call export_model first"}
        )
    fm: ForecastModel = joblib.load(pkl_path)
    future_exog = None
    if future_exog_path:
        if not Path(future_exog_path).exists():
            return json.dumps({"error": f"no data file at '{future_exog_path}'"})
        future_exog = pd.read_csv(future_exog_path)
        if isinstance(fm, PanelForecastModel):
            counts = future_exog.groupby(fm.series_column).size() if fm.series_column in future_exog else None
            if counts is None or (counts != horizon).any():
                return json.dumps({"error": f"future_exog_path needs a {fm.series_column} column and exactly "
                                            f"horizon={horizon} rows per series"})
        elif len(future_exog) != horizon:
            return json.dumps(
                {
                    "error": f"future_exog_path has {len(future_exog)} row(s), expected exactly horizon={horizon}"
                }
            )

    forecast = fm.forecast(horizon, future_exog)
    out_path = out_path or str(
        Path(pkl_path).with_name(f"{Path(pkl_path).stem}_forecast.csv")
    )
    forecast.to_csv(out_path, index=False)
    # pd.Timestamp has no .item() (unlike numpy scalars), so _to_native
    # can't coerce it — stringify up front instead of fighting json.dumps.
    forecast[fm.date_column] = forecast[fm.date_column].astype(str)
    return json.dumps(
        {
            "model": fm.model_type,
            "horizon": horizon,
            "out_path": out_path,
            "forecast_summary": forecast["forecast"].describe().round(4).to_dict(),
            "interval": "lower/upper = P10-P90 from backtest errors" if "lower" in forecast else None,
            "preview": forecast.head(5).to_dict(orient="records"),
        },
        default=_to_native,
    )
