"""regression agent — `diagnostics` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import (
    _build_pipeline,
    _cv_sample,
    _evaluate,
    _load_meta,
    _load_split,
    _run_dir,
    _save_meta,
    _cv_scoring,
    _validation_folds,
    objective_of,
    split_keys,
)  # noqa: F401


@mcp.tool()
def record_business_context(
    run_id: str,
    domain: str,
    business_objective: str,
    target_definition: str,
    success_metric: str = "",
    success_threshold: float | None = None,
    success_criteria: str = "",
    target_provenance: str = "",
    assumptions: str = "",
    clarifications: str = "",
) -> str:
    """Record what this run is FOR, before modelling: the domain, the decision
    the prediction feeds, what the target column actually measures, and
    optionally the bar the model has to clear.

    The success bar is the part that matters most. If you record one, the
    readiness gate judges the model against it; if you don't, the gate reports
    that no bar exists. A threshold is NEVER derived at report time — a report
    announcing "target missed: 0.71 vs 0.80" for a target nobody set is worse
    than one with no bar at all. Ask the user for the number; don't invent it.

    success_metric: one of r2 / rmse / mae / median_ae / pinball_loss (the last
    for a quantile objective). Direction is implied (r2 higher is better, the
    others lower).
    success_criteria: the bar in business words ("MAE under 5 per subscriber-
    month, because a retention budget is set from it").
    target_provenance: how the target is produced and when it settles (billed
    revenue final 30 days after month end; claims reserved, then paid).
    assumptions / clarifications: one per line — every choice made without
    asking, and every question the user answered; the report shows both."""
    meta = _load_meta(run_id)
    try:
        target = gate_engine.record_success_target(
            meta,
            success_metric or None,
            success_threshold,
            SUPPORTED_SUCCESS_METRICS,
            direction=SUCCESS_METRIC_DIRECTION.get(success_metric, "at_least"),
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    meta["business_understanding"] = {
        "domain": domain,
        "business_objective": business_objective,
        "target_definition": target_definition,
        "success_target": target,
        "success_criteria": success_criteria or None,
        "target_provenance": target_provenance or None,
        "assumptions": [a.strip() for a in assumptions.split("\n") if a.strip()],
        "clarifications": [c.strip() for c in clarifications.split("\n") if c.strip()],
    }
    _save_meta(run_id, meta)
    return json.dumps(
        {
            "run_id": run_id,
            "recorded": meta["business_understanding"],
            "note": (
                "no success bar recorded — the readiness gate will report that none was set, and no threshold "
                "will be invented later"
                if target is None
                else "success bar recorded"
            ),
        }
    )


@mcp.tool()
def detect_data_leakage(run_id: str) -> str:
    """Screen the training fold for columns that give away the target: a
    feature correlating near-perfectly with it (a transformed copy, a
    post-outcome field), and identifier-shaped columns that can only memorize.

    Read-only. Re-run it after any step that changes the feature set —
    otherwise it describes columns that are no longer the ones being modelled,
    which the evidence_ordering gate will catch."""
    train, _, meta = _load_split(run_id)
    target = meta["target"]
    # Split keys never reach the model (the pipeline drops them) — screening
    # them would only tempt a drop that breaks grouped/temporal validation.
    train = train.drop(columns=split_keys(meta), errors="ignore")
    suspicious = {}
    y = train[target]
    for col in train.columns:
        if col == target or not pd.api.types.is_numeric_dtype(train[col]):
            continue
        series = train[col]
        if series.nunique(dropna=True) <= 1:
            continue
        rho = stats.spearmanr(series, y, nan_policy="omit").statistic
        if rho is not None and not np.isnan(rho) and abs(rho) >= LEAKAGE_CORRELATION:
            suspicious[col] = (
                f"|Spearman rho| {abs(round(float(rho), 4))} against the target — a copy of the "
            )
            suspicious[col] += "target rather than a predictor of it"
    identifiers = {c: s["reason"] for c, s in identifier_columns(train, target).items()}
    meta["leakage"] = {
        "clear": not suspicious and not identifiers,
        "target_correlated_columns": suspicious,
        "identifier_columns": identifiers,
    }
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, **meta["leakage"]})


@mcp.tool()
def acknowledge_identifier_column(run_id: str, column: str, justification: str) -> str:
    """Record that an identifier-shaped column is genuinely a feature in this
    domain. The only way past the no_identifier_in_model gate without dropping
    the column — and it needs a real reason, because the usual outcome of
    keeping one is a model that memorizes rows."""
    meta = _load_meta(run_id)
    try:
        gate_engine.acknowledge(meta, "identifiers", column, justification)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    _save_meta(run_id, meta)
    return json.dumps(
        {"run_id": run_id, "acknowledged": column, "justification": justification}
    )


@mcp.tool()
def train_baseline(run_id: str) -> str:
    """The naive predictors the model has to beat, scored on the held-out fold:
      mean         the training mean (the training quantile on a quantile run)
      persistence  each entity's PREVIOUS value — only when the run has both
                   an entity and a time column. On entity-per-period data
                   (next month's spend, this week's usage) "same as last time"
                   is the honest bar: beating the mean there is trivial.
    Every candidate is reported; the readiness gate judges the model against
    the STRONGEST. A model that loses to persistence has learned less than
    the obvious rule."""
    train, test, meta = _load_split(run_id)
    target = meta["target"]
    objective, q = objective_of(meta)
    strategy = {"strategy": "quantile", "quantile": q} if objective == "quantile" else {"strategy": "mean"}
    dummy = DummyRegressor(**strategy).fit(train.drop(columns=[target]), train[target])
    candidates = {strategy["strategy"]: _evaluate(dummy, test.drop(columns=[target]), test[target], meta)}

    group, time_col = meta.get("group_column"), meta.get("time_column")
    if group and time_col and {group, time_col} <= set(train.columns):
        both = pd.concat([train.assign(_fold=0), test.assign(_fold=1)], ignore_index=True)
        order = both[time_col] if pd.api.types.is_numeric_dtype(both[time_col]) \
            else shared_validation._parse_dates(both[time_col])
        both = both.assign(_order=order).sort_values([group, "_order"], kind="mergesort")
        previous = both.groupby(group, sort=False)[target].shift(1)
        held = both["_fold"] == 1
        fallback = float(dummy.predict(test.drop(columns=[target]).head(1))[0])

        class _Persistence:  # the previous value, or the naive constant for an entity's first period
            def predict(self, X):
                return previous[held].fillna(fallback).to_numpy()

        candidates["persistence"] = _evaluate(_Persistence(), None, both.loc[held, target], meta)
        candidates["persistence"]["rows_with_history_pct"] = round(float(previous[held].notna().mean()) * 100, 2)

    key, better = ("pinball_loss", min) if objective == "quantile" else ("r2", max)
    kind = better(candidates, key=lambda k: candidates[k][key])
    meta["baseline_comparison"] = {**candidates[kind], "kind": kind, "candidates": candidates}
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, "strongest": kind, "baselines": candidates,
                       "note": "every naive baseline scored on the held-out fold; the model is judged against "
                               "the strongest"})


@mcp.tool()
def check_residuals(run_id: str) -> str:
    """Diagnose HOW this run's current model is wrong, not just how much.

    Two failures a single RMSE hides: a systematic bias (the model is off in
    one direction everywhere), and heteroscedastic error (accurate for small
    values, wild for large ones — the average looks fine and the model is
    unusable at the end of the range that usually matters)."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    _, test, meta = _load_split(run_id)
    target = meta["target"]
    y_true = test[target]
    y_pred = joblib.load(pipeline_path).predict(test.drop(columns=[target]))
    residuals = y_true - y_pred

    spread = float(y_true.std()) or 1.0
    mean_residual = float(residuals.mean())
    bias_ratio = round(abs(mean_residual) / spread, 4)
    rho = stats.spearmanr(np.abs(residuals), y_pred, nan_policy="omit").statistic
    rho = 0.0 if rho is None or np.isnan(rho) else float(rho)

    result = {
        "mean_residual": round(mean_residual, 4),
        "bias_ratio": bias_ratio,
        "biased": bias_ratio > RESIDUAL_BIAS_RATIO,
        "abs_residual_vs_prediction_rho": round(rho, 4),
        "heteroscedastic": abs(rho) > HETEROSCEDASTICITY_RHO,
        "residual_std": round(float(residuals.std()), 4),
    }
    meta["residuals"] = result
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, **result})


@mcp.tool()
def check_model_stability(run_id: str, repeats: int = 3, folds: int = 5) -> str:
    """Refit this run's model across repeated CV splits — folds that obey the
    run's split, as every CV here does — and report the spread of R^2. One
    split's score is one split's luck; a model whose R^2 swings 0.2 across
    splits has not been measured, it has been sampled once. On a temporal
    run the per-fold scores are a time series: read them for decay."""
    train, _, meta = _load_split(run_id)
    if not meta.get("model"):
        return json.dumps(
            {"error": "no model trained for this run yet — call train_model first"}
        )
    target = meta["target"]
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_cv, y_cv, cv_n_rows = _cv_sample(X_train, y_train, meta)
    scoring, metric, sign = _cv_scoring(meta)
    scores = []
    # Forward-chained folds on a temporal run have no randomness: one pass, and
    # the spread is across time — a score falling fold after fold is decay.
    for seed in range(1 if meta.get("time_column") else repeats):
        cv, cv_scheme = _validation_folds(X_cv, y_cv, meta, n_splits=folds, seed=seed)
        scores.extend((sign * cross_val_score(_build_pipeline(meta, meta["model"], X_cv), X_cv, y_cv,
                                              cv=cv, scoring=scoring)).tolist())
    scores = np.array(scores)
    # R^2 spread is absolute; a loss has no fixed scale, so its spread is relative.
    spread = scores.std() if metric == "r2" else scores.std() / max(abs(scores.mean()), 1e-12)
    result = {
        "cv_metric": metric,
        "cv_scheme": cv_scheme,
        "cv_computed_on_n_rows": cv_n_rows,
        f"{metric}_mean": round(float(scores.mean()), 4),
        f"{metric}_std": round(float(scores.std()), 4),
        f"{metric}_min": round(float(scores.min()), 4),
        f"{metric}_max": round(float(scores.max()), 4),
        "per_fold": [round(float(v), 4) for v in scores] if meta.get("time_column") else None,
        "splits": int(scores.size),
        "stable": bool(spread <= (STABILITY_STD_CONCERN if metric == "r2" else 2 * STABILITY_STD_CONCERN)),
    }
    meta["stability"] = result
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, **result})




@mcp.tool()
def analyze_errors(run_id: str, top: int = 8) -> str:
    """WHERE the current model is wrong, not just how much: mean absolute
    error and bias per segment (each feature's quartiles, or its most
    frequent categories), ranked by how much worse than average the segment
    is, plus the largest single misses. Measured on OUT-OF-FOLD predictions
    for training rows (folds that obey the run's split) — acting on segments
    found on the test fold would be choosing on the test fold.

    bias > 0 means the model UNDER-predicts that segment. A segment far above
    average points at a missing feature (the mechanism that segment runs on)
    or a mixed population that needs its own model; a bias that grows with
    the target is the heteroscedastic case — a log1p target or a poisson
    objective usually fixes it. Read-only."""
    from sklearn.base import clone

    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps({"error": "no fitted model for this run yet — call train_model first"})
    pipeline = joblib.load(pipeline_path)
    train, _, meta = _load_split(run_id)
    target = meta["target"]
    X, y, n_rows = _cv_sample(train.drop(columns=[target]), train[target], meta)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    folds, scheme = _validation_folds(X, y, meta)
    pred = np.full(len(X), np.nan)
    for fit_idx, val_idx in folds:
        pred[val_idx] = clone(pipeline).fit(X.iloc[fit_idx], y.iloc[fit_idx]).predict(X.iloc[val_idx])
    scored = ~np.isnan(pred)
    X, actual, pred = X[scored].reset_index(drop=True), y[scored].to_numpy(dtype=float), pred[scored]
    err = actual - pred
    overall = float(np.mean(np.abs(err)))
    min_rows = max(30, int(0.02 * len(X)))
    keys = set(split_keys(meta)) | set(meta.get("dropped_columns") or [])
    segments = []
    for col in X.columns:
        if col in keys:
            continue
        x = X[col]
        if pd.api.types.is_numeric_dtype(x) and x.nunique() > 10:
            labels = pd.qcut(x.rank(method="first"), 4, labels=["q1 (lowest)", "q2", "q3", "q4 (highest)"])
        else:
            common = x.astype(str).value_counts().index[:10]
            labels = x.astype(str).where(x.astype(str).isin(common), "(other)")
        frame = pd.DataFrame({"segment": labels, "abs": np.abs(err), "err": err})
        for seg, grp in frame.groupby("segment", observed=True):
            if len(grp) >= min_rows:
                segments.append({"segment": f"{col} = {seg}", "rows": int(len(grp)),
                                 "mae": round(float(grp["abs"].mean()), 4),
                                 "mae_vs_overall": round(float(grp["abs"].mean()) / overall, 3) if overall else None,
                                 "bias": round(float(grp["err"].mean()), 4)})
    segments.sort(key=lambda r: -(r["mae_vs_overall"] or 0))
    worst = np.argsort(-np.abs(err))[:5]
    result = {
        "measured_on": f"{int(scored.sum())} out-of-fold training predictions — {scheme}",
        "cv_computed_on_n_rows": n_rows,
        "overall_mae": round(overall, 4),
        "overall_bias": round(float(err.mean()), 4),
        "worst_segments": segments[:top],
        "largest_misses": [{"actual": round(float(actual[i]), 4), "predicted": round(float(pred[i]), 4)}
                           for i in worst],
    }
    meta["error_analysis"] = result
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, **result})

@mcp.tool()
def calibrate_intervals(run_id: str, coverage: float = 0.9) -> str:
    """Prediction intervals for this run's CURRENT model, by split-conformal
    calibration: refit the model on each validation fold's fitting rows
    (folds that obey the run's split), collect the absolute residuals on the
    rows it did not see, and take their `coverage` quantile as the interval
    half-width. Then MEASURE the promised coverage once on the test fold —
    an interval claimed at 90% that covers 78% of test rows is reported as
    such, not as 90%.

    Use it whenever a number will be acted on (a budget, a capacity plan, a
    price): a point prediction without its uncertainty invites false
    precision. The half-width is constant across rows — honest on average,
    too wide for easy rows and too narrow for hard ones when errors grow with
    the prediction (check_residuals' heteroscedasticity flag says when); a
    log1p target transform helps exactly that case. Exported with the model;
    predict adds prediction_lower / prediction_upper."""
    from sklearn.base import clone

    if not 0.5 <= coverage < 1:
        return json.dumps({"error": "coverage must be in [0.5, 1)"})
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps({"error": "no fitted model for this run yet — call train_model first"})
    pipeline = joblib.load(pipeline_path)
    train, test, meta = _load_split(run_id)
    target = meta["target"]
    X, y, n_rows = _cv_sample(train.drop(columns=[target]), train[target], meta)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    folds, scheme = _validation_folds(X, y, meta)
    residuals = []
    for fit_idx, val_idx in folds:
        fold_model = clone(pipeline).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        residuals.append(np.abs(y.iloc[val_idx].to_numpy() - fold_model.predict(X.iloc[val_idx])))
    residuals = np.concatenate(residuals)
    level = min(1.0, np.ceil((len(residuals) + 1) * coverage) / len(residuals))
    half_width = float(np.quantile(residuals, level))
    test_pred = pipeline.predict(test.drop(columns=[target]))
    covered = float(np.mean(np.abs(test[target].to_numpy() - test_pred) <= half_width))
    result = {
        "coverage_target": coverage,
        "half_width": round(half_width, 6),
        "test_coverage": round(covered, 4),
        "calibrated_on": f"{len(residuals)} out-of-fold residuals — {scheme}",
        "cv_computed_on_n_rows": n_rows,
        "undercovers": bool(covered < coverage - 0.05),
    }
    meta["prediction_interval"] = result
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, **result})

@mcp.tool()
def check_readiness(run_id: str) -> str:
    """Deterministic readiness gate — the answer to "is this run actually
    finished and safe to believe", computed from this run's artifacts rather
    than from anything the agent asserts.

    Returns every required gate's status (pass / fail / not_run /
    not_applicable) with the evidence behind it, plus overall_status:
      `blocked`    — a gate found positive evidence of a problem
      `incomplete` — required evidence is missing (NOT the same as passing)
      `ready`      — every mechanical gate passed

    Call before export_model, which refuses a blocked run.
    `mechanical_completeness` is the fraction of deterministic gates satisfied
    — execution correctness only, never a claim that the modelling choices or
    the business framing are correct."""
    return json.dumps(compute_readiness(run_id))


# ── The standard report (core/stages.py) ─────────────────────────────────────
from core import stages  # noqa: E402

_, _, _, _, generate_report = stages.register_tools(
    mcp,
    "regression",
    load_meta=_load_meta,
    save_meta=_save_meta,
    run_dir=_run_dir,
    required_gates=REQUIRED_GATES,
    supported_metrics=SUCCESS_METRIC_DIRECTION,
    compute_readiness=compute_readiness,
    report_metrics=lambda meta: (
        meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
    ),
    report_sections=lambda meta: [
        (
            "Diagnostics",
            [
                f"- Baseline (predict the mean): {meta.get('baseline_comparison') or 'not run'}",
                f"- Residuals: {meta.get('residuals') or 'not checked'}",
                f"- Stability: {meta.get('stability') or 'not checked'}",
                f"- Top features: {(meta.get('explainability') or {}).get('top_features') or 'explain_model not run'}",
            ],
        )
    ],
    include=("generate_report",),
)
