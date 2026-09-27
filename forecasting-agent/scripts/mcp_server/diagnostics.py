"""forecasting agent — `diagnostics` tools: the business framing, the
exogenous-leak screen, the naive baseline every forecast must beat, and the
deterministic readiness gate. Mechanism is core/gates.py + core/stages.py;
what "finished and safe to believe" MEANS for a forecaster lives here."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import (
    _backtest,
    _score_on_test,
    _current_metrics,
    _exog_columns,
    _fit_forecast_model,
    _load_meta,
    _load_split,
    _mase_scale,
    _rmse_mae_mape,
    _run_dir,
    _save_meta,
)  # noqa: F401

from core import gates as gate_engine  # noqa: E402
from core import stages  # noqa: E402

REQUIRED_GATES = (
    "business_context",
    "leakage_screened",
    "baseline_beaten",
    "no_overfitting",
    "explainability_run",
    "evidence_ordering",
    "success_criteria",
)
SUCCESS_METRICS = {"rmse": "at_most", "mae": "at_most", "mape": "at_most", "wape": "at_most", "mase": "at_most"}
LEAK_CORRELATION = 0.98


def measured_metric(meta: dict, metric: str):
    return _current_metrics(meta).get(metric)


@mcp.tool()
def detect_data_leakage(run_id: str) -> str:
    """Screen exogenous columns before training: one that moves in lockstep
    with the target at the SAME timestamp (a copy or a sum that contains it),
    or with the target one step AHEAD (it encodes the future — at forecast
    time it will not be known). Univariate runs pass trivially: the model only
    sees the target's own history. Call BEFORE train_model."""
    train, _, meta = _load_split(run_id)
    target = train[meta["target"]].astype(float)
    # the NEXT value of the same series, never the next row of another one
    following = target.groupby(train[meta["series_column"]]).shift(-1) if meta.get("panel") else target.shift(-1)
    same_time, future = {}, {}
    for col in _exog_columns(meta):
        x = pd.to_numeric(train[col], errors="coerce")
        if x.notna().sum() < 3 or x.nunique() < 2:
            continue
        same, ahead = abs(x.corr(target)), abs(x.corr(following))
        if same >= LEAK_CORRELATION:
            same_time[col] = round(float(same), 4)
        elif ahead >= LEAK_CORRELATION:
            future[col] = round(float(ahead), 4)
    result = {
        "run_id": run_id,
        "exogenous_columns": _exog_columns(meta),
        "target_copy_columns": same_time,
        "future_leak_columns": future,
        "clear": not same_time and not future,
    }
    meta["leakage"] = result
    _save_meta(run_id, meta)
    return json.dumps(result, default=str)


@mcp.tool()
def train_baseline(run_id: str) -> str:
    """Score the naive seasonal forecast (repeat the last seasonal cycle) on
    the same held-out period AND the same backtest origins as the models.
    Every learned model must beat it; one that does not is complexity that
    buys nothing. Doesn't touch the run's model."""
    train, test, meta = _load_split(run_id)
    fm = _fit_forecast_model(train, meta, "naive_seasonal", {})
    metrics, _ = _score_on_test(fm, train, test, meta)
    result = {"model": "naive_seasonal", **metrics}
    bt = _backtest(train, meta, "naive_seasonal", {})
    result.update(backtest_rmse_mean=bt["rmse_mean"], backtest_origins=bt["origins"])
    meta["baseline_comparison"] = result
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, **result})


def pooled_rmse(metrics: dict) -> float | None:
    """RMSE over every window a forecaster was judged on — the backtest
    origins and the held-out period — so one lucky or unlucky 4-week window
    does not decide whether a model beats the naive."""
    test, bt, k = metrics.get("rmse"), metrics.get("backtest_rmse_mean"), metrics.get("backtest_origins") or 0
    if test is None:
        return None
    if bt is None or not k:
        return test
    return round(float(np.sqrt((k * bt**2 + test**2) / (k + 1))), 4)


def compute_readiness(run_id: str) -> dict:
    """Every required gate's status for this run, computed from its artifacts."""
    train, _, meta = _load_split(run_id)
    metrics = _current_metrics(meta)
    acknowledged = set(meta.get("acknowledged_identifiers") or {})
    checks: dict = {}
    bu = meta.get("business_understanding") or {}
    checks["business_context"] = (
        gate_engine.gate(
            gate_engine.PASS,
            f"objective recorded; domain='{bu.get('domain') or 'not declared'}'",
        )
        if bu.get("business_objective") and bu.get("target_definition")
        else gate_engine.gate(
            gate_engine.NOT_RUN,
            "record_business_context was not called — no horizon, unit or decision on file",
        )
    )

    leakage = meta.get("leakage")
    live = [
        c
        for c in list((leakage or {}).get("target_copy_columns") or {})
        + list((leakage or {}).get("future_leak_columns") or {})
        if c in _exog_columns(meta) and c not in acknowledged
    ]
    if not leakage:
        checks["leakage_screened"] = gate_engine.gate(
            gate_engine.NOT_RUN, "detect_data_leakage was never called for this run"
        )
    elif live:
        checks["leakage_screened"] = gate_engine.gate(
            gate_engine.FAIL,
            f"exogenous column(s) that copy the target or encode its future still feed the model: {live} — "
            "drop them and retrain, or acknowledge with the reason they are known at forecast time",
        )
    else:
        checks["leakage_screened"] = gate_engine.gate(
            gate_engine.PASS, "detect_data_leakage ran; no leaking exogenous column"
        )

    baseline = meta.get("baseline_comparison")
    if not metrics:
        checks["baseline_beaten"] = gate_engine.gate(
            gate_engine.NOT_RUN, "no model has been trained for this run"
        )
    elif meta.get("model") == "naive_seasonal":
        checks["baseline_beaten"] = gate_engine.gate(
            gate_engine.NOT_APPLICABLE, "the model IS the naive seasonal baseline"
        )
    elif not baseline:
        checks["baseline_beaten"] = gate_engine.gate(
            gate_engine.NOT_RUN,
            "train_baseline was never called — no naive forecast to beat",
        )
    else:
        model_err, naive_err = pooled_rmse(metrics), pooled_rmse(baseline)
        windows = (metrics.get("backtest_origins") or 0) + 1
        checks["baseline_beaten"] = gate_engine.gate(
            gate_engine.PASS if model_err is not None and naive_err is not None and model_err < naive_err
            else gate_engine.FAIL,
            f"RMSE over {windows} window(s) at the horizon (backtest origins + held-out period): {model_err} vs "
            f"naive seasonal {naive_err}; held-out alone {metrics.get('rmse')} vs {baseline['rmse']} "
            f"(MASE {metrics.get('mase')} vs {baseline.get('mase')}; WAPE {metrics.get('wape')}% vs {baseline.get('wape')}%)",
        )

    checks["no_overfitting"] = (
        gate_engine.gate(gate_engine.NOT_RUN, "no model has been trained for this run")
        if not metrics
        else gate_engine.gate(
            (
                gate_engine.FAIL
                if metrics.get("overfitting_warning")
                else gate_engine.PASS
            ),
            f"backtest-vs-test RMSE gap {metrics.get('cv_to_test_gap')} (threshold {OVERFIT_GAP_THRESHOLD})",
        )
    )

    checks["explainability_run"] = (
        gate_engine.gate(gate_engine.PASS, "explain_model explained the current model")
        if stages.ran_after_training(meta, "explain_model")
        else gate_engine.gate(
            gate_engine.NOT_RUN, "explain_model has not explained the current model"
        )
    )

    checks["evidence_ordering"] = gate_engine.check_screen_ordering(
        meta, "detect_data_leakage", ("apply_drop_columns", "apply_imputation")
    )
    checks["success_criteria"] = gate_engine.evaluate_success_criteria(
        meta, measured_metric, tuple(SUCCESS_METRICS)
    )
    return gate_engine.readiness(
        run_id, stages.apply_acknowledgements(checks, meta), REQUIRED_GATES
    )


def _report_sections(meta: dict) -> list:
    m = _current_metrics(meta)
    return [
        (
            "Forecast setup",
            [
                f"- Frequency: {meta.get('freq')}, seasonal period {meta.get('seasonal_periods')} "
                f"(chosen by {meta.get('seasonal_periods_basis') or 'frequency default'}; autocorrelation at candidate "
                f"seasons: {meta.get('seasonality') or 'n/a'})",
                f"- Horizon: {meta.get('horizon')} steps — the held-out period and every backtest origin forecast "
                "this far ahead",
                *([f"- Rows: {meta.get('duplicates_aggregated')} duplicate-timestamp rows aggregated by "
                   f"{meta.get('aggregate')}"] if meta.get("duplicates_aggregated") else []),
                *([f"- Series: {meta.get('series')}"] if meta.get("series") else []),
                *([f"- Multi-series: {m.get('series')} series scored; the model beats each series' own seasonal "
                   f"naive on {m.get('series_beating_naive_share')} of them (median MASE {m.get('mase')}); worst by "
                   f"WAPE: {[w.get('series') for w in m.get('worst_series_by_wape') or []]}; left out: "
                   f"{meta.get('series_left_out')}"] if meta.get("panel") else []),
                *([f"- {meta.get('gaps_inserted')} missing period(s) inserted to make the grid regular, filled by "
                   f"{meta.get('imputation', {}).get(meta.get('target'), 'NOT YET FILLED')}"]
                  if meta.get("gaps_inserted") else []),
                f"- Prediction interval: P{int(50 - 50 * (m.get('interval_level') or 0.8))}-"
                f"P{int(50 + 50 * (m.get('interval_level') or 0.8))}, held-out coverage {m.get('interval_coverage')} "
                "(should be near the nominal level)",
                f"- Scale-free error: MASE {m.get('mase')} (< 1 beats the seasonal naive), WAPE {m.get('wape')}%, "
                f"bias {m.get('bias_pct')}% (+ = over-forecast)",
                f"- Lags {meta.get('lags')}, rolling windows {meta.get('rolling_windows')}",
                f"- Naive seasonal baseline on the same period: {meta.get('baseline_comparison') or 'not run'}",
                "- The exported model is refit on the FULL history so it forecasts from the last "
                "observed date; the metrics above are from the train-only evaluation fit.",
            ],
        )
    ]


(
    record_business_context,
    acknowledge_identifier_column,
    acknowledge_gate,
    check_readiness,
    generate_report,
) = stages.register_tools(
    mcp,
    "forecasting",
    load_meta=_load_meta,
    save_meta=_save_meta,
    run_dir=_run_dir,
    required_gates=REQUIRED_GATES,
    supported_metrics=SUCCESS_METRICS,
    compute_readiness=compute_readiness,
    report_metrics=_current_metrics,
    report_sections=_report_sections,
)
