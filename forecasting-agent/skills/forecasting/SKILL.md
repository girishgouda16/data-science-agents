---
name: forecasting
description: Compare, train, tune, explain, export and forecast a time-series model once the series is clean — rolling-origin backtests at the business horizon, MASE / WAPE / bias against the seasonal naive, a P10-P90 prediction interval calibrated on backtest errors, model-specific explanations, and the overfitting check. Load once eda/preprocessing are done and the user wants a forecast.
---

# Modeling, Export & Forecasting

Tools: `compare_models(run_id, models)`, `train_model(run_id, model)`,
`tune_hyperparams(run_id, model, n_trials, cv_folds)`,
`explain_model(run_id, method)`, `compare_runs(run_ids)`,
`export_model(run_id, out_path, force)`,
`predict(pkl_path, horizon, future_exog_path, out_path)`.

Domain model choice and metrics (retail, logistics, ops, telecom):
`references/domain-notes.md`, appended below.

## How the model is judged

Every score is a forecast made from history only, `horizon` steps ahead —
the held-out period is the last `horizon` steps, and the backtest forecasts
the same distance from each of the last few origins in the training fold,
recursively (step 2 uses step 1's prediction, as production will). Read:

- **MASE** — error relative to the seasonal naive; < 1 means the model
  earns its complexity. **WAPE** — total absolute error / total volume,
  the number to quote for traffic and demand. **bias_pct** — + means
  systematically high; a plan built on a biased forecast misses by that
  much. RMSE / MAE in the target's units; MAPE only when no actual is near
  zero (it is withheld otherwise — say why).
- **interval_coverage** — the share of held-out actuals inside the P10-P90
  band; near 0.8 is calibrated, well below means the band is too narrow
  to plan on.
- **cv_to_test_gap** — test RMSE vs the backtest; a large gap
  (`overfitting_warning`) means the backtest flattered the model or the
  held-out period is different (a break, an event).

## Models

- **`naive_seasonal`** — repeats the last season (`train_baseline` scores
  it on the held-out period). The bar every model must clear: with the
  right season it is strong — hourly traffic's weekly naive often lands
  within 10% WAPE.
- **`exponential_smoothing`** — Holt-Winters: level, trend, season.
  Default when there are no exogenous drivers; its components explain
  themselves ("rising 3% a week").
- **`xgboost`** — lag, rolling and calendar features (hour, weekday,
  weekend, week, month) plus exogenous columns, forecast recursively. For
  known-future drivers (events, holidays, launches) and interactions a
  decomposition misses.

## Many series in one run (`each_series`)

- `naive_seasonal` and `exponential_smoothing` fit **each series on its
  own**; `xgboost` fits **one global model on all of them**, each series
  scaled by its own mean, so hundreds of short or noisy series share the
  daily/weekly shape and calendar effects (the M5-style approach). Compare
  both: per-series ETS wins on long, clean, distinct series; the global
  model wins on many short or noisy ones and when series share drivers.
- Metrics are pooled across series (RMSE, WAPE, bias); `mase` is the
  median series' MASE against its own seasonal naive, with
  `series_beating_naive_share` and `worst_series_by_wape` — name the worst
  series to the user, they are where the forecast needs a human.
- The interval is calibrated in each series' own scale; `predict` returns
  one row per series per step. Gaps are filled within each series
  (`apply_imputation`), never across series.

## Steps

1. **Compare** — `compare_models(run_id)` backtests every model at the
   horizon and ranks by backtest RMSE (read-only). Always include
   `naive_seasonal`; the gap to it IS the case for the model.
2. **Train** — `train_model(run_id, model)` with the best-ranked model
   (decide and record why). Report MASE, WAPE, bias, coverage and the gap.
3. **Tune** (optional) — `tune_hyperparams(run_id, model)` searches on the
   backtest and overwrites the current model. Worth it only when the
   model is close to the bar.
4. **Explain** — `explain_model(run_id)`: xgboost → permutation importance
   of lags / calendar / drivers; exponential_smoothing → its fitted
   level / trend / season; naive → nothing to rank, and say so plainly.
5. **Export** — two steps, always: `export_model(run_id)` returns a
   suggested path; confirm it with the user, then write. The export is
   refit on the full history (it forecasts from the last observed date)
   and carries the calibrated interval. Refused on `overfitting_warning`
   or a blocked readiness unless `force=true` — surface why and ask; never
   pass `force=true` yourself.
6. **Forecast** — `predict(pkl_path, horizon)`: `forecast`, `lower`,
   `upper` per step (clipped at 0 for a series that never went below it).
   Exogenous columns need their future values (`future_exog_path`);
   without them the last value is carried flat — say it is an
   approximation.
7. `compare_runs(run_ids)` compares runs (models, cleaning choices) by
   RMSE, MASE, WAPE and coverage.
