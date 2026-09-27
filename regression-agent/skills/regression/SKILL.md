---
name: regression
description: Train, tune, explain, and export a regressor once the data is prepared. Autonomous by default — compare_models' split-respecting CV ranking picks the model. Covers five model families (linear, ridge, random forest, XGBoost, histogram gradient boosting), log-space targets, hyperparameter search, explainability (permutation and SHAP), overfitting checks via CV-vs-test R^2 gap, and export with calibrated prediction intervals. Load once eda/preprocessing are done.
---

# Modeling & Export

Tools: `set_objective(run_id, objective, quantile)`, `compare_models(run_id, models)`, `train_model(run_id, model)`,
`tune_hyperparams(run_id, model, n_trials, cv_folds)`,
`explain_model(run_id, method, background_size)`, `compare_runs(run_ids)`,
`export_model(run_id, out_path, force)`, `predict(pkl_path, data_path)`.

Domain metric priorities and explainability expectations:
`references/domain-notes.md`.

## How the pipeline object works

Every tool operates on the run's **current pipeline** — one fitted object
(drop → impute → encode → [scale, for linear models] → regressor, wrapped
in a log1p/expm1 transform when `apply_target_transform` set one), saved by
`train_model` and **overwritten in place** by `tune_hyperparams`. Whichever
ran last for this `run_id` IS the current model for every tool downstream.
A refit clears the previous model's explanation, residuals, stability and
intervals (`invalidated_by_refit`) — re-run them; the readiness gate asks.

`explain_model` returns `feature_importance: [[feature, value], ...]`; to
plot it, pass that exact array to the visualization agent's
`plot_feature_importance`.

## Every CV respects the split — quote `cv_scheme`

compare / train / tune / stability / intervals all cross-validate on folds
that obey the run's split: forward-chained on a temporal run, grouped on an
entity run, shuffled otherwise. Quote `cv_scheme` beside every CV score. The
entity id and time column stay in the data and never reach the model.

## Choose the loss from the decision — before comparing models

`set_objective(run_id, objective, quantile)`:
- **squared_error** (default) — the conditional mean.
- **poisson** — also the mean, through a log link, for counts and
  non-negative, zero-heavy targets (calls, claims, usage). When predictions
  will be **summed** into a budget or plan, prefer it to a log1p target:
  log1p's back-transform under-predicts totals (`sum_ratio` below 1 shows
  by how much); poisson does not.
- **quantile** — the q-th quantile, when errors cost asymmetrically. If
  under-predicting costs k times over-predicting, the loss-minimising
  prediction is the k/(k+1) quantile (5:1 → 0.83). Scored on pinball loss;
  `quantile_hit_rate` (share of actuals at or below the prediction) must
  land near q. `random_forest` has no quantile loss; compare_models skips it.

Decide it, state the reason in one line, record it as an assumption.

## Metrics — and which one leads

R^2 (scale-free, the CV metric for mean objectives; pinball loss for
quantile ones), RMSE and MAE (target units), median absolute error (robust
to a few huge misses), `sum_ratio` (predicted total / actual total — what a
budget is set from), and MAPE only when the
target stays clear of zero (otherwise `mape_withheld` says why). Lead with
the metric the business decision is priced in — usually MAE in the
target's units ("off by 4.20 a month per subscriber"), RMSE when large
misses cost disproportionately. With a log1p target every metric is still
reported in the original units.

## Autonomous flow

1. **Compare** — `compare_models(run_id)` ranks `linear_regression`,
   `ridge`, `random_forest`, `xgboost`, `hist_gradient_boosting` on
   split-respecting CV R^2. That ranking IS the model choice; don't ask.
   Prefer the simpler model when the top two are within one CV std, or
   when the stakeholder needs coefficients (ridge).
2. **Train** — `train_model(run_id, winner)`. Surface CV R^2 ± std (with
   `cv_scheme`), test R^2 / MAE / RMSE, and flag `overfitting_warning` or a
   CV std above 0.1.
3. **Tune when the metric matters** — `tune_hyperparams(run_id, model)`
   overwrites the pipeline; state that you are doing it and why (the gap
   to the success bar, or a large CV std). The defaults are a fast pass;
   raise `n_trials` when the bar is close.
4. **Explain** — `explain_model` (permutation on the test fold by default;
   `method="shap"` for per-prediction reasoning — with a log1p target its
   values are in log units: rankings hold, magnitudes don't). Stop and
   flag an ID-like column, timestamp or row index among the top features,
   or one feature above 3× the next — likely leakage or a proxy.
5. **Residuals, stability, intervals** (`diagnostics`) — `check_residuals`,
   `check_model_stability`, and `calibrate_intervals` whenever the
   prediction will be acted on as a number.
6. `check_readiness` → `generate_report` → export → `log_run_to_mlflow`.

## Where the user decides

- **Export is always two-step.** `export_model(run_id)` with no `out_path`
  returns a suggested path without writing; ask the user to confirm it,
  then call again with the confirmed path. Never write silently.
- **`overfitting_warning` or a `blocked` readiness refuses export** unless
  `force=true` — surface `cv_to_test_gap` / the failed gates and ask; never
  pass `force=true` on your own.

## Predict and compare runs

`predict(pkl_path, data_path)` runs an exported model on new data in any
supported format (target column optional). It adds `prediction_lower` /
`prediction_upper` when the export carries calibrated intervals. Confirm
the new data has the training columns — the pipeline encodes and imputes
automatically but cannot recover missing or renamed columns.

`compare_runs(run_ids)` ranks already-trained runs by R^2 from their saved
metrics, no retraining — use it when several runs tried different cleaning
or features, and say which won and why.
