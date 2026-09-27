---
name: diagnostics
description: Business context, leakage screen, baseline, residual and stability checks, and the deterministic readiness gate for a regression run. Load right after prepare_dataset — before training — because export_model refuses a run whose gates found a problem.
---

# Diagnostics & the readiness gate

Tools: `detect_data_leakage(run_id)`, `analyze_errors(run_id)`, `calibrate_intervals(run_id, coverage)`,
`acknowledge_identifier_column(run_id, column, justification)`,
`train_baseline(run_id)`, `check_residuals(run_id)`,
`check_model_stability(run_id)`, `check_readiness(run_id)`.

`check_readiness` computes from the run's artifacts whether the work behind a
model actually happened. `export_model` (and so champion promotion) refuses a
`blocked` run. Three statuses: `ready`; `blocked` — a gate found positive
evidence of a problem (fix it, don't narrate around it); `incomplete` —
required evidence was never produced (absence of evidence is NOT a pass).

The order matters — the `evidence_ordering` gate checks it:

1. `record_business_context` (the `business-understanding` skill) — what
   the prediction is FOR, what the target measures, when it settles, and the
   success bar. Ask for the bar; never invent one.
2. `detect_data_leakage` — BEFORE training, and again after any column is
   dropped, imputed or added. Training without a screen blocks the run. Drop
   strong identifier columns; a flagged column the user says is a genuine
   feature needs `acknowledge_identifier_column` with their reason.
3. `train_baseline` — the naive predictors: the mean (or the training
   quantile on a quantile run) and, when the run has an entity AND a time
   column, **persistence** — each entity's previous value. On
   entity-per-period data persistence is the honest bar; the gate judges the
   model against the strongest baseline.
4. train / tune (the `regression` skill), explain_model.
5. `analyze_errors` — where the model is wrong: MAE and bias per segment
   on out-of-fold predictions (never the test fold). A segment far above
   average is a missing mechanism or a mixed population; act on it with a
   feature, not a filter. Then `check_residuals` (bias, heteroscedasticity), `check_model_stability`
   (same split-respecting folds as every CV; on a temporal run the per-fold
   R^2 is a time series — falling fold after fold is decay), and
   `calibrate_intervals(run_id, coverage=0.9)` whenever the number will be
   acted on: split-conformal intervals from out-of-fold residuals, with the
   coverage MEASURED on test (`undercovers` flags a promise the test fold
   doesn't keep). Heteroscedastic residuals make a constant-width interval
   too narrow for large predictions — a log1p target is the usual fix.
6. `check_readiness` — loop on `not_run_gates` until it returns `ready`;
   then export / log to MLflow (the `mlops` skill).
