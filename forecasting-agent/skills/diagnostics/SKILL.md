---
name: diagnostics
description: Business context, the forecasting data screen, the naive baseline, and the deterministic readiness gate. Load right after prepare_dataset — BEFORE train_model — because the gates check the order steps ran in and export_model refuses a blocked run.
---

# Diagnostics & the readiness gate

Tools: `record_business_context(run_id, business_objective, target_definition, success_criteria, domain, success_metric, success_threshold, direction, assumptions, clarifications)`,
`detect_data_leakage(run_id)`, `train_baseline(run_id)`, `acknowledge_identifier_column(run_id, column, justification)`,
`acknowledge_gate(run_id, gate, justification)`, `check_readiness(run_id)`.

Work the way a senior data scientist does — in this order (the
`evidence_ordering` gate checks it):

1. **`record_business_context`** — what the output is FOR, the unit, and the
   success bar (wape / mase / rmse / mae / mape — lower is better). Ask the user for the bar in one question; never
   invent one. Record every assumption you made instead of asking.
2. **`detect_data_leakage`** — BEFORE training, and again after any column
   is dropped or imputed. Flags exogenous columns that copy the target or track its NEXT value (unknown at forecast time).
3. Clean, **`train_baseline`** (naive seasonal on the same held-out period — every model must beat it), `compare_models` (walk-forward backtest), `train_model`/`tune_hyperparams`, `explain_model`.
4. **`check_readiness`** — `blocked` = a gate found a problem (fix it; export
   refuses); `incomplete` = evidence never produced — NOT a pass, go run it;
   `ready` = every mechanical gate passed. Loop on `not_run_gates`.
5. `generate_report` (the `reporting` skill), then `log_run_to_mlflow` (the
   `mlops` skill) and tell the user the registry version.

Gates: business_context, leakage_screened, baseline_beaten, no_overfitting, explainability_run, evidence_ordering, success_criteria.

A FAILED gate is passed only with a person's reason: `acknowledge_gate` (or
`acknowledge_identifier_column` for a flagged column). Ask the user; never
record an acknowledgement nobody gave you.
