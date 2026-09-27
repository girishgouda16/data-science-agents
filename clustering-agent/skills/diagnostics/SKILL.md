---
name: diagnostics
description: Business context, the clustering data screen, and the deterministic readiness gate. Load right after prepare_dataset — BEFORE train_model — because the gates check the order steps ran in and export_model refuses a blocked run.
---

# Diagnostics & the readiness gate

Tools: `detect_data_leakage(run_id)`, `acknowledge_identifier_column(run_id, column, justification)`,
`acknowledge_gate(run_id, gate, justification)`, `check_readiness(run_id)`.

Work the way a senior data scientist does — in this order (the
`evidence_ordering` gate checks it):

1. **`record_business_context`** (the `business-understanding` skill) —
   what each segment will get, how many segments can be acted on, the
   outcome they must differ on, and the bar. Never invent a bar.
2. **`detect_data_leakage`** — BEFORE training, and again after any column
   is dropped or imputed. Flags identifier-shaped columns (distance on an ID is meaningless) and constant columns — drop them.
3. Clean, `propose_k`, `train_model`, then **`explain_model`** — what each
   segment IS (distinguishing features), whether it matters (outcome eta²
   on the profile columns) and whether it can be described (rules
   accuracy).
4. **`check_readiness`** — `blocked` = a gate found a problem (fix it; export
   refuses); `incomplete` = evidence never produced — NOT a pass, go run it;
   `ready` = every mechanical gate passed. Loop on `not_run_gates`.
5. `generate_report` (the `reporting` skill), then `log_run_to_mlflow` (the
   `mlops` skill) and tell the user the registry version.

Gates: business_context, leakage_screened, no_identifier_in_model, structure_found (silhouette > 0.25 — below it the segments are arbitrary cuts; tell the user rather than presenting them), holdout_stability (bootstrap stability ARI >= 0.7 — segments that don't come back on resampled data are artefacts of one sample), explainability_run, evidence_ordering, success_criteria.

A FAILED gate is passed only with a person's reason: `acknowledge_gate` (or
`acknowledge_identifier_column` for a flagged column). Ask the user; never
record an acknowledgement nobody gave you.
