---
name: diagnostics
description: Business context, the anomaly data screen and label-leak screen, and the deterministic readiness gate. Load right after prepare_dataset — BEFORE train_model — because the gates check the order steps ran in and export_model refuses a blocked run.
---

# Diagnostics & the readiness gate

Tools: `record_business_context(run_id, business_objective, target_definition, success_criteria, domain, success_metric, success_threshold, direction, assumptions, clarifications)`,
`detect_data_leakage(run_id)`, `acknowledge_identifier_column(run_id, column, justification)`,
`acknowledge_gate(run_id, gate, justification)`, `check_readiness(run_id)`.

Work the way a senior data scientist does — in this order (the
`evidence_ordering` gate checks it):

1. **`record_business_context`** — what the output is FOR, the unit, and the
   success bar — only when a label column exists: `precision_at_budget` /
   `recall_at_budget` (the review queue), `average_precision`, or roc_auc /
   precision / recall / f1. Ask the user for the bar in one question; never
   invent one. Record every assumption you made instead of asking.
2. **`detect_data_leakage`** — BEFORE training, and again after any column
   is dropped or imputed. Flags identifier columns and, with labels, any feature that alone nearly IS the label (a case-status field filled in after the investigation) — drop it or ask where it comes from.
3. Clean, **`propose_contamination`** before training (it decides how many rows get flagged — from review capacity), `compare_detectors`, `train_model`, `explain_model`.
4. **`check_readiness`** — `blocked` = a gate found a problem (fix it; export
   refuses); `incomplete` = evidence never produced — NOT a pass, go run it;
   `ready` = every mechanical gate passed. Loop on `not_run_gates`.
5. `generate_report` (the `reporting` skill), then `log_run_to_mlflow` (the
   `mlops` skill) and tell the user the registry version.

Gates: business_context, leakage_screened, no_identifier_in_model, contamination_justified, detection_quality (with labels: held-out ROC-AUC >= 0.55 AND the review queue at the budget >= 2x the base rate; without labels not_applicable — a person reviews the queue before anyone acts), alerts_stable (the top alerts overlap >= 50% across refits on resampled data), explainability_run, evidence_ordering, success_criteria.

A FAILED gate is passed only with a person's reason: `acknowledge_gate` (or
`acknowledge_identifier_column` for a flagged column). Ask the user; never
record an acknowledgement nobody gave you.
