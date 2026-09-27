---
name: reporting
description: Generate the standard anomaly report — verdict first, business framing, data, model, metrics, every readiness gate with its evidence, and the execution ledger. Load at the end of a run or whenever the user asks for a report or summary.
---

# Reporting

Tool: `generate_report(run_id)` — writes `report.md` into the run (and
`log_run_to_mlflow` ships it with the version). Built only from the run's
artifacts: it states the readiness verdict in its first line, so an
unfinished run reads as unfinished. Hand it to the user as-is; don't
paraphrase the verdict into something rosier. Call it after
`check_readiness`, and again after anything that changes the model.
