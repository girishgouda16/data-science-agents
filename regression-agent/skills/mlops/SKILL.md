---
name: mlops
description: Track a trained regression run in MLflow, compare it with earlier versions, and promote one to champion/challenger for serving. Load after a model is trained (and exported or ready to export) or whenever the user asks to log, compare, version, promote, roll back or serve a model.
---

# MLOps — log, compare, promote

Tools: `log_run_to_mlflow(run_id)`, `list_model_versions(name|run_id)`,
`compare_model_versions(name|run_id, version_a, version_b)`,
`promote_model(version, alias, name|run_id, confirmed, force, reason, approved_by)`,
`demote_model(alias, name|run_id, confirmed)`.

**Log every finished run.** `log_run_to_mlflow` registers the run as the next
VERSION of `regression-<project>-<target>` and records its lineage (data hash, model
hash, whether the code had uncommitted edits). Always tell the user the
version number it returns — it is what they quote back to promote.

**Compare from the registry, never from memory.** `compare_model_versions`
returns signed metric deltas and `changed_params`, and deliberately no
verdict: which metric justifies a promotion is the user's call. Report the
deltas (and any metric that moved the other way); don't announce a winner.

**Promote only on the user's word.**
* `challenger` is free — it serves nothing.
* `champion` needs `confirmed=true` AND the user's `reason` (their words, one
  line — ask for both in the same question, naming the version it
  displaces). The request that opened the conversation is not confirmation.
* Champion promotion EXPORTS the version to `data/artifacts/<name>-v<N>.pkl`
  and points `data/artifacts/<name>-champion.pkl` — the file serving loads —
  at it. Relay `serving_path` and the `rollback` call. It refuses (alias
  untouched) if the run was refit since logging or its directory is gone:
  re-log and promote the new version instead; never export by hand.
* A run the readiness gate BLOCKED is refused unless `force=true` — only when
  the user explicitly accepts shipping it.
* `demote_model('champion', confirmed=true)` withdraws it: nothing serves
  until a new champion is promoted.
