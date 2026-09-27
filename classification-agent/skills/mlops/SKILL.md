---
name: mlops
description: Log a run to MLflow, version its data with DVC, and emit a CI workflow for retraining. Load at the end of every autonomous modeling run (log_run_to_mlflow is automatic) and whenever the user separately asks to version data or set up CI. No HITL — these are infra side-effects, not data changes.
---

# MLOps

**MCP module:** `mcp_server/mlops.py`

Tools: `log_run_to_mlflow(run_id)`, `dvc_track(path)`, `generate_ci_workflow(out_path=".github/workflows/ml-ci.yml")`.
Registry: `list_model_versions(target)`, `compare_model_versions(target)`, `promote_model(version, alias, target, confirmed, force, reason, approved_by)`, `demote_model(alias, target, confirmed)`.

None of these gate — they don't touch the model or the data, only record/
version/scaffold around it. **`log_run_to_mlflow` runs automatically at the
end of every autonomous modeling run** (see `modeling`'s autonomous-mode
flow) — the user asking for "a model" gets a trained pickle, a report, and
an MLflow-tracked run out of one request, not three. `dvc_track` and
`generate_ci_workflow` stay opt-in — run them only when asked, since they
touch files outside the run's own directory (a tracked data file, a CI
workflow file) rather than just recording what already happened.

**`log_run_to_mlflow(run_id)`** — logs this run's model, params, and
metrics (from `meta.json`) to a local MLflow file store (`./mlruns`, no
server required). Returns the run's MLflow run_id. Requires `mlflow`
installed; returns an error string (not a raised exception) if missing.

## Registry: inspect, then promote

`log_run_to_mlflow` now also **registers** the run's pipeline as a version of
`classification-<project>-<target>`, so successive experiments stack as v1, v2, v3 under
one name instead of becoming unrelated entries. **Always tell the user the
version number it returns** — that number is what they read in the MLflow UI
and quote back to promote. A registration whose version you never surfaced is
one the user cannot act on.

The loop is deliberately human-in-the-middle: the agent registers and compares,
the user inspects in the UI, the user decides what gets promoted.

**`list_model_versions(target)`** — every version of that model, newest first,
with metrics and current aliases. Run this before any promotion so the user and
the agent are looking at the same version numbers.

**`compare_model_versions(target, version_a, version_b)`** — signed per-metric
deltas, defaulting to newest vs previous. Reads the registry, not the local run
dirs, so it still works after old runs are swept. It returns **no verdict** on
purpose: which metric justifies a promotion is the user's call. Report the
deltas and let them choose; don't announce a winner. Read `comparability`
first: when both versions were scored on the same test rows each delta carries
a paired 95% `delta_ci` and `distinguishable` — a delta whose interval spans 0
is **not a difference**, say so plainly. Without paired scores there is no
interval; point to `cv_score` instead of a small test delta.

Every logged version also records its lineage as tags — `data_sha256`,
`test_fingerprint`, `git_dirty` — plus `test_scores.csv`,
`tuning_trials.json`, `cv_score` and `roc_auc_ci_low/high`.

**`promote_model(version, alias, target, confirmed, force, reason, approved_by)`** — points
`champion` or `challenger` at a version.

* `challenger` promotes freely — it serves no traffic.
* `champion` requires `confirmed=true`, because it changes what serves.
  Ask the user which version they want as champion, name what it
  displaces, and only then retry with `confirmed=true`. **The request that
  opened the conversation is not confirmation.**
* `champion` also needs `reason` — ask the user why, in the same question
  as the confirmation — and takes `approved_by` if you know who confirmed.
  Both are written as `champion.*` tags on the version with `promoted_at`
  and `forced`; demotion adds `champion.demoted_at`. `list_model_versions`
  shows them as `version_tags`.
* `champion` needs the evaluation charts (confusion matrix, ROC, PR) for the
  exact fit behind the version (the `evaluation_charts` gate). Missing →
  refused with the run_id to chart; you can't draw them, so say so and let
  the orchestrator get them. On success the charts and a regenerated report
  are logged to the version's MLflow run.
* Promoting to `champion` exports that version to
  `data/artifacts/classification-<project>-<target>-v<N>.pkl` and points
  `data/artifacts/classification-<project>-<target>-champion.pkl` (+ `.monitoring.json`)
  at it — that stable path is what serving and drift load. Relay
  `serving_path`. The alias alone serves nothing; never say "serving now
  resolves models:/…@champion".
* It refuses (alias untouched) if the run was refit after registration or
  its run directory is gone — say so; the fix is to re-log and promote the
  new version. Registered runs are pinned against the 7-day run sweep.
* `demote_model('champion')` also removes the champion links, so nothing
  serves until a new champion is promoted.
* `list_model_versions` returns `readiness` (snapshot at logging) and
  `readiness_effective` (what promotion enforces). Report the effective one.
  Each version also records `success_bar` — flag versions judged against
  different bars (or none) instead of ranking them as if comparable.
* Either alias is refused if the readiness gate BLOCKED the run behind that
  version, unless `force=true` — the same rule `export_model` applies. Ask
  before overriding.

The result carries `previous_version` and a ready-made `rollback` call; pass
those on, so the user knows a promotion is reversible.

**`dvc_track(path)`** — shells out to `dvc add <path>`. Requires the `dvc`
CLI on PATH and an initialized DVC repo (`dvc init`); returns an error
string with the exact command to run if either precondition isn't met,
rather than trying to init DVC for the user silently.

**`generate_ci_workflow(out_path)`** — writes a GitHub Actions workflow
(lint + pytest + a smoke train on a sample) to `out_path`. Overwrites if it
already exists — treat this as scaffolding, not something to hand-edit and
expect preserved; tell the user before overwriting an existing workflow
file with different content.
