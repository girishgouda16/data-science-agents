---
name: classification-routing
description: When and how to delegate to the remote classification agent — EDA, imputation, imbalance, training, tuning, explainability, model export, or any human-in-the-loop question it asks back. Load before the first delegate_to_classification_agent call in a conversation Also covers the MLflow experiment loop: log a run, compare it against the previous experiment, and promote a version to challenger/champion.
---

# Classification Routing

Delegate here for ANY data classification / EDA / model training request —
call `delegate_to_classification_agent` with the user's request (or their
follow-up answer, if the agent asked a question) as the `message` argument.

This agent may pause mid-task awaiting an answer (e.g. which column is the
target, how to handle missing values). Relay its question to the user
verbatim — don't paraphrase it into something vaguer — the task stays
open until they respond, and their next message continues the same remote
task/context automatically. A paused task is where the chain stops: never
answer for the user to keep things moving.

## What to do with its reply

A training request isn't finished when a model exists. Once this agent
reports a trained run:

1. `inspect_runs` for the real `run_id` and phase — don't copy a uuid out
   of the reply text.
2. Delegate to `visualization-routing` with that literal `run_id` for the
   evaluation charts (confusion matrix, ROC, PR, calibration,
   feature importance) — **also when readiness is blocked**. They read the
   same fitted pipeline this agent trained, the confusion matrix is drawn at
   the model's shipped threshold, and each chart lands in the run folder
   where it satisfies the `evaluation_charts` gate. The agent's own reply
   will say "incomplete — evaluation charts pending" until then; that is
   expected, not a problem to relay as one.
3. Relay the report, the charts, and the readiness verdict.

Two things this agent says that you must pass through, never soften:

- **`readiness: blocked`** — its own artifacts contradict something. Get
  the charts, then stop the chain and relay why. Don't route around it by
  delegating an export.
- **an INCONCLUSIVE success verdict** — the metric's 95% interval
  straddles the bar, so the test fold cannot say whether the model meets it.
  Relay it as inconclusive, never round it to a pass or a miss, and don't ask
  for re-tuning to "fix" it; only more labelled test data narrows it.
- **the positive class and the decision threshold** — "recall 0.90" means
  nothing without which class and at what cutoff. If it reports a
  calibration caveat on a tuned threshold, that caveat travels with every
  downstream mention of that number.

## Experiments: train -> log -> compare -> promote

Successive models on the same target are EXPERIMENTS on one registered
model, not unrelated runs. Drive that loop explicitly:

1. **Experiment 1.** Train as usual, then delegate `log_run_to_mlflow` for
   that `run_id`. Its reply carries a registry **version number** — relay it
   verbatim ("registered as `classification-<project>-<target>` version 1"). A version
   you never surfaced is one the user cannot act on. Then ask whether they
   want a second experiment, and what to change.
2. **Experiment 2.** Train the variant, log it the same way, relay its
   version number.
3. **Explain both, from the registry — not from memory.** Delegate
   `compare_model_versions` and `list_model_versions`. They return each
   version's metrics, algorithm, hyperparameters and readiness verdict. Walk
   through BOTH experiments: what each one was, what changed between them
   (`changed_params`), and each metric's delta with its sign. Never
   reconstruct these numbers from earlier turns — the registry is the record,
   the transcript is not.
4. **Do not declare a winner.** `compare_model_versions` deliberately returns
   no verdict, because which metric justifies a promotion is the user's call.
   Relay its `comparability` line and each delta's `delta_ci`: a delta whose
   interval spans 0 is not a demonstrated difference, however it is signed.
   Report the deltas, name any metric that moved the OTHER way, then ask what
   they want promoted.
5. **Promote only on their answer.** `challenger` is free. `champion` needs
   `confirmed=true` plus a `reason` (the user's why, one line, asked in the
   same question) and changes what serves traffic — ask in plain words,
   naming the version it displaces. It also needs the evaluation charts for
   that version's run; if promotion says they're missing, get them from
   visualization-routing with that run_id, show them, then retry. The
   request that started the conversation is never confirmation. Relay the returned `rollback` call so they know it
   is reversible. A champion promotion also exports the version and returns
   `serving_path` (`data/artifacts/classification-<project>-<target>-champion.pkl`) —
   that file is what serves, so relay it. If promotion is refused because
   the run was refit or swept, relay that verbatim; do not work around it by
   exporting the run by hand.

Readiness: quote `readiness_effective` from `list_model_versions`, not
`readiness` (a logging-time snapshot). Versions with different
`success_bar` params were judged against different bars — say so.

Metrics that disagree (ROC-AUC up, PR-AUC down) are a finding to report, not
a tie to break silently. So is a version whose `readiness` is not `ready`.

Exporting, serving and monitoring are further steps: do them when the user
asked for them, or offer them in a line. See the base SKILL.md.
