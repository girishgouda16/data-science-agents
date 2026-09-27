---
name: explain-routing
description: When and how to delegate to the remote explain agent — why a model made one specific decision about one specific case, what would have had to differ to change it, or what a standalone .pkl pays attention to overall. Load before the first delegate_to_explain_agent call in a conversation.
---

# Explain Routing

Delegate here whenever the question is **why** — why was this transaction
flagged, why was this subscriber predicted to churn, what would have had
to be different, what does this `.pkl` actually pay attention to. Call
`delegate_to_explain_agent` with the user's request (or their follow-up)
as the `message` argument.

It needs a path to an exported `.pkl`, and for a case-level question also
the data file and which row. `inspect_runs` lists recent runs and exported
artifacts if the user didn't give a path.

## Which agent owns "explain"

Two agents can answer an explainability question and they are not
interchangeable:

- **the training agent** (`classification-routing` / `regression-routing`)
  for a run you have a `run_id` for. Its `explain_model` runs on the
  held-out test fold and its result is gated evidence for that run. This
  is the right route for "what drives the model we just built".
- **the explain agent** for a specific *case*, and for any model you only
  have as a `.pkl` — typically because the run directory has been swept or
  the model came from elsewhere.

Rule of thumb: **a question about a row goes here; a question about a run
in progress goes to the agent that owns the run.** Routing a case-level
question to a training agent gets you a global ranking, which sounds
responsive and answers something else.

## Relay its caveats intact

This agent deliberately refuses to produce explanations it can't stand
behind, and marks the shaky ones. Three things must reach the user
unedited:

- **`additivity_ok: false`** — the contributions don't reconstruct the
  model's score, so there are no reasons to give. Don't summarise this
  into "the model found X important anyway".
- **low `explanation_stability`** — near-identical cases get different
  drivers. The caveat goes *above* the reasons, not after them.
- **counterfactual `limitations`** — greedy, median-targeted, correlation-
  blind, and association not causation. Never relay a counterfactual as
  advice to a person.

If it refuses (unsupported model family, no reference dataset, a
clustering/anomaly/forecasting artifact), relay the refusal and its stated
next step. Don't route the same question to another agent to get a more
agreeable answer — none of them can answer it either, they'd just be less
careful about saying so.

This agent is stateless and read-only. It never pauses for human input,
so a reply from it always completes the task.
