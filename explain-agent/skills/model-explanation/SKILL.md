---
name: model-explanation
description: Global driver ranking for a standalone exported model — what a .pkl pays attention to overall, when there is no training run to ask. Load this only when the question is about the model in general and no run_id is available.
---

# Model Explanation (global)

Tool: `explain_model(pkl_path, top_n, sample_size)`.

Read-only.

## Check this first

If the model was trained in this system and the caller has (or can get)
its `run_id`, **the training agent's own `explain_model` is the better
answer** and you should say so rather than quietly serving a worse one:

- it runs on the **held-out test fold**; this tool runs on the training
  reference shipped with the `.pkl`, which flatters a model that memorised
- its result is persisted to the run and is what that run's
  `explainability_run` / `explainability_clean` gates read — this tool's
  output is not gated evidence and must never be presented as if it were

Use this tool when there is no run to ask: a `.pkl` from elsewhere, or one
whose run directory has been swept.

## Reading the result

- `feature_importance` is **mean |SHAP|** — magnitude of influence, not
  direction and not correlation with the target.
- `direction` is an average tendency. Individual rows are routinely pushed
  the other way. If someone asks about a specific case, that's
  `decision-explanation`, not this.
- `scope_caveat` says what the ranking was computed on. Relay it.
- On a classifier the ranking is for the model's **positive class**, named
  in the result. State which class, every time — "the top driver of fraud"
  and "the top driver of not-fraud" are different sentences.

## The thing not to do

Do not answer "why was this customer flagged?" with a global ranking. It
is the single most common way an explanation misleads: it sounds
responsive, it is about a different question, and the person receiving it
usually can't tell. If the question is about a case, load
`decision-explanation` and explain that case.
