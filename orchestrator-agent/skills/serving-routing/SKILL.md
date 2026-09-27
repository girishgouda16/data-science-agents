---
name: serving-routing
description: When and how to delegate to the remote serving agent — getting predictions, forecasts, or cluster/anomaly assignments from a model another agent already exported (a .pkl path), or asking what a given .pkl actually is. Load before the first delegate_to_serving_agent call in a conversation.
---

# Serving Routing

Delegate here whenever the request is about USING an already-exported model
(a `.pkl` path someone already has, or wants you to find) to get
predictions/forecasts on new data — not about training, tuning, or
exporting one. Call `delegate_to_serving_agent` with the user's request (or
their follow-up) as the `message` argument.

This agent may pause mid-task awaiting an answer (e.g. it doesn't recognize
the artifact shape, or needs to know which CSV to predict on / what horizon
to forecast). Relay its question to the user verbatim — the task stays
open until they respond, and their next message continues the same remote
task/context automatically.

"Serve the champion" means the stable path
`data/artifacts/<agent>-<target or dataset>-champion.pkl` (e.g.
`classification-churn-champion.pkl`, `forecasting-sales-champion.pkl`) — a
link that each training agent's `promote_model(..., 'champion')` maintains. If it doesn't exist, no
champion has been promoted since this was added (or it was demoted): say
so and route the promotion to the classification agent rather than
guessing a versioned file.

Every `predict` reply names the `model` (run_id + registry version) that
produced it — also stamped on every output row — and the `predictions_log`
it appended to. Relay the version, so "which model scored this file" is
answerable.

Every `predict` reply carries `readiness_at_export`. When it comes with a
`warning` (the model was exported from a blocked or incomplete run), relay
that warning alongside the predictions — never drop it.

Every `predict` reply carries a `decision_rule` saying which threshold was
applied and to which class. Relay it. If it reports no tuned operating
point, say so — that model is being served at sklearn's 0.5, which on an
imbalanced target is a cutoff nobody chose, and the fix is `tune_threshold`
back at the training agent rather than serving it anyway.

When predictions come back and someone asks why a particular row was
flagged, that is `explain-routing` — the serving agent scores rows, it
does not explain them. The explain agent takes the same `.pkl` and the
same CSV plus a row index.

If the request is instead "build/train/fit a model" — even if it also
mentions exporting one at the end — that belongs to whichever training
agent matches the target (classification/regression/clustering/anomaly/
forecasting-routing), not this one. The distinguishing question: does a
`.pkl` already exist that just needs to be *used*, or does one still need
to be *produced*?
