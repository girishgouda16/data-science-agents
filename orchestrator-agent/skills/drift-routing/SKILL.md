---
name: drift-routing
description: When and how to delegate to the remote drift-detection agent — comparing a reference/baseline dataset against a current/newer one to check for distribution shift, before deciding whether a model needs retraining. Load before the first delegate_to_drift_agent call in a conversation.
---

# Drift Routing

Delegate here for any "has this data drifted", "compare these two
datasets", "is it still safe to use the trained model on this new data"
request — call `delegate_to_drift_agent` with the user's request (or their
follow-up) as the `message` argument. The request needs two file paths (a
reference/baseline dataset and a current/newer one); if the user only gave
one, ask which second file to compare it against before delegating — unless
a monitoring profile exists for the model (see below).

This agent is stateless and read-only — it never pauses awaiting a
human-in-the-loop decision the way the training agents do, so a reply from
it always completes the task immediately.

Prefer the monitoring profile. If the model was trained here, its export
wrote a `*.monitoring.json` beside the `.pkl` — pass that as
`profile_path` along with the current file, instead of two bare CSVs. It
carries the training fold (which outlives the run directory's retention
window), the columns the model actually depends on, and which column is
the label. Without it the verdict weights every column equally, and one
drifting column nobody models reads the same as a broken feature.
`inspect_runs` lists the profiles it can see.

"Has production drifted" needs no second file: pass `profile_path` alone
and the drift agent compares the training fold against the serving agent's
predictions log for that model — every row it has actually scored.

Its verdict (none / moderate / severe drift, plus a recommendation) is
advisory, not an action — it never retrains anything itself.

**Retraining on a "severe" verdict is the user's call, not yours.** Drift
severity is a statement about inputs, not about model quality: this agent
never scores the model and never sees a current-period label, so "severe"
is a reason to investigate, not a measurement showing the model got worse.
Relay the verdict, relay its `what_this_did_not_check`, and ask whether to
retrain. Delegate to `classification-routing`/`regression-routing`/
`forecasting-routing` once they say yes — not before, and never as an
automatic reflex to the word "severe".
