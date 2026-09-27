---
name: anomaly-routing
description: When and how to delegate to the remote anomaly-detection agent — finding outliers/anomalies/fraud-like rows in a dataset (with or without a rare label column for evaluation), model export, or any human-in-the-loop question it asks back. Load before the first delegate_to_anomaly_agent call in a conversation.
---

# Anomaly Routing

Delegate here for ANY "find the outliers/anomalies/suspicious rows" request
— the model is unsupervised (fit without labels), even if a rare binary
label happens to exist and can be used afterward just to *evaluate* how
well it worked. Call `delegate_to_anomaly_agent` with the user's request
(or their follow-up answer) as the `message` argument.

This agent may pause mid-task awaiting an answer (e.g. which column is the
optional label, how to handle missing values, what contamination rate to
assume). Relay its question to the user verbatim — the task stays open
until they respond, and their next message continues the same remote
task/context automatically.

If the user wants to predict a known, well-represented class label (not
just flag rare oddities) that's `classification-routing`, not this one —
the boundary is "is the interesting class common enough to model directly
with a classifier, or so rare it needs anomaly detection instead".

## What a finished anomaly run looks like

The agent works like a senior data scientist — relay, don't shortcut it:
business context and the success bar first (it asks the user for the bar and
never invents one), the leakage screen BEFORE training, the contamination rate reasoned before fitting, train,
explain, then `check_readiness` and `generate_report`. Relay the report's
verdict line as written: `blocked` means its own evidence found a problem
(export and promotion refuse it); `incomplete` means evidence is missing — not
a pass. Without a label column detection quality cannot be measured — relay that flagged rows need human review before anyone acts on them.

After a trained run, delegate to `visualization-routing` with the literal
`run_id` (from `inspect_runs`) for the anomaly-score distribution with the flagging threshold — also when
readiness is blocked, since that is the run someone needs to look at. It is
evidence to relay, not a gate: this agent has no chart gate, so a missing or
failed chart never blocks its log / promote steps.

## Experiments: log -> compare -> promote

1. After a run, delegate `log_run_to_mlflow` for that `run_id` and relay the
   registry **version number** it returns (`anomaly-<project>`).
2. To compare, delegate `compare_model_versions` / `list_model_versions` —
   from the registry, never from memory — and report the deltas without
   declaring a winner.
3. `challenger` is free. `champion` needs the user's explicit confirmation AND
   their reason, asked in one plain question naming what it displaces. It
   exports the version and returns `serving_path`
   (`data/artifacts/anomaly-<project>-champion.pkl`) — that file is what
   serving loads; relay it and the `rollback` call.
