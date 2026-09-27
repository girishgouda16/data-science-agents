---
name: forecasting-routing
description: When and how to delegate to the remote forecasting agent — EDA, imputation, training/tuning/backtesting a time-series model, explainability, forecasting future periods, model export, or any human-in-the-loop question it asks back. Load before the first delegate_to_forecasting_agent call in a conversation.
---

# Forecasting Routing

Delegate here for any request that asks to predict FUTURE values of a
metric that's indexed by time (demand, revenue, traffic, sensor readings,
"forecast the next N days/weeks/months of X") — call
`delegate_to_forecasting_agent` with the user's request (or their follow-up
answer, if the agent asked a question) as the `message` argument.

This agent may pause mid-task awaiting an answer (e.g. which column is the
date, how to handle missing values, which model to use). Relay its question
to the user verbatim — the task stays open until they respond, and their
next message continues the same remote task/context automatically.

If the request is predicting a value from OTHER columns on the same row
(no time dimension involved — e.g. "predict this house's price from its
features"), that's `regression-routing`, not this one. The distinguishing
question: does the answer depend on *when* it is (forecasting), or on
*what the other columns say* (regression)?

## What a finished forecasting run looks like

The agent works like a senior data scientist — relay, don't shortcut it:
business context and the success bar first (it asks the user for the bar and
never invents one), the leakage screen BEFORE training, the naive seasonal baseline every model must beat, train,
explain, then `check_readiness` and `generate_report`. Relay the report's
verdict line as written: `blocked` means its own evidence found a problem
(export and promotion refuse it); `incomplete` means evidence is missing — not
a pass. 

After a trained run, delegate to `visualization-routing` with the literal
`run_id` (from `inspect_runs`) for the forecast-vs-actual chart — also when
readiness is blocked, since that is the run someone needs to look at. It is
evidence to relay, not a gate: this agent has no chart gate, so a missing or
failed chart never blocks its log / promote steps.

## Experiments: log -> compare -> promote

1. After a run, delegate `log_run_to_mlflow` for that `run_id` and relay the
   registry **version number** it returns (`forecasting-<project>-<target>`).
2. To compare, delegate `compare_model_versions` / `list_model_versions` —
   from the registry, never from memory — and report the deltas without
   declaring a winner.
3. `challenger` is free. `champion` needs the user's explicit confirmation AND
   their reason, asked in one plain question naming what it displaces. It
   exports the version and returns `serving_path`
   (`data/artifacts/forecasting-<project>-<target>-champion.pkl`) — that file is what
   serving loads; relay it and the `rollback` call. A forecasting champion is refit on the full history at export, so its forecasts start after the last observed date.
