---
name: clustering-routing
description: When and how to delegate to the remote clustering agent — unsupervised grouping/segmentation of a dataset with no target column, choosing k, cluster profiling, model export, or any human-in-the-loop question it asks back. Load before the first delegate_to_clustering_agent call in a conversation.
---

# Clustering Routing

Delegate here for ANY "group/segment/find natural clusters in this data"
request — there is no target column to predict, just structure to discover.
Call `delegate_to_clustering_agent` with the user's request (or their
follow-up answer) as the `message` argument.

This agent may pause mid-task awaiting an answer (e.g. how to handle missing
values, which k to use). Relay its question to the user verbatim — the task
stays open until they respond, and their next message continues the same
remote task/context automatically.

If the user actually has a target column they want predicted (a label or a
number), that's `classification-routing`/`regression-routing`, not this one.

## What a finished clustering run looks like

The agent works like a senior data scientist — relay, don't shortcut it:
business context and the success bar first (it asks the user for the bar and
never invents one), the leakage screen BEFORE training, train,
explain, then `check_readiness` and `generate_report`. Relay the report's
verdict line as written: `blocked` means its own evidence found a problem
(export and promotion refuse it); `incomplete` means evidence is missing — not
a pass. A `structure_found` failure (silhouette <= 0.25) means the segments are arbitrary cuts — say so plainly; only the user can accept them (`acknowledge_gate`).

After a trained run, delegate to `visualization-routing` with the literal
`run_id` (from `inspect_runs`) for the cluster map (2-D, coloured by cluster, with sizes) — also when
readiness is blocked, since that is the run someone needs to look at. It is
evidence to relay, not a gate: this agent has no chart gate, so a missing or
failed chart never blocks its log / promote steps.

## Experiments: log -> compare -> promote

1. After a run, delegate `log_run_to_mlflow` for that `run_id` and relay the
   registry **version number** it returns (`clustering-<project>`).
2. To compare, delegate `compare_model_versions` / `list_model_versions` —
   from the registry, never from memory — and report the deltas without
   declaring a winner.
3. `challenger` is free. `champion` needs the user's explicit confirmation AND
   their reason, asked in one plain question naming what it displaces. It
   exports the version and returns `serving_path`
   (`data/artifacts/clustering-<project>-champion.pkl`) — that file is what
   serving loads; relay it and the `rollback` call.
