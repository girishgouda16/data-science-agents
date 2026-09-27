---
name: feature-engineering
description: Derive BUSINESS-meaning features from the mechanism behind the label and BUILD them — seven archetypes (share, own baseline, counterparties, timing, trend/velocity, streak, peers) mapped to tools: per-entity or decision-time aggregates from event rows (gap statistics, shares, diversity, lookback windows), backward-looking history where every row is a decision (lags, time-window rolling, velocity, first-time counts, streaks), peer comparison fitted on the training fold, and a pre-model signal/leakage check. Domain procedures for telecom, fraud and credit on demand. Load before prepare_dataset when the file is event rows, otherwise right after it.
---

# Feature Engineering

**MCP module:** `mcp_server/feat_engineering.py`

Tools:
- `aggregate_events(path, entity_column, time_column, features, period, before, at, window, out_path)`
  — event rows -> one row per entity, per entity × period, or per decision
  event (`at` + `window`). Raw file, **before** `prepare_dataset`.
- `apply_entity_features(run_id, features)` — lags, rolling stats over `n`
  rows or a time `window`, velocity (`roll_count`), own-baseline deviation,
  recency, first-time counts (`prior_count`), streaks. Right **after**
  `prepare_dataset`, **before** data-cleaning.
- `apply_peer_features(run_id, features)` — value against its peers (same
  cell, plan, device model), fitted on the training fold. Before
  data-cleaning.
- `apply_custom_feature(run_id, name, expression, rationale)` — row-wise
  ratios/differences between columns that exist.
- `profile_features(run_id, columns)` — read-only: per-class medians, the
  AUC each feature reaches alone, and flags (`leakage_suspect`, `constant`,
  `mostly_missing`, `weak_alone`). Run it before modelling.
- `declare_feature_engineering_not_applicable(run_id, reason)`
- `propose_features(run_id, top_k=5)` + `apply_features(run_id, features)`
  (correlation fallback).

The method — which tool for which data shape, the seven archetypes, the
derive → build → check → revise loop, and the rules every feature follows
— is `references/domain-features.md`, appended below. Domain procedures
are on demand: `load_skill("feature-engineering", reference="telecom")`
(plus `telecom-traces` for worked examples), `"fraud"`, `"credit"` —
fetch only the one matching `meta["business_understanding"]["domain"]`.

## Decide and apply — don't gate the list

Per the top-level HITL rule, adding features is a decision the next steps
measure (`profile_features`, importance, `feature-selection`), so asking
about each one costs a round-trip to learn something the data will tell
you anyway. State in ONE message before applying:

> *Features for [domain], pass [n]: [name — spec — archetype — mechanism —
> innocent twin]. Pruned: [name — reason]. Data requests: [what the schema
> lacks and which feature it would unlock].*

then apply them and record the choice in `record_business_context(
assumptions=[...])`. Ask only when the answer is business framing: the
aggregation grain or decision point, how long labels take to settle, or
domain semantics the data doesn't carry (what `status=3` means).

**Always pass `rationale`** on every spec and every `apply_custom_feature`
call — the mechanism in a sentence, the legitimate population that looks
the same, and what separates them. It is persisted to `meta.json` and
rendered in the report's Feature engineering section — the only way the
reasoning outlives the conversation.

**Read what the tools return.** `aggregate_events` returns a `summary` per
feature and the columns it did not carry; `apply_entity_features` returns
each feature's missing share; `apply_peer_features` the share of test rows
whose peer group never appeared in training. Then `profile_features`.

**This step is gated, and the gate is about visibility, not volume.**
`check_readiness` accepts either applied features or a recorded reason via
`declare_feature_engineering_not_applicable(run_id, reason)` — a real
reason: the file arrived with the domain features already computed and
nothing between windows is missing, or every archetype needs a column this
schema lacks. "Didn't get to it" is not one.

## Correlation-driven features (last resort, after the archetypes)

`propose_features(run_id, top_k=5)` ranks the `top_k` training-fold numeric
columns by absolute correlation with the target, then proposes pairwise
ratio and product features among them. Read-only. Surface the candidates
with each pair's correlation to the target, then
`apply_features(run_id, features)` (comma-separated names, or `"all"`).
