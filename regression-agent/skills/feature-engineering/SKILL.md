---
name: feature-engineering
description: Derive BUSINESS-meaning features for a continuous target from the process that produces it, and BUILD them — per-entity or decision-time aggregates from event rows, backward-looking history where every row is a decision (lags, time-window rolling, velocity, first-time counts, streaks), peer comparison fitted on the training fold, and a pre-model signal/leakage check (|Spearman| alone). Same tools and method as the classification agent. Load before prepare_dataset when the file is event rows, otherwise right after it.
---

# Feature Engineering (regression)

**MCP module:** `mcp_server/feat_engineering.py` (tools shared with the
classification agent — `core/feature_tools.py`)

Tools: `aggregate_events` (event rows -> entity / entity × period /
decision rows, before `prepare_dataset`), `apply_entity_features` and
`apply_peer_features` (right after `prepare_dataset`, before
preprocessing), `apply_custom_feature` (row-wise ratios),
`profile_features` (read-only check), `propose_features` +
`apply_features` (correlation fallback, last resort).

The method — data shape → tool, the seven archetypes, derive → build →
check → revise, and the rules every feature follows — is
`references/domain-features.md`, appended below. Domain procedures on
demand: `load_skill("feature-engineering", reference="telecom")` (plus
`telecom-traces`), `"credit"`, `"fraud"`.

## What changes for a continuous target

- **Ask what produces the NUMBER, not what makes a row positive.** Next
  month's ARPU is this month's usage times price, bent by lock-in expiry,
  experience and competitor offers; a claim amount is severity times
  exposure. Write that decomposition first — each factor is a feature
  family, and a factor you cannot measure is a data request.
- **The target's own period is off limits.** Predicting next month's
  revenue from this month's revenue is fine (a lag); predicting this
  month's revenue from this month's minutes and price is an accounting
  identity, not a model — the leakage screen will flag it near-perfect, and
  it is.
- **Ratios over levels, levels over nothing.** Own baseline (archetype 2)
  and trend (5) carry most of the signal for spend and usage targets; a
  level alone mostly restates the entity's size.
- **Skewed, non-negative targets** (money, usage, counts): decide the
  target transform (`apply_target_transform("log1p")`, preprocessing
  skill) before judging features — under log1p, multiplicative features
  (ratios, growth rates) line up with the target; additive ones don't.
- **`profile_features` reports |Spearman| alone** and the feature's median
  per target quartile. A monotone climb from q1 to q4 is the shape of a
  useful feature; `leakage_suspect` (≥ 0.98) usually means the feature is
  the target restated.

## Decide and apply — don't gate the list

State in ONE message before applying: each feature's spec, archetype,
mechanism and innocent twin; what you pruned and why; what the schema
lacks. Then apply and record the choice in `record_business_context(
assumptions=[...])`. Ask only about business framing: the grain, the
decision point, how long the target takes to settle.

**Always pass `rationale`** on every spec — it is persisted and shown in
the report.
