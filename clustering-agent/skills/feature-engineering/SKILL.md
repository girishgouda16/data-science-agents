---
name: feature-engineering
description: Build the BEHAVIOUR a segmentation should be measured on — per-entity aggregates of event rows (gap statistics, rhythm, shares, diversity, recency), ratios between them, and comparison with peers — normalised for exposure so segments reflect how customers behave, not how big or how long-tenured they are. Load before prepare_dataset when the data is event rows.
---

# Feature Engineering (clustering)

Tools (shared with the supervised agents, `core/feature_tools.py`):
`aggregate_events` (event rows -> one row per entity, before
`prepare_dataset`), `apply_custom_feature` (row-wise ratios),
`apply_peer_features` (value vs peers of the same kind).

The method — archetypes, data shape → tool, rules — is
`references/domain-features.md`, appended below; telecom's feature
vocabulary is on demand: `load_skill("feature-engineering",
reference="telecom")`. There is no target here, so the archetypes are used
to DESCRIBE behaviour, not to predict an outcome.

## What changes for segmentation

- **Every feature is a coordinate in the distance.** A feature that is
  just scale (total calls, total spend, tenure) will dominate and split
  customers by size. Prefer shares, ratios and rates: calls per active
  day, share of data vs voice, night share, share of international calls.
- **Normalise by exposure** — divide counts by `span_days` or active days,
  or a 3-day-old subscriber and a 3-year one land in different segments
  for no behavioural reason.
- **Rhythm and spacing segment well**: gap CV and periodicity separate
  machine from human, hour entropy separates always-on from 9-to-5,
  recency in units of the entity's own rhythm separates drifting from
  steady.
- **Fewer, meaningful features beat many.** Ten correlated windows of one
  quantity count ten times in the distance; keep one or two per mechanism.
- **Outcomes and protected attributes are not features** — they go to
  `set_profile_columns` (preprocessing) and describe the segments.

Always pass `rationale` on every spec — it is shown in the report.
