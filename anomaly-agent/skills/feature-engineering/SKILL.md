---
name: feature-engineering
description: Build the behaviour anomalies show up in — per-entity aggregates of event rows (gap regularity, bursts, destination diversity), deviation from the entity's own history (vs its rolling baseline, velocity, first-time values) and from its peers — so contextual and collective anomalies become visible to a detector. Load before prepare_dataset when the data is event rows (xDRs, transactions, logs), otherwise right after it.
---

# Feature Engineering (anomaly)

Tools (shared with every agent, `core/feature_tools.py`):
`aggregate_events` (event rows -> one row per entity / entity × period /
decision time, before `prepare_dataset`), `apply_entity_features` and
`apply_peer_features` (right after `prepare_dataset` with `group_column`
and `time_column`, before preprocessing), `apply_custom_feature` (row-wise
ratios), `profile_features` (labelled runs only: each feature's AUC alone
against the label, and leakage suspects).

The method — data shape → tool, the seven archetypes, the rules — is
`references/domain-features.md`, appended below. Telecom fraud and fault
vocabulary on demand: `load_skill("feature-engineering",
reference="telecom")` (Wangiri, IRSF, SIM-box, SIM swap, network faults),
`"telecom-traces"` (worked traces), `"fraud"` (payments / AML).

## What changes for anomaly detection

- **Deviation beats level.** A detector on raw levels flags the biggest
  customers. Most telco anomalies are *contextual*: unusual for this line
  (`apply_entity_features` `vs_roll_mean`, `roll_count` over "1h"/"24h"
  for bursts, `prior_count` = 0 for a first-ever destination country) or
  for its kind (`apply_peer_features` zscore by cell, plan or device).
- **Collective anomalies need aggregation.** One short call is nothing;
  300 in an hour with identical gaps is Wangiri or SIM-box.
  `aggregate_events` per SIM over a window: count, `_gap` cv (≈ 0 =
  machine rhythm), distinct destinations, share of international/premium
  destinations, share of calls under 5 s, hour entropy.
- **Normalise by exposure** — per active day or per call, or every
  high-volume line (a PBX, a call centre) is the top alert forever.
- **Few features per mechanism.** Ten windows of one quantity make a
  distance detector see one thing ten times; keep the one or two that
  describe the mechanism.
- **The label is never a feature,** and neither is anything written after
  the case was decided (case status, block reason, refund flag) —
  `detect_data_leakage` screens for them.

Always pass `rationale` on every spec — it is shown in the report and
makes the alert reasons readable.
