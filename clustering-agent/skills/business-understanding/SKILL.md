---
name: business-understanding
description: Settle what the segments are FOR before touching data — the decision each segment will get, how many segments the business can act on, the smallest segment worth acting on, what the segments must differ on (outcomes kept out of the distance), which attributes must not define them, and the bar. Load FIRST on any new segmentation request.
---

# Business Understanding (clustering)

Tool: `record_business_context(run_id, business_objective,
target_definition, success_criteria, domain, assumptions, clarifications,
success_metric, success_threshold)` — call it right after
`prepare_dataset`; everything settled before goes in. (`target_definition`
here is what a row is and what a segment will be used for.)

A segmentation can be perfectly separated and useless. Work through these
before `eda`:

1. **What will each segment GET?** A different offer, a retention call, a
   network investment, a tariff. "Segment our customers" is not a
   decision; "choose one of five retention treatments per subscriber" is.
2. **How many segments can the business act on?** Each needs an owner, a
   treatment and a measurement. This caps k, whatever the statistics say —
   record the range (e.g. 4-6).
3. **Smallest segment worth acting on** — a share or a count. It sets
   `min_cluster_size` for hdbscan and makes a 0.5% segment a merge
   candidate, not a finding.
4. **What must the segments differ on?** The outcome that makes a segment
   matter — churn, ARPU, NPS, drop rate. It must be kept OUT of the
   distance (`set_profile_columns`) and used to judge the segments:
   clustering on the outcome just finds the outcome back.
5. **What must not define the segments?** Protected attributes (age,
   gender, ethnicity) and their proxies (postcode) when segments drive
   offers or treatment. Keep them out of the distance too
   (`set_profile_columns`) — the profiles then show whether segments
   correlate with them anyway, which is the fairness question to report.
6. **Which features SHOULD define them?** Behaviour (usage, rhythm,
   channel, experience) segments differently from value or demographics.
   Agree on the lens — it changes the answer more than the algorithm does.
7. **Will the segments be tracked over time?** Monthly movement between
   segments (who drifts from high value towards churn) needs one row per
   subscriber per period, and the ID and period kept as profile columns.
8. **Bar** — usually stability plus a minimum outcome difference, in
   business words. A numeric bar (`silhouette_score` / `davies_bouldin_score`
   / `calinski_harabasz_score`) only if the user gives one; never invent it.

## Ask, or decide and record

**Ask** what defines "useful": the decision, the actionable range of k, the
smallest segment, the outcome, the excluded attributes, the lens. **Decide
and record** everything a measurement settles: imputation, identifier
drops, logging skewed features, the algorithm, k inside the agreed range.
Put each in `assumptions` (one per line), each answered question in
`clarifications`; the report shows both.
