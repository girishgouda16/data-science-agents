---
name: clustering
description: Choose k from stability AND the business's capacity to act, fit a clustering pipeline (kmeans, minibatch_kmeans, kmedoids for mixed data, gmm, hierarchical, hdbscan, dbscan), prove the segments reproduce, describe them (distinguishing features, outcomes per segment, simple rules), track how customers move between segments over time, and export/predict. Load after eda/preprocessing when the user wants segments.
---

# Clustering & Export

Tools: `propose_k(run_id, k_range, algorithm)`,
`train_model(run_id, algorithm, n_clusters, eps, min_samples, linkage, min_cluster_size, covariance_type)`,
`explain_model(run_id)`, `segment_migration(run_id, entity_column, period_column, data_path)`,
`compare_runs(run_ids)`,
`export_model(run_id, out_path, force)`, `predict(pkl_path, data_path)`.

Every tool operates on the run's **current pipeline** — drop (dropped and
profile columns) → impute → log the skewed numerics → standardise numerics /
one-hot categoricals → clusterer — saved by `train_model`. Whatever
`train_model` last wrote IS the current model.

## What makes a segmentation real

1. **Stability** — the segments come back when the data is resampled
   (`stability_ari_mean` ≥ 0.7 across refits on 80% subsamples). This is
   the check; a silhouette on one sample is not. An unstable segmentation
   is an artefact, however clean its profiles look — say so.
2. **Separation** — silhouette above 0.25 (below it there is no
   substantial structure; the segments are cuts of a continuum).
3. **Usefulness** — segments differ on the outcome the business cares
   about (`explain_model`'s eta² on the profile columns), are big enough to
   act on, and can be described (`rules` accuracy). Separation without
   usefulness is a finding to report, not a deliverable.

## Choosing k — statistics AND capacity

`propose_k(run_id, k_range)` reports per k: silhouette, Davies-Bouldin,
inertia (elbow) and stability. `recommended_k` is the best-silhouette k
among the stable ones. Intersect with how many segments the business can
act on (from `business-understanding` — e.g. "marketing can run 4-6
treatments"): take the best stable k inside that range and say why. Ask
only when no stable k falls inside it — then the honest options are fewer
segments, different features, or telling them the data doesn't split that
finely.

## Algorithm

- `kmeans` — default for (mostly) numeric data; compact, similar-sized
  segments.
- `kmedoids` — **mixed data**: plan, device, region or channel next to
  usage. Gower distance: a numeric gap counts by its share of the range, a
  category mismatch counts 1, so neither dummies nor standardised noise
  drown the categories that carry meaning. Each segment's centre is a real
  subscriber, new rows are scored, and it is sampled so it scales. Run
  `propose_k(algorithm="kmedoids")` for its k. When categoricals matter,
  fit both and keep the one with the better stability and outcome eta².
- `minibatch_kmeans` — the same above a few hundred thousand rows.
- `gmm` — elliptical, overlapping segments of different spread
  (`covariance_type` full/diag); use when kmeans splits one elongated
  group in two. **Numeric features only**: one-hot dummies make a
  component collapse onto a rare category (`model_warnings` says so, and
  stability drops) — move categoricals to profile columns or use kmeans.
- `hdbscan` — density-based; finds k itself and labels noise
  (`min_cluster_size` ≈ the smallest segment worth acting on). Good for
  "is there natural structure at all?"; cannot score new rows.
- `hierarchical` — up to 20k rows; `dbscan` — density with a fixed `eps`,
  rarely better than hdbscan.

## Flow

1. `propose_k` with the algorithm you will fit → pick k as above (skip
   for hdbscan/dbscan).
2. `train_model` — surface stability, silhouette, sizes, `size_warnings`
   (a segment under 1% or one over 70%), and which columns were logged.
3. `explain_model` — name each segment from its **distinguishing**
   features ("+1.8 sd data_gb, x3 plan=max"), not its means; report the
   outcomes per segment with eta² and the rules accuracy.
4. `check_readiness` → `generate_report` → export → `log_run_to_mlflow`.
5. `compare_runs` ranks alternatives (different features, k or algorithm)
   by silhouette and shows each one's stability.

## Migration over time

When the data holds one row per subscriber per period (monthly
snapshots), `segment_migration(run_id, entity_column, period_column)`
assigns every row with the ONE fitted model — never refit per period, or
"segment 2" means something different each month — and reports the stay
rate, the from → to matrix, the largest moves and the share that moved at
each step. Read it as: a low stay rate marks a transient state
(onboarding, a promotion), a high one a durable identity; the flow from a
high-value segment towards an at-risk one is the finding to act on; a jump
in `by_period` points at a tariff change, an outage or drift. Only a
stable segmentation has meaningful moves.

## Where the user decides

- **Export is always two-step.** `export_model(run_id)` without `out_path`
  returns a suggested path; confirm it with the user, then write.
- **An unstable model (`overfitting_warning`) or a blocked readiness
  refuses export** unless `force=true` — surface why and ask; never pass
  `force=true` yourself. When you ask, recommend the honest option first
  (a stable k, other features, or "the data doesn't split that finely") —
  never "ship it anyway".
- **Predict**: `predict(pkl_path, data_path)` assigns new rows (any data
  format) for kmeans / minibatch_kmeans / gmm natively and hierarchical by
  nearest centroid; density-based exports refuse — don't work around it.
