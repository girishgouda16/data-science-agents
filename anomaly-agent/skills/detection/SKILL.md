---
name: detection
description: Set the alert share from review capacity, compare detectors, fit one (on known-normal rows when labels exist), prove the alert list is stable, explain every alert, and export/predict. Load once eda/preprocessing are done and the user wants a detector.
---

# Detection & Export

Tools: `propose_contamination(run_id, review_capacity, rows_per_period)`,
`compare_detectors(run_id, algorithms, contamination, train_on)`,
`train_model(run_id, algorithm, contamination, train_on)`,
`explain_model(run_id, top_n)`, `compare_runs(run_ids)`,
`export_model(run_id, out_path, force)`, `predict(pkl_path, data_path)`.

Every tool operates on the run's **current pipeline** — drop (dropped
columns, split keys) → impute → log the skewed numerics → standardise
numerics / one-hot categoricals (not scaled, so a rare but normal category
is not an anomaly by itself) → detector — saved by `train_model`.

## 1. The alert share is an operating decision

`propose_contamination(run_id, review_capacity, rows_per_period)` — the
share of rows flagged. From review capacity it is exactly what the team
can work; from labels it is the prevalence (then ask for capacity); with
neither it is a 5% guess — say so. When the budget is below the labelled
prevalence, `recall_ceiling` is the most the queue can catch: tell the
user before they judge the detector against a recall bar it cannot reach.

## 2. Compare, then train

`compare_detectors(run_id)` fits iforest, ecod, knn and lof on the same
split (read-only) and ranks them — with labels by average precision, without
by stability and agreement with the consensus. Train the winner with
`train_model(run_id, algorithm, contamination)`:

- `iforest` — default; fast, no distance assumptions, good on mixed scales
- `ecod` — parameter-free, per-feature tails; strong for point anomalies
- `knn` — far from its neighbours; `lof` — sparse compared with its
  neighbourhood (contextual: unusual for its kind)
- `ocsvm` — a kernel boundary; rarely better, slowest
knn/lof/ocsvm fit on a 20k-row sample above that; every row is scored.

`train_on="auto"` fits on **known-normal rows only** when labels exist —
the detector learns normal undiluted by what it must find (novelty
detection); the labels still never enter `.fit()` as targets, and the
threshold is set on the whole training fold so the alert share holds.

## 3. Read the evidence

- **With labels** — the review queue at the budget: `precision_at_budget`
  (share of alerts that are real), `recall_at_budget` (share of anomalies
  caught), `lift_at_budget` (× the base rate; the gate needs ≥ 2), and
  `average_precision` (PR-AUC). Report ROC-AUC last: at a 1% base rate
  0.9 can still mean most alerts are false.
- **Always** — `stability.top_alert_jaccard_mean`: the overlap of the top
  alerts across refits on resampled data. Below 0.5 the alert list is an
  artefact of the sample (the `alerts_stable` gate fails): change
  features, algorithm or alert share.
- `model_warnings` naming a categorical: the alerts are mostly rows with a
  rare value of it. iforest and ecod isolate a rare category in one split —
  if those values are normal business (a niche plan, a device model), drop
  the column from the detector or make it a peer key, and compare knn/lof.
- `holdout_alert_rate` far from the alert share means the held-out rows
  differ from training (drift, or a temporal split crossing a change).

## 4. Explain every alert

`explain_model(run_id)` — global drivers plus the **review queue**: the top
held-out rows, each with its reasons ("calls_to_premium=41 (+12.3 robust sd
vs typical 0); country=XX (never seen in training)") and, with labels,
whether it was a known anomaly. Show the user the first few. Without
labels this queue IS the deliverable: a person must review it before
anyone acts on the detector.

## 5. Export and predict

- **Export is always two-step.** `export_model(run_id)` without `out_path`
  returns a suggested path; confirm it with the user, then write. A
  blocked readiness refuses unless `force=true` — surface why and ask;
  never pass `force=true` yourself.
- `predict(pkl_path, data_path)` scores new data in any format: `anomaly`,
  `anomaly_score` and `anomaly_reasons` for every flagged row.
- `compare_runs(run_ids)` ranks labelled runs by average precision.
