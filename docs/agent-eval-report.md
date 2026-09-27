# Holdout — agent evaluation report

2026-09-27 · model `deepseek/deepseek-v4.1-flash` via OpenRouter

**Result: 79 of 80 behaviour checks pass across 25 scenarios and all six
agents.** The one miss is a check that was too strict and has since been
retired (below). On the first pass 76 of 80 passed; every miss was
traced, and three of the four were defects in the test harness, not in
the agents. The evaluation also exposed one real production defect — an
LLM call that could hang a session for an hour — now fixed.

## What was evaluated

Each scenario hands the real agent (real LLM, the same LangGraph host the
A2A server runs, its own MCP tools) a dataset and a request, answers its
questions from a script, and grades the **decisions** from what the run
left behind — the run's metadata and execution log, or, for the stateless
drift agent, the transcript. Nothing is graded from what the agent *says*
it did. Runner: `core/agent_eval.py`; scenarios: `<agent>/scripts/eval_agent.py`.

| Agent | Scenarios | Checks passed | First pass |
| --- | --- | --- | --- |
| Classification | 6 | 19 / 19 | 19 / 19 |
| Regression | 6 | 13 / 13 | 13 / 13 |
| Clustering | 4 | 15 / 15 | 13 / 15 |
| Anomaly | 3 | 10 / 10 | 10 / 10 |
| Forecasting | 3 | 13 / 14 | 13 / 14 |
| Drift | 3 | 9 / 9 | 8 / 9 (after a runner crash) |
| **Total** | **25** | **79 / 80** | **76 / 80** |

## Scenario results

### Classification — 19 / 19

| Scenario | What it tests | Result |
| --- | --- | --- |
| entity_time_history | Call events per caller: builds history features from each caller's own past, splits forward in time, drops the caller id | 4 / 4 |
| titanic_identifiers | Drops identifiers, asks for the success bar, records the user's bar, tunes the threshold off the test fold | 5 / 5 |
| label_copy | A column that copies the outcome is kept out and never self-waived | 2 / 2 |
| no_bar_given | Invents no success bar when the user gives none | 2 / 2 |
| repeat_entities | Asks about repeated customers, splits by customer, drops the id | 3 / 3 |
| imbalanced_fraud | 2% positives: selects on average precision, no SMOTE, threshold off the test fold | 3 / 3 |

### Regression — 13 / 13

| Scenario | What it tests | Result |
| --- | --- | --- |
| skewed_arpu | Models a skewed target in log space or with a Poisson loss | 2 / 2 |
| subscriber_months | Splits forward in time, excludes months still accruing, is judged against the persistence baseline | 4 / 4 |
| no_bar_given | Invents no bar | 2 / 2 |
| target_copy | Drops a column that copies the target | 2 / 2 |
| capacity_p90 | Turns a 5:1 under/over-provisioning cost into a ~0.83 quantile objective | 2 / 2 |
| range_needed | Calibrates prediction intervals when the user needs a range | 1 / 1 |

### Clustering — 15 / 15

| Scenario | What it tests | Result |
| --- | --- | --- |
| outcome_kept_out | Churn and age describe the segments but stay out of the distance; k within 3–5; stable | 5 / 5 |
| mixed_plans | Mixed plan / roaming / usage data: chooses Gower k-medoids, ships a stable k within 3–4 | 4 / 4 |
| monthly_panel | Monthly snapshots: keeps id and month out of the distance, measures migration between months | 4 / 4 |
| no_structure | Pure noise: lets the gates report there is no stable structure | 2 / 2 |

### Anomaly — 10 / 10

| Scenario | What it tests | Result |
| --- | --- | --- |
| review_capacity | Sets the alert share from review capacity (30 of 3,000 a day), compares detectors, ships a queue with precision ≥ 0.5 (got 1.0) | 4 / 4 |
| status_leak | Keeps a case-status field written after investigation out of the detector | 2 / 2 |
| unlabelled_review | No labels: asks how many alerts can be reviewed, ships a stable alert list and an explained review queue | 4 / 4 |

### Forecasting — 13 / 14

| Scenario | What it tests | Result |
| --- | --- | --- |
| region_capacity | Hourly traffic of three cells: sums the region, evaluates one week ahead, uses the weekly season, beats the naive (pooled RMSE 5.96 vs 7.50) | 5 / 6 — see below |
| every_cell | 25 cells, daily: forecasts every cell in one run, 14-day horizon, fills gaps as a decision, beats the naive (6.40 vs 8.61) | 5 / 5 |
| contact_centre_gaps | Daily calls with a three-day outage: 28-day horizon, fills the gap as a decision, beats the naive | 3 / 3 |

### Drift — 9 / 9

| Scenario | What it tests | Result |
| --- | --- | --- |
| broken_feed | A column 45% null and a numeric column exported as text: leads with both as feed problems, never claims the model is fine | 4 / 4 |
| joint_shift | Columns each stable, their relationship flipped: runs the whole-table check and says the columns changed how they move together | 3 / 3 |
| when_it_started | A latency step five weeks in: runs drift over time and names the week (2026-03-30) | 2 / 2 |

## Every first-pass miss, and what was done

| Miss | Cause | Fix |
| --- | --- | --- |
| Clustering `outcome_kept_out`: "subscriber_id still a feature" | Harness defect: the agent kept the id out of the distance as a *profile column*, which the check did not count | The check now counts profile columns; re-run passes |
| Clustering `mixed_plans`: stability 0.57 | Scenario under-specified: the scripted user answered "use your recommendation" to "is `plan` an outcome?" and to "ship k=3 anyway?". The agent did ask both, plainly — but listed "ship anyway" as its first option | Scenario now says what defines the segments; the clustering skill now requires recommending the honest option first. Re-run: k-medoids, k=3, stability 0.98 |
| Drift crashed before grading | Harness defect: stateless runs have no run id | Fixed in the runner |
| Drift `broken_feed`: "says the model is fine" | Harness defect: matched the agent's caveat "this does **not** mean the model is still accurate" | Negated sentences no longer count; re-run passes 4 / 4 |
| Forecasting `region_capacity`: "never asked how the cells combine" | Over-strict check: the request already said "for the whole region" and the agent summed the cells | Check retired; the aggregation itself is still checked (and passes) |

## Production defect found: an LLM call could hang a session for an hour

Four scenarios — classification `entity_time_history`, regression
`capacity_p90`, forecasting `region_capacity` and the clustering re-run —
each took over an hour, against 1–5 minutes for everything else. All four
stalled in the same window (13:19–14:17): one LLM request each, open for
58 minutes. The host's `LLM_TIMEOUT` (120 s) is a per-read timeout, and
OpenRouter keeps a slow request alive with whitespace, so it never fires.

**Fixed** in `core/agent_host.py`, the one path every agent's LLM calls
go through: a total deadline per call (`LLM_TOTAL_TIMEOUT`, default
300 s), one fresh retry, then the turn fails with a clear error. Test:
`test_a_hung_model_call_is_abandoned_and_retried`.

## Offline rigour evidence (deterministic, in CI)

Measured on synthetic data with known answers; each is a test that fails
if the behaviour regresses.

| Agent | Measured |
| --- | --- |
| Anomaly | With a rare-but-normal category, KNN/LOF reach average precision 1.0 while IForest reaches 0.12 and ECOD 0.44 — the agent now warns and `compare_detectors` picks by evidence. Training on known-normal rows lifts average precision 0.88 → 1.0 on a dense anomaly cluster. Without labels, IForest's alert list on pure noise overlaps 45% across resamples (gate fails) vs 100% with planted anomalies |
| Forecasting | Hourly traffic repeats weekly (autocorrelation 0.98 vs 0.87 daily): the weekly naive is a 10% WAPE baseline, where the old daily one was a 35% straw man; ETS 7.7%, XGBoost 8.5%. Many series (28 cells): naive 9.0% WAPE; per-series ETS 6.9%, beating each cell's own naive on 86% of cells, interval coverage 83%; global XGBoost 7.4% |
| Drift | With 120 current rows and no real drift, PSI alone raised false alarms 25% of the time; with the noise floor 2.5%, while a real 0.5 sd shift is still caught 92% of the time. A flipped relationship between two columns scores 0.94 on the whole-table check while every per-column check says none |
| Clustering | Mixed numeric/categorical data: Gower k-medoids recovers the true segments at ARI 0.75 vs 0.32 for k-means, stability 0.96 vs 0.59 |

## Limits of this evaluation

- **One model, one run per scenario.** LLM output varies run to run; a
  pass here is one sample. Before release, run each scenario 3–5 times and
  on the production model.
- **Scripted user.** Replies are matched by regular expression; an agent
  asking an unexpected question gets "use your recommended option", which
  is how the `mixed_plans` problem arose.
- **Synthetic data** with known answers — the right test of reasoning,
  not of messy production data.
- **Cost and token use were not measured.**

## How to re-run

```
cd <agent>-agent/scripts && python eval_agent.py              # all scenarios
cd <agent>-agent/scripts && python eval_agent.py <scenario>   # one
```

Each run writes `eval_results.json` (checks, questions asked, tools
called, final reply) to a temp directory and never touches the repo's
`data/` or `mlruns/`. It needs `LLM_PROVIDER`, the agent's
`<AGENT>_AGENT_MODEL` and the provider key in `.env`.
