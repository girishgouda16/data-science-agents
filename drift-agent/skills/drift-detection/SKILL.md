---
name: drift-detection
description: Compare a reference (baseline) dataset against a current (new) one and produce a per-column and overall drift verdict. Load this once you have both file paths and the user wants to know if the data has drifted.
---

# Drift Detection

Tools: `detect_drift(reference_path="", current_path="", columns="",
profile_path="", key_columns="", target_column="")`,
`detect_multivariate_drift(reference_path, current_path, profile_path, columns)`,
`drift_over_time(reference_path, current_path, time_column, freq, profile_path, columns)`,
`compare_distributions(reference_path, current_path, column)`.
Files in any format (csv, parquet, compressed, json, xlsx); production data
in a database or ClickHouse comes in with `load_dataset` (the `eda` skill).

Both are read-only — no `ask_user` needed, nothing here changes either file.

## Ask for the monitoring profile first

If the drift check is about a model someone trained here, there is almost
certainly a `*.monitoring.json` sitting next to its exported `.pkl`. Ask for
it, and call `detect_drift(profile_path=..., current_path=...)`.

It is not a convenience. It supplies three things this agent cannot work out
for itself:

- **the real baseline.** The training run's `data/runs/<run_id>/train.csv` is
  swept after a retention window. The profile keeps its own copy, so a model
  more than a few days old is still checkable at all.
- **which columns matter.** The verdict is decided by the columns the
  model's own explainer said it uses. Without them, every column votes
  equally and one drifting column nobody models produces a permanent
  "retrain now" — an alarm that fires always is an alarm nobody reads.
- **which column is the label.** Scored separately, because a moved label
  distribution is prior shift and breaks a tuned threshold even when every
  feature is stable.

With no profile, `key_columns` and `target_column` can be passed by hand.
With neither, say plainly in the report that the verdict weighted every
column equally.

## Steps

1. **Full check** — `detect_drift(profile_path=..., current_path=...)`, or
   `detect_drift(reference_path=..., current_path=...)` if there is no
   profile. Read `overall_severity`, `verdict_basis` and `recommendation`
   first, then `drifted_key_columns` for what drove it.
   `drifted_incidental_columns` moved but the model doesn't use them —
   mention them as context, never as the reason to retrain.
   `missing_key_columns` is a blocking finding, not a drift finding: the
   model cannot score that feed at all. Lead with it if present.
2. **Zoom in** — if the user wants more detail on a specific flagged
   column (or you want to sanity-check one before reporting it),
   `compare_distributions(reference_path, current_path, column)` returns
   the actual summary stats/proportions behind that column's PSI.
2b. **Say what you did not check.** Every result carries
   `what_this_did_not_check` and `scope_caveat`. Relay them. This tool
   compares two tables of inputs; it never runs the model, so it cannot see
   prediction drift, concept drift, or current accuracy. "No drift" means
   the inputs still look like training data — it does **not** mean the model
   is still accurate, and reporting it as though it did is the single
   easiest way to mislead someone with this agent. If the user wants to know
   whether the model is still any good, they need current-period labels and
   a backtest from the training agent, not a drift check.

2c. **Read what else it found, first.** `type_changes` (a numeric column
   now arrives as text — a locale or export change: the model cannot use
   it) and `missingness_changes` (a column going null — a broken join or
   a feed that stopped; invisible to PSI, which only sees non-missing
   values) are feed problems, not drift: lead with them.
   `excluded_structural_columns` (timestamps, ids) always differ and are
   left out of the verdict — say so.

2d. **Check the table as a whole** — `detect_multivariate_drift`: a
   classifier tries to tell reference rows from current rows; AUC 0.5 =
   indistinguishable, >= 0.6 moderate, >= 0.75 severe, and `top_columns`
   say which columns give it away. It catches what no per-column check
   can: columns each stable on their own whose relationship changed. Run it
   whenever `detect_drift` says none but someone still suspects a change,
   and on every scheduled check.

2e. **When did it start?** — `drift_over_time(..., time_column, freq)` on a
   current file with a timestamp: per-period severity per column,
   `first_drift`. A step that persists = a change on that date (release,
   tariff, feed); a slow climb = gradual drift; one spike = an event.

3. **Report, don't act** — state the overall verdict and which columns
   drove it. `recommendation` is a suggestion for the user, never an
   instruction you act on yourself — this agent has no train/retrain tools,
   and `detect_drift` never modifies either input file. If the user decides
   to retrain, that's a new, separate request to classification/regression/
   forecasting-agent (or the orchestrator, if one is available) — not
   something to chain from here.

## Compare like with like

A reference from December against a January current shows seasonality,
not decay; a weekday against a weekend shows the weekly cycle. Telecom
usage moves with the calendar, tariff launches and network changes — when
the reference is a training window, say which season it covers, and prefer
a current window of the same length and kind.

## Reading severity

- `severity: "none"` (PSI < 0.1) — no meaningful shift for that column.
- `severity: "moderate"` (0.1 ≤ PSI < 0.25) — real shift, not yet extreme.
- `severity: "severe"` (PSI ≥ 0.25) — substantial shift; treat the current
  model's predictions on this data with real suspicion.
- A PSI above 0.1 but below the column's `psi_noise_floor` is sampling
  noise for these sample sizes (`noise_note`) and is not flagged — with a
  few hundred current rows, PSI alone raises false alarms a quarter of the
  time. `small_sample_warning` says when the current file is that small.
- A missing share that moved 5+ points (and doubled) is at least moderate,
  20+ points severe; a type change is always severe.

`p_value` (from the KS-test for numeric columns, chi-square for categorical
ones) is supporting context, not what severity is based on — it gets
noisier as row counts grow or shrink, while PSI stays comparable across
dataset sizes. Trust the PSI-based `severity`/`drifted` fields; mention the
p-value only if it adds something (e.g. it's non-significant despite a
moderate PSI, worth flagging as "borderline").
