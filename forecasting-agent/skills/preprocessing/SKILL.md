---
name: preprocessing
description: Impute missing values (forward-fill, not median/mode — order matters here) and drop junk columns. Every tool here changes the run, so it's always propose → ask_user → apply, never a silent one-shot. Load this after eda shows missingness or ID-like/constant columns.
---

# Preprocessing

Tools: `propose_imputation(run_id)` / `apply_imputation(run_id, strategies)`,
`propose_drop_columns(run_id)` / `apply_drop_columns(run_id, columns)`.

Needs `run_id` from `prepare_dataset` (see the `eda` skill).

There's no class-imbalance/SMOTE step and no datetime-decomposition step
here — the date column already drives the split and the model's seasonal
features directly; there's no separate "engineer datetime features" action
the user needs to approve.

## Steps

1. **Missing values** — `propose_imputation` → `apply_imputation`. For the
   target (including periods `prepare_dataset` inserted): `0` for a count
   series where a missing period means nothing happened (confirm the feed
   was up), `interpolate` for a level, `ffill` for a value that holds
   until it changes (a price, a tariff). Never fill a long outage — cut the
   history after it instead, and ask. Exogenous columns: `ffill`.
2. **Column pruning** — `propose_drop_columns` → `apply_drop_columns`
   (constants, ID-like). Never the date column or the target.
3. **Outliers** — for what `detect_outliers` flagged: an outage, an event
   or a launch is usually real and often the most important row; cap or
   drop only with the user, never silently.

Decide and record (`record_business_context(assumptions=...)`) a fill of a
few periods; ask before filling more than a handful of target periods.
