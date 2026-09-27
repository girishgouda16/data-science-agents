---
name: preprocessing
description: Impute missing values, drop identifier/constant columns, and keep outcomes and protected attributes OUT of the distance (profile columns) for clustering. Decide-and-record by default; scaling, one-hot and logging of skewed features happen inside the clustering pipeline.
---

# Preprocessing

Tools: `propose_imputation` / `apply_imputation`,
`propose_drop_columns` / `apply_drop_columns`,
`set_profile_columns(run_id, columns)`.

Needs a `run_id` from `prepare_dataset` — always split before cleaning.

## Decide, then record

Run each `propose_*` and apply its recommendation, then echo one line:
*"Applied: [what changed] — [why]."* Record each choice in
`record_business_context(assumptions=...)`. Ask only when the data can't
settle it: a name-only identifier match with no cardinality signal, or a
column whose role (feature, outcome, protected) is unclear.

## Steps

1. **Profile columns first** — `set_profile_columns(run_id, "churned,arpu,
   age")`: outcomes and protected attributes stay in the data, never enter
   the distance, and are reported per segment by `explain_model`.
2. **Missing values** — `propose_imputation` → `apply_imputation`, fill
   values from the fit fold only.
3. **Identifier and constant columns** — `propose_drop_columns` (shared
   identifier rules) → `apply_drop_columns`. Distance on an ID splits
   every row into its own cluster. Tracking segments over time? Keep the
   subscriber ID and the period with `set_profile_columns` instead — out
   of the distance, still in the data for `segment_migration`.
4. **The distance space is built for you** (inside `train_model`):
   numerics imputed and standardised, right-skewed non-negative numerics
   (usage, spend, counts) logged first so heavy users don't become their
   own segments, categoricals one-hot and NOT scaled (a scaled rare dummy
   would dominate). `train_model` reports `log_scaled_columns`. For
   `kmedoids` (mixed data) it builds the Gower space instead: numerics
   min-max scaled, each category weighted so a mismatch counts 1.
