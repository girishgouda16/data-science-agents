---
name: preprocessing
description: Impute missing values and drop identifier/constant columns before detection — decide and record by default; scaling, logging of skewed numerics and one-hot encoding happen inside the detector pipeline. Load after eda shows missingness or ID-like/constant columns.
---

# Preprocessing

Tools: `propose_imputation(run_id)` / `apply_imputation(run_id, strategies)`,
`propose_drop_columns(run_id)` / `apply_drop_columns(run_id, columns)`.

Needs `run_id` from `prepare_dataset`. Build entity and peer features
(`feature-engineering`) BEFORE this — dropping the id or the timestamp
leaves nothing to group or order by (split keys are kept anyway).

1. **Missing values** — `propose_imputation` → `apply_imputation`. Fill
   values come from the training fold only. Missingness can itself be the
   anomaly (a probe that stopped reporting): when a column's missing share
   differs by label, or when missing means "none" (no international
   calls), fill with 0 or add a flag rather than the median.
2. **Identifiers and constants** — `propose_drop_columns` →
   `apply_drop_columns`. A detector keyed on an id flags what is rare in
   the id, not in behaviour.
3. **The distance space is built for you** (inside `train_model`):
   right-skewed non-negative numerics are logged, numerics standardised,
   categoricals one-hot and NOT scaled. `train_model` reports
   `log_scaled_columns`.
4. **Leave labels alone** — a label column is never a feature.

Decide and record (`record_business_context(assumptions=...)`); ask only
when a column's meaning is unclear (is it written after the case was
decided?).
