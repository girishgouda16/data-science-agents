---
name: preprocessing
description: Impute missing values, drop identifier/constant columns, decompose datetime columns, and choose the target's modelling space (log1p for skewed money/usage targets). Decide-and-record by default — mechanical choices are measured downstream — with a question only where the data can't settle it (outliers that may be the signal). Load after eda.
---

# Preprocessing

Tools: `propose_imputation` / `apply_imputation`,
`propose_drop_columns` / `apply_drop_columns`,
`propose_datetime_features` / `apply_datetime_features`,
`apply_target_transform(run_id, transform)`.

Needs a `run_id` from `prepare_dataset` — always split before cleaning.
There is no class imbalance step: the target is continuous.

## Decide, then record — ask only where the data can't settle it

Run each `propose_*` and apply its recommendation, then echo one line:
*"Applied: [what changed] — [why]."* Record the choices in
`record_business_context(assumptions=[...])` so the report shows them.
Ask only when the answer is a domain judgment no measurement can make:

- **Outliers in the target or a key driver** — a heavy-tailed target makes
  legitimate extremes (the enterprise account, the flagship store) look
  like outliers. Recommend (usually: keep them, or log1p the target), then
  ask; never cap or drop silently.
- **A name-only identifier match** with no cardinality signal behind it —
  it may be a real feature in this domain.

## Steps

1. **Missing values** — `propose_imputation` → `apply_imputation`. Fill
   values come from the training fold only. First-period NaNs from lag
   features are expected; impute them (median) or leave them for
   `hist_gradient_boosting`, which handles NaN natively.
2. **Identifier and constant columns** — `propose_drop_columns` uses the
   shared identifier rules (near-unique, repeating high-cardinality,
   identifier-ratio, name pattern) → `apply_drop_columns`. The run's split
   keys are never proposed and never dropped (`kept_split_keys`) —
   validation needs them and the pipeline already excludes them.
3. **Datetime columns** — `propose_datetime_features` →
   `apply_datetime_features` adds month / day / weekday / hour (and year,
   except for the run's own time column, which is kept and gets no year —
   the test period's year is one the model never saw).
4. **Target space** — for a non-negative target with `target_skew` above
   ~1, `apply_target_transform(run_id, "log1p")`: a squared-error model on
   the raw scale spends its capacity on the few huge values and fits every
   typical row badly. Metrics and predictions stay in the original units.
   Say why in one line; `"none"` reverts. Retrain afterwards. One catch:
   log1p's back-transform predicts something closer to the median than the
   mean, so SUMMED predictions come out low (`sum_ratio` < 1). If the
   numbers will be totalled into a budget, use `set_objective("poisson")`
   (modeling) instead — the two don't combine.
