---
name: domain-notes
description: Domain-specific metric priority and explainability requirements for retail pricing/demand, logistics ETAs, fintech risk scoring, and asset valuation. Referenced automatically by the regression skill — do not load separately.
---

# Domain Notes

This reference is appended to the regression skill automatically. Use it to
decide what to optimize for and how much explainability a given domain
expects. There's no decision threshold or class imbalance here — every
domain below has a continuous target, scored on R^2/RMSE/MAE.

---

## Retail — pricing / demand forecasting

**Primary metric:** RMSE/MAE in the target's own units (dollars, units
sold) — easier for a business stakeholder to reason about than R^2 alone.
Lead with R^2 for model comparison, but always report RMSE/MAE alongside it
for the "how far off, in real terms" question.

**Model choice:** `random_forest`/`xgboost` are usually fine — pricing and
demand curves are rarely linear (seasonality, promo effects, category
interactions). Reach for `linear_regression` only if the stakeholder needs
transparent, additive coefficients (e.g. "how much does a $1 price change
move demand").

**Explainability:** Permutation importance is usually sufficient — internal
merchandising/pricing decisions, not customer-facing. Watch for a date/SKU
ID feature ranking high in `explain_model` — likely leakage, not signal.

**Outliers:** A target outlier (e.g. a holiday demand spike) is often the
most important row in the dataset, not noise — flag `target_outlier_count`
from `detect_outliers` explicitly rather than defaulting to "cap or drop."

---

## Logistics — delivery time / ETA

**Primary metric:** MAE in minutes/hours — the number ops teams actually
plan around. R^2 is secondary context, not what gets surfaced first.

**Model choice:** `xgboost` typically wins on ETA problems (nonlinear
interactions between distance, traffic, time-of-day) if the baseline
`random_forest`'s MAE isn't tight enough. Reach for `linear_regression`
only for a fully transparent baseline comparison.

**Explainability:** Permutation importance for internal ops use. If ETA
predictions are surfaced to customers, consider SHAP so a specific
late-delivery prediction can be explained rather than just ranked globally.

**Outliers:** Extreme delivery times often indicate a genuine operational
failure (route exception, warehouse delay) worth surfacing on their own,
not just averaged into MAE — check `target_outlier_count`.

---

## Fintech — risk scoring / LTV / default-amount estimation

**Primary metric:** R^2 for overall fit, but MAE is what a risk or finance
team will ask for directly (expected dollar error per account).

**Explainability:** SHAP is often required by the same regulatory pressure
as classification credit models (adverse-action explanations) — default to
`explain_model(run_id, method="shap")` for any model that feeds a pricing,
limit, or risk-tier decision. Favor `linear_regression` over tree ensembles
when a fully auditable, coefficient-level model is required.

**Outliers:** Do not silently cap extreme target values (large loss/LTV
amounts) — these are frequently the cases the business most needs the
model to get right. Surface `target_outlier_count` and ask before any
capping/dropping.

---

## Telecom — ARPU, customer value, usage, network KPIs

**Primary metric:** MAE per subscriber-month (or per cell-hour for network
KPIs), in the target's units. ARPU, data usage and customer value are
non-negative and heavily right-skewed — `apply_target_transform("log1p")`
is usually right, and a relative view (error as a share of the typical
bill) is what finance reads.

**Settling:** billed revenue finalises after the billing run and
adjustments; "next month's usage" is only known once that month closes —
`immature_after` on every forward split.

**Grain and split:** subscriber-month data is entity × period — split
forward in time with the subscriber as `group_column`, and build history
features (`apply_entity_features`) before cleaning. Multi-line accounts:
model at the level the decision is made (usually the account).

**Intervals:** capacity and budget decisions need the range, not the point
— `calibrate_intervals`. Network KPIs are heteroscedastic across cells;
check residuals before trusting a constant-width interval.

**Peers:** throughput or drop rate against the same cell / technology
(`apply_peer_features`) separates a site problem from a customer one.

## Real estate / asset valuation

**Primary metric:** RMSE/MAE in the target's currency units — an R^2 of
0.9 sounds good until MAE reveals the model is routinely off by $50k on a
$300k property.

**Model choice:** `random_forest`/`xgboost` handle non-linear
location/size/condition interactions well. `linear_regression` only if the
stakeholder wants a transparent per-feature dollar contribution.

**Explainability:** SHAP is expected for any valuation that feeds an
appraisal, loan, or listing-price decision — a single dominant feature (see
the leakage-signal note in the regression skill) is a red flag worth
investigating before export, not a headline result.
