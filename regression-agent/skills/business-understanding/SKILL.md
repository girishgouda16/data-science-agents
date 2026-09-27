---
name: business-understanding
description: Settle what the predicted NUMBER is for before touching data — the decision it feeds, what the target measures and in which units, the horizon, when the target settles, which errors cost most, and the success bar in business terms. Ask when the answer changes what "better" means; record an assumption otherwise. Load FIRST on any new regression request.
---

# Business Understanding (regression)

Tool: `record_business_context(run_id, domain, business_objective,
target_definition, success_metric, success_threshold, success_criteria,
target_provenance, assumptions, clarifications)` — call it right after
`prepare_dataset` returns a `run_id`; everything settled before that goes
in.

A regressor can be well fitted and still answer the wrong question. Work
through these, one at a time, before `eda`/`prepare_dataset`:

1. **What decision does the number feed?** A budget, a price, a capacity
   plan, a retention offer sized by predicted value, a reserve. "Predict
   revenue" is not a decision; "set next quarter's collections budget per
   region" is. The decision decides everything below.
2. **What exactly is the target, in which units, over which horizon?**
   Gross or net? Per subscriber-month or per account-year? Billed or paid?
   Next 30 days, or lifetime? Two readings of one column are two models.
3. **When does the target settle?** Revenue finalises after billing runs,
   claims after they are paid, "next month's usage" after next month ends.
   Rows newer than that are not low values, they are unfinished ones — ask
   how long, and pass `immature_after` to `prepare_dataset`. Record it as
   `target_provenance`.
4. **Which errors cost most?** Under-forecasting capacity drops calls;
   over-forecasting wastes spend. Symmetric costs → MAE; a few large misses
   are disastrous → RMSE; errors matter relative to size (small and large
   accounts alike) → consider the log1p target and a relative metric. If
   costs are clearly asymmetric, the answer is a QUANTILE, not the mean:
   under-predicting k times as costly as over-predicting → predict the
   k/(k+1) quantile (`set_objective("quantile", ...)`, 5:1 → 0.83). If the
   predictions will be summed into a budget, the mean must be unbiased on
   totals — a poisson objective for non-negative targets. When the
   decision needs a range, `calibrate_intervals` is part of the deliverable.
5. **Prediction unit** — one row = one what? This decides whether the run
   needs a `group_column` (entity) and a `time_column`.
6. **Success bar** — in business terms first ("MAE under 5 per
   subscriber-month, because the offer budget is set within ±5"), then as
   `success_metric` (r2 / rmse / mae / median_ae / pinball_loss) + `success_threshold`. **Never invent
   a threshold**: no bar stated and none implied by the domain means record
   none — the report then says none was set.

## Ask, or decide and record

**Ask** when the answer defines what "better" means or nothing downstream
can check it: the decision, the target definition and units, the horizon,
the settling time, the cost asymmetry, the success bar.

**Decide and record** everything a measurement settles later: imputation,
dropping identifier columns, the target transform, the model (CV ranks
it), standard features. Put each in `assumptions` (one per line); put each
question the user answered in `clarifications`. The report renders both,
which is what makes deciding safe rather than presumptuous.
