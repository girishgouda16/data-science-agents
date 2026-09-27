---
name: business-understanding
description: Settle what the forecast is FOR before touching data — the decision, the horizon and granularity it needs, whether the plan uses the expected value or an upper band, which series (one, a total, many), which drivers are known in advance, and known breaks. Load FIRST on any new forecasting request.
---

# Business Understanding (forecasting)

Tool: `record_business_context(run_id, business_objective,
target_definition, success_criteria, domain, success_metric,
success_threshold, direction, assumptions, clarifications)` — call it right
after `prepare_dataset`; everything settled before goes in.

Work through these before `eda`:

1. **What decision does the forecast drive?** A capacity upgrade, a
   staffing roster, a budget, a stock order. It sets everything below.
2. **Horizon and granularity.** How far ahead (an upgrade takes 8 weeks:
   the horizon is at least 8 weeks) and at what step (hourly for staffing,
   monthly for budgets). The horizon goes into `prepare_dataset(horizon=)`
   — a model judged one step ahead says little about week eight.
3. **Expected value or an upper band?** Capacity and staffing are planned
   on a high percentile (the busy hour's P90 = `upper`), budgets on the
   expected value with its bias. Say which one the user will read.
4. **Which series?** One cell, a region's total, or each of many. Several
   rows per timestamp need a decision: `aggregate` ("sum" for volumes,
   "mean" for levels), `series_column` + `series_value` for one, or
   `each_series=true` for every series in one run (all cells of a region).
5. **Drivers known in advance.** Planned events, holidays, tariff launches,
   a site going live: columns whose FUTURE values are known can be used;
   anything only known after the fact (actual weather, actual promotions
   uptake) cannot — the leakage screen flags columns that track the next
   value.
6. **Known breaks.** A migration, a network swap, a pandemic, a tariff
   change: history before a structural break may describe a different
   system. Ask; cut the history or record it.
7. **Bar** — `wape` (%), `mase` (< 1 beats the seasonal naive), `rmse` /
   `mae` in the target's units, `mape` only when the series never nears
   zero. Only if the user gives one; never invent it.

## Ask, or decide and record

**Ask** what the business sets: the decision, horizon, granularity, band,
series, known-future drivers, breaks, the bar, the export path. **Decide
and record** what the data settles: the frequency, the seasonal period
(from the autocorrelation), gap filling for a few periods, the model (from
the backtest). Put each in `assumptions` (one per line), each answered
question in `clarifications`; the report shows both.
