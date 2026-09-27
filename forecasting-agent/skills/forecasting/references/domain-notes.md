---
name: domain-notes
description: Domain-specific model choice and evaluation expectations for retail demand, logistics volumes, and ops/infra metrics. Referenced automatically by the forecasting skill — do not load separately.
---

# Domain Notes

This reference is appended to the forecasting skill automatically.

---

## Retail — demand / revenue forecasting

**Primary metric:** MAPE — merchandising and finance teams think in
percentage terms ("we're usually within 10%"), not raw units. Report
RMSE/MAE alongside it for the "how many units, in absolute terms" question,
especially for low-volume SKUs where MAPE can look artificially bad on a
single-digit true value.

**Model choice:** `xgboost` with promo/holiday flags as exogenous columns
usually wins — retail demand is rarely pure trend+seasonality, promotions
and holidays create real non-additive spikes. `exponential_smoothing` is a
fine fast baseline for a stable, non-promoted SKU.

**Seasonality:** daily retail data usually has BOTH weekly (weekday vs.
weekend) and annual (holiday) seasonality — a single `seasonal_periods`
value only captures one. If annual seasonality matters and you have >2
years of daily history, flag to the user that this agent's single-period
seasonal model may underfit the annual cycle.

---

## Logistics — parcel / shipment volume forecasting

**Primary metric:** MAE in shipment counts — ops capacity planning cares
about absolute headcount/truck-count implications, not percentage error.

**Model choice:** `exponential_smoothing` is often sufficient for volume at
the network level (strong weekly seasonality, gentle trend). Reach for
`xgboost` when forecasting at a finer grain (per-warehouse, per-route)
where local exogenous effects (weather, regional promotions) matter.

**Outliers:** a volume spike right before a known peak event (holiday
shipping) is signal, not noise — check `detect_outliers`' flagged rows
against a calendar of known events before assuming they should be
smoothed.

---

## Ops / infrastructure — traffic, latency, resource-usage forecasting

**Primary metric:** RMSE in the metric's own units (requests/sec, ms,
CPU%) — capacity planning decisions are made on absolute headroom, not
percentage error.

**Model choice:** `naive_seasonal` is a surprisingly strong baseline for
infra metrics with clean daily/weekly cycles (traffic today ≈ traffic
exactly one cycle ago) — always compare against it before committing to
`xgboost`'s extra complexity and slower retraining.

**Explainability:** for capacity-planning decisions that feed
provisioning, prefer `exponential_smoothing`'s trend/seasonal decomposition
over `xgboost`'s feature importances — "the trend is rising 3%/week" is
directly actionable in a way a lag-feature ranking isn't.

---

## Telecom — network traffic, subscriber base, revenue, contact volume

**Network traffic (per cell, site, region; hourly or daily).** The decision
is capacity: an upgrade takes weeks to months, so the horizon is long and
the number that matters is the busy hour's upper band, not the mean —
plan on `upper` (P90) of the busiest hour, not on `forecast`. Hourly
traffic repeats weekly (weekends differ), so the season is usually 168,
not 24; `prepare_dataset` picks it from the autocorrelation. Forecast the
total of a region (`series_column` + `aggregate="sum"`) or one cell
(`series_value`); a cell whose neighbour was just switched on has a
structural break — cut the history after it. Known-future drivers:
planned events (concerts, matches), holidays, a site going live —
calendar columns known at forecast time; last week's weather is not.

**Subscriber base, gross adds, churned lines (daily / monthly).** Driven by
tariff launches, handset launches and competitor moves — each is a level
shift, not noise. A monthly series has few points: prefer
`exponential_smoothing`, report WAPE and bias (finance cares whether the
plan is systematically high), and say plainly when a launch is inside the
held-out period.

**Revenue / ARPU.** Revenue = base × ARPU — forecasting the two separately
and multiplying usually beats forecasting revenue directly, because they
move for different reasons. Watch bias: a +3% bias on revenue is a budget
miss.

**Contact-centre volume (hourly / half-hourly).** Staffing is set per
interval, so accuracy per interval (WAPE) and the upper band matter;
outages and bill runs cause predictable spikes — pass them as known-future
columns when they are scheduled.

**Metrics:** WAPE for volumes (total error / total volume — no division by
near-zero hours at night), MASE to show the model beats the seasonal
naive, bias to show it is not systematically high or low, interval
coverage to show the band can be planned on.
