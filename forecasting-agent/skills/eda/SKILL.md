---
name: eda
description: Load the series from wherever it lives (files in any format, SQL, ClickHouse, object stores), look at it, then build a regular series — one, a total, or every series of a long table in one run — missing periods inserted, the season chosen from the autocorrelation — and split it CHRONOLOGICALLY at the business horizon into a run. Load this first on any new dataset.
---

# EDA & Dataset Setup

Tools: `list_data_sources()`, `load_dataset(source, query, name)`,
`eda(path)`, `prepare_dataset(path, date_column, target, test_size, freq,
horizon, aggregate, series_column, series_value, seasonal_periods,
each_series)`,
`detect_outliers(run_id)`.

1. **Intake** — a file path works directly (csv, parquet, compressed csv,
   json, xlsx, archives). Data in a database, ClickHouse or a bucket:
   `list_data_sources()`, then `load_dataset(source, query=...)` — push the
   aggregation to the period grain into the query (`GROUP BY toStartOfHour
   (ts)`); never ask for credentials.
2. **EDA** — `eda(path)`: the timestamp, the target, other columns
   (exogenous drivers or other series?), missingness.
3. **Prepare** — `prepare_dataset(path, date_column, target, horizon=...)`:
   - **horizon** from `business-understanding` — the held-out period and
     every backtest origin forecast exactly that far.
   - **several rows per timestamp** → it refuses and asks: `aggregate`
     ("sum" for traffic, calls, revenue; "mean" for price, latency), or
     `series_column` + `series_value` for one series, `series_column` +
     `aggregate` for the total, or `series_column` + `each_series=true` for
     **every series in one run** (each cell, region or product forecast;
     `aggregate` then combines duplicates within a series). The user's call
     — relay it. Series that stopped reporting before the held-out period
     or are too short to backtest are left out and listed
     (`series_left_out`) — say which and why.
   - **missing periods** are inserted so lags mean "one day ago", not "one
     row ago"; `gaps_inserted` says how many. The target is empty there:
     fill it (`preprocessing`) before training. A long outage is better cut
     than filled — ask.
   - **season** — `seasonality_autocorrelation` at each candidate (hourly:
     24 and 168) and the chosen `seasonal_periods` (the strongest with 3+
     cycles of history). Pass `seasonal_periods` to override with a reason.
   - `freq` is inferred, also with a few gaps; pass it when timestamps are
     irregular.
   Returns a `run_id` — every later tool takes it.
4. **Outliers** — `detect_outliers(run_id)` (target, training fold). An
   outage or an event is often the most important row: decide with the
   user, never smooth silently.
