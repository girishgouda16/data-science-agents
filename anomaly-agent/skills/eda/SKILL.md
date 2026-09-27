---
name: eda
description: Load the data from wherever it lives (files in any format, SQL, ClickHouse, object stores), explore it — shape, dtypes, missingness, whether a label exists — and split it into a run (by time, by entity, or stratified) before any preprocessing or detection. Load this first on any new dataset.
---

# EDA & Dataset Setup

Tools: `list_data_sources()`, `load_dataset(source, query, name)`,
`eda(path)`, `prepare_dataset(path, label_column, test_size, time_column,
group_column)`.

1. **Intake** — a file path works directly (csv, parquet, compressed csv,
   json, xlsx, archives). Data in a database, ClickHouse or a bucket:
   `list_data_sources()` for the configured names, then
   `load_dataset("sql:<name>" | "clickhouse:<name>" | "store:<name>",
   query=...)` — one read-only SELECT; push filters and aggregation into
   it. Never ask for credentials.
2. **Event rows?** (one row per call, session, transaction) — build the
   per-entity behaviour first (`feature-engineering`, `aggregate_events`),
   then prepare the aggregated file.
3. **EDA** — `eda(path)`: shape, dtypes, missingness, a candidate label
   column, an entity id and a timestamp.
4. **Split** — `prepare_dataset(path, label_column, ...)`:
   - `time_column` when rows carry a time: the detector is judged on
     LATER rows, as it will be used. Default for fraud, faults and
     anything that evolves.
   - `group_column` (subscriber, device, card) when entities repeat: with
     a time column it is kept for entity features; without one the split
     keeps each entity on one side.
   - neither: stratified on the label, or random.
   Both keys stay in the data and never reach the detector. A label is
   used only to evaluate. Returns a `run_id` — every later tool takes it.
