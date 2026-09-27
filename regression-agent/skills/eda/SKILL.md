---
name: eda
description: Get the data (any file format, or a named database / ClickHouse / object-store source), explore it — shape, dtypes, missingness, target distribution and skew, outliers — and split it into a run (random, by entity, or forward in time, with unsettled targets excluded) before any cleaning or modeling. Load this first on any new dataset.
---

# EDA & the split

Tools: `list_data_sources()`, `load_dataset(source, query, name, max_rows)`,
`eda(path)`, `propose_group_column(path, target)`,
`propose_time_column(path)`,
`prepare_dataset(path, target, test_size, group_column, time_column, immature_after)`,
`detect_outliers(run_id)`.

0. **Get the data.** A file path works directly in every tool: csv/tsv
   (also .gz/.bz2/.xz/.zst), parquet, feather, json/jsonl, xlsx, and
   tar/tgz/zip archives whose members share a schema. For a database,
   warehouse or bucket, `list_data_sources()` shows the configured names
   (`sql:…`, `clickhouse:…`, `store:…`) and `load_dataset(source, query)`
   snapshots it to parquet — use that path from here on. **Never ask for a
   password or connection string**; an unconfigured source is an ops
   request. Push filters, joins and aggregation into the SQL. A
   `truncated` result is the first N rows the database returned — often the
   oldest — not a random sample: say so, or narrow the query.
1. **EDA** — `eda(path)`: shape, dtypes, missingness, target distribution.
2. **How to split — the decision that cannot be corrected later.**
   - `propose_group_column(path, target)`: a repeating entity (customer,
     store, device)? If its rows could land on both sides, metrics grade
     the model on entities it already saw. Ask whether to split by it.
   - `propose_time_column(path)`: rows ordered in time? A random split
     trains on the future; any lag or trend feature crosses the boundary.
     Ask whether to split forward in time. Pass both for entity-per-period
     data; `entity_overlap_pct` then reports how many test rows belong to
     entities seen in training.
   - **Target maturity.** A target that accrues after the row's date
     (revenue still coming in, claims still open, next month's usage)
     is unsettled for the newest rows — pass `immature_after` (newest
     date minus the settling time; ask how long if nobody said).
3. **Split** — `prepare_dataset(...)`. Exact duplicates go first; returns a
   `run_id` that every later tool takes. Quote `split_type` with any
   metric. The entity id and time column stay in the data for validation
   and never reach the model.
4. **Target shape** — `target_skew` and `target_min` come back with the
   split. A non-negative target with skew above ~1 (money, usage, counts)
   is a candidate for `apply_target_transform("log1p")` in
   `preprocessing` — decide it there, record the choice.
5. **Outliers** — `detect_outliers(run_id)` (z-score, training fold
   only). Report; what to do about them is a `preprocessing` decision.
