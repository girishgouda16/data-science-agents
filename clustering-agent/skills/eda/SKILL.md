---
name: eda
description: Get the data (any file format, or a named database / ClickHouse / object-store source), explore it — shape, dtypes, missingness, skew — and split it into a fit fold and a holdout fold before any learned preprocessing. Load this first on any new dataset.
---

# EDA & the split

Tools: `list_data_sources()`, `load_dataset(source, query, name, max_rows)`,
`eda(path)`, `prepare_dataset(path, test_size=0.2)`.

Read-only steps — report findings and move on.

0. **Get the data.** A file path works in every tool: csv/tsv (also
   compressed), parquet, feather, json/jsonl, xlsx, tar/tgz/zip archives
   whose members share a schema. For a database, warehouse or bucket,
   `list_data_sources()` shows the configured names and
   `load_dataset(source, query)` snapshots it to parquet. Never ask for a
   password or connection string. If the data is EVENT rows (calls,
   sessions, transactions), build per-entity behaviour first with
   `feature-engineering`'s `aggregate_events` and split its output.
1. **EDA** — `eda(path)`: shape, dtypes, missingness, and skew. Heavily
   skewed non-negative columns are logged automatically in the pipeline;
   note which ones.
2. **Split** — `prepare_dataset(path)`: exact duplicates dropped, then an
   80/20 fit/holdout split before any learned imputation, encoding or
   scaling. Returns a `run_id` every later tool takes.
3. **What the holdout is for** — there is no label to predict: the holdout
   checks that the model can place unseen rows consistently with how a
   clustering of those rows would group them (`holdout_ari`). The main
   evidence that segments are real is bootstrap stability, computed at
   training time.
