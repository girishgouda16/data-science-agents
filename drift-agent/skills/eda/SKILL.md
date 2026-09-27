---
name: eda
description: Look at one CSV's shape, dtypes, missingness, and summary stats before comparing it against another. Optional first step — load this when the user hands you a file and wants a look at it before/instead of a drift check.
---

# EDA

Tools: `eda(path)`, `list_data_sources()`, `load_dataset(source, query, name)`.

Production data usually lives in a database or ClickHouse: `list_data_sources()`
for the configured names, then `load_dataset("clickhouse:<name>", query=...)`
— one read-only SELECT for the current window (the same columns and grain
as the reference) — and pass the snapshot path to the drift tools. Never
ask for credentials.

Read-only — report findings and move on, no `ask_user` needed (this agent
has no data-changing tools at all, so there's nothing to gate here).

Call `eda(path)` on either the reference or current file to see shape,
dtypes, missingness, and summary stats before running `detect_drift`. This
is optional — if the user already knows both files and just wants the drift
verdict, skip straight to the `drift-detection` skill.
