---
name: eda
description: Explore a tabular dataset before touching it — shape, dtypes, missingness, target distribution, class imbalance, outliers — then split it into a run before any cleaning or modeling. Load this right after business-understanding on any new dataset.
---

# EDA

**MCP module:** `mcp_server/eda.py`

Tools: `eda(path)`, `inspect_target(path, target)`,
`propose_group_column(path, target)`, `propose_time_column(path)`,
`prepare_dataset(path, target, test_size=0.2, group_column="", time_column="")`,
`check_imbalance(run_id)`, `detect_outliers(run_id)`.

Steps 1, 2, 5, 6 are read-only — run them without stopping to ask the user;
they change nothing. Steps 3 and 3b (how to split) are the gates that matter
most in this skill: they are the only decisions here that cannot be
corrected later, because every metric downstream is computed against
whatever split they produce.

0. **Get the data.** A file path works directly in every tool: csv/tsv
   (also .gz/.bz2/.xz/.zst), parquet, feather, json/jsonl, xlsx, and
   tar/tgz/zip archives whose members share a schema. When the data lives
   in a database, warehouse or bucket, `list_data_sources()` shows the
   configured names (`sql:…`, `clickhouse:…`, `store:…`) and
   `load_dataset(source, query)` snapshots it to parquet — use that path
   from here on; the run records the source and query. **Never ask the user
   for a password or connection string** — an unconfigured source is an ops
   request. Push filters, joins and heavy aggregation into the SQL. If the
   result says `truncated`, the row cap cut it: a capped table is its first
   N rows in whatever order the database returned — often the oldest — not
   a random sample; say so, or narrow the query.
1. **EDA** — `eda(path)`. Look at shape, dtypes, missingness, target
   distribution before touching anything else. This decides whether
   `data-cleaning` is even needed. Also generate the standard EDA plots
   here via the visualization agent — `plot_missingness(path, out_path)`,
   `plot_class_distribution(path, target, out_path)`,
   `plot_correlation_heatmap(path, out_path)` — and keep their returned
   `markdown` for `reporting/SKILL.md`'s EDA section; these take the raw
   `path`, not a `run_id`, so they only need calling once per dataset.
2. **Target shape** — `inspect_target(path, target)`. Dtype, class count,
   binary vs multiclass, class balance % — on the RAW file, before the
   split. Decides stratification and which metrics (binary ROC/PR vs
   macro-averaged) apply for the rest of the run.
3. **Group column check** — `propose_group_column(path, target)`. Flags any
   column that looks like a repeating ENTITY id (a customer, caller,
   account...) — the same identifier signals as `data-cleaning`'s
   `propose_drop_columns`, but only for columns that actually repeat.
   **Why this matters:** if the same entity's rows can land in both train
   and test (e.g. one row per caller per day — a real telco file this was
   found on had 52.6% of rows belonging to a repeated caller), a plain
   random split lets the model partially recognize an entity it already saw
   in training, inflating every metric downstream in a way nothing else in
   this agent would catch after the fact. **Never assume a column name** —
   the entity id is different in every dataset (a caller in telco data, an
   account in fintech data, a patient in healthcare data); always ask:

   > *"Column(s) [name(s)] look like a repeating entity id ([N] distinct
   > values, [M] rows share a repeated one). If the same entity can appear
   > in both train and test, metrics will look better than true
   > generalization. Split by entity instead of by row?"*
   > - "Yes, split by [recommended column] — prevents entity leakage"
   > - "Yes, but a different column — I'll specify"
   > - "No, split by row as usual (default, and correct when there's no
   >   repeating entity — e.g. row_id was already unique)"

   Skip this gate only when `propose_group_column` returns no candidates —
   nothing to ask about.
3b. **Time axis check** — `propose_time_column(path)`. Flags any column that
   orders the rows in time (a date/timestamp, or an integer period index
   whose name says it is one). **This is a separate leak from step 3's, and
   nothing else in the agent catches it.** A stratified random split on
   time-ordered rows puts later rows in training and earlier rows in test,
   so every backward-looking feature — a lag, a rolling mean, a trend, a
   "days since" recency — is computed partly from rows the model also
   trained on, and the test fold stops being a forecast of anything. The
   metrics look *better*, which is what makes it dangerous.

   Ask whenever a candidate exists AND any feature in the dataset looks
   backwards (or is about to — `feature-engineering` will add trend and
   window features later, so a dataset with a time axis usually ends up
   with them even if it has none yet):

   > *"Column [name] orders these rows in time ([span]). If the split is
   > random, training data includes rows from after the test period, and
   > any trailing-window or trend feature is computed across that boundary.
   > Split forward in time instead — train on the earlier rows, test on the
   > later ones?"*
   > - "Yes, split forward on [column] — test becomes a real hold-out future"
   > - "Yes, and also group by [entity] — report how much entity overlap remains"
   > - "No, random split (correct when rows are independent snapshots with
   >   no window features and no time ordering to respect)"

   A temporal split cannot be stratified — the class mix of the future is
   not ours to choose — so `prepare_dataset` verifies both classes survive
   in both folds and errors out rather than handing back a single-class
   fold. **That error is a finding, not an obstacle:** it means the
   positives are concentrated in one period, which tells you something real
   about the label before any model exists.

   Passing both `time_column` and `group_column` is the honest option for
   entity-per-period data. The split stays temporal and the result reports
   `entity_overlap_pct` — how many test rows belong to an entity also seen
   in training. Do not treat a high number as a failure: a production
   scorer really does see yesterday's callers again. Do surface it, because
   it means the score is evidence about *returning* entities and not about
   new ones.

4. **Split** — `prepare_dataset(path, target, group_column=..., time_column=...)`.
   If the rows are EVENTS (one per call, session, transaction) and the
   label is per entity, the split comes after `feature-engineering`'s
   `aggregate_events` — split the file it writes, not the raw one.
   **Label maturity:** when outcomes are confirmed late (fraud, chargebacks,
   default, "churned next month"), pass `immature_after` = newest date minus
   the settling time, so unsettled rows — negatives only because nobody has
   looked yet — never reach the test fold. Ask how long outcomes take to
   settle if the user hasn't said; it is not inferable from the data.
   Train/test split done *before* any imputation, encoding, or SMOTE —
   that ordering is what keeps every later step leakage-safe. Exact duplicate
   rows are dropped first (reported as `duplicates_dropped` — mention it when
   non-zero), so no row can sit in train with its twin in test. Pass whatever
   steps 3 and 3b settled (empty strings for a plain stratified per-row
   split). Returns a `run_id`, plus `split_type` — quote that word when you
   report results, since "PR-AUC 0.62" means a different thing under each
   of the three. **Every tool from here on — in this
   skill, `data-cleaning`, and `modeling` — takes `run_id`, not the
   original file path.** Do this immediately after step 3, even if no
   cleaning turns out to be needed. If `business-understanding` ran first
   (it should, on any new dataset), call its `record_business_context` tool
   right here, now that a `run_id` exists to persist it to.
5. **Class balance** — `check_imbalance(run_id)`. Reads the *training* fold
   only (matches what the model will actually be fit on). If the minority
   class is under ~10%, note it — `train_model`/`tune_hyperparams`
   (in the `modeling` skill) already apply `class_weight="balanced"`
   automatically, that's free and always on. Only the `data-cleaning` skill's
   SMOTE step is a bigger hammer, and only worth reaching for if balanced
   class weights aren't enough.
6. **Outliers** — `detect_outliers(run_id)` (z-score, training fold only).
   Don't decide what to do about them here — that's a data-changing
   decision, hand it to the `data-cleaning` skill.
7. **Phase gate (autonomous by default)** — after all read-only steps above,
   surface a concise findings summary in plain language: row/column count,
   which columns have missingness and at what %, class imbalance ratio,
   outlier count and which columns are affected. Then proceed automatically —
   no `ask_user`:
   - If *any* of: missingness > 0, ID-like or constant columns present,
     minority class < 10% and SMOTE may help, or outliers present — go
     straight to `data-cleaning` with this `run_id`.
   - Otherwise go straight to `modeling`.

   This is a routine "which skill runs next" decision, not a genuinely
   ambiguous one — only call `ask_user` here if the user asked for
   step-by-step control, or the findings are contradictory enough that
   neither path is clearly right.

   **Exception — the split has both a group_column and a time_column:**
   load `feature-engineering` and run `apply_entity_features` (and
   `apply_peer_features`) FIRST.
   History features leave NaN where an entity has no earlier rows;
   running them before cleaning lets imputation fill those gaps. (The entity
   id and time column themselves stay in the data for validation — the
   model pipeline excludes them — so cleaning never removes them.)