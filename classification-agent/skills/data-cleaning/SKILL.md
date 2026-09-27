---
name: data-cleaning
description: Impute missing values, drop junk columns, and rebalance via SMOTE. Auto-applies recommended defaults; only asks for name-only ID-column drops and outlier handling, where there's no dataset-agnostic default. Load this after eda shows missingness, ID-like/constant columns, or imbalance that class weights alone won't fix.
---

# Data Cleaning

**MCP module:** `mcp_server/data_cleaning.py`

Tools: `propose_imputation(run_id)` / `apply_imputation(run_id, strategies)`,
`propose_drop_columns(run_id)` / `apply_drop_columns(run_id, columns)`,
`propose_datetime_features(run_id)` / `apply_datetime_features(run_id, columns)`,
`apply_frequency_encoding(run_id, columns)`,
`propose_smote(run_id)` / `apply_smote(run_id)`.

Needs `run_id` from `prepare_dataset` (see the `eda` skill) — always split
before cleaning, never clean the original file directly.

## Autonomous mode (default)

Unless the user asks for step-by-step control, run each applicable step by
calling the `propose_*` tool and then the matching `apply_*` tool with the
recommended values, back to back, without an `ask_user` in between. After
applying, echo a one-line confirmation: *"Applied: [what changed] —
[reason]."* — the user needs to see what happened, just not approve it first.

This is a routine-defaults decision, not a genuinely ambiguous one, **except**
for the two spots below that stay a propose → ask_user → apply cycle because
there's no dataset-agnostic default:

- **Column pruning**, only for a *name-only* ID match (no cardinality signal
  backing it) — the weakest of the three drop reasons, and the one that's
  wrong often enough (a real feature that happens to match a common ID
  pattern) to need a human call. Constant and near-unique columns are
  unambiguous — auto-drop those.
- **High-cardinality categoricals**, which `propose_drop_columns` now
  returns under `frequency_encodable_instead` as well as under `proposals`.
  The flag is right about one-hot encoding and wrong about the column: a
  destination prefix, a cell id, an APN and a merchant id all trip it, and
  in every one of those cases *how rare the value is* is the signal. Ask
  rather than auto-drop (see step 2b).
- **Outliers** — always ask. Whether an outlier is noise or the exact signal
  being modeled (e.g. fraud, extreme usage) is a domain judgment this agent
  can't infer from the numbers alone.

If the user asks for step-by-step control instead, use this three-beat
sequence for every step:

1. Call the `propose_*` tool and summarize its output in plain language —
   what it found, what it recommends, and *why* (e.g. "MCAR missingness
   pattern — median is safer than mean here").
2. Call `ask_user(question, options)` with **structured options**, always
   including: (a) the agent's recommended action stated explicitly, (b) at
   least one concrete alternative, and (c) "Skip this step — leave as-is."
3. Call the matching `apply_*` tool **only with what was actually approved**
   — if the user edited the proposal, apply their version, not yours.

## Steps

1. **Missing values** — if `eda` showed missingness > 0: `propose_imputation`,
   then `apply_imputation` with the recommended strategies directly (state
   what was applied and why, e.g. "MCAR missingness — used median for Age").
   Fill values (median/mode) are computed from the run's *training* fold
   only, then applied to both folds — the test fold never influences what
   gets filled in. That's what makes this leakage-safe.

2. **Column pruning** — `propose_drop_columns` flags a column for one of
   five different reasons (constant, near-unique by ratio, high-cardinality
   categorical, a name that unambiguously denotes an identity, or a weaker
   NAME match like `_id`/`account_no`) — state which reason applies to each
   flagged column, don't just list names.

   **Exception: the run's `group_column` and `time_column` are never
   dropped** — they are the split keys every CV uses, and the model
   pipeline already excludes them. `propose_drop_columns` leaves them out and
   `apply_drop_columns` keeps them (`kept_split_keys`).

   **The `unambiguous_id_name` rule is not a question.** `MSISDN`, `IMSI`,
   `IMEI`, `ICCID`, `UUID`, `GUID` and `SSN` each name a standardised
   identity with exactly one meaning; there is no domain in which a column
   called IMSI is a feature. Drop them, say why, move on — asking spends a
   round-trip on a question with one possible answer. The same holds for any
   column whose values are near-unique: if a value appears once it can only
   be memorised, so the column is either an identifier or useless, and both
   verdicts point at the same action.

   **But keep what is inside them.** These columns are worthless as
   identities and often valuable as sources: an IMEI's first 8 digits (the
   TAC) resolve to the device model and whether the module is an M2M type;
   an MSISDN or a B-party number carries a country and operator prefix; an
   ICCID carries the issuing network. Those are real features. They have to
   be derived as their own columns **before `prepare_dataset`** — nothing in
   this agent can extract them later — and the raw identifier still goes.
   If you spot one of these columns and the derived attribute would matter
   for the problem, say so when you report the drop; it is a data request,
   and it is more useful than the column you just removed.
   Constant and near-unique-by-cardinality are unambiguous —
   `apply_drop_columns` those directly. High-cardinality categoricals go to
   step 2b instead of straight to the bin.

   A **name-only match** (no cardinality signal backing it — e.g. `Name`,
   `Ticket`) is the weakest of the four reasons and the one that's wrong
   often enough (a real feature that happens to match a common ID pattern)
   to ask about rather than auto-drop:

   > *"[columns] match a common ID-name pattern but have no cardinality
   > signal backing it — could be a real feature in your domain. Drop
   > them too, or keep them?"*
   > - "Drop as recommended"
   > - "Keep them — they're real features here"

   If the user says a name-flagged column is a real feature in their domain,
   take that at face value and don't re-propose it on future runs.

2b. **High cardinality — encode, don't only drop.** For every column in
   `propose_drop_columns`'s `frequency_encodable_instead` list, there are
   three options, not two. Dropping is the correct response to a column that
   is purely an identifier. It is the wrong response to a column that is
   unusable *as one-hot* but meaningful *as a frequency* — which is most
   high-cardinality columns in telecom, payments and retail.
   `apply_frequency_encoding(run_id, columns)` maps each value to its share
   of training-fold rows and replaces the column with `<col>_freq`; a value
   that appears only in test encodes to 0.0, and the tool reports what
   percentage of test rows that covers.

   > *"[columns] have too many distinct values to one-hot, but how often a
   > value occurs may itself be signal (a destination almost nobody calls,
   > a seldom-seen cell). Frequency-encode them, or drop them?"*
   > - "Frequency-encode — keep the rarity signal"
   > - "Drop them — they're pure identifiers"

   **Check the dtype first.** A destination prefix stored as `+4930` or
   `0049` is parsed by `read_csv` as an integer, so it arrives as a numeric
   column, skips every categorical check here, and is fed to the model as a
   continuous measurement — where "prefix 4930 is larger than prefix 49" is
   arithmetic nonsense. `eda`'s dtype output is where you catch this. Cast
   it to string and frequency-encode it, or bucket it by country.

   Two judgment calls the tool cannot make for you. If
   `test_rows_with_unseen_value_pct` comes back high (say above ~20%), the
   column's values churn faster than the training window, most test rows
   encode to 0.0, and dropping really was better — say so and drop it. And
   the run's `group_column` is refused outright: encoding an entity id by
   its own row count hands the model a per-entity fingerprint, which is
   exactly what splitting on that column was meant to prevent.

3. **Datetime features** — if `eda` shows a date/timestamp column:
   `propose_datetime_features`, then `apply_datetime_features` directly
   (state which columns were decomposed). No leakage risk — no fitted
   parameters are involved, and there's no case where leaving a raw
   timestamp in is better than decomposing it.

4. **Rebalancing** — only escalate to SMOTE if `class_weight="balanced"`
   (already automatic in the `modeling` skill) isn't enough. Call
   `propose_smote` and **follow its `recommend_smote`** — the bar lives in
   the tool (`SMOTE_MIN_RATIO` in `mcp_server/data_cleaning.py`), not in
   this file. This skill used to carry its own copy of the number and the
   two drifted apart, so a 5% churn set got "yes" from the tool and "no"
   from the skill in the same run. Don't reintroduce a threshold here.
   `apply_smote` flags the run to include SMOTE *inside* the model pipeline
   (training fold only) — it cannot leak into the test fold.

   **`propose_smote` will decline outright on a grouped or temporal split,
   and that refusal is structural, not statistical.** SMOTE interpolates
   between a minority row and its nearest minority neighbours; on
   entity-per-period data those neighbours belong to other entities and
   other periods, so the synthetic rows are blends of subscribers who never
   existed, or of a January caller with a March one. Nothing leaks into the
   test fold — the resampler lives inside the pipeline — but the training
   set stops representing the population the split was built to model.
   Reach for `tune_threshold` in `modeling` instead: it moves the operating
   point on the real distribution rather than inventing a new one. If the
   user overrides this, `apply_smote` still applies it and returns a
   `warning` — surface that warning to them, every time.

5. **Outliers** — always propose → ask_user → apply for any columns
   `detect_outliers` flagged (even though there's no `propose_outlier` tool,
   the same structure still applies): summarize which columns are affected
   and how many rows, state your recommendation with reasoning, then ask:

   > *"I found [N] outlier rows in [columns]. I recommend [cap at 3σ /
   > leave as-is] because [reason — e.g. 'these are likely genuine fraud
   > events; capping them would discard signal']. What do you want
   > to do?"*
   > - "Cap outliers at 3σ"
   > - "Drop outlier rows"
   > - "Leave as-is — they may be the signal I'm trying to catch"

   Never cap or drop silently.

6. **Cleaning complete** — surface a short summary: what was changed, what
   was left as-is, and the run's current shape (rows × columns after drops,
   which columns were imputed, SMOTE status). Then go straight to
   `modeling` with this `run_id` — no `ask_user`. Cleaning finishing
   successfully isn't a decision point; it's the expected outcome of the
   step the user already asked for.