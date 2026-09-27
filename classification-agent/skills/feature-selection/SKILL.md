---
name: feature-selection
description: Drop columns the CURRENT trained model has shown it doesn't need, based on permutation importance — not the missingness/cardinality-based dropping in data-cleaning. Load after a first train_model, before the final tune/export. HITL — dropping a column is a data change.
---

# Feature Selection

**MCP module:** `mcp_server/feat_selection.py`

Tool: `propose_feature_selection(run_id, bottom_k=3)`. Read-only — ranks
this run's CURRENT model's raw input columns by permutation importance on
held-out TRAINING folds (the model refit per fold) and returns the `bottom_k`
lowest-importance ones. Never the test fold: dropping what looks weakest on
the test rows and then reporting on those rows is selecting on the test set.

This answers a different question than `data-cleaning`'s
`propose_drop_columns` (which flags columns by missingness/cardinality
before any model exists) — this is "the model itself has shown it doesn't
need this column," which you can only ask once a model has been trained at
least once.

## Cycle

Same `propose_*` -> **ask_user** -> `apply_*` shape as every other
data-changing step, reusing the tool that already does the write:

1. `propose_feature_selection(run_id)` — surface the candidates and their
   importance scores.
2. **ask_user**: *"[N] columns show near-zero importance to the current
   model: [list with scores]. Drop any of these?"*
   - "Drop all of them"
   - "Drop these specific ones — I'll list them"
   - "Keep everything — skip feature selection"
3. On approval, call `apply_drop_columns(run_id, columns)` (the
   `data-cleaning` tool — dropping columns is the same operation regardless
   of why they're being dropped) with the approved names.
4. Re-run `train_model` — the pipeline was fit on the old column set and
   needs to be refit without the dropped ones.

**Why HITL:** a near-zero-importance column can still be one the
stakeholder wants kept (regulatory field, a column that will carry signal
once more data arrives, interpretability). Never auto-drop.

**Check correlation before believing a low score.** Permutation importance
shuffles one column at a time, so when two columns carry the same
information, shuffling either leaves the model's score intact via the other
— and BOTH rank near zero despite the pair being jointly essential. This is
not a rare edge case in engineered feature sets: a window level and the
ratio built from it, a count and its share of total, a 3-day and a 7-day
version of the same statistic are all near-duplicates by construction. Look
at `plot_correlation_heatmap` (or the EDA correlation output) for the
proposed columns first. If a bottom-`k` candidate is highly correlated with
a column that ranked high, the low score is the twin masking it, not
evidence the column is useless — drop at most one of a correlated pair,
re-run `train_model`, and check the score actually held.
