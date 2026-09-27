Full step-by-step scripts for `modeling` — exact `ask_user` wording, what
to surface after each tool, and the complete "common failure modes"
checklist the reasoning-and-reflection section requires walking. Fetched
via `load_skill("modeling", reference="step-playbook")`, not auto-loaded
with the base skill: autonomous mode (the default) makes its own judgment
calls from the compact step overview in `SKILL.md` and never needs this
file's exact question wording. Fetch this when: the user asked for
step-by-step control, or the reflection step needs the full failure-modes
checklist and just the two-line summary in `SKILL.md` isn't enough.

---

## Steps

---

### Step 0 — Compare models (optional)

**When to use:** Use `compare_models` instead of guessing a model when
there is no strong domain reason to pick one upfront. If the user already
knows which model they want, skip to Step 1.

**Tool:** `compare_models(run_id, models="")`.

- Omit `models` to benchmark all candidates.
- Pass a comma-separated subset (e.g. `models="random_forest,xgboost"`)
  to narrow the search.
- Cross-validates every candidate on the training fold and ranks by CV
  F1-weighted. **Read-only — no `pipeline.pkl` is written.**

**What to surface after compare:**
Show the full ranking table — model name, CV F1 mean, CV F1 std, and any
other metrics returned. Do not paraphrase or truncate. Flag:
- High std (> 0.05) on the top candidate — the model is sensitive to fold
  composition; consider more `cv_folds` at tune time.
- A close gap between first and second — the simpler model may be
  preferable for explainability even if it ranks slightly lower.

**Gate after compare — call `ask_user`:**

> *"Compare complete. Rankings: [full table]. Which model do you want
> to train?"*
> - "Train [top-ranked model] — recommended"
> - "Train [second-ranked model] — simpler / more explainable"
> - "I want a different model — I'll specify"
> - "Skip compare — I already know which model I want"

Do not call `train_model` until the user has made a choice.

---

### Step 1 — Train

**Tool:** `train_model(run_id, model)`.

**Model choice guidance:**
- Default to `random_forest` or `logistic_regression`. These are the right
  starting point for most tabular classification problems — robust, fast
  to train, and straightforward to explain to a non-technical stakeholder.
- Reach for `xgboost` only when the simple baseline's metric is not good
  enough after tuning. XGBoost is more accurate on many benchmarks but
  slower to tune, more sensitive to hyperparameters, and harder to explain
  to non-technical stakeholders.
- `logistic_regression` is the right default when the stakeholder will need
  to audit or challenge individual predictions. Its coefficients map
  directly to feature contributions in a way tree ensembles do not.
- `class_weight="balanced"` is applied automatically to every model at
  train and tune time — this is free, always on, and the first line of
  defence against class imbalance. SMOTE (if enabled in cleaning) is the
  bigger hammer and only worth it at extreme imbalance.

**What `train_model` reports:**
- 5-fold cross-validated F1 (mean ± std) on the training fold — the honest
  variance estimate of model stability across different slices of training
  data.
- Held-out test metrics: precision, recall, F1, ROC-AUC, PR-AUC — how the
  model performs on data it never trained on.

**What to surface after train — full metrics, then flag:**
- **PR-AUC vs ROC-AUC:** For imbalanced targets, PR-AUC is the more honest
  metric. A model that predicts all-negative can still score well on
  ROC-AUC when the negative class dominates — PR-AUC does not have this
  problem. Always lead with PR-AUC on imbalanced data.
- **CV F1 std > 0.05:** Flag explicitly — the model's performance varies
  meaningfully across folds, which is a stability concern at deployment
  time.
- **Precision/recall spread:** If recall is high but precision is low (or
  vice versa), flag it. Threshold tuning (Step 4) can rebalance this
  without retraining.
- **Train vs test gap:** If CV F1 is substantially higher than test F1,
  flag overfitting. Consider a simpler model, fewer features, or stronger
  regularisation before tuning.

**Gate after train — call `ask_user`:**

> *"Training complete. Metrics: [full table]. CV F1: [mean] ± [std].
> What do you want to do next?"*
> - "Tune hyperparameters"
> - "Explain the model first (feature importance)"
> - "Tune the decision threshold"
> - "Export this model as-is"
> - "Try a different model"

---

### Step 2 — Tune hyperparameters

**Tool:** `tune_hyperparams(run_id, model, n_trials=8, cv_folds=5)`.
Uses Optuna with TPE sampler.

**Important — this overwrites the pipeline:** `tune_hyperparams` refits
the best candidate and replaces the run's current pipeline in place.
This is irreversible within the run. Always gate before running.

**Budget guidance:**
- `n_trials=8, cv_folds=5` is a fast first pass — good for quickly checking
  whether tuning helps at all. Not an exhaustive search.
- Raise `n_trials` (30–50) and `cv_folds` (10) when the metric actually
  matters and compute is affordable.
- If the fast pass shows zero or negative improvement, a larger search
  budget rarely reverses that — the model choice or feature set is more
  likely the problem.

**Gate before tune — call `ask_user` (destructive, always gate first):**

> *"Tuning will replace the current pipeline with the best candidate found.
> This cannot be undone within this run. Defaults: n_trials=8, cv_folds=5
> (fast pass). How do you want to run it?"*
> - "Run with defaults (fast)"
> - "Run a thorough search — I'll specify n_trials and cv_folds"
> - "Cancel — keep the current model"

**What to surface after tune:**
- `roc_auc_improvement_vs_baseline` — surface this number prominently. If
  it is negative or approximately zero, say so explicitly. Do not oversell
  a small search budget.
- Full metric table at the tuned model vs the untuned baseline — same
  fields as Step 1 so the user can compare directly.
- Best hyperparameters found — surface these even if the user did not ask;
  useful context if the model is retrained later.

**Gate after tune — call `ask_user`:**

> *"Tuning complete. ROC-AUC improvement vs baseline: [delta]. Best
> params: [params]. Metrics: [full table]. What next?"*
> - "Explain the model"
> - "Tune the decision threshold"
> - "Export this model"
> - "Re-tune with a larger budget — I'll specify n_trials"
> - "Discard tuning and go back to the untuned model"

---

### Step 3 — Explain

**Tool:** `explain_model(run_id, method="permutation", background_size=200)`.
Read-only — runs without prior approval.

**Method choice:**
- `method="permutation"` (default): Cheap, model-agnostic. Measures how
  much permuting each feature hurts test-fold performance. Reflects
  generalization, not memorisation. Use this as the default.
- `method="shap"`: Exact TreeExplainer for `random_forest` / `xgboost`;
  LinearExplainer for `logistic_regression`. Use when the stakeholder needs
  per-prediction-shaped reasoning rather than just a global ranking — this
  is a regulated-domain ask, not a default. See `domain-notes.md` for when
  SHAP is expected.

**`background_size` — auto-detected for imbalanced targets:**
`background_size` is a **stratified** sample of the test fold used as
SHAP's reference distribution. The server auto-detects the right value
when the default (200) is left unchanged — it computes the minimum sample
size needed to guarantee at least 10 positive rows in the background,
clamped to the full test fold. You do not need to compute or pass this
manually for imbalanced datasets.

Override only when you have a specific reason:
- Pass a smaller value to cap compute time on a very large test fold
  where a representative sample is enough.
- Pass a larger explicit value if you want more than the auto-detected
  floor (e.g. for a final production explanation run).
- The auto-detection only fires on the unchanged default — any explicit
  value you pass is always respected as-is.
- Note: `background_size` only affects `method="shap"`. For permutation
  importance it has no effect.

**What to surface after explain:**
- Full ranked feature importance list — all features, not just top N. The
  user may have domain knowledge that makes a low-ranked or high-ranked
  feature surprising.
- **Flag anomalies explicitly:**
  - An ID-like column, timestamp, or row-index feature in the top features
    — likely a leakage signal. Stop and flag before exporting.
  - A single feature dominating by a large margin (> 3× the second) —
    possible leakage or proxy variable.
  - A feature the domain expert expects to rank high appearing near the
    bottom — worth surfacing before shipping.
**Plotting — do this automatically, don't wait to be asked:**
- Global ranking: immediately pass the literal `[[feature, value], ...]`
  pairs from `explain_model`'s result to the visualization agent's
  `plot_feature_importance` and show the chart in the same turn. **This
  tool has no `run_id` argument** — pass the array directly, never the
  `run_id`.
- If `method="shap"` was used: also call `plot_shap_beeswarm(run_id,
  out_path)` in the same turn and show it alongside the ranking. Don't
  make the user ask for the plot separately — they already asked for the
  explanation.

**Gate after explain — call `ask_user`:**

> *"Explanation complete. Feature importance (all features): [full list].
> [chart(s) shown above] [Flag any anomalies here.] What next?"*
> - "Tune the decision threshold"
> - "Export this model"
> - "Re-run explain with SHAP for per-prediction breakdown" (omit if SHAP
>   was already used)
> - "Go back and retrain — I want to revisit feature selection"

---

### Step 4 — Tune the decision threshold (optional, binary targets only)

**Tool:** `tune_threshold(run_id, target_recall=0.0)`.

**When to use:** sklearn's default of 0.5 is rarely optimal for imbalanced
targets. If post-train metrics show an unfavourable precision/recall split,
threshold tuning can rebalance without retraining.

**Check calibration first:** If you are not confident the model's predicted
probabilities are trustworthy, check `plot_calibration_curve(run_id)` first.
Thresholding on poorly calibrated probabilities is unreliable. Logistic
regression is usually well-calibrated by default; tree ensembles often are
not without explicit calibration.

**Two modes:**
- Default (`target_recall=0.0`): finds the threshold that maximises F1 on
  the test fold. Use when there is no hard recall floor.
- `target_recall=X`: finds the highest-precision threshold that still meets
  the recall floor. Returns the best-precision threshold that clears that
  recall, plus the same metrics at 0.5 for comparison.

For the right `target_recall` value for your domain, see `domain-notes.md`.

**Gate before tune_threshold — call `ask_user`:**

> *"What do you want to optimize the threshold for?"*
> - "Maximize F1 on the test fold (default)"
> - "Hit a minimum recall floor — I'll specify the target"
> - "Skip — keep the sklearn default of 0.5"

**What to surface after tune_threshold:**
Show metrics at three points side by side — never just the new threshold:

    Metric        Threshold 0.5    Tuned threshold [value]
    Precision     XX%              XX%
    Recall        XX%              XX%
    F1            XX%              XX%

Flag trade-offs explicitly: if tuning gained recall but lost meaningful
precision (or vice versa), say so and let the user decide whether the
trade-off is acceptable before proceeding.

**Gate after tune_threshold — call `ask_user`:**

> *"Threshold tuned to [value]. Comparison: [table above]. Accept?"*
> - "Accept — this is the threshold I want"
> - "Try a different recall floor — I'll specify"
> - "Revert to 0.5"

---

### Step 5 — Export

**Tool:** `export_model(run_id, out_path)`.

Saves the run's current pipeline as one self-contained `.pkl` (joblib).
Includes all preprocessing — loading it and calling `.predict(new_raw_df)`
on data shaped like the original CSV just works, no manual re-encoding.

**Two-step gate — no exceptions, no silent writes:**

**Step 5a:** Call `export_model(run_id)` with *no* `out_path` — returns a
suggested path without writing. Surface it and call `ask_user`:

> *"Ready to export. Suggested path: [path]. This saves the full pipeline
> including all preprocessing. Confirm?"*
> - "Export to this path"
> - "Use a different path — I'll specify"
> - "Cancel"

**Step 5b:** Only on explicit confirmation, call
`export_model(run_id, confirmed_path)`. Never guess a path and write
silently. Never skip Step 5a.

**After successful export, surface:**
- The exact path written.
- What is inside: model type, preprocessing steps, SMOTE status, tuned
  threshold if applicable.
- Predict syntax: `predict(pkl_path="[path]", data_path="new_data.csv")`.

---

### Step 6 — Predict on new data

**Tool:** `predict(pkl_path, data_path)`.

Loads an exported `.pkl` and runs it on a new CSV. Target column is
optional — dropped if present. Returns one prediction per row plus the
predicted class's confidence score.

**Before calling predict, confirm:**
- The new CSV is shaped like the original training CSV — same column names,
  same dtypes. The pipeline handles encoding and imputation automatically
  but cannot recover from missing or renamed columns.
- The `.pkl` is from a confirmed export, not a mid-run snapshot.

**What to surface after predict:**
- Row count: how many predictions were made.
- Predicted class distribution: how many rows landed in each class and at
  what average confidence. A wildly skewed distribution on new data (e.g.
  99% positive when training rate was 5%) is a signal worth flagging —
  either the new data is genuinely different or something is wrong upstream.
- Confidence scores clustering near 0.5: flag low model certainty — the
  threshold may need revisiting or the new data may be out of distribution.

---

### Step 7 — Compare across runs (optional)

**Tool:** `compare_runs(run_ids)` (comma-separated).

Reads each run's saved metrics from `meta.json` — no retraining, no
pipeline changes. Ranks by PR-AUC. Use when the user has iterated over
multiple `run_id`s and wants to know which configuration won.

**What to surface after compare_runs:**
- Full ranking table: run ID, model type, key cleaning flags (SMOTE on/off,
  columns dropped), PR-AUC, ROC-AUC, F1.
- Large gap between PR-AUC and ROC-AUC — high ROC / low PR on an imbalanced
  target usually means the model is defaulting to the majority class.
- Two runs within noise of each other (< 0.005 PR-AUC difference) —
  recommend the simpler model. Performance parity favours interpretability.

**Gate after compare_runs — call `ask_user`:**

> *"Run comparison complete. Rankings: [full table]. Which run do you
> want to export or continue working with?"*
> - "Export run [ID] — the top-ranked one"
> - "Export run [ID] — I prefer this one for a reason I'll state"
> - "Go back and tune [run ID] further"
> - "None — I'll start a new run"

---

## Common failure modes — detect and flag before exporting

**Leakage signal:**
Run `detect_data_leakage(run_id)` before training (it flags near-perfect
target correlation and ID-like columns automatically) and again if an ID
column, timestamp, row-index, or post-event feature appears in the top
features after `explain_model` — the tool's heuristics and a human-visible
importance ranking catch different shapes of the same problem. Either
signal: stop and flag it before exporting. Re-run data-cleaning with that
column dropped and retrain. Do not export a pipeline with a leakage
feature in it.

**Threshold 0.5 on an imbalanced target:**
The default 0.5 threshold is calibrated for balanced datasets. On a 1%
positive rate dataset it almost always produces near-zero recall. Always
run Step 4 on imbalanced targets — do not ship with the sklearn default
unless the user explicitly accepts it after seeing the comparison.

**High ROC-AUC, low PR-AUC:**
A model predicting all-negative can score > 0.95 ROC-AUC on a 1% positive
rate dataset. PR-AUC exposes this. Always lead with PR-AUC for imbalanced
targets and flag any large gap between the two scores explicitly.

**Tuning improved ROC-AUC but not PR-AUC:**
Common when the search found better majority-class separation but not better
minority-class detection. Flag this — do not treat a ROC gain as an
unqualified win on an imbalanced target.

**SMOTE / test-fold contamination check:**
`apply_smote` flags SMOTE to run inside the pipeline on the training fold
only — it cannot leak into the test fold. If the user questions why test
metrics seem optimistic, verify SMOTE was applied via the pipeline flag,
not on the raw dataset before `prepare_dataset`. The label distribution log
surfaced at the start of the modeling skill is the checkpoint — if the
test-fold class ratio also changed, that is a contamination signal.

**Confidence scores clustering near 0.5 at predict time:**
Signals that the model is uncertain on the new data — either the data is
out of distribution, the threshold needs revisiting, or the model needs
recalibration. Flag this and do not present predictions as reliable without
that caveat.
