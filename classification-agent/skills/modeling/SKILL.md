---
name: modeling
description: Train, tune, explain, and export a classifier once the data is clean. Covers model choice, hyperparameter search, explainability (permutation and real SHAP), and threshold tuning. Load this once eda/data-cleaning are done and the user wants a model.
---

# Modeling & Export

**MCP module:** `mcp_server/modeling.py`

Tools: `train_model(run_id, model)`, `compare_models(run_id, models)`,
`tune_hyperparams(run_id, model, n_trials, cv_folds)`,
`explain_model(run_id, method, background_size)`,
`tune_threshold(run_id, target_recall)`, `calibrate_model(run_id, method)`,
`compare_runs(run_ids)`, `export_model(run_id, out_path)`,
`predict(pkl_path, data_path)`. From the `diagnostics` skill (load it
alongside this one): `train_baseline(run_id)`, `detect_data_leakage(run_id)`,
`evaluate_model(run_id)`, `error_analysis(run_id)`,
`analyze_segments(run_id, segment_column)`,
`check_model_stability(run_id)`, `assess_fairness(run_id,
protected_attribute)`, `record_reflection(run_id, checks, critical_issues,
recommended_actions)`. From the `mlops` skill: `log_run_to_mlflow(run_id)`.

**First action once this skill loads:** call `load_skill("modeling",
reference="domain-notes")` — domain-specific metric priority, threshold
floors, explainability requirements, and fairness-relevant attributes for
fraud/healthcare/credit/telecom. It's on-demand, not auto-appended, so it
won't be in context unless you fetch it — do this before `train_model`,
not after, since it decides what to optimize for. If `business-
understanding` already fetched it and set `meta["business_understanding"]
["domain"]`, use that domain directly instead of re-guessing.

For the exact `ask_user` wording and full per-step guidance below — needed
in step-by-step mode, or when reflection needs the complete failure-modes
checklist — call `load_skill("modeling", reference="step-playbook")`;
autonomous mode doesn't need it.

---

## How the pipeline object works

Every tool here operates on the run's **current pipeline** — one fitted
object with a fixed stage order:

    drop columns → impute → one-hot encode → [SMOTE if enabled] → classifier

`train_model` creates and saves this pipeline. `tune_hyperparams`
**overwrites it in place** with the best candidate it finds. That is why
`explain_model` and `export_model` take no `model` or `hyperparams`
arguments — whichever pipeline ran most recently for this `run_id` **is**
the current model automatically for every tool downstream, including the
visualization agent's confusion-matrix and ROC-curve charts for the same
`run_id`. There is nothing to pass by hand between steps.

One exception: `explain_model` returns `feature_importance:
[[feature, value], ...]`. This array must be passed **literally** to the
visualization agent's `plot_feature_importance` tool — that tool has no
`run_id` argument at all; it only accepts the literal pairs. Pass that exact
array (not the `run_id`) when calling it. For per-sample SHAP spread instead
of a collapsed global ranking, use `plot_shap_beeswarm(run_id, out_path)`
instead, which does take `run_id` directly.

**Default `explain_model` to `method="shap"`, not the tool's own
`"permutation"` default.** `domain-notes.md` marks permutation "OK" only
for churn/telecom and spam — never *required* over SHAP — and permutation's
magnitude-only ranking can't say whether a feature pushes the prediction up
or down, which is exactly what `reporting/SKILL.md`'s Executive Summary
needs for its "top driver" line and what `explain_model` now persists as
`direction` (SHAP-only). Only fall back to permutation when the dataset is
large enough that SHAP's cost is a genuine concern and the domain is one of
the two "Permutation OK" rows in `domain-notes.md`'s table.

**Plots are on by default, not on request — but you do not draw them.**
The `plot_*` tools live in the visualization agent, which only the
orchestrator can reach. After your reply, it asks that agent for
`plot_confusion_matrix`, `plot_roc_curve`, `plot_pr_curve` (and feature
importance / SHAP beeswarm) with your `run_id`. Each chart is saved into
`<run>/charts/`, keyed to the current fit, and satisfies the
`evaluation_charts` readiness gate. So end your reply with the `run_id` and
"evaluation charts pending", and never loop on that gate — it is the one
gate you cannot satisfy yourself.

---

## Autonomous mode (default)

Unless the user asks to go step-by-step, run the full flow back-to-back on
your own judgment, using the guidance in each step overview below (and
`domain-notes.md`) to pick models/params/thresholds yourself:

    detect_data_leakage -> compare -> train -> tune ->
    run_standard_diagnostics -> threshold -> assess_fairness (if a
    protected attribute applies, else declare_fairness_not_applicable) ->
    record_reflection -> check_readiness -> generate_report ->
    log_run_to_mlflow

**`run_standard_diagnostics(run_id)`** runs the judgment-free middle in ONE
call, in the one correct order (baseline, calibration — it refits, so first —
label rule, SHAP, error analysis, stability, readiness). Read its `steps` for
the stop conditions below as if you had called each; `remaining` lists what
needs your judgment.

**Decide on validation, report on test.** Threshold, calibration and feature
selection all read out-of-fold training predictions; the test fold is scored
once, with 95% bootstrap intervals — so a recall target can land UNDER target
on test: report it, don't re-tune to hide it. An interval straddling the
success bar is **INCONCLUSIVE** (gate `not_run`): not a miss, not a pass, and
not something to loop on — only more labelled data moves it.

Two steps in that sequence are not where a beginner would put them, and the
placement is the point:

- **`check_label_rule` runs right after the model exists, before you invest
  in explaining it.** `train_baseline` establishes the floor — what no skill
  scores. This establishes the ceiling from the other side: if a depth-2
  tree on one column recovers most of the model's PR-AUC, the likeliest
  explanation is that a rule on that column wrote the labels and the model
  has rediscovered it. Every number after that point would measure agreement
  with the rule rather than detection of the behaviour. Learn this before
  you spend a SHAP pass explaining a model built on it, not after.
- **`threshold` is a decision step, not a formatting step.** It now persists
  the operating point, which travels into the export and is applied by
  `predict`. Skipping it does not leave the question open — it answers it
  with sklearn's 0.5, which on an imbalanced target is a different model
  with materially different precision.

`check_readiness` is the loop condition, not a formality. Call it, read
`not_run_gates`, go satisfy them, call it again. You are finished when it
returns `ready` — not when you feel finished. An `incomplete` result means
required evidence was never produced, which is not the same as passing, and
`generate_report` will say so in its first paragraph.

Skip the per-step `ask_user` gates in this mode. Still stop and call
`ask_user` immediately if you hit any of:

- an `overfitting_warning` from `train_model`/`tune_hyperparams`
- a **weak** (name-pattern) identifier flag you think is a real feature —
  a **strong** one is not a question: drop it, record why, and carry on. It
  is cardinality-shaped, so it cannot generalise through a one-hot encoder,
  only memorise, and a round-trip to confirm that buys nothing. This is the
  `Ticket` case, which twice reached a model and dominated SHAP because the
  agent waited to be told to drop an obviously-useless column.
- a `near_perfect_predictors` entry from `detect_data_leakage` (one column,
  any type, that nearly decides the label alone) — that one IS ambiguous (a
  legitimate strong predictor and a leaked label look alike), so ask where the
  column comes from. The `leakage_screened` gate FAILS until it is dropped or
  the user's answer is recorded via `acknowledge_identifier_column`; never
  record an acknowledgement nobody gave you. Also ask about any
  `possible_post_event_columns` (fields that sound filled in after the outcome)
- `high_variance_warning` from `check_model_stability`
- a **temporal** run (`meta["split_type"] == "temporal"`) whose test-fold
  score falls far below its CV score. Under a random split that gap reads
  as overfitting; under a forward-chained one it is usually the world
  moving — a relationship learned on earlier periods no longer holding
  later. The two need opposite fixes (regularize, versus retrain on recent
  data and prefer features that stay stable), so don't silently file it as
  overfitting
- the real model failing to clearly beat `train_baseline` (if a dummy
  classifier scores within noise of the real one, the data may not support
  this target — that's a stop, not a metric to quietly report)
- a `disparate_impact_concern` or `equal_opportunity_concern` from
  `assess_fairness`
- any `critical_issues` entry from `record_reflection`
- a `label_rule_suspicion` from `check_label_rule`. **This one is not a
  modelling problem and cannot be fixed by modelling.** It is a question for
  whoever produced the labels: did a rule on this column generate them? If
  yes, the column has to go and the run restarts. If no — the domain
  genuinely has one dominant driver — record that answer with
  `acknowledge_label_rule`. Do not acknowledge it because the run is
  otherwise finished; an acknowledgement you cannot attribute to a person is
  an assumption with a signature on it
- ambiguous domain requirements not covered by `domain-notes.md`
- ambiguous business framing (target definition, prediction unit/horizon,
  success criteria) not already resolved by `business-understanding`

Otherwise proceed straight through: call `generate_report` and then
`log_run_to_mlflow` automatically at the end of every autonomous run — the
user asked for a model, a report, and an MLflow-tracked run out of one
request, not three separate asks — and hand the user the report instead of
a play-by-play. If `mlflow` isn't installed, `log_run_to_mlflow` returns an
error string; surface it once and move on, don't retry. Export always
keeps its two-step gate regardless of mode (see the step-playbook
reference) — that step writes a file, never make it silent.

If the user explicitly asks for step-by-step control, fetch the
step-playbook reference and use the gate rule below instead.

## The CV metric adapts to the class balance — don't fight it

`compare_models`, `train_model`, `tune_hyperparams` and
`check_model_stability` all score with **PR-AUC (`average_precision`) when the
binary target is imbalanced** (minority:majority below 0.2), and f1_weighted
otherwise. Every result reports which one it used under `cv_metric`; quote
that name when you report the score, never "CV F1" by habit.

This is the difference between selecting a model and selecting noise: on
the 3.5M-row Wangiri file (2.5% positive) f1_weighted scored three models
0.971–0.982 — the rate of always predicting "not fraud" — while PR-AUC was
0.62 against a ROC-AUC of 0.98. So: **lead with PR-AUC on an imbalanced
target** (never "98% accuracy" as a headline), and treat **a large
ROC-AUC/PR-AUC gap as a finding** — the model ranks well overall and is
much weaker on the class you care about.

## Every CV respects the split — quote `cv_scheme`

Every CV here (compare/train/tune/stability, threshold, calibration,
feature selection) obeys the split — forward-chained on a temporal run,
grouped on an entity run, stratified otherwise — and reports `cv_scheme`:
quote it beside the score. A temporal stability score falling fold after
fold is a model decaying with time. Split keys never reach the model.
Models: logistic_regression, random_forest, xgboost, hist_gradient_boosting
(native NaN, fastest at scale); all class-balanced, all take text labels.

## Branching on genuinely open decisions (tree-of-thoughts, made concrete)

Some decisions have no defensible default. For those, don't deliberate in
prose and pick — **materialise the branches and let measurement close them.**
Prose comparison of options you never ran is just a confident guess with
extra steps; this agent has tools that evaluate real alternatives:

- competing model families -> `compare_models` IS the branch-and-evaluate
  step. It fits each candidate under the identical pipeline and ranks them on
  CV. Never argue a model family is better in text when one call measures it.
- competing preprocessing choices (SMOTE vs class weights alone, keep vs drop
  a borderline column, one feature set vs another) -> run the branches as
  separate `run_id`s and settle it with `compare_runs`, which ranks them on
  held-out PR-AUC and writes the comparison into every participating run's
  meta, so the report shows what was rejected and why.
- competing operating points -> `tune_threshold` returns the chosen threshold
  AND the 0.5 default side by side. That is the trade-off, measured. Its
  three modes are three different questions, not three ways of asking one:
  a recall floor asks how much you catch, an alert budget asks how much work
  you create, a cost ratio asks what the errors are worth. Ask the one the
  business actually has. When a recall floor and an alert budget disagree,
  the budget usually wins — a model emitting more alerts than anyone can
  review has an effective recall of whatever gets reviewed.

Branch when the alternatives are genuinely close and the choice is material;
don't branch to look thorough. Three runs to settle a decision that
`compare_models` already answered is waste, and so is branching on something
the success criteria make obvious. When you do branch, say in the report
which alternatives were evaluated and on what number they lost — a rejected
branch with a measured score is evidence; a rejected branch you only thought
about is not, and must not be written up as though it were.

## Reasoning approach — chain-of-thought before, reflection after

This applies in every mode, autonomous or step-by-step. It's what makes
autonomous mode trustworthy instead of just fast.

**Before each state-changing call** (`train_model`, `tune_hyperparams`,
`tune_threshold`), state your reasoning out loud in 1-3 sentences before
making the call: what the data/prior metrics show, what you're choosing
and why, and what alternative you're rejecting and why. "Training
random_forest because eda showed mixed numeric/categorical features and
this stakeholder needs feature importances" is reasoning; "Training a
model" is not. This is chain-of-thought, not a formality — if you can't
articulate the reasoning in a sentence, you don't have a real basis for
the choice yet, go back and look at the data again.

**After each result** (train, tune, explain, threshold), before moving to
the next step or reporting, run an explicit self-critique:

1. Does this result match what I expected given the data characteristics I
   noted in EDA? If not, why not — is that a red flag or a legitimate
   surprise worth mentioning to the user?
2. Walk the failure-modes checklist explicitly: leakage signal, threshold
   0.5 on an imbalanced target, high-ROC/low-PR gap, ROC improved but PR
   didn't, SMOTE/test-fold contamination, near-0.5 confidence clustering.
   State which ones you checked and cleared, not just the ones that fired.
   The two-line summary above covers the common cases; fetch
   `load_skill("modeling", reference="step-playbook")` for the full
   checklist with the reasoning behind each one if you need it spelled out.
3. If reflection surfaces a real concern, **do not proceed past it**:
   revise the plan (different model, drop the leaking column, re-tune) and
   re-run the affected step, then reflect again. Cap this at 2 revision
   cycles — if the second attempt still fails the same check, that's a
   stop-and-`ask_user` case (add it to the autonomous-mode stop list
   above), not a third silent retry.

Once the flow reaches its end (right before `generate_report`), call
`record_reflection(run_id, checks, critical_issues, recommended_actions)`
(`diagnostics` skill) to turn the running self-critique above into a
persisted, auditable record — one entry per dimension in
`REFLECTION_CHECKS`, each citing the actual tool result it's based on.
**This call is not optional in autonomous mode**: stating the reflection
outcome only in chat ("checked for leakage, clear") is exactly the
unauditable-prose problem this tool exists to fix — if it isn't in
`meta["reflection"]`, `generate_report` has no way to show a reviewer that
it happened at all.

## HITL gate rule (step-by-step mode only)

Apply this to every step without exception — fetch the step-playbook
reference for the exact `ask_user` wording per step; this is the general
rule that wording follows.

**Read-only tools** — `compare_models`, `explain_model`, `compare_runs`:
run without prior approval, they change nothing. Always surface their full
output and then call `ask_user` before the next state-changing step.

**State-changing tools** — `train_model`, `tune_hyperparams`,
`tune_threshold`: require a gate **before** running when called as a next
step — unless the user explicitly chose that action in the immediately
preceding gate (in which case the before and after gates collapse into one
ask). When in doubt, gate before. A redundant question costs one round
trip; a silently overwritten pipeline costs a full re-run.

**Export** — always two-step, no exceptions. Dry-run first (no `out_path`),
confirm path before writing.

---

## Label distribution carry-forward from cleaning

If SMOTE was applied in the data-cleaning phase, call `check_imbalance(run_id)`
at the start of this skill and echo the current training-fold distribution
before any training begins — so the user knows exactly what class ratio the
model will be fit on:

> *"Reminder — training fold label distribution going into modeling:"*
>
>     Class 0: N rows (XX%)
>     Class 1: N rows (XX%)
>     Ratio: 1 : X   [SMOTE applied / class_weight='balanced' only]

This is read-only and automatic — no `ask_user` needed. Purpose: catch any
mismatch between what the user expected SMOTE to do and what it actually
did, before the first training run.

---

## Step overview

Full detail (exact `ask_user` scripts, what to surface, budget/method
guidance) is in `load_skill("modeling", reference="step-playbook")` —
autonomous mode makes its own judgment calls from this table alone.

| Step | Tool | When |
|---|---|---|
| 0. Compare (optional) | `compare_models(run_id, models="")` | No strong prior on which model — read-only, ranks by the run's adaptive `cv_metric` (PR-AUC when imbalanced, f1_weighted otherwise — see "The CV metric adapts to the class balance" above; never call it "CV F1" by habit) |
| 1. Train | `train_model(run_id, model)` | Always — establishes the current pipeline and baseline metrics |
| 2. Tune | `tune_hyperparams(run_id, model, n_trials, cv_folds)` | Metric matters enough to spend the extra compute — overwrites the pipeline, irreversible within the run |
| 3. Explain | `explain_model(run_id, method, background_size)` | Always in autonomous mode — read-only, catches leakage signals too. Default `method="shap"` (see below) |
| 3a. Mechanical diagnostics | `run_standard_diagnostics(run_id)` | Always in autonomous mode, after train/tune — replaces calling 3c/3d/3/error_analysis/stability one by one |
| 3b. Visualize | `plot_confusion_matrix`, `plot_roc_curve`, `plot_pr_curve`, `plot_feature_importance`, `plot_shap_beeswarm` (visualization agent — requested by the ORCHESTRATOR after your reply, not callable from here) | Always — satisfies the `evaluation_charts` gate; `generate_report` embeds whatever charts exist for the current fit |
| 3d. Calibration (binary only) | `check_calibration(run_id)` (`diagnostics`), then `calibrate_model(run_id, method)` if it warns | Always **before** step 4 on a binary target. `tune_threshold` reads precision/recall off the probability distribution and the cost mode does arithmetic on it — a cutoff picked from uncalibrated scores encodes a tradeoff nobody chose. RF is over-confident at the extremes, XGBoost under-confident in the middle; neither is calibrated by default. Calibrating refits the pipeline, so it invalidates SHAP/fairness/threshold — do it here, not after |
| 4. Threshold (binary only) | `tune_threshold(run_id, target_recall \| max_alert_rate \| fn_to_fp_cost_ratio)` | Always, on a binary target — skipping it ships 0.5. Pick the mode that matches the actual constraint: a recall floor when the domain sets one, `max_alert_rate` when a review queue has finite capacity, `fn_to_fp_cost_ratio` when the business can state relative costs |
| 3c. Label-rule screen | `check_label_rule(run_id)` (`diagnostics`) | Always, right after the first trained model — the ceiling check that pairs with `train_baseline`'s floor |
| 4d. Backtest (optional) | `backtest_on_new_data(run_id, path)` (`diagnostics`) | A later labelled period exists — the only way to answer "does it still work", which no CV can |
| 4b. Fairness (if applicable) | `assess_fairness(run_id, protected_attribute)` (`diagnostics` skill) | A protected/sensitive attribute is present in the data — always in autonomous mode when one applies, skip and say so when none does |
| 4c. Reflect | `record_reflection(run_id, checks, critical_issues, recommended_actions)` (`diagnostics` skill) | Always, near the end of an autonomous run — persists the self-critique above |
| 5. Export | `export_model(run_id, out_path)` | User wants the model as a standalone file — always two-step, no exceptions |
| 6. Predict | `predict(pkl_path, data_path)` | Scoring new data with an already-exported model |
| 7. Compare runs (optional) | `compare_runs(run_ids)` | User has iterated over multiple `run_id`s and wants to know which won |
