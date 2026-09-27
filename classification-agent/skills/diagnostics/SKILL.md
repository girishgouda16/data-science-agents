---
name: diagnostics
description: The checks that aren't cleaning or modeling — data leakage, a baseline to beat, error analysis, per-segment performance, and CV stability. Load alongside modeling in an autonomous run (target-shape checking is eda's inspect_target, not this skill).
---

# Diagnostics

**MCP module:** `mcp_server/diagnostics.py`

Tools: `detect_data_leakage(run_id)`, `train_baseline(run_id)`,
`evaluate_model(run_id)`, `error_analysis(run_id, top_n=10)`,
`analyze_segments(run_id, segment_column)`, `check_model_stability(run_id,
n_repeats=3)`, `check_label_rule(run_id)`, `acknowledge_label_rule(run_id,
justification)`, `backtest_on_new_data(run_id, path)`, `assess_fairness(run_id, protected_attribute,
min_group_size=20)`, `record_reflection(run_id, checks, critical_issues,
recommended_actions)`. All read-only with respect to the model/data — no
HITL gate — but every one of them now **persists** its result onto
`meta.json` (see each docstring for the exact key), specifically so
`generate_report` can render it without the live conversation. Re-running
one overwrites its own key, so the report always reflects the most recent
call, not a stale one from earlier in a long session.

(`inspect_target(path, target)` — the raw-file target-shape check — lives
in `mcp_server/eda.py` under the `eda` skill instead, since it runs
alongside `eda(path)` before `prepare_dataset`, before this skill's tools
are relevant.)

## Before training

**`detect_data_leakage(run_id)`** — call this right after `prepare_dataset`
(and cleaning), before `train_model` — and again after any column is added,
dropped or imputed (the ordering gate checks). Flags ID-like columns,
`near_perfect_predictors` (ANY column type that nearly decides the label on
its own, scored out-of-fold — the `leakage_screened` gate FAILS on these
until dropped or acknowledged with the user's reason), and
`possible_post_event_columns` (names that sound filled in after the outcome
— ask, don't gate). Not a substitute for the human judgment call in
`modeling`'s leakage checklist — a flagged column can be legitimate (e.g. a
pre-computed risk score in a workflow where using it is the point) — but it
catches the two most common shapes automatically instead of relying on
`explain_model` surfacing it after the fact.

**`train_baseline(run_id)`** — call once, before or right after your first
`train_model`. Fits a stratified `DummyClassifier` and evaluates it on the
test fold; does not touch `pipeline.pkl`/`meta.json`. This is the number
every real model must clearly beat — if it doesn't, that's a stop-and-ask
signal in `modeling`'s autonomous mode, not something to report as a normal
result.

**`check_label_rule(run_id)`** — the other half of `train_baseline`. The
dummy gives the floor: what no skill scores. This gives the ceiling from the
opposite direction: fit a depth-1 and depth-2 tree on each numeric column
alone, and report the best one's PR-AUC as a fraction of the full model's.

Call it right after the first trained model. If one column's stump recovers
≥90% of the model, the likeliest explanation is not that the model is
elegant — it is that a rule on that column generated the labels and the
model has rediscovered it. From that point every downstream number measures
agreement with the rule: PR-AUC is inflated, error analysis describes
disagreements with the rule, and a recall floor is met against labels that
already contain the answer.

**Nothing else in this agent can see this.** `detect_data_leakage` looks for
near-perfect correlation and identifier shape. A detection rule is usually a
threshold on a *ratio*, and correlation with the target can sit around 0.6
while a stump on the same column reaches 0.97 — clear on one check,
decisive on the other.

The finding is genuinely ambiguous and the gate treats it that way. A
dominant column means either a rule-written label or a domain with one real
physical driver, and no statistic separates them — only the person who
produced the labels can. So `label_rule_screened` fails until either the
column goes or `acknowledge_label_rule(run_id, justification)` records the
answer. Acknowledging it to unblock a finished run, without having asked
anyone, converts an open question into a documented assumption and is worse
than leaving it open.

## After training

**`evaluate_model(run_id)`** — re-states the CURRENT model's metrics on
demand, without retraining. Use after a step that doesn't retrain (e.g.
`apply_drop_columns` from a `propose_feature_selection` result) when you
want fresh numbers before deciding whether to re-run `train_model`.

**`error_analysis(run_id, top_n=10)`** — the model's most-confident WRONG
predictions on the test fold. Use this to say something concrete about
*where* the model fails, not just its aggregate score — surface a few rows
verbatim when reporting to a stakeholder who needs to trust the model, not
just see a number.

**`analyze_segments(run_id, segment_column)`** — precision/recall/F1 broken
out by a categorical column (region, customer type, channel...). Run this
whenever the domain has an known segment that matters (e.g. churn by
customer tier, fraud by channel) — an aggregate F1 of 0.9 can hide one
segment at 0.4. Ask the user which column if none is obvious from the
domain, otherwise pick the most business-meaningful categorical column
`eda` surfaced. For which segment a domain cares about, read
`modeling/references/on_demand/domain-notes.md`'s per-domain section
(fetch via `load_skill("modeling", reference="domain-notes")`) — that is
the single source of domain business-facts in this agent. (This used to
point at `feature-engineering/references/domain-features.md`, which has
never contained a segment list; that file is about predictive-feature
archetypes, and the stale pointer sent the agent to fetch guidance that
was not there.)

**`check_model_stability(run_id, n_repeats=3)`** — repeats 5-fold CV with
different random splits and reports the spread of F1 across all folds.
`high_variance_warning` (std > 0.1) means the single CV number `train_model`
reported was partly a fluke of that one split — treat it the same as an
`overfitting_warning`: a stop-and-ask signal before exporting, not a
footnote.

**Know what this does not test.** It resamples *randomly*, so it answers
"was that one split lucky?" and nothing else. It cannot see temporal
degradation: on a time-ordered dataset it will confidently report stability
for a model that collapses next month, because the random folds mix periods
together — precisely the structure a temporal problem does not have. For a
run whose `meta["split_type"]` is `temporal`, a stable result here is
necessary and not sufficient; the real question is whether performance holds
on a period later than the test fold, which means a fresh run on newer data
or the separate `drift-agent`. Say which of the two you actually checked
rather than letting a passing stability check stand in for a claim about
the future.

## Fairness — a distinct question from segment performance

**`assess_fairness(run_id, protected_attribute, min_group_size=20)`** —
`analyze_segments` above answers "does the model perform equally well
across groups" (precision/recall/F1 per value). This answers a different,
narrower question: "does the model *treat* groups differently in a way a
fairness/bias review would flag" — demographic parity difference,
disparate impact ratio (the standard 80%/four-fifths rule), and an
equal-opportunity (true-positive-rate) gap. Binary targets only. Call this
whenever the dataset has a column that is, or proxies, a protected
attribute (gender, age band, region-as-proxy-for-ethnicity, etc.) relevant
to the domain — check `meta["business_understanding"]["domain"]` (set by
`business-understanding`) and `modeling/references/on_demand/domain-notes.md`'s
per-domain "Fairness-relevant attributes" note (fetch via `load_skill(
"modeling", reference="domain-notes")` if it hasn't been read yet this
conversation) for which column a domain typically cares about — this is a
different question from `analyze_segments`'s segment column, even though
both are now informed by the same `domain-notes.md` section: a segment is
any group whose performance you want broken out, a protected attribute is
one the model is not allowed to treat differently. Do not substitute one
check for the other. A group with fewer than `min_group_size` rows
in the test fold is marked `insufficient_sample` and excluded from the
ratio math, not silently averaged in — say so plainly rather than reporting
a number with no real support. This is a screening tool, not a legal
certification; a flagged `disparate_impact_concern` or
`equal_opportunity_concern` is a stop-and-ask signal in `modeling`'s
autonomous mode, same tier as an overfitting or leakage warning. If no
column in the dataset is a plausible protected attribute, skip this tool
and say so in the reflection step below (`fairness_assessment_status:
not_applicable`) — don't force a fairness check onto a dataset that has no
demographic dimension.

**`backtest_on_new_data(run_id, path)`** — score the current model against a
*later* labelled file, at the run's operating point. The test fold answers
"does this work on data I held out"; only this answers "does this still
work", and in an adversarial domain they are different questions. Report
`pr_auc_change` alongside the headline whenever a later period exists, and
read the positive-rate shift separately from the metric shift: a population
that moved is a different problem from a model that decayed, and they call
for different responses.

## Calibration — only when a score, not a decision, leaves the model

If the output is thresholded into a yes/no (block this call, flag this
account), calibration does not matter — `tune_threshold` picks the
operating point on the raw scores and only the ranking is used. It starts
mattering the moment a *number* is handed onward: a risk tier, a score
shown to an analyst, a probability feeding a cost calculation. There a
predicted 0.8 has to mean roughly 80%, and tree ensembles are routinely
badly calibrated even when their ranking is excellent.

`check_calibration(run_id)` measures whether the predicted probabilities
mean what they say — Brier score plus expected calibration error (ECE), with
the per-bin table behind it. Run it after training and **before**
`tune_threshold`: every threshold mode reads precision/recall off the
probability distribution, so tuning on uncalibrated scores picks a cutoff
whose stated tradeoff isn't the one you get. ECE above 0.05 raises
`miscalibration_warning`; the fix is `calibrate_model` (modeling skill),
then re-run `check_calibration` and everything the refit invalidated.
ECE is measured on out-of-fold predictions for the training fold, because it
DECIDES whether and how to calibrate; the test fold's ECE is reported beside
it as `test_expected_calibration_error` — read it, never pick a method by it.

The readiness gate records this as `calibration_checked`. It FAILS a run
that tuned a threshold without ever measuring calibration, and resolves to
`not_applicable` (with the reason stated) on a run that never tuned one —
a model read purely as a ranking doesn't need calibrated probabilities, and
a gate that fires where it doesn't apply teaches people to ignore gates.

The visualization agent's
`plot_calibration_curve(run_id, out_path)` is how you look. Check it
whenever `domain-notes.md`'s domain section asks for it (healthcare does
explicitly) or whenever this run's output is a score rather than a
decision, and if the curve is visibly off the diagonal, say so in the
reflection and in the report's limitations rather than passing the
probabilities on as though they were meaningful.

## Reflection — the self-critique, made auditable

**`record_reflection(run_id, checks, critical_issues, recommended_actions)`**
— converts the chain-of-thought self-critique `modeling/SKILL.md` already
asks for into a structured record instead of prose that only exists in the
chat transcript. Call once near the end of an autonomous run, after
`explain_model`/`error_analysis`/`check_model_stability` (and
`assess_fairness`, if it applies) have run. `checks` covers ten fixed
dimensions (see `mcp_server/diagnostics.py`'s `REFLECTION_CHECKS`) — cite
the actual tool result as evidence for each ("detect_data_leakage: clear"),
not a restated opinion, and mark anything you didn't actually check as
`not_assessed` rather than skipping it silently. This is what lets a human
reviewer trust that "the agent verified this" is true, not just claimed.
