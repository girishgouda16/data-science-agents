---
name: reporting
description: Generate a full markdown report for a completed run — EDA, feature transforms, feature engineering, model, SHAP/explainability, and evaluation. Load after train_model (and optionally tune/explain/threshold) has produced a current pipeline. No HITL — read-only, summarizes what already ran.
---

# Reporting

**MCP module:** `mcp_server/reporting.py`

Tool: `generate_report(run_id)`.

`generate_report` is self-sufficient — it does **not** depend on anything
from earlier in the conversation. It recomputes EDA stats from
`meta["source_path"]`, reads `meta.json` for what every phase actually did
(with reasons/formulas/evidence, not just column names), and replays
`pipeline.pkl` on `test.csv` for metrics/confusion/ROC/PR (never refit).
Calling it alone, in a fresh session, with only a `run_id`, reproduces the
same report — that's the point: nothing here is conversation memory. This
is also why every phase's tools (`business-understanding`,
`diagnostics`'s leakage/baseline/segments/fairness/reflection,
`modeling`'s `compare_models`/`compare_runs`) persist their result onto
`meta.json` rather than only returning it — a result that only ever lived
in the chat transcript can't end up here.

Sections, always in this order:

0. **Executive Summary** — the section a manager actually reads: headline
   (model + tuned/baseline + accuracy + ROC-AUC), top driver with direction
   (from `meta["explain"]`, SHAP only — permutation has no direction to
   report), risk caveats (overfitting gap, CV instability from
   `check_model_stability` if it was run, class imbalance, small-sample
   size), and a rule-based ✅ Proceed / ⚠ Hold recommendation anchored in
   the overfitting/stability gates the rest of the agent already computes
   — never a hallucinated judgment call. Every claim in this section traces
   to a field computed elsewhere in the report; if a field is missing (e.g.
   `explain_model` never ran) it says so explicitly rather than guessing.
1. **Business Understanding & Assumptions** — from `meta["business_understanding"]`
   (`business-understanding` skill's `record_business_context`): the
   business objective, target definition, **label provenance** (how the
   labels were produced, and whether the negatives are confirmed or merely
   un-flagged), success criteria, and any assumption documented instead of
   asking. Says "not recorded" rather than
   silently omitting the section if that tool was never called.
2. **HITL Clarification History** — every `ask_user` round actually used to
   resolve a business ambiguity (question, answer, and its impact on the
   plan), from the same `business_understanding.clarifications` field. This
   is what makes the agent's business-side judgment calls reviewable
   without needing the raw chat transcript.
3. **EDA** — raw shape, missingness by column (recomputed from the
   original file), training-fold class balance, outlier counts (z-score).
4. **Feature transforms** — dropped columns with the actual reason
   (constant / near-unique / high-cardinality / name-match / >50% missing),
   per-column imputed value, datetime decomposition, frequency-encoded
   columns (with how many test rows hold a value never seen in training),
   SMOTE on/off, and **the split type** — random, grouped, or temporal,
   with the column it used and, for a temporal split on entity data, the
   percentage of test rows whose entity also appears in training. The split
   line belongs here rather than buried in the model section because it
   determines whether every metric below it means anything.
5. **Feature engineering** — each applied feature's name, formula, and the
   recorded rationale (the mechanism, and the legitimate population that
   looks the same). Says so explicitly when a feature was applied with no
   rationale, rather than printing the formula alone as if it were
   self-explanatory.
6. **Model** — source, target, train shape, split type, model +
   tuned/baseline, best hyperparameters.
7. **Explainability** — the last `explain_model` result: method, positive
   class, and a top-features table annotated with direction ("increases"/
   "decreases" the prediction) when `method="shap"` was used — see
   `modeling/SKILL.md`'s default-to-SHAP rule. Says "not run yet" rather
   than a blank/missing section if `explain_model` was never called for
   this run — call it before `generate_report` in autonomous mode (see
   `modeling/SKILL.md`'s flow) so this section (and the Executive
   Summary's top-driver line) is never empty.
7b. **Operating point** — the threshold that ships, how it was chosen, and
   precision/recall/F1/alert-rate against the 0.5 default side by side. Says
   plainly when no threshold was tuned, because then the exported model
   decides at 0.5 and a reader who assumes otherwise is wrong about what was
   built.
7c. **Label-rule screen** — from `meta["label_rule_check"]`: the best
   single-column rule's PR-AUC as a fraction of the model's, and whether
   that was flagged and acknowledged. A model that a two-split tree can
   reproduce is a finding about the labels, not about the model.
8. **Diagnostics** — data leakage check, dummy-baseline comparison,
   `compare_models`'s ranked alternatives (with a note if the trained model
   deviated from CV's top pick), error analysis, and per-segment
   performance — from `meta["leakage"]`/`baseline_comparison"`/
   `"model_comparison"`/`"error_analysis"`/`"segment_analysis"`.
9. **Fairness & Bias Assessment** — from `meta["fairness"]`
   (`assess_fairness`): demographic parity difference, disparate impact
   ratio, equal-opportunity gap, per-group rates, and its own limitations.
   Distinct from Diagnostics' segment performance — see `diagnostics/SKILL.md`.
   Says "not run" if no protected attribute applied and `assess_fairness`
   was skipped.
10. **Classification metrics, Confusion matrix, ROC curve, PR curve** — the
    numeric backbone, from `pipeline.pkl` replayed on `test.csv`.
10b. **Forward backtest** — from `meta["backtest"]`, when a later labelled
    period was scored: PR-AUC change, precision/recall at the operating
    point, and the positive-rate shift. Says "not run" otherwise, since the
    test fold cannot speak to a later period.
11. **Known Limitations** — run-specific limitations built from this run's
    actual evidence (overfitting/stability warnings, imbalance, sample
    size, missing fairness assessment, unresolved reflection issues, an
    unrecorded business-understanding gate) plus this agent's fixed scope
    limitations (3-model zoo, no nested CV, no calibration, etc. — see
    `mcp_server/reporting.py`'s `STATIC_LIMITATIONS`) — always visible here,
    not only in the separate `PRODUCTION_READINESS.md`.
12. **Reflection & Self-Critique** — from `meta["reflection"]`
    (`record_reflection`): every checklist dimension's status and cited
    evidence, critical issues, and recommended next steps. Says "not
    recorded" if the modeling skill's self-critique step never persisted
    one — a report missing this section is a report that hasn't actually
    been reviewed, not just a shorter one.

**Plots are not rendered by `generate_report`, and not by you.** They
come from the separate visualization agent, which only the orchestrator can
reach; it saves each run-keyed chart into `<run>/charts/pv<N>_<kind>.png`.
`generate_report` embeds every chart that exists for the CURRENT fit in its
"Evaluation charts" section (relative links, so they also resolve in
MLflow), and says "none rendered for this fit" otherwise — which is also
what the `evaluation_charts` gate reports. A report generated before the
charts exist is regenerated at champion promotion, so the version's MLflow
record carries the charts the reviewer actually saw.

This instruction has been skipped in a real run, so to be unambiguous: a
report that hands the user ROC/PR coordinates as raw number arrays and a
SHAP ranking as a bullet list has NOT satisfied this step. Those are the
plots' inputs, not a substitute for them — nobody should have to graph a
report in their head. Being asked "where are the charts?" is proof this step
was missed.

If a visualization call genuinely errors, quote the error. Don't report
charts as unavailable without having tried — and note that the visualization
agent is a SEPARATE MCP server from this one, so establish which is actually
unreachable before calling anything missing, and never generalise its
absence into a claim about the tools that live in this server.

Call this as the last step of an autonomous run (see `modeling/SKILL.md`'s
autonomous mode), or any time the user asks for "a report" / "a summary" on
a run_id. Writes `report.md` next to the run's other artifacts and returns
its content — surface that (plus the spliced-in plots) directly to the user
instead of paraphrasing it.

It never refuses to run, but it is not a way to make an unfinished run look
finished: its Run Readiness section states every deterministic gate's status
up front, its ship/hold call comes from that gate rather than from whichever
warnings happen to be set, and its Execution Log lists every tool call that
actually happened. Call `check_readiness` BEFORE this and clear the
outstanding gates — a report is meant to be the summary of finished work,
not the place you find out what you skipped.
