---
name: decision-explanation
description: Explain why a model made one specific decision about one specific row — the reasons behind a fraud flag, a churn alert, a credit denial. Load this whenever the question is about a particular case rather than the model in general.
---

# Decision Explanation

Tools: `inspect_explainability(pkl_path)`, `explain_prediction(pkl_path,
data_path, row, top_n)`.

Both read-only. Nothing here needs `ask_user` — there is no decision to
gate, only a model to read.

## Steps

1. **Check what you're working with** — `inspect_explainability(pkl_path)`
   before your first explanation of a model you didn't just see exported.
   It costs one artifact load and tells you the task type, the explainer
   that applies, whether a reference dataset exists, whether the model is
   calibrated, and the threshold its decisions actually use. If it returns
   `explainable: false`, relay `reason_if_not` and stop — don't try
   `explain_prediction` anyway to see what happens.

2. **Explain the row** — `explain_prediction(pkl_path, data_path, row)`.
   `row` is 0-based into `data_path`. If the caller identified the case by
   an account id or a transaction reference rather than a row number, say
   you need the row index, or ask which row it is — do not guess, because
   explaining the wrong row produces a confident answer about somebody
   else.

3. **Read the caveats before the contributions.** `additivity_ok` and
   `explanation_stability` come back with every call. See the base
   SKILL.md's rule 2 — these decide whether you have an answer at all, so
   read them first, not last.

4. **Report the decision, then the reasons.** `margin_over_threshold` is
   the fact that matters: how far over the line this case fell. A case at
   +0.002 over the threshold and a case at +0.4 are different stories even
   when their top drivers are identical, and only one of them is worth a
   human's time to review.

## Reading contributions honestly

- **Positive contribution = pushed toward the positive class**, which the
  result names explicitly. Don't assume the positive class is "1" or the
  second label; on a `{"churn","no_churn"}` model the positive class is
  `churn`, and it sorts first.
- **`base_value` is the model's average output over the reference data** —
  what it would say knowing nothing about this row. Contributions are
  movements away from that, not absolute quantities of blame.
- **A feature absent from the top list is not unimportant** — it may be
  important to the model overall and simply unremarkable for this row.
  That distinction matters when someone asks "so my account age didn't
  matter?" The honest answer is "it didn't move *your* score much."
- **On a calibrated model**, contributions are reported on the
  uncalibrated score and the result says so. Relay the calibrated
  probability as the decision, and the contributions as the reasons; don't
  claim the contributions add up to the calibrated number, because they
  don't.

## When the answer is "I can't tell you"

Say it plainly, with the reason, and offer the next step:

- no reference dataset → the model was exported before its training agent
  shipped one; re-export it
- unsupported model family → name the family and say it's refused rather
  than approximated
- `additivity_ok: false` → the explanation doesn't describe this
  prediction; report no reasons
- very low stability → give the explanation with the caveat first, and say
  it should not be quoted to the affected party

None of these are failures to apologise for. An explanation you can't
stand behind is the one thing this agent must never produce.
