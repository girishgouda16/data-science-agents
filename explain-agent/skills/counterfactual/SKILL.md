---
name: counterfactual
description: Work out what would have had to be different for a model's decision to go the other way — the adverse-action question behind a denial, a flag, or an alert. Load this when the caller asks what would change the outcome, not just what caused it.
---

# Counterfactual

Tool: `explain_counterfactual(pkl_path, data_path, row, max_features)`.

Read-only. Classifiers only — a regression model has no decision to flip,
so use `explain_prediction`'s contributions there instead.

## When to reach for this instead of attribution

Attribution answers "what drove this score". A counterfactual answers
"what would have had to be different". People asking about a decision that
went against them almost always want the second one, even when they phrase
it as the first. A denied applicant asking "why was I rejected" is rarely
asking for a SHAP ranking.

Run `explain_prediction` first anyway when both are wanted: the
attribution tells you whether the explanation is sound at all
(`additivity_ok`, `explanation_stability`), and a counterfactual built on
an unstable model deserves the same caveat.

## Reading the result

- **`decision_flipped: true`** — the tool found a set of changes that
  crosses the threshold. Report which features, from what to what, and the
  new score.
- **`decision_flipped: false`** — this is informative, not a failure. It
  means no small change to the most influential *numeric* features moves
  the decision. Either the case is strongly determined, or it's driven by
  categorical values this search doesn't vary. Say which you think it is,
  and don't pad the answer by listing changes that didn't work as though
  they were findings.

## Always relay the limitations

The tool returns them; they are not boilerplate, and dropping them is how
this output becomes harmful:

- **greedy, not minimal** — a smaller set of changes may exist
- **targets the reference median** — a plausible value, not an achievable
  one. "Your account would need to be 4 years old" is arithmetic, not
  advice, and nobody can act on it.
- **ignores correlations** — the counterfactual row may not be a realistic
  customer at all
- **association, not causation** — "the score would differ if this value
  differed" is not "changing this changes the outcome"

Never phrase the output as a recommendation to a person ("to avoid being
flagged, reduce X"). Phrase it as a property of the model ("the model's
decision on this case turns mainly on X; at a value near the population
median it would not have flagged"). The first is advice this agent has no
standing to give and no causal evidence to support. The second is true.

If the case looks like a regulated adverse-action decision — credit,
insurance, employment — say explicitly that this output is input to a
human-written reason, not a compliant notice in itself.
