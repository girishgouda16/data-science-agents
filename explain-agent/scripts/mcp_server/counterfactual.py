"""explain agent — `counterfactual` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import _decision, _native, _prepare, _score  # noqa: F401


@mcp.tool()
def explain_counterfactual(
    pkl_path: str, data_path: str, row: int = 0, max_features: int = 3
) -> str:
    """**What would have had to be different for this decision to change?**

    The adverse-action question. Attribution says which features pushed the
    score; a counterfactual says what would have flipped it — which is what
    a regulator, or the person the decision was about, actually asks for.

    Greedy search: takes the features contributing most toward the current
    decision and moves them, one at a time, to their median in the
    reference data, stopping as soon as the decision flips. Reports the
    minimal set it found and the new score.

    Honest about its limits, and they matter:
      - this is a greedy search over at most `max_features`, not a proven
        minimal counterfactual
      - it moves features to a reference median, so the result is a
        plausible value but not necessarily an ACTIONABLE one (nobody can
        change their account age), and it does not respect correlations
        between features
      - it says nothing about causation: "the score would differ if this
        value differed" is not "changing this would change the outcome"
    Treat the output as a starting point for a human explanation, never as
    advice to give someone directly."""
    ctx, err = _prepare(pkl_path, data_path)
    if err:
        return json.dumps(err)
    if ctx["task"] != "classification":
        return json.dumps(
            {
                "error": "counterfactuals here are defined for a classifier's decision; for a "
                "regression model use explain_prediction's contributions"
            }
        )
    if ctx["data"] is None or not 0 <= row < len(ctx["data"]):
        return json.dumps({"error": f"row {row} is out of range for '{data_path}'"})

    X_row = ctx["data"].iloc[[row]].copy()
    original_score = float(_score(ctx, X_row)[0])
    original = _decision(ctx, original_score)
    threshold = original["threshold"]
    flagged = original_score >= threshold

    # Rank the raw (pre-encoding) columns by how much moving them to the
    # reference median shifts the score — a one-at-a-time sensitivity on the
    # columns a human can actually talk about, not on one-hot fragments.
    numeric = [
        c
        for c in X_row.columns
        if c in ctx["reference"].columns
        and pd.api.types.is_numeric_dtype(ctx["reference"][c])
    ]
    deltas = []
    for col in numeric:
        probe = X_row.copy()
        probe[col] = ctx["reference"][col].median()
        try:
            shifted = float(_score(ctx, probe)[0])
        except Exception:
            continue
        move = (original_score - shifted) if flagged else (shifted - original_score)
        if move > 0:
            deltas.append(
                (
                    col,
                    move,
                    _native(X_row[col].iloc[0]),
                    _native(ctx["reference"][col].median()),
                )
            )
    deltas.sort(key=lambda d: -d[1])

    applied, probe = [], X_row.copy()
    new_score = original_score
    for col, _move, was, becomes in deltas[:max_features]:
        probe[col] = becomes
        applied.append({"feature": col, "from": was, "to": becomes})
        new_score = float(_score(ctx, probe)[0])
        if (new_score < threshold) if flagged else (new_score >= threshold):
            break

    flipped = (new_score < threshold) if flagged else (new_score >= threshold)
    return json.dumps(
        {
            "pkl_path": pkl_path,
            "row": row,
            "original": original,
            "changes_applied": applied,
            "new_probability_of_positive": round(new_score, 6),
            "decision_flipped": flipped,
            "result": (
                f"changing {len(applied)} feature(s) moves the score from {original_score:.4f} to {new_score:.4f}, "
                f"across the {threshold} threshold — the decision changes"
                if flipped
                else f"moving the {len(applied)} most influential numeric feature(s) to their reference median only moves "
                f"the score from {original_score:.4f} to {new_score:.4f}, which does not cross {threshold}. This "
                "decision is not driven by any small number of numeric features — it is either strongly determined or "
                "driven by categorical values this search does not vary."
            ),
            "limitations": [
                "greedy one-at-a-time search over numeric features only, not a proven minimal counterfactual",
                "targets the reference median — a plausible value, not necessarily an achievable or actionable one",
                "ignores correlations between features, so the counterfactual row may not be realistic",
                "association, not causation — do not present this as 'change X and the outcome changes'",
            ],
        },
        default=_native,
    )
