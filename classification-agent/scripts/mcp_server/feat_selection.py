"""Model-driven feature selection — a different question from
feature-engineering's "what to add": this is "what to remove because the
CURRENT trained model shows it isn't earning its place". Read-only
proposal; the write step reuses core.apply_drop_columns rather than adding
a second drop tool, since the effect (drop columns from train/test) is
identical."""

import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from .core import _cv_sample, _load_split, _run_dir, _validation_folds, mcp, split_keys


@mcp.tool()
def propose_feature_selection(run_id: str, bottom_k: int = 3) -> str:
    """Ranks this run's CURRENT fitted model's raw input columns by
    permutation importance on held-out TRAINING folds (a copy of the model
    refit per fold, scored on the fold it didn't see) and returns the
    bottom_k lowest-importance ones as drop candidates — noise columns a
    tree/linear model learned to ignore anyway, kept only because nobody
    removed them. Read-only. Gate with the user (a near-zero-importance
    column can still be the one a stakeholder insists on keeping for
    interpretability), then call apply_drop_columns with the approved
    names and re-run train_model.

    Not the test fold: dropping the columns that look weakest on the test
    rows and then reporting on those same rows is selecting on the test set."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    pipeline = joblib.load(pipeline_path)
    train, _, meta = _load_split(run_id)
    X, y, _ = _cv_sample(train.drop(columns=[meta["target"]]), train[meta["target"]], meta)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    folds, scheme = _validation_folds(X, y, meta)
    per_fold = []
    for fit_idx, val_idx in folds:
        model = clone(pipeline).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        per_fold.append(
            permutation_importance(
                model, X.iloc[val_idx], y.iloc[val_idx], n_repeats=3, random_state=42
            ).importances_mean
        )
    # The split keys never reach the model, so their importance is exactly 0 —
    # proposing them for removal would only break grouped/temporal validation.
    keys = set(split_keys(meta))
    ranked = sorted(
        ((c, v) for c, v in zip(X.columns, np.mean(per_fold, axis=0)) if c not in keys),
        key=lambda kv: kv[1],
    )
    candidates = [
        {"column": c, "importance": round(float(v), 4)} for c, v in ranked[:bottom_k]
    ]
    return json.dumps(
        {
            "run_id": run_id,
            "drop_candidates": candidates,
            "measured_on": scheme,
            "next_step": "ask_user to approve, then apply_drop_columns(run_id, columns) with the approved names, then re-run train_model",
        }
    )
