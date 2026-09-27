"""explain agent — `model_explanation` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import (
    _encode,
    _explainer_for,
    _final_step_name,
    _native,
    _positive_index,
    _prepare,
    _shap_matrix,
)  # noqa: F401


@mcp.tool()
def explain_model(pkl_path: str, top_n: int = 10, sample_size: int = 300) -> str:
    """**What drives this model in general?** Global mean-|SHAP| ranking
    over a sample of the reference data.

    Use this for a model you have only as a .pkl — an export from somewhere
    else, or one whose training run is long gone. If the model was trained
    here and you have its run_id, the training agent's own `explain_model`
    is the one to use instead: it runs on the held-out TEST fold (this one
    runs on the training reference, which flatters a model that memorised),
    and its result is what that run's explainability gate reads.

    sample_size caps the rows SHAP runs on. On a classifier the sample is
    stratified toward keeping the positive class represented, so a rare
    class like fraud is not averaged away."""
    ctx, err = _prepare(pkl_path)
    if err:
        return json.dumps(err)
    pipeline, bundle = ctx["pipeline"], ctx["bundle"]
    reference = ctx["reference"]
    if 0 < sample_size < len(reference):
        reference = reference.sample(sample_size, random_state=42)

    X_enc, feature_names = _encode(pipeline, reference)
    estimator = pipeline.named_steps[_final_step_name(pipeline)]
    explainer, explainer_kind = _explainer_for(ctx["model_name"], estimator, X_enc)
    if explainer is None:
        return json.dumps(
            {"error": f"no exact SHAP explainer for model '{ctx['model_name']}'"}
        )

    classes = list(getattr(pipeline, "classes_", []))
    pos_idx = _positive_index(bundle, classes) if ctx["task"] == "classification" else 0
    try:
        raw = explainer.shap_values(X_enc)
    except Exception as exc:
        if "additivity" not in str(exc).lower():
            return json.dumps({"error": f"SHAP failed: {type(exc).__name__}: {exc}"})
        raw = explainer.shap_values(X_enc, check_additivity=False)
    values = _shap_matrix(raw, pos_idx)

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    order = np.argsort(-mean_abs)[:top_n]
    return json.dumps(
        {
            "pkl_path": pkl_path,
            "model": ctx["model_name"],
            "task": ctx["task"],
            "explainer": explainer_kind,
            "positive_class": (
                str(bundle.get("positive_label"))
                if ctx["task"] == "classification"
                else None
            ),
            "feature_importance": [
                [feature_names[i], round(float(mean_abs[i]), 6)] for i in order
            ],
            "direction": {
                feature_names[i]: ("increases" if mean_signed[i] > 0 else "decreases")
                for i in order
            },
            "direction_caveat": "a global average direction — individual rows can be pushed the other way; use "
            "explain_prediction for a specific case",
            "computed_on": f"{len(reference)} rows of {ctx['reference_path']}",
            "scope_caveat": "computed on the TRAINING reference shipped with this model, not a held-out test fold — "
            "it describes what the model uses, and flatters a model that memorised. For gated "
            "evidence about a run, use the training agent's own explain_model on its test fold.",
        },
        default=_native,
    )
