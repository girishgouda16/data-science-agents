"""explain agent — `decision` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import (
    _decision,
    _encode,
    _explainer_for,
    _final_step_name,
    _narrate,
    _native,
    _positive_index,
    _prepare,
    _score,
    _shap_matrix,
    _stability,
)  # noqa: F401

# ─────────────────────────────────────────────────────────────────── tools


@mcp.tool()
def inspect_explainability(pkl_path: str) -> str:
    """Reports whether a given exported .pkl can be explained, and how,
    WITHOUT computing anything. Call this first on a model you didn't
    export yourself: it names the task type, the explainer that applies,
    whether a reference/background dataset is available, whether the model
    is wrapped in a calibrator, and which decision threshold its
    predictions actually use. Cheap; loads the artifact and nothing else."""
    ctx, err = _prepare(pkl_path)
    if err:
        # Still useful to say WHY it can't be explained, including the
        # missing-reference case, which is fixable by re-exporting.
        return json.dumps(err)
    supported = any(t in ctx["model_name"] for t in TREE_MODELS + LINEAR_MODELS)
    op = ctx["bundle"].get("operating_point") or {}
    return json.dumps(
        {
            "pkl_path": pkl_path,
            "task": ctx["task"],
            "model": ctx["model_name"],
            "explainer": (
                "TreeExplainer (exact)"
                if any(t in ctx["model_name"] for t in TREE_MODELS)
                else (
                    "LinearExplainer (exact)"
                    if any(l in ctx["model_name"] for l in LINEAR_MODELS)
                    else None
                )
            ),
            "explainable": supported,
            "reason_if_not": (
                None
                if supported
                else f"no exact SHAP explainer for model '{ctx['model_name']}' — refused rather than approximated with "
                "KernelExplainer, whose sampling error is not reported and which can take minutes per row"
            ),
            "reference_path": ctx["reference_path"],
            "reference_rows": len(ctx["reference"]),
            "calibrated": ctx["calibrated"],
            "calibration_note": (
                "this model is wrapped in a calibrator — contributions are additive in the UNCALIBRATED score, so "
                "explanations are reported on that scale and the calibrated probability is shown alongside, not "
                "decomposed"
                if ctx["calibrated"]
                else None
            ),
            "positive_label": ctx["bundle"].get("positive_label"),
            "decision_threshold": op.get("threshold", 0.5),
            "run_id": ctx["bundle"].get("run_id"),
        }
    )


@mcp.tool()
def explain_prediction(
    pkl_path: str, data_path: str, row: int = 0, top_n: int = 8
) -> str:
    """**Why did this model decide THIS row the way it did?**

    Computes exact per-feature SHAP contributions for one row of data_path
    and reports them against the decision the model actually makes — for a
    classifier, the distance to its *tuned* threshold, not to 0.5. "Why was
    this flagged" means "why did the score cross the cutoff this model
    ships with", and those are different questions whenever a threshold was
    tuned.

    row: 0-based index into data_path. data_path is shaped like the
    training data (target column optional, dropped if present).

    The result carries an additivity check: base_value plus the
    contributions must reconstruct the model's score. SHAP is exact for
    these model families, so a failed check means the explanation does not
    describe this prediction — it is reported as `additivity_ok: false` and
    should not be shown to anyone as a reason.

    Contributions are measured against the training fold shipped beside the
    .pkl. A contribution is always relative to something; that reference is
    named in the result.

    Also returns a stability check: whether rows near this one in the
    reference data get a similar explanation. A local attribution on a
    high-variance model can swing between near-identical cases, and an
    unstable explanation presented confidently is worse than none —
    especially if it is going to the person the decision was about."""
    ctx, err = _prepare(pkl_path, data_path)
    if err:
        return json.dumps(err)
    if ctx["data"] is None or ctx["data"].empty:
        return json.dumps({"error": f"'{data_path}' has no rows"})
    if not 0 <= row < len(ctx["data"]):
        return json.dumps(
            {
                "error": f"row {row} is out of range — '{data_path}' has {len(ctx['data'])} rows (0-based)"
            }
        )

    X_row = ctx["data"].iloc[[row]]
    pipeline, bundle = ctx["pipeline"], ctx["bundle"]

    X_bg, feature_names = _encode(pipeline, ctx["reference"])
    X_enc, _ = _encode(pipeline, X_row)
    estimator = pipeline.named_steps[_final_step_name(pipeline)]
    explainer, explainer_kind = _explainer_for(ctx["model_name"], estimator, X_bg)
    if explainer is None:
        return json.dumps(
            {
                "error": f"no exact SHAP explainer for model '{ctx['model_name']}' — refused rather than approximated",
            }
        )

    classes = list(getattr(pipeline, "classes_", []))
    pos_idx = _positive_index(bundle, classes) if ctx["task"] == "classification" else 0
    try:
        raw = explainer.shap_values(X_enc)
    except Exception as exc:
        if "additivity" not in str(exc).lower():
            return json.dumps({"error": f"SHAP failed: {type(exc).__name__}: {exc}"})
        raw = explainer.shap_values(X_enc, check_additivity=False)
    values = _shap_matrix(raw, pos_idx)[0]

    expected = explainer.expected_value
    base = (
        float(np.asarray(expected).flatten()[pos_idx])
        if np.asarray(expected).size > 1
        else float(np.asarray(expected).flatten()[0])
    )

    score = float(_score(ctx, X_row)[0])
    decision = _decision(ctx, score)

    # Additivity: base + Σcontributions must reconstruct the model's raw
    # output. For a classifier that raw output is a margin/log-odds, not the
    # probability, so the reconstruction is compared against the estimator's
    # own score on the encoded row rather than against predict_proba.
    reconstructed = base + float(values.sum())
    try:
        if ctx["task"] == "classification" and hasattr(estimator, "predict_proba"):
            raw_out = float(np.asarray(estimator.predict_proba(X_enc))[0, pos_idx])
            # TreeExplainer on a probability-output forest reconstructs the
            # probability directly; on a margin-output booster it does not.
            additivity_ok = (
                abs(reconstructed - raw_out) < ADDITIVITY_TOLERANCE
                or abs(1 / (1 + np.exp(-reconstructed)) - raw_out)
                < ADDITIVITY_TOLERANCE
            )
        else:
            additivity_ok = (
                abs(reconstructed - float(np.asarray(estimator.predict(X_enc))[0]))
                < ADDITIVITY_TOLERANCE
            )
    except Exception:
        additivity_ok = None

    order = np.argsort(-np.abs(values))[:top_n]
    contributions = [
        {
            "feature": feature_names[i],
            "value": _native(X_enc[0, i]),
            "contribution": round(float(values[i]), 6),
            "direction": "toward" if values[i] > 0 else "away from",
        }
        for i in order
    ]
    stability = _stability(
        ctx, explainer, pipeline, X_enc, values, pos_idx, feature_names
    )

    positive_name = decision.get("positive_class", "the predicted value")
    return json.dumps(
        {
            "pkl_path": pkl_path,
            "row": row,
            "model": ctx["model_name"],
            "explainer": explainer_kind,
            **decision,
            "base_value": round(base, 6),
            "base_value_meaning": f"the model's average output over the reference data — what it would say knowing "
            f"nothing about this row",
            "contributions": contributions,
            "summary": _narrate(contributions, decision, positive_name, ctx["task"]),
            "additivity_ok": additivity_ok,
            "additivity_note": (
                None
                if additivity_ok is not False
                else "base value plus contributions does NOT reconstruct this model's score — this explanation does not "
                "describe this prediction and must not be presented as a reason for it"
            ),
            "reference_path": ctx["reference_path"],
            "reference_rows": len(ctx["reference"]),
            "calibrated": ctx["calibrated"],
            "calibration_note": (
                "contributions are on the UNCALIBRATED score; the calibrated probability is reported above but is not "
                "decomposed, because SHAP values are not additive through the calibration map"
                if ctx["calibrated"]
                else None
            ),
            **stability,
        },
        default=_native,
    )
