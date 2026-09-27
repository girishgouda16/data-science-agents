"""anomaly agent — `diagnostics` tools: the business framing, the data and
label-leak screen, and the deterministic readiness gate a senior fraud /
anomaly review asks for. Mechanism is core/gates.py + core/stages.py; what
"finished and safe to believe" MEANS for a detector lives here."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import split_keys  # noqa: F401
from .core import (
    _current_metrics,
    _feature_frame,
    _label_to_binary,
    _load_meta,
    _load_split,
    _run_dir,
    _save_meta,
)  # noqa: F401

from core import gates as gate_engine  # noqa: E402
from core import stages  # noqa: E402
from core.data_quality import identifier_columns  # noqa: E402

REQUIRED_GATES = (
    "business_context",
    "leakage_screened",
    "no_identifier_in_model",
    "contamination_justified",
    "detection_quality",
    "alerts_stable",
    "explainability_run",
    "evidence_ordering",
    "success_criteria",
)
SUCCESS_METRICS = {
    "roc_auc": "at_least",
    "precision": "at_least",
    "recall": "at_least",
    "f1": "at_least",
    "average_precision": "at_least",
    "precision_at_budget": "at_least",
    "recall_at_budget": "at_least",
}
NEAR_PERFECT_AUC = 0.98


def measured_metric(meta: dict, metric: str):
    return (
        (meta.get("metrics") or {}).get(metric)
        if meta.get("evaluation_available")
        else None
    )


@mcp.tool()
def detect_data_leakage(run_id: str) -> str:
    """Screen the training fold before fitting a detector: identifier-shaped
    columns (a detector keyed on an ID flags whatever is rare in the ID, not
    in behaviour) and — when a label column exists — any STATUS-SHAPED
    feature (categorical, or a 0/1-style flag) that on its own nearly IS the
    label: a case-status or investigation-outcome field filled in after the
    fact. Continuous features that separate anomalies are the signal, not a
    leak. Call BEFORE train_model, and again after any column is dropped or
    imputed (the evidence_ordering gate)."""
    train, _, meta = _load_split(run_id)
    label = meta.get("label_column")
    features = _feature_frame(train, meta).drop(columns=split_keys(meta))
    signals = identifier_columns(features, None)
    strong = [c for c, s in signals.items() if s["strength"] == "strong"]
    single = {}
    if label and meta.get("anomaly_label") is not None:
        # Only status-shaped columns: a categorical or a 0/1-style flag that
        # mirrors the label is a field filled in after the case was decided. A
        # CONTINUOUS feature separating anomalies perfectly is the phenomenon
        # itself (anomalies are extreme) — flagging it would reject the signal.
        status_like = [
            c
            for c in features.columns
            if c not in strong
            and (
                not pd.api.types.is_numeric_dtype(features[c])
                or features[c].nunique(dropna=True) <= 5
            )
        ]
        single = stages.single_column_auc(
            features, _label_to_binary(train[label], meta), status_like
        )
    result = {
        "run_id": run_id,
        "identifier_columns": signals,
        "strong_identifier_columns": strong,
        "label_copy_columns": sorted(
            c for c, a in single.items() if a >= NEAR_PERFECT_AUC
        ),
        "single_column_auc_top": dict(
            sorted(single.items(), key=lambda kv: -kv[1])[:5]
        ),
    }
    result["clear"] = not signals and not result["label_copy_columns"]
    meta["leakage"] = result
    _save_meta(run_id, meta)
    return json.dumps(result, default=str)


def compute_readiness(run_id: str) -> dict:
    """Every required gate's status for this run, computed from its artifacts."""
    train, _, meta = _load_split(run_id)
    metrics = _current_metrics(meta)
    acknowledged = set(meta.get("acknowledged_identifiers") or {})
    checks: dict = {}
    bu = meta.get("business_understanding") or {}
    checks["business_context"] = (
        gate_engine.gate(
            gate_engine.PASS,
            f"objective recorded; domain='{bu.get('domain') or 'not declared'}'",
        )
        if bu.get("business_objective") and bu.get("target_definition")
        else gate_engine.gate(
            gate_engine.NOT_RUN,
            "record_business_context was not called — nothing says what an anomaly IS here",
        )
    )

    leakage = meta.get("leakage")
    live_copies = [
        c
        for c in (leakage or {}).get("label_copy_columns") or []
        if c in train.columns and c not in acknowledged
    ]
    if not leakage:
        checks["leakage_screened"] = gate_engine.gate(
            gate_engine.NOT_RUN, "detect_data_leakage was never called for this run"
        )
    elif live_copies:
        checks["leakage_screened"] = gate_engine.gate(
            gate_engine.FAIL,
            f"feature(s) that on their own nearly ARE the label still feed the detector: {live_copies} — "
            "almost always a field filled in after the outcome; drop and retrain, or acknowledge with the reason",
        )
    else:
        checks["leakage_screened"] = gate_engine.gate(
            gate_engine.PASS,
            "detect_data_leakage ran; no label-copy column in the model",
        )

    live_ids = {
        c: s
        for c, s in identifier_columns(_feature_frame(train, meta).drop(columns=split_keys(meta)), None).items()
        if s["strength"] == "strong" and c not in acknowledged
    }
    checks["no_identifier_in_model"] = (
        gate_engine.gate(
            gate_engine.FAIL,
            "identifier-shaped columns still feed the detector: "
            + "; ".join(f"`{c}` ({s['reason']})" for c, s in live_ids.items())
            + " — drop them and retrain",
        )
        if live_ids
        else gate_engine.gate(
            gate_engine.PASS, "no identifier-shaped column feeds the detector"
        )
    )

    checks["contamination_justified"] = (
        gate_engine.gate(
            gate_engine.PASS,
            f"propose_contamination reasoned the rate before training (used {meta.get('contamination')})",
        )
        if stages.ran_before_training(meta, "propose_contamination")
        and meta.get("algorithm")
        else gate_engine.gate(
            gate_engine.NOT_RUN,
            "the contamination rate was never reasoned — call propose_contamination before train_model; "
            "it decides how many rows get flagged",
        )
    )

    if not meta.get("algorithm"):
        checks["detection_quality"] = gate_engine.gate(
            gate_engine.NOT_RUN, "no detector has been trained for this run"
        )
    elif not meta.get("evaluation_available"):
        checks["detection_quality"] = gate_engine.gate(
            gate_engine.NOT_APPLICABLE,
            "no label column — detection quality cannot be measured here; a human must review a "
            "sample of flagged rows before anyone acts on them",
        )
    else:
        auc, lift = metrics.get("roc_auc"), metrics.get("lift_at_budget")
        good = auc is not None and auc >= EXPORT_MIN_ROC_AUC and (lift is None or lift >= MIN_LIFT_AT_BUDGET)
        checks["detection_quality"] = gate_engine.gate(
            gate_engine.PASS if good else gate_engine.FAIL,
            f"held-out ROC-AUC {auc} vs the {EXPORT_MIN_ROC_AUC} floor; the review queue at the budget is "
            f"{lift}x the base rate (needs >= {MIN_LIFT_AT_BUDGET}x) — precision {metrics.get('precision_at_budget')}, "
            f"recall {metrics.get('recall_at_budget')}, average precision {metrics.get('average_precision')}",
        )

    stability = (metrics.get("stability") or {}).get("top_alert_jaccard_mean")
    if not meta.get("algorithm"):
        checks["alerts_stable"] = gate_engine.gate(gate_engine.NOT_RUN, "no detector has been trained for this run")
    elif stability is None:
        checks["alerts_stable"] = gate_engine.gate(gate_engine.NOT_RUN, "stability was not measured — retrain")
    else:
        checks["alerts_stable"] = gate_engine.gate(
            gate_engine.PASS if stability >= STABLE_JACCARD else gate_engine.FAIL,
            f"the top alerts overlap {stability:.0%} (Jaccard) across refits on resampled data "
            f"(needs >= {STABLE_JACCARD:.0%}) — below it the alert list is an artefact of the sample",
        )

    checks["explainability_run"] = (
        gate_engine.gate(
            gate_engine.PASS, "explain_model ranked what drives the current detector"
        )
        if stages.ran_after_training(meta, "explain_model", training=("train_model",))
        else gate_engine.gate(
            gate_engine.NOT_RUN, "explain_model has not explained the current detector"
        )
    )

    checks["evidence_ordering"] = gate_engine.check_screen_ordering(
        meta, "detect_data_leakage", ("apply_drop_columns", "apply_imputation", "apply_custom_feature",
                                      "apply_entity_features", "apply_peer_features")
    )
    checks["success_criteria"] = (
        gate_engine.evaluate_success_criteria(
            meta, measured_metric, tuple(SUCCESS_METRICS)
        )
        if meta.get("evaluation_available")
        or (bu.get("success_target") or {}).get("metric")
        else gate_engine.gate(
            gate_engine.NOT_APPLICABLE,
            "unlabelled data — no measurable bar; success is judged by reviewing flagged rows",
        )
    )
    return gate_engine.readiness(
        run_id, stages.apply_acknowledgements(checks, meta), REQUIRED_GATES
    )


def _report_sections(meta: dict) -> list:
    m = meta.get("metrics") or {}
    stab = m.get("stability") or {}
    return [
        (
            "Detector",
            [
                f"- Algorithm: {meta.get('algorithm')}; trained on {m.get('trained_on')}"
                + (" (fitted on a sample)" if m.get("fit_sampled") else ""),
                f"- Alert share (contamination): {meta.get('contamination')} — the review queue is the top "
                f"{m.get('alerts_at_budget') or 'share'} held-out rows",
                *([f"- Review queue at the budget: precision {m.get('precision_at_budget')}, recall "
                   f"{m.get('recall_at_budget')}, lift {m.get('lift_at_budget')}x over the base rate "
                   f"{m.get('test_label_prevalence')}; average precision {m.get('average_precision')}, ROC-AUC "
                   f"{m.get('roc_auc')}"] if meta.get("evaluation_available") else
                  ["- No labels: detection quality is unmeasured — a person must review the queue before acting"]),
                f"- Stability (top-alert overlap across resampled refits): {stab.get('top_alert_jaccard_mean')}",
                f"- Held-out alert rate: {m.get('holdout_alert_rate')} (a rate far from the alert share means the "
                "held-out data differs from training)",
                f"- Logged before scaling (skewed): {m.get('log_scaled_columns') or 'none'}",
                *[f"- Model warning: {w}" for w in m.get("model_warnings") or []],
                f"- Split: {meta.get('split_type')}"
                + (f" on `{meta.get('time_column')}`" if meta.get("time_column") else "")
                + (f", entity `{meta.get('group_column')}`" if meta.get("group_column") else ""),
                f"- Review queue: {meta.get('review_queue') or 'explain_model not run'}",
                f"- Labelled evaluation available: {meta.get('evaluation_available')} "
                f"(label column `{meta.get('label_column')}`, anomaly label `{meta.get('anomaly_label')}`)",
                f"- Top drivers: {meta.get('feature_importance') or 'explain_model not run'}",
            ],
        )
    ]


(
    record_business_context,
    acknowledge_identifier_column,
    acknowledge_gate,
    check_readiness,
    generate_report,
) = stages.register_tools(
    mcp,
    "anomaly",
    load_meta=_load_meta,
    save_meta=_save_meta,
    run_dir=_run_dir,
    required_gates=REQUIRED_GATES,
    supported_metrics=SUCCESS_METRICS,
    compute_readiness=compute_readiness,
    report_metrics=_current_metrics,
    report_sections=_report_sections,
)
