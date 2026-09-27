"""clustering agent — `diagnostics` tools: the business framing, the data
screen, and the deterministic readiness gate every senior segmentation
review asks for. The gate engine and the shared tools (business context,
acknowledgements, readiness, report) are core/gates.py and core/stages.py;
what "finished and safe to believe" MEANS for a clustering lives here."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import _load_meta, _load_split, _run_dir, _save_meta  # noqa: F401

from core import gates as gate_engine  # noqa: E402
from core import stages  # noqa: E402
from core.data_quality import identifier_columns  # noqa: E402

REQUIRED_GATES = (
    "business_context",
    "leakage_screened",
    "no_identifier_in_model",
    "structure_found",
    "holdout_stability",
    "explainability_run",
    "evidence_ordering",
    "success_criteria",
)
# Kaufman & Rousseeuw: silhouette <= 0.25 is "no substantial structure" — the
# segments are arbitrary cuts of a continuum, however clean the profiles look.
MIN_SILHOUETTE = 0.25
SUCCESS_METRICS = {
    "silhouette_score": "at_least",
    "davies_bouldin_score": "at_most",
    "calinski_harabasz_score": "at_least",
}


def measured_metric(meta: dict, metric: str):
    """A metric of the current model, from the HOLDOUT refit when available
    (the honest one), else the fit fold."""
    tm = meta.get("training_metrics") or {}
    for part in ("holdout_refit", "fit"):
        value = (tm.get(part) or {}).get(metric)
        if value is not None:
            return value
    return None


@mcp.tool()
def detect_data_leakage(run_id: str) -> str:
    """Screen the fit fold before clustering: identifier-shaped columns (IDs,
    ticket/order numbers — distance on them is meaningless and they split
    every row into its own cluster) and constant columns (no information).
    Call after prepare_dataset and cleaning, BEFORE train_model — and again
    after any column is dropped or imputed (the evidence_ordering gate)."""
    fit, _, meta = _load_split(run_id)
    signals = identifier_columns(fit, None)
    constant = [c for c in fit.columns if fit[c].nunique(dropna=True) <= 1]
    result = {
        "run_id": run_id,
        "identifier_columns": signals,
        "strong_identifier_columns": [
            c for c, s in signals.items() if s["strength"] == "strong"
        ],
        "constant_columns": constant,
        "clear": not signals and not constant,
    }
    meta["leakage"] = result
    _save_meta(run_id, meta)
    return json.dumps(result, default=str)


def compute_readiness(run_id: str) -> dict:
    """Every required gate's status for this run, computed from its artifacts."""
    fit, _, meta = _load_split(run_id)
    tm = meta.get("training_metrics") or {}
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
            "record_business_context was not called — nothing says what these segments are FOR",
        )
    )

    leakage = meta.get("leakage")
    checks["leakage_screened"] = (
        gate_engine.gate(
            gate_engine.PASS,
            "detect_data_leakage ran; "
            + (
                "nothing flagged"
                if leakage.get("clear")
                else f"flagged identifiers {leakage.get('strong_identifier_columns')}, constant {leakage.get('constant_columns')}"
            ),
        )
        if leakage
        else gate_engine.gate(
            gate_engine.NOT_RUN, "detect_data_leakage was never called for this run"
        )
    )

    acknowledged = set(meta.get("acknowledged_identifiers") or {})
    live = {
        c: s
        for c, s in identifier_columns(fit, None).items()
        if s["strength"] == "strong"
        and c not in (meta.get("dropped_columns") or [])
        and c not in (meta.get("profile_columns") or [])  # described by, never in, the distance
        and c not in acknowledged
    }
    checks["no_identifier_in_model"] = (
        gate_engine.gate(
            gate_engine.FAIL,
            "identifier-shaped columns still feed the distance metric: "
            + "; ".join(f"`{c}` ({s['reason']})" for c, s in live.items())
            + " — drop them (apply_drop_columns) and retrain",
        )
        if live
        else gate_engine.gate(
            gate_engine.PASS, "no identifier-shaped column feeds the model"
        )
    )

    silhouette = (tm.get("fit") or {}).get("silhouette_score")
    if not tm:
        checks["structure_found"] = gate_engine.gate(
            gate_engine.NOT_RUN, "no model has been trained for this run"
        )
    elif silhouette is None:
        checks["structure_found"] = gate_engine.gate(
            gate_engine.NOT_RUN,
            "silhouette not computable (fewer than 2 clusters, or everything labelled noise)",
        )
    else:
        checks["structure_found"] = gate_engine.gate(
            gate_engine.PASS if silhouette > MIN_SILHOUETTE else gate_engine.FAIL,
            f"silhouette {silhouette} vs {MIN_SILHOUETTE} (Kaufman & Rousseeuw: at or below it there is no substantial "
            "structure — the segments would be arbitrary cuts)",
        )

    stability = (tm.get("stability") or {}).get("stability_ari_mean")
    if not tm:
        checks["holdout_stability"] = gate_engine.gate(gate_engine.NOT_RUN, "no model has been trained for this run")
    elif stability is None:
        checks["holdout_stability"] = gate_engine.gate(
            gate_engine.NOT_RUN, "stability not measured — retrain (older run, or too few non-noise rows)")
    else:
        checks["holdout_stability"] = gate_engine.gate(
            gate_engine.PASS if stability >= STABLE_ARI else gate_engine.FAIL,
            f"stability ARI {stability} across {(tm.get('stability') or {}).get('resamples')} refits on 80% "
            f"subsamples (threshold {STABLE_ARI}) — the segments "
            f"{'come back' if stability >= STABLE_ARI else 'do NOT come back'} when the data is resampled"
            + (f"; holdout ARI {tm.get('holdout_ari')}" if tm.get("holdout_ari") is not None else ""),
        )

    checks["explainability_run"] = (
        gate_engine.gate(
            gate_engine.PASS, "explain_model profiled the current model's clusters"
        )
        if stages.ran_after_training(meta, "explain_model")
        else gate_engine.gate(
            gate_engine.NOT_RUN,
            "explain_model has not profiled the current model — nobody knows what the segments ARE",
        )
    )

    checks["evidence_ordering"] = gate_engine.check_screen_ordering(
        meta, "detect_data_leakage",
        ("apply_drop_columns", "apply_imputation", "set_profile_columns", "apply_custom_feature", "apply_peer_features"),
    )
    checks["success_criteria"] = gate_engine.evaluate_success_criteria(
        meta, measured_metric, tuple(SUCCESS_METRICS)
    )
    return gate_engine.readiness(
        run_id, stages.apply_acknowledgements(checks, meta), REQUIRED_GATES
    )


def _report_sections(meta: dict) -> list:
    tm = meta.get("training_metrics") or {}
    fit = tm.get("fit") or {}
    return [
        (
            "Segments",
            [
                f"- Clusters found: {fit.get('n_clusters_found')}",
                f"- Sizes: {fit.get('cluster_size_distribution')}",
                f"- Noise points: {fit.get('noise_points')}",
                f"- Stability (ARI, resampled refits): {(tm.get('stability') or {}).get('stability_ari_mean')}",
                f"- Holdout assignment counts: {tm.get('holdout_assignment_counts')}",
                f"- Logged before scaling (skewed): {tm.get('log_scaled_columns') or 'none'}",
                f"- Kept out of the distance, used to profile: {tm.get('profile_columns_excluded') or 'none'}",
                *[f"- Size warning: {w}" for w in tm.get("size_warnings") or []],
                *[f"- Model warning: {w}" for w in tm.get("model_warnings") or []],
                *[f"- Outcome `{col}`: segments explain eta^2 = {o.get('eta_squared')} of its variance"
                  for col, o in ((meta.get("explain") or {}).get("outcomes") or {}).items() if "eta_squared" in o],
                f"- Rules reproduce the segments with accuracy {(meta.get('explain') or {}).get('rules_accuracy')}",
                *([f"- Migration: {m['stay_rate']:.0%} of {m['transitions']} {m['entity_column']} period steps stay "
                   f"in their segment; largest moves {m['top_moves'][:3]}"] if (m := meta.get("migration")) else []),
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
    "clustering",
    load_meta=_load_meta,
    save_meta=_save_meta,
    run_dir=_run_dir,
    required_gates=REQUIRED_GATES,
    supported_metrics=SUCCESS_METRICS,
    compute_readiness=compute_readiness,
    report_metrics=lambda meta: meta.get("training_metrics") or {},
    report_sections=_report_sections,
)
