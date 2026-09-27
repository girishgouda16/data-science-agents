"""forecasting agent — `mlops` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import _load_meta, _run_dir, _save_meta  # noqa: F401
from .diagnostics import compute_readiness  # noqa: F401
from .modeling import export_model  # noqa: F401

# ── MLOps: MLflow tracking, registry, champion export for serving ─────────────
# Shared with every non-classification agent (core/mlops.py): adds
# log_run_to_mlflow, list_model_versions, compare_model_versions,
# promote_model and demote_model to this server.
(
    log_run_to_mlflow,
    list_model_versions,
    compare_model_versions,
    promote_model,
    demote_model,
) = core_mlops.register_tools(
    mcp,
    "forecasting",
    load_meta=_load_meta,
    save_meta=_save_meta,
    run_dir=_run_dir,
    model_file="model.pkl",
    export_model=export_model,
    metrics_of=lambda meta: (
        meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
    ),
    readiness=lambda run_id: compute_readiness(run_id)["overall_status"],
)
