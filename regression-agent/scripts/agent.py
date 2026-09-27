"""A2A server for the regression agent — senior-data-scientist agent for tabular regression across domains (pricing, demand forecasting, risk scoring, delivery times).

The model/tool loop, human-in-the-loop pauses, durable (LangGraph-checkpointed)
conversation state and the A2A server itself live in core/agent_host.py; this
file is only what makes this agent itself.

Auth: callers send `Authorization: Bearer <REGRESSION_AGENT_API_KEY>`.
Run: python agent.py  (serves on http://localhost:9300)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for core.*
from core import runtime  # noqa: E402,F401
from core.agent_host import (
    Agent,
    Host,
    build_app,
    discover_skills,
    litellm,
    resolve_model,
    run_footer,
    serve,
)  # noqa: E402,F401

HOME = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = HOME / "scripts"
MCP_SERVER_ARGS = ["-m", "mcp_server.server"]

# Which skill gates each MCP tool — a call is refused until its skill is
# loaded. Keep in sync with mcp_server/'s @mcp.tool() functions
# (test_skill_gating.py checks it).
TOOL_TO_SKILL = {
    "generate_report": "reporting",
    "record_business_context": "business-understanding",
    "detect_data_leakage": "diagnostics",
    "acknowledge_identifier_column": "diagnostics",
    "train_baseline": "diagnostics",
    "check_residuals": "diagnostics",
    "check_model_stability": "diagnostics",
    "calibrate_intervals": "diagnostics",
    "analyze_errors": "diagnostics",
    "check_readiness": "diagnostics",
    "log_run_to_mlflow": "mlops",
    "list_model_versions": "mlops",
    "compare_model_versions": "mlops",
    "promote_model": "mlops",
    "demote_model": "mlops",
    "eda": "eda",
    "prepare_dataset": "eda",
    "detect_outliers": "eda",
    "propose_group_column": "eda",
    "propose_time_column": "eda",
    "list_data_sources": "eda",
    "load_dataset": "eda",
    "propose_imputation": "preprocessing",
    "apply_imputation": "preprocessing",
    "propose_drop_columns": "preprocessing",
    "apply_drop_columns": "preprocessing",
    "apply_target_transform": "preprocessing",
    "aggregate_events": "feature-engineering",
    "apply_entity_features": "feature-engineering",
    "apply_peer_features": "feature-engineering",
    "apply_custom_feature": "feature-engineering",
    "profile_features": "feature-engineering",
    "propose_features": "feature-engineering",
    "apply_features": "feature-engineering",
    "propose_datetime_features": "preprocessing",
    "apply_datetime_features": "preprocessing",
    "train_model": "regression",
    "set_objective": "regression",
    "tune_hyperparams": "regression",
    "compare_models": "regression",
    "compare_runs": "regression",
    "explain_model": "regression",
    "export_model": "regression",
    "predict": "regression",
}

SPEC = Agent(
    name="regression",
    port=9300,
    home=HOME,
    card={
        "name": "Regression Agent",
        "description": "Senior-data-scientist agent for tabular regression across domains (pricing, demand forecasting, risk scoring, delivery times). Confirms with the human before imputing, dropping data, or picking a final model.",
        "version": "0.2.0",
        "skill_id": "tabular_regression",
        "skill_name": "Tabular Regression",
        "skill_description": "Interactive EDA, imputation, outlier handling, training, tuning, and explainability for any tabular regression problem with a continuous target — asks before any data-changing step.",
        "tags": ["regression", "ml", "data-science", "human-in-the-loop"],
        "examples": [
            "Predict house prices in housing.csv",
            "What drives delivery_time in logistics.csv?",
        ],
    },
    tool_to_skill=TOOL_TO_SKILL,
    ask_user="Pause and ask the human a question before applying a data-changing step (imputation, dropping columns/rows, model choice). Always include a 'go with your recommendation' option.",
    load_skill="Load one phase's detailed playbook (eda, preprocessing, or regression) before using its tools for the first time this conversation. Its tools are unusable until this is called.",
    max_rounds=30,
    footer=run_footer,
)

SKILLS = discover_skills(HOME / "skills")
_resolve_model = resolve_model
HOST = Host(SPEC)
app = build_app(SPEC, HOST)

if __name__ == "__main__":
    serve(SPEC, app)
