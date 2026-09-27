"""A2A server for the anomaly agent — senior-data-scientist agent for tabular anomaly and outlier detection across domains (fraud, intrusion, telemetry faults, quality escapes).

The model/tool loop, human-in-the-loop pauses, durable (LangGraph-checkpointed)
conversation state and the A2A server itself live in core/agent_host.py; this
file is only what makes this agent itself.

Auth: callers send `Authorization: Bearer <ANOMALY_AGENT_API_KEY>`.
Run: python agent.py  (serves on http://localhost:9500)
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
    "record_business_context": "business-understanding",
    "detect_data_leakage": "diagnostics",
    "acknowledge_identifier_column": "diagnostics",
    "acknowledge_gate": "diagnostics",
    "check_readiness": "diagnostics",
    "generate_report": "reporting",
    "log_run_to_mlflow": "mlops",
    "list_model_versions": "mlops",
    "compare_model_versions": "mlops",
    "promote_model": "mlops",
    "demote_model": "mlops",
    "eda": "eda",
    "prepare_dataset": "eda",
    "list_data_sources": "eda",
    "load_dataset": "eda",
    "aggregate_events": "feature-engineering",
    "apply_custom_feature": "feature-engineering",
    "apply_entity_features": "feature-engineering",
    "apply_peer_features": "feature-engineering",
    "profile_features": "feature-engineering",
    "propose_imputation": "preprocessing",
    "apply_imputation": "preprocessing",
    "propose_drop_columns": "preprocessing",
    "apply_drop_columns": "preprocessing",
    "propose_contamination": "detection",
    "train_model": "detection",
    "compare_detectors": "detection",
    "explain_model": "detection",
    "export_model": "detection",
    "predict": "detection",
    "compare_runs": "detection",
}

SPEC = Agent(
    name="anomaly",
    port=9500,
    home=HOME,
    card={
        "name": "Anomaly Agent",
        "description": "Senior-data-scientist agent for tabular anomaly and outlier detection across domains (fraud, intrusion, telemetry faults, quality escapes). Confirms with the human before imputing, dropping data, or exporting a detector.",
        "version": "0.2.0",
        "skill_id": "tabular_anomaly_detection",
        "skill_name": "Tabular Anomaly Detection",
        "skill_description": "Interactive EDA, preprocessing, anomaly detection, explainability, and export for tabular outlier problems — asks before any data-changing step.",
        "tags": [
            "anomaly-detection",
            "outliers",
            "ml",
            "data-science",
            "human-in-the-loop",
        ],
        "examples": [
            "Find anomalous transactions in transactions.csv",
            "Detect outlier sensor readings in telemetry.csv",
        ],
    },
    tool_to_skill=TOOL_TO_SKILL,
    ask_user="Pause and ask the human a question before applying a data-changing step (imputation, dropping columns, exporting a final detector). Always include a 'go with your recommendation' option.",
    load_skill="Load one phase's detailed playbook (business-understanding, eda, feature-engineering, preprocessing, detection, diagnostics) before using its tools for the first time this conversation. Its tools are unusable until this is called.",
    max_rounds=30,
    footer=run_footer,
)

SKILLS = discover_skills(HOME / "skills")
_resolve_model = resolve_model
HOST = Host(SPEC)
app = build_app(SPEC, HOST)

if __name__ == "__main__":
    serve(SPEC, app)
