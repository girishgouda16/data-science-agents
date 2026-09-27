"""A2A server for the classification agent — senior-data-scientist agent for tabular classification across domains (fraud, spam, churn, diagnosis, credit risk).

The model/tool loop, human-in-the-loop pauses, durable (LangGraph-checkpointed)
conversation state and the A2A server itself live in core/agent_host.py; this
file is only what makes this agent itself.

Auth: callers send `Authorization: Bearer <CLASSIFICATION_AGENT_API_KEY>`.
Run: python agent.py  (serves on http://localhost:9000)
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
    "eda": "eda",
    "prepare_dataset": "eda",
    "check_imbalance": "eda",
    "detect_outliers": "eda",
    "inspect_target": "eda",
    "propose_group_column": "eda",
    "propose_time_column": "eda",
    "load_dataset": "eda",
    "list_data_sources": "eda",
    "detect_data_leakage": "diagnostics",
    "train_baseline": "diagnostics",
    "evaluate_model": "diagnostics",
    "error_analysis": "diagnostics",
    "analyze_segments": "diagnostics",
    "check_model_stability": "diagnostics",
    "assess_fairness": "diagnostics",
    "record_reflection": "diagnostics",
    "check_calibration": "diagnostics",
    "check_readiness": "diagnostics",
    "declare_fairness_not_applicable": "diagnostics",
    "check_label_rule": "diagnostics",
    "backtest_on_new_data": "diagnostics",
    "acknowledge_label_rule": "diagnostics",
    "propose_imputation": "data-cleaning",
    "apply_imputation": "data-cleaning",
    "propose_drop_columns": "data-cleaning",
    "apply_drop_columns": "data-cleaning",
    "propose_smote": "data-cleaning",
    "apply_smote": "data-cleaning",
    "propose_datetime_features": "data-cleaning",
    "apply_datetime_features": "data-cleaning",
    "apply_frequency_encoding": "data-cleaning",
    "acknowledge_identifier_column": "data-cleaning",
    "train_model": "modeling",
    "tune_hyperparams": "modeling",
    "tune_threshold": "modeling",
    "calibrate_model": "modeling",
    "run_standard_diagnostics": "modeling",
    "compare_models": "modeling",
    "compare_runs": "modeling",
    "explain_model": "modeling",
    "export_model": "modeling",
    "predict": "modeling",
    "declare_feature_engineering_not_applicable": "feature-engineering",
    "propose_features": "feature-engineering",
    "apply_features": "feature-engineering",
    "apply_custom_feature": "feature-engineering",
    "aggregate_events": "feature-engineering",
    "apply_entity_features": "feature-engineering",
    "apply_peer_features": "feature-engineering",
    "profile_features": "feature-engineering",
    "propose_feature_selection": "feature-selection",
    "generate_report": "reporting",
    "log_run_to_mlflow": "mlops",
    "dvc_track": "mlops",
    "generate_ci_workflow": "mlops",
    "list_model_versions": "mlops",
    "compare_model_versions": "mlops",
    "promote_model": "mlops",
    "demote_model": "mlops",
}

SPEC = Agent(
    name="classification",
    port=9000,
    home=HOME,
    card={
        "name": "Classification Agent",
        "description": "Senior-data-scientist agent for tabular classification across domains (fraud, spam, churn, diagnosis, credit risk). Confirms with the human before imputing, dropping data, or picking a final model.",
        "version": "0.2.0",
        "skill_id": "tabular_classification",
        "skill_name": "Tabular Classification",
        "skill_description": "Interactive EDA, imputation, outlier/imbalance handling, training, tuning, and explainability for any tabular classification problem — asks before any data-changing step.",
        "tags": ["classification", "ml", "data-science", "human-in-the-loop"],
        "examples": ["Classify fraud in transactions.csv", "Is churn.csv imbalanced?"],
    },
    tool_to_skill=TOOL_TO_SKILL,
    ask_user="Pause and ask the human a question before applying a data-changing step (imputation, dropping columns/rows, model choice), OR to resolve a business ambiguity that matters (target definition, prediction unit/horizon, feature meaning, success criteria) before proceeding. Always include a 'go with your recommendation' / 'proceed with a documented assumption' option.",
    load_skill="Load one phase's playbook (eda, data-cleaning, modeling, ...) before using its tools for the first time this conversation — its tools are unusable until this is called. Some skills also name situational deep-dive references in their body (e.g. modeling's full step-by-step script, only needed in step-by-step mode) — pass `reference` to fetch one of those by name when the skill's own text tells you to; omit it for the normal load.",
    max_rounds=30,
    footer=run_footer,
)

SKILLS = discover_skills(HOME / "skills")
_resolve_model = resolve_model
HOST = Host(SPEC)
LOAD_SKILL_TOOL = next(
    t for t in HOST.control_tools if t["function"]["name"] == "load_skill"
)
app = build_app(SPEC, HOST)

if __name__ == "__main__":
    serve(SPEC, app)
