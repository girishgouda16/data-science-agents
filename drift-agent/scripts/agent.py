"""A2A server for the drift agent — senior-data-scientist agent for tabular dataset drift detection — compares a reference and current CSV, reports per-column severity and an overall safe/retrain verdict.

The model/tool loop, human-in-the-loop pauses, durable (LangGraph-checkpointed)
conversation state and the A2A server itself live in core/agent_host.py; this
file is only what makes this agent itself.

Auth: callers send `Authorization: Bearer <DRIFT_AGENT_API_KEY>`.
Run: python agent.py  (serves on http://localhost:9700)
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
    "eda": "eda",
    "detect_drift": "drift-detection",
    "compare_distributions": "drift-detection",
    "detect_multivariate_drift": "drift-detection",
    "drift_over_time": "drift-detection",
    "list_data_sources": "eda",
    "load_dataset": "eda",
}

SPEC = Agent(
    name="drift",
    port=9700,
    home=HOME,
    card={
        "name": "Drift Agent",
        "description": "Senior-data-scientist agent for tabular dataset drift detection — compares a reference and current CSV, reports per-column severity and an overall safe/retrain verdict. Read-only, no human-in-the-loop needed.",
        "version": "0.2.0",
        "skill_id": "tabular_drift_detection",
        "skill_name": "Tabular Drift Detection",
        "skill_description": "Compares a reference dataset against a current one and reports which columns have drifted, how severely (PSI + KS-test/chi-square), and an overall retrain recommendation.",
        "tags": ["drift-detection", "monitoring", "ml", "data-science"],
        "examples": [
            "Has customers_2024.csv drifted from customers_2023.csv?",
            "Compare training.csv and production.csv for drift",
        ],
    },
    tool_to_skill=TOOL_TO_SKILL,
    load_skill="Load one phase's detailed playbook (eda or drift-detection) before using its tools for the first time this conversation. Its tools are unusable until this is called.",
    max_rounds=6,
)

SKILLS = discover_skills(HOME / "skills")
_resolve_model = resolve_model
HOST = Host(SPEC)
app = build_app(SPEC, HOST)

if __name__ == "__main__":
    serve(SPEC, app)
