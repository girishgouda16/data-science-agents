"""A2A server for the serving agent — serves predictions/forecasts from an already-exported model (.pkl) — no training, tuning, or export of its own.

The model/tool loop, human-in-the-loop pauses, durable (LangGraph-checkpointed)
conversation state and the A2A server itself live in core/agent_host.py; this
file is only what makes this agent itself.

Auth: callers send `Authorization: Bearer <SERVING_AGENT_API_KEY>`.
Run: python agent.py  (serves on http://localhost:9800)
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
    "list_exported_models": "serving",
    "inspect_model": "serving",
    "predict": "serving",
}

SPEC = Agent(
    name="serving",
    port=9800,
    home=HOME,
    card={
        "name": "Serving Agent",
        "description": "Serves predictions/forecasts from an already-exported model (.pkl) — no training, tuning, or export of its own. Handles both sklearn-Pipeline bundles and forecasting-agent's ForecastModel artifacts.",
        "version": "0.2.0",
        "skill_id": "model_serving",
        "skill_name": "Model Serving",
        "skill_description": "Loads a .pkl exported by classification/regression/clustering/anomaly/forecasting-agent and serves predictions/forecasts from it on new data — the consumption side of every other agent's export_model.",
        "tags": ["serving", "inference", "ml", "predict"],
        "examples": [
            "Predict on new_customers.csv using fraud_model.pkl",
            "Forecast the next 30 days with sales_model.pkl",
            "What kind of model is model.pkl?",
        ],
    },
    tool_to_skill=TOOL_TO_SKILL,
    load_skill="Load the 'serving' playbook before using its tools for the first time this conversation. Its tools are unusable until this is called.",
    max_rounds=6,
)

SKILLS = discover_skills(HOME / "skills")
_resolve_model = resolve_model
HOST = Host(SPEC)
app = build_app(SPEC, HOST)

if __name__ == "__main__":
    serve(SPEC, app)
