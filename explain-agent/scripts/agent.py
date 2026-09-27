"""A2A server for the explain agent — explains why a model made a specific decision, and what drives it overall.

The model/tool loop, human-in-the-loop pauses, durable (LangGraph-checkpointed)
conversation state and the A2A server itself live in core/agent_host.py; this
file is only what makes this agent itself.

Auth: callers send `Authorization: Bearer <EXPLAIN_AGENT_API_KEY>`.
Run: python agent.py  (serves on http://localhost:9900)
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
    "inspect_explainability": "decision-explanation",
    "explain_prediction": "decision-explanation",
    "explain_counterfactual": "counterfactual",
    "explain_model": "model-explanation",
}

SPEC = Agent(
    name="explain",
    port=9900,
    home=HOME,
    card={
        "name": "Explain Agent",
        "description": "Explains why a model made a specific decision, and what drives it overall. Supervised models only (classification and regression); read-only, trains nothing.",
        "version": "0.2.0",
        "skill_id": "model_explanation",
        "skill_name": "Model & Decision Explanation",
        "skill_description": "Explains an exported classification/regression model: SHAP attribution for one specific decision (why was THIS row flagged, measured against the model's own tuned threshold), a counterfactual for what would have changed it, and a global driver ranking for a standalone .pkl.",
        "tags": ["explainability", "shap", "interpretability", "xai", "counterfactual"],
        "examples": [
            "Why was row 42 of flagged.csv marked as fraud by wangiri_model.pkl?",
            "What would have had to be different for this subscriber not to be flagged?",
            "What drives churn_model.pkl overall?",
        ],
    },
    tool_to_skill=TOOL_TO_SKILL,
    load_skill="Load one explanation playbook (decision-explanation, counterfactual, or model-explanation) before using its tools for the first time this conversation. Its tools are unusable until this is called.",
    max_rounds=6,
)

SKILLS = discover_skills(HOME / "skills")
_resolve_model = resolve_model
HOST = Host(SPEC)
app = build_app(SPEC, HOST)

if __name__ == "__main__":
    serve(SPEC, app)
