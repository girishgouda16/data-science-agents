"""A2A server for the visualization agent — renders tabular data as charts (matplotlib/seaborn) and saves them as PNG files.

The model/tool loop, human-in-the-loop pauses, durable (LangGraph-checkpointed)
conversation state and the A2A server itself live in core/agent_host.py; this
file is only what makes this agent itself.

Auth: callers send `Authorization: Bearer <VISUALIZATION_AGENT_API_KEY>`.
Run: python agent.py  (serves on http://localhost:9200)
"""

import os
import subprocess
import sys
from pathlib import Path

from starlette.staticfiles import StaticFiles

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

CHARTS_DIR = HOME / "charts"
CHARTS_DIR.mkdir(exist_ok=True)
PORT = int(os.environ.get("VISUALIZATION_AGENT_PORT", "9200"))

# Fail at startup, not on the first chat, if the tool package cannot import.
_check = subprocess.run(
    [sys.executable, "-c", "import mcp_server"],
    capture_output=True,
    cwd=str(SCRIPTS_DIR),
)
if _check.returncode != 0:
    raise RuntimeError(
        f"the mcp_server package failed to import:\n{_check.stderr.decode()}"
    )

# Which skill gates each MCP tool — a call is refused until its skill is
# loaded. Keep in sync with mcp_server/'s @mcp.tool() functions
# (test_skill_gating.py checks it).
TOOL_TO_SKILL = {
    "plot_histogram": "exploratory-plots",
    "plot_boxplot": "exploratory-plots",
    "plot_scatter": "exploratory-plots",
    "plot_correlation_heatmap": "exploratory-plots",
    "plot_class_distribution": "exploratory-plots",
    "plot_missingness": "exploratory-plots",
    "plot_feature_importance": "model-eval-plots",
    "plot_confusion_matrix": "model-eval-plots",
    "plot_roc_curve": "model-eval-plots",
    "plot_pr_curve": "model-eval-plots",
    "plot_shap_beeswarm": "model-eval-plots",
    "plot_calibration_curve": "model-eval-plots",
    "plot_regression_diagnostics": "model-eval-plots",
    "plot_forecast": "model-eval-plots",
    "plot_clusters": "model-eval-plots",
    "plot_anomaly_scores": "model-eval-plots",
}

SPEC = Agent(
    name="visualization",
    port=PORT,
    home=HOME,
    card={
        "name": "Visualization Agent",
        "description": "Renders tabular data as charts (matplotlib/seaborn) and saves them as PNG files.",
        "version": "0.2.0",
        "skill_id": "tabular_visualization",
        "skill_name": "Tabular Visualization",
        "skill_description": "Generates matplotlib/seaborn charts for tabular data — histograms, boxplots, scatter plots, correlation heatmaps, class-distribution and missingness charts, feature-importance bars.",
        "tags": ["visualization", "matplotlib", "seaborn", "data-science"],
        "examples": [
            "Plot a histogram of Age in titanic.csv",
            "Show the correlation heatmap for iris.csv",
        ],
    },
    tool_to_skill=TOOL_TO_SKILL,
    load_skill="Load one chart category's detailed playbook (exploratory-plots or model-eval-plots) before picking a chart from it for the first time this conversation. Its tools are unusable until this is called.",
    max_rounds=6,
    tool_env=lambda: {**runtime.mcp_env(), "VISUALIZATION_AGENT_PORT": str(PORT)},
    # <img> tags cannot send a bearer key, so the PNGs are public (unguessable names).
    public_paths=("/charts",),
    mounts={"/charts": StaticFiles(directory=CHARTS_DIR)},
)

SKILLS = discover_skills(HOME / "skills")
_resolve_model = resolve_model
HOST = Host(SPEC)
app = build_app(SPEC, HOST)

if __name__ == "__main__":
    serve(SPEC, app)
