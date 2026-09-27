"""Every agent's MCP server must SERVE every tool its source defines.

Unit tests import mcp_server as a module, which defines every tool. Production
runs it as a script: `mcp.run()` blocks, so any tool defined BELOW the
`if __name__ == "__main__":` line is never registered. That shipped: the
regression server served 16 of its 24 tools — its whole readiness layer — and
export_model hit a NameError on compute_readiness. No import-based test can
see it; this starts each server exactly the way its agent.py does and lists
what it actually offers.

Run:  python -m pytest -q scripts/test_mcp_servers_serve_every_tool.py
"""

import ast
import asyncio
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
AGENTS = [
    "classification",
    "regression",
    "clustering",
    "anomaly",
    "forecasting",
    "visualization",
    "serving",
    "drift",
    "explain",
]


def _defined_tools(agent: str) -> set[str]:
    """Function names decorated with @mcp.tool() in the agent's source, plus
    tools registered through core.mlops.register_tools."""
    scripts = ROOT / f"{agent}-agent" / "scripts"
    files = sorted((scripts / "mcp_server").glob("*.py"))
    names = set()
    for f in files:
        source = f.read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and any(
                "mcp.tool" in ast.unparse(d) for d in node.decorator_list
            ):
                names.add(node.name)
        if "core_mlops.register_tools(" in source:
            names |= {
                "log_run_to_mlflow",
                "list_model_versions",
                "compare_model_versions",
                "promote_model",
                "demote_model",
            }
    return names


async def _served_tools(agent: str, tmp: Path) -> set[str]:
    scripts = ROOT / f"{agent}-agent" / "scripts"
    args = [
        "-m",
        "mcp_server.server",
    ]  # every agent: scripts/mcp_server/ package, started as agent.py does
    env = {**os.environ, "AGENTIC_ML_DATA_DIR": str(tmp)}
    async with stdio_client(
        StdioServerParameters(
            command=sys.executable, args=args, cwd=str(scripts), env=env
        )
    ) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            return {t.name for t in (await session.list_tools()).tools}


@pytest.mark.parametrize("agent", AGENTS)
def test_server_serves_every_defined_tool(agent, tmp_path):
    defined = _defined_tools(agent)
    served = asyncio.run(_served_tools(agent, tmp_path))
    assert defined, (
        f"found no @mcp.tool definitions for {agent} — source layout changed?"
    )
    missing = defined - served
    assert not missing, (
        f"{agent}'s server does not serve {sorted(missing)} — defined below mcp.run()?"
    )
