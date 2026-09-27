#!/bin/sh
# Starts one agent's MCP server for Claude Code (.mcp.json): sh scripts/mcp.sh classification
# Python: the repo's .venv (a symlink to a conda env works too), else python3 on PATH.
root="$(cd "$(dirname "$0")/.." && pwd)"
py="$root/.venv/bin/python"
[ -x "$py" ] || py=python3
cd "$root/$1-agent/scripts" && exec "$py" -m mcp_server.server
