"""Stdio entry point. Run: python -m mcp_server.server (cwd=scripts/, which
agent.py's StdioServerParameters sets) — importing the package registers
core + feat_engineering + feat_selection + reporting's tools onto `mcp`."""

from . import mcp

if __name__ == "__main__":
    mcp.run()
