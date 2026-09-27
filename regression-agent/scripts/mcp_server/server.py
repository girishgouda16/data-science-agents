"""Stdio entry point: python -m mcp_server.server (cwd=scripts/, which
agent.py sets). Importing the package registers every module's tools."""

from . import mcp

if __name__ == "__main__":
    mcp.run()
