"""Self-check for skill gating — no MCP transport, no Ollama needed.
Run: python test_skill_gating.py"""

import agent as a


def test_every_mcp_tool_is_mapped_to_a_skill():
    import asyncio
    import sys
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def _list_tool_names():
        params = StdioServerParameters(
            command=sys.executable, args=a.MCP_SERVER_ARGS, cwd=str(a.SCRIPTS_DIR)
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return {t.name for t in (await session.list_tools()).tools}

    tool_names = asyncio.run(_list_tool_names())
    assert tool_names, "no MCP tools found — mcp_server/ import shape changed?"
    unmapped = tool_names - set(a.TOOL_TO_SKILL)
    assert not unmapped, f"tools missing from TOOL_TO_SKILL: {unmapped}"
    unknown_skills = set(a.TOOL_TO_SKILL.values()) - set(a.SKILLS)
    assert not unknown_skills, (
        f"TOOL_TO_SKILL references skills that don't exist: {unknown_skills}"
    )


def test_gated_tool_hidden_until_skill_loaded():
    loaded = set()
    gated_names = {
        name for name, skill in a.TOOL_TO_SKILL.items() if skill == "model-eval-plots"
    }
    visible = {name for name in a.TOOL_TO_SKILL if a.TOOL_TO_SKILL[name] in loaded}
    assert not (gated_names & visible), (
        "model-eval-plots tools visible before load_skill('model-eval-plots')"
    )

    loaded.add("model-eval-plots")
    visible = {name for name in a.TOOL_TO_SKILL if a.TOOL_TO_SKILL[name] in loaded}
    assert gated_names <= visible, (
        "model-eval-plots tools still hidden after load_skill('model-eval-plots')"
    )


if __name__ == "__main__":
    test_every_mcp_tool_is_mapped_to_a_skill()
    test_gated_tool_hidden_until_skill_loaded()
    print("skill gating OK")
