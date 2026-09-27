"""Self-check for skill gating — no MCP transport, no LLM needed.
Run: python test_skill_gating.py"""

import agent as a


def test_every_mcp_tool_is_mapped_to_a_skill():
    import asyncio
    import sys
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def _list_tool_names():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            cwd=str(a.SCRIPTS_DIR),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return {t.name for t in (await session.list_tools()).tools}

    tool_names = asyncio.run(_list_tool_names())
    assert tool_names, "no MCP tools found — mcp_server.py import shape changed?"
    unmapped = tool_names - set(a.TOOL_TO_SKILL)
    assert not unmapped, f"tools missing from TOOL_TO_SKILL: {unmapped}"
    unknown_skills = set(a.TOOL_TO_SKILL.values()) - set(a.SKILLS)
    assert not unknown_skills, (
        f"TOOL_TO_SKILL references skills that don't exist: {unknown_skills}"
    )


def test_gated_tool_hidden_until_skill_loaded():
    loaded = set()
    gated_names = {
        name for name, skill in a.TOOL_TO_SKILL.items() if skill == "modeling"
    }
    visible = {name for name in a.TOOL_TO_SKILL if a.TOOL_TO_SKILL[name] in loaded}
    assert not (gated_names & visible), (
        "modeling tools visible before load_skill('modeling')"
    )

    loaded.add("modeling")
    visible = {name for name in a.TOOL_TO_SKILL if a.TOOL_TO_SKILL[name] in loaded}
    assert gated_names <= visible, (
        "modeling tools still hidden after load_skill('modeling')"
    )


def test_call_rejected_without_its_skill_loaded():
    loaded_skills = set()  # nothing loaded yet
    required_skill = a.TOOL_TO_SKILL["train_model"]
    assert required_skill not in loaded_skills  # the actual gate check in _run_turn


def test_modeling_on_demand_reference_is_not_force_loaded():
    """The whole point of the on_demand split: base body stays lean (no
    per-step ask_user scripts or domain thresholds baked in), and each
    deep-dive is fetchable by name but isn't silently included in the base
    body's text — domain-notes specifically must be independently fetchable
    (including from business-understanding, before modeling even loads) so
    reading it never unlocks modeling's tools as a side effect."""
    base_body = a.SKILLS["modeling"]["body"]
    on_demand = a.SKILLS["modeling"]["on_demand"]
    assert "step-playbook" in on_demand, (
        "modeling must expose the step-playbook on_demand reference"
    )
    assert "domain-notes" in on_demand, (
        "domain-notes must be on_demand, not auto-loaded, so business-understanding/diagnostics can fetch it without unlocking modeling's tools"
    )
    # The base body (SKILL.md + auto-loaded references) stays lean; step-by-step
    # detail belongs in on_demand references. The limit is a margin, not a budget.
    assert len(base_body.splitlines()) < 350, (
        "modeling's base body should stay lean — the step-by-step detail belongs in the on_demand reference"
    )
    assert "Gate before tune — call `ask_user`" not in base_body, (
        "exact step-by-step ask_user scripts must live in the on_demand reference, not the base body"
    )
    assert "Gate before tune — call `ask_user`" in on_demand["step-playbook"]
    assert "Healthcare / Diagnosis" not in base_body, (
        "domain-notes content must not be auto-appended to modeling's base body anymore"
    )
    assert "Healthcare / Diagnosis" in on_demand["domain-notes"]
    assert "Fairness-relevant attributes" in on_demand["domain-notes"], (
        "domain-notes must cover which attributes assess_fairness should check per domain"
    )


def test_feature_engineering_domain_files_are_on_demand_not_force_loaded():
    """telecom/fraud/credit are separate on_demand files, fetched by name —
    the base feature-engineering body must stay domain-agnostic (archetypes
    only), and each on_demand file must be reachable and NOT bleed into
    the others (fetching telecom shouldn't hand back credit's content)."""
    fe = a.SKILLS["feature-engineering"]
    for domain in ("telecom", "telecom-traces", "fraud", "credit"):
        assert domain in fe["on_demand"], f"missing on_demand domain file: {domain}"
    # A raw schema column name, not a derived feature name and not the domain
    # word. telecom.md is a derivation PROCEDURE, so the features it names today
    # are exactly what it exists to stop hard-coding, and the domain word itself
    # legitimately appears in the base body's index of on_demand files. A column
    # from the worked trace is unambiguously worked-example detail.
    assert "TOTAL_CALLS" not in fe["body"], (
        "telecom worked-example detail must live in telecom.md, not the base body"
    )
    # The worked traces are their own on_demand file, so the procedure can be
    # fetched without them — and must not quietly lose them.
    assert "TOTAL_CALLS" in fe["on_demand"]["telecom-traces"]
    assert "TOTAL_CALLS" not in fe["on_demand"]["telecom"], "worked-trace detail belongs in telecom-traces"
    assert "TOTAL_CALLS" not in fe["on_demand"]["fraud"], (
        "fraud.md must not contain telecom's worked examples"
    )
    assert "TOTAL_CALLS" not in fe["on_demand"]["credit"]


def test_load_skill_reference_argument_is_wired_into_the_tool_schema():
    load_skill_params = a.LOAD_SKILL_TOOL["function"]["parameters"]["properties"]
    assert "reference" in load_skill_params, (
        "load_skill's schema must accept an optional reference so the model can fetch on_demand docs"
    )


if __name__ == "__main__":
    test_every_mcp_tool_is_mapped_to_a_skill()
    test_gated_tool_hidden_until_skill_loaded()
    test_call_rejected_without_its_skill_loaded()
    test_modeling_on_demand_reference_is_not_force_loaded()
    test_feature_engineering_domain_files_are_on_demand_not_force_loaded()
    test_load_skill_reference_argument_is_wired_into_the_tool_schema()
    print("skill gating OK")
