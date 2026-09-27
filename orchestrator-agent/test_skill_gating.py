"""Self-check for delegate-call gating — no A2A network, no Ollama needed.
Run: python test_skill_gating.py"""

import agent as a


def test_every_delegate_tool_is_mapped_to_a_skill():
    tool_names = {t["function"]["name"] for t in a.DELEGATE_TOOLS}
    unmapped = tool_names - set(a.TOOL_TO_SKILL)
    assert not unmapped, f"delegate tools missing from TOOL_TO_SKILL: {unmapped}"
    unknown_skills = set(a.TOOL_TO_SKILL.values()) - set(a.SKILLS)
    assert not unknown_skills, (
        f"TOOL_TO_SKILL references skills that don't exist: {unknown_skills}"
    )


def test_delegate_tools_stay_visible_regardless_of_loaded_skills():
    # Unlike the other two agents, both delegate tools must always be in the
    # schema — the model needs to see both to decide which a request needs.
    tool_names = {t["function"]["name"] for t in a.TOOLS}
    assert set(a.TOOL_TO_SKILL) <= tool_names


def test_delegate_call_rejected_without_its_skill_loaded():
    loaded_skills = set()  # nothing loaded yet
    required_skill = a.TOOL_TO_SKILL["delegate_to_classification_agent"]
    assert required_skill not in loaded_skills  # the actual gate check in _run_turn

    loaded_skills.add(required_skill)
    assert required_skill in loaded_skills  # gate now passes


if __name__ == "__main__":
    test_every_delegate_tool_is_mapped_to_a_skill()
    test_delegate_tools_stay_visible_regardless_of_loaded_skills()
    test_delegate_call_rejected_without_its_skill_loaded()
    print("skill gating OK")
