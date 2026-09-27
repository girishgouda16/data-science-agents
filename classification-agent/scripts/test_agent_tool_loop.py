"""Proves the LangGraph tool loop (core/agent_host.py) keeps the contracts a
live model call can't verify here: ask_user/tool_call_id pairing, resume
after a question, on-demand skill references — and that a question asked
before a restart is still answerable after it. Mocks litellm.completion only;
the MCP server subprocess is the real thing. No live LLM call, no network.

Run: python -m pytest -q test_agent_tool_loop.py
"""

import asyncio
import json
import tempfile
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from litellm.types.utils import Message

import agent as a
from core import runtime
from core.agent_host import AUTOPILOT_ANSWER, Host


def _msg(content=None, tool_calls=None) -> Message:
    return Message(role="assistant", content=content, tool_calls=tool_calls)


def _tool_call(call_id: str, name: str, **kwargs) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(kwargs)},
    }


def _script(*responses):
    """litellm.completion replaced by a fixed script of replies."""
    queue = list(responses)

    def fake_completion(**kwargs):  # sync — the host runs it via asyncio.to_thread
        class R:
            choices = [type("C", (), {"message": queue.pop(0)})]

        return R()

    a.litellm.completion = fake_completion


def _messages(host: Host, thread: str) -> list[dict]:
    return asyncio.run(host.state(thread))["messages"]


def test_ask_user_alongside_other_calls_pauses_the_whole_batch():
    """ask_user + load_skill in the SAME assistant turn: load_skill must get
    a placeholder result (not actually run), the thread must pause on the
    question, and every tool_call must have a matching tool-role message
    once answered — the contract every provider enforces."""
    host = Host(a.SPEC, checkpointer=InMemorySaver())
    _script(
        _msg(
            tool_calls=[
                _tool_call("call_load", "load_skill", skill="eda"),
                _tool_call(
                    "call_ask", "ask_user", question="which column is the target?"
                ),
            ]
        ),
        _msg(content="ok"),
    )
    text, awaiting = asyncio.run(host.turn("t1", "build me a model"))
    assert awaiting is True and "which column is the target?" in text
    assert "eda" not in (asyncio.run(host.state("t1")).get("loaded_skills") or []), (
        "load_skill must NOT run when batched with ask_user"
    )

    asyncio.run(host.turn("t1", "churned"))
    tool_msgs = {
        m["tool_call_id"]: m["content"]
        for m in _messages(host, "t1")
        if m.get("role") == "tool"
    }
    assert set(tool_msgs) == {"call_load", "call_ask"}
    assert "skipped" in tool_msgs["call_load"] and tool_msgs["call_ask"] == "churned"


def test_resume_after_ask_user_then_reaches_a_final_answer():
    host = Host(a.SPEC, checkpointer=InMemorySaver())
    _script(
        _msg(
            tool_calls=[
                _tool_call("call_ask", "ask_user", question="binary or multiclass?")
            ]
        ),
        _msg(content="Got it, proceeding with binary."),
    )
    _, awaiting1 = asyncio.run(host.turn("t2", "build me a model"))
    assert awaiting1 is True
    text2, awaiting2 = asyncio.run(host.turn("t2", "binary"))
    assert awaiting2 is False and "binary" in text2


def test_autopilot_answers_the_question_instead_of_pausing():
    host = Host(a.SPEC, checkpointer=InMemorySaver())
    _script(
        _msg(
            tool_calls=[
                _tool_call("call_ask", "ask_user", question="split by ticket group?")
            ]
        ),
        _msg(content="Split by ticket group (my recommendation)."),
    )
    runtime.bind("dana", autopilot=True)
    try:
        text, awaiting = asyncio.run(host.turn("t-auto", "build me a model"))
    finally:
        runtime.bind(None)
    assert awaiting is False and "recommendation" in text
    answers = {
        m["tool_call_id"]: m["content"]
        for m in _messages(host, "t-auto")
        if m.get("role") == "tool"
    }
    assert answers["call_ask"] == AUTOPILOT_ANSWER


def test_load_skill_reference_fetches_on_demand_without_reloading_base_body():
    host = Host(a.SPEC, checkpointer=InMemorySaver())
    _script(
        _msg(tool_calls=[_tool_call("call_load", "load_skill", skill="modeling")]),
        _msg(
            tool_calls=[
                _tool_call(
                    "call_ref",
                    "load_skill",
                    skill="modeling",
                    reference="step-playbook",
                )
            ]
        ),
        _msg(content="Using the step-by-step script now."),
    )
    _, awaiting = asyncio.run(host.turn("t3", "go step by step"))
    assert awaiting is False
    fetched = {
        m["tool_call_id"]: m["content"]
        for m in _messages(host, "t3")
        if m.get("role") == "tool"
    }["call_ref"]
    assert "Gate before tune" in fetched, (
        "must return the step-playbook's actual content"
    )
    assert fetched != a.SKILLS["modeling"]["body"], (
        "must not just re-send the base body"
    )


def test_a_question_survives_a_restart_and_is_answered():
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db = Path(tempfile.mkdtemp()) / "threads.db"

    async def turn(text):
        async with aiosqlite.connect(db) as conn:  # each call = a new process lifetime
            return await Host(a.SPEC, checkpointer=AsyncSqliteSaver(conn)).turn(
                "t5", text
            )

    _script(
        _msg(
            tool_calls=[
                _tool_call("call_ask", "ask_user", question="drop the id column?")
            ]
        ),
        _msg(content="Dropped it."),
    )
    assert asyncio.run(turn("train on churn.csv")) == ("drop the id column?", True)
    assert asyncio.run(turn("yes")) == ("Dropped it.", False)


def test_a_hung_model_call_is_abandoned_and_retried():
    """A provider that holds the connection open (keep-alive whitespace)
    never trips the per-read timeout; the total deadline must cut it and a
    fresh call must answer."""
    import os
    import time

    host = Host(a.SPEC, checkpointer=InMemorySaver())
    calls = []

    def hang_then_answer(**kwargs):
        calls.append(time.time())
        if len(calls) == 1:
            time.sleep(3)  # the hung call
        return type("R", (), {"choices": [type("C", (), {"message": _msg(content="answered")})]})()

    a.litellm.completion = hang_then_answer
    os.environ["LLM_TOTAL_TIMEOUT"] = "0.5"
    try:
        assert asyncio.run(host.turn("t7", "hello")) == ("answered", False)
        assert len(calls) == 2 and calls[1] - calls[0] < 1.5, "the retry waited for the hung call"
    finally:
        os.environ.pop("LLM_TOTAL_TIMEOUT")


def test_a_turn_cut_off_mid_batch_does_not_poison_the_thread():
    """A crash or cancel between two tool calls of one batch leaves a call
    with no result; the next message must still produce a history every
    provider accepts, and say the call never ran."""
    host = Host(a.SPEC, checkpointer=InMemorySaver())
    real_load, seen = host._load_skill, []

    def crash_on_second(args, loaded):
        seen.append(args["skill"])
        if len(seen) == 2:
            raise RuntimeError("process died mid-batch")
        return real_load(args, loaded)

    host._load_skill = crash_on_second
    _script(
        _msg(
            tool_calls=[
                _tool_call("c1", "load_skill", skill="eda"),
                _tool_call("c2", "load_skill", skill="modeling"),
            ]
        )
    )
    try:
        asyncio.run(host.turn("t6", "start"))
    except Exception:  # arrives wrapped in the MCP session's ExceptionGroup
        pass
    host._load_skill = real_load
    _script(_msg(content="recovered"))
    assert asyncio.run(host.turn("t6", "continue")) == ("recovered", False)
    messages = _messages(host, "t6")
    results = {
        m["tool_call_id"]: m["content"] for m in messages if m.get("role") == "tool"
    }
    assert "interrupted" in results["c2"] and "interrupted" not in results["c1"], (
        results
    )
    for i, m in enumerate(messages):
        for call in m.get("tool_calls") or []:
            assert any(
                r.get("tool_call_id") == call["id"] for r in messages[i + 1 :]
            ), f"unanswered {call['id']}"
