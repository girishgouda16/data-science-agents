"""Self-check for the delegation/error-handling logic in agent.py — the
piece every user request actually flows through, and the one thing
test_skill_gating.py never touches (it checks the tool<->skill mapping, not
a live call through _run_turn/_call_remote_agent).

No live Ollama or remote agent needed: ollama.chat is monkeypatched (it's
invoked via asyncio.to_thread, so the replacement must be a plain sync
function, not a coroutine), and _call_remote_agent's A2AClient is replaced
with a fake whose send_message returns/raises whatever each test queues —
proves the three FIX-N behaviors already in agent.py (clear a stale task_id
on transport failure / on a JSON-RPC error / feed a delegate exception back
to the LLM instead of crashing the request) actually hold, not just that
the code exists.

Run: python test_delegation.py
"""

import os as _os
import tempfile as _tempfile

# Tests never write into the real data/ (runs, artifacts, predictions) that
# the orchestrator reports as "recent runs"; CI may point this elsewhere.
_os.environ.setdefault(
    "AGENTIC_ML_DATA_DIR", _tempfile.mkdtemp(prefix="agentic-ml-test-")
)
# ...nor register test models into the real MLflow registry (the UI on :5000).
_os.environ.setdefault(
    "MLFLOW_TRACKING_URI", "file:" + _tempfile.mkdtemp(prefix="agentic-ml-test-mlruns-")
)

import asyncio
import json
import os

os.environ.setdefault("ORCHESTRATOR_API_KEY", "test-key")

from types import SimpleNamespace

from a2a.types import Message, Role, Task, TaskState, TaskStatus, TextPart

import agent as a

# ── Fakes ────────────────────────────────────────────────────────────────
_NEXT_OUTCOME = [None]  # what the next _FakeA2AClient.send_message call returns/raises
_SENT = []  # every request _FakeA2AClient was handed


class _FakeA2AClient:
    """Stand-in for a2a.client.A2AClient — constructed the same way
    _call_remote_agent does (A2AClient(httpx_client, agent_card=card)), but
    send_message returns/raises whatever the test queued in _NEXT_OUTCOME,
    no network involved."""

    def __init__(self, *_args, **_kwargs):
        pass

    async def send_message(self, _request):
        _SENT.append(_request)
        outcome = _NEXT_OUTCOME[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def _fake_get_agent_card(_agent_name, _url, _httpx_client):
    return None  # unused — _FakeA2AClient ignores agent_card entirely


def _task_response(state, text, task_id="remote-t1", context_id="ctx-1"):
    """A REAL a2a.types.Task — _call_remote_agent does `isinstance(result,
    Task)`, so a duck-typed stand-in wouldn't take this branch."""
    status = TaskStatus(
        state=state,
        message=Message(
            role=Role.agent,
            parts=[TextPart(text=text)],
            message_id="m1",
            context_id=context_id,
            task_id=task_id,
        ),
    )
    task = Task(id=task_id, context_id=context_id, status=status)
    return SimpleNamespace(root=SimpleNamespace(result=task))


def _message_response(text, context_id="ctx-1"):
    """The non-Task branch (a plain message reply) — duck-typed is fine
    here since _call_remote_agent never isinstance-checks this shape."""
    result = SimpleNamespace(
        context_id=context_id,
        parts=[SimpleNamespace(root=SimpleNamespace(kind="text", text=text))],
    )
    return SimpleNamespace(root=SimpleNamespace(result=result))


def _error_response(code, message):
    return SimpleNamespace(
        root=SimpleNamespace(error=SimpleNamespace(code=code, message=message))
    )


class _FakeMsg(dict):
    """Stands in for litellm's message object, which supports attribute
    access (msg.tool_calls, msg.content) and .model_dump(). A dict subclass
    with __getattr__ covers both without pulling in the real types."""

    def model_dump(self, exclude_none=True):
        return dict(self)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None


class _FakeToolCall:
    """litellm hands tool calls back as objects with .id and
    .function.name/.arguments (arguments a JSON *string*), not as dicts."""

    def __init__(self, name, arguments, call_id="call-1"):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))


def _fake_completion_returning(tool_calls=None, content=None):
    """litellm.completion(...) -> resp.choices[0].message"""
    msg = _FakeMsg(role="assistant", content=content, tool_calls=tool_calls)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    return (
        lambda **_kwargs: resp
    )  # litellm.completion is called with keywords only (model, messages, tools, metadata)


# ── _call_remote_agent: FIX 1/2/3's actual effects on `remotes` state ─────


def test_call_remote_agent_keeps_task_id_on_transport_failure():
    """A transport failure is NOT evidence the remote task died.

    This assertion used to be the opposite, and the code changed under it
    deliberately. A timeout or reset most often means the remote is alive
    and still working — or paused server-side awaiting human input. Clearing
    task_id there silently abandons a paused HITL task and starts a fresh
    one, losing the question the user was about to answer. Keeping it lets
    the next call resume.

    The unrecoverable case stays covered: if the remote really is gone, its
    task store comes back empty on restart and the next send gets a JSON-RPC
    error, which DOES clear the id — see the test below."""
    a._get_agent_card, a.A2AClient = _fake_get_agent_card, _FakeA2AClient
    remotes = {"classification": {"task_id": "stale-task-123", "context_id": "ctx-1"}}
    _NEXT_OUTCOME[0] = ConnectionError("connection refused")
    try:
        asyncio.run(a._call_remote_agent("classification", "hello", remotes))
        assert False, "a transport failure must propagate, not be swallowed"
    except ConnectionError:
        pass
    assert remotes["classification"]["task_id"] == "stale-task-123", (
        "a transport failure must keep the task_id for resume"
    )


def test_call_remote_agent_clears_task_id_on_json_rpc_error():
    remotes = {"classification": {"task_id": "stale-task-456", "context_id": "ctx-1"}}
    _NEXT_OUTCOME[0] = _error_response(code=-32000, message="unknown tool")
    try:
        asyncio.run(a._call_remote_agent("classification", "hello", remotes))
        assert False, "a JSON-RPC error response must raise"
    except RuntimeError as e:
        assert "unknown tool" in str(e)
    assert remotes["classification"]["task_id"] is None, (
        "a JSON-RPC error must clear the stale task_id"
    )


def test_call_remote_agent_resets_task_id_when_remote_task_completes():
    remotes = {"classification": {"task_id": None, "context_id": None}}
    _NEXT_OUTCOME[0] = _task_response(TaskState.completed, "all done")
    reply = asyncio.run(a._call_remote_agent("classification", "hello", remotes))
    assert reply == "all done"
    assert (
        remotes["classification"]["task_id"] is None
    )  # completed -> reset, not held onto
    assert (
        remotes["classification"]["context_id"] == "ctx-1"
    )  # kept, so the next turn remembers this thread


def test_call_remote_agent_keeps_task_id_when_remote_awaits_input():
    remotes = {"classification": {"task_id": None, "context_id": None}}
    _NEXT_OUTCOME[0] = _task_response(TaskState.input_required, "which strategy?")
    reply = asyncio.run(a._call_remote_agent("classification", "hello", remotes))
    assert reply == "which strategy?"
    assert (
        remotes["classification"]["task_id"] == "remote-t1"
    )  # kept, so the next call continues this exact task


def test_call_remote_agent_handles_non_task_message_reply():
    remotes = {"classification": {"task_id": None, "context_id": None}}
    _NEXT_OUTCOME[0] = _message_response("plain reply")
    reply = asyncio.run(a._call_remote_agent("classification", "hello", remotes))
    assert reply == "plain reply"
    assert remotes["classification"]["context_id"] == "ctx-1"


# ── the host loop: the LLM-facing side of the same failures ────────────────


def _host():
    from langgraph.checkpoint.memory import InMemorySaver

    return a.Host(a.SPEC, checkpointer=InMemorySaver())


def _fake_completion_script(*messages):
    """Each call returns the next message; the last one repeats forever."""
    queue = list(messages)

    def completion(**_kwargs):
        msg = queue.pop(0) if len(queue) > 1 else queue[0]
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    return completion


def _tool_messages(host, thread):
    return [
        m["content"]
        for m in asyncio.run(host.state(thread))["messages"]
        if m.get("role") == "tool"
    ]


def test_run_turn_feeds_delegate_failure_back_to_llm_not_crash():
    async def _failing_call_remote_agent(_agent_name, _message, _remotes):
        raise ConnectionError("classification agent is down")

    orig_completion, orig_call = a.litellm.completion, a._call_remote_agent
    a.litellm.completion = _fake_completion_script(
        _FakeMsg(
            role="assistant",
            tool_calls=[
                _FakeToolCall("load_skill", {"skill": "classification-routing"}, "c0")
            ],
        ),
        _FakeMsg(
            role="assistant",
            tool_calls=[
                _FakeToolCall("delegate_to_classification_agent", {"message": "hi"})
            ],
        ),
    )
    a._call_remote_agent = _failing_call_remote_agent
    try:
        host = _host()
        reply, _ = asyncio.run(host.turn("d1", "classify fraud.csv"))
        failures = [c for c in _tool_messages(host, "d1") if "agent error" in c]
        assert failures and "ConnectionError" in failures[-1], (
            "the failure must reach the LLM as a tool result"
        )
        # Budget exhausts because the fake model never stops calling the tool;
        # the reply must still name what was attempted rather than nothing.
        assert "classification" in reply
    finally:
        a.litellm.completion, a._call_remote_agent = orig_completion, orig_call


def test_run_turn_gates_delegate_call_until_skill_loaded():
    """The gate must hold for EVERY hop, not just the first — a model that
    keeps calling a gated tool must keep being refused."""
    orig_completion = a.litellm.completion
    a.litellm.completion = _fake_completion_script(
        _FakeMsg(
            role="assistant",
            tool_calls=[
                _FakeToolCall("delegate_to_classification_agent", {"message": "hi"})
            ],
        )
    )
    try:
        host = _host()
        reply, _ = asyncio.run(host.turn("d2", "classify"))
        assert "Stopped after" in reply, "an exhausted hop budget must say so"
        refused = _tool_messages(host, "d2")
        assert refused and all(
            "requires the 'classification-routing' skill" in c for c in refused
        )
        assert len(refused) == a.MAX_HOPS
    finally:
        a.litellm.completion = orig_completion


def test_a_paused_specialist_survives_an_orchestrator_restart():
    """Which remote agent holds the open question lives in the checkpointed
    thread state — not process memory — so a restart can still route the
    user's answer back to it."""
    import aiosqlite
    import tempfile
    from pathlib import Path
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db = Path(tempfile.mkdtemp()) / "orchestrator.db"
    a._get_agent_card, a.A2AClient = _fake_get_agent_card, _FakeA2AClient

    async def turn(text):
        async with aiosqlite.connect(db) as conn:
            host = a.Host(a.SPEC, checkpointer=AsyncSqliteSaver(conn))
            return await host.turn("d3", text), await host.state("d3")

    orig_completion = a.litellm.completion
    a.litellm.completion = _fake_completion_script(
        _FakeMsg(
            role="assistant",
            tool_calls=[
                _FakeToolCall("load_skill", {"skill": "classification-routing"}, "c0")
            ],
        ),
        _FakeMsg(
            role="assistant",
            tool_calls=[
                _FakeToolCall(
                    "delegate_to_classification_agent", {"message": "train"}, "c1"
                )
            ],
        ),
        _FakeMsg(role="assistant", content="Which column is the target?"),
    )
    _NEXT_OUTCOME[0] = _task_response(
        TaskState.input_required, "Which column is the target?"
    )
    try:
        (reply, awaiting), state = asyncio.run(turn("train on churn.csv"))
        assert (
            awaiting is True
            and state["remotes"]["classification"]["task_id"] == "remote-t1"
        )
        a.litellm.completion = _fake_completion_script(
            _FakeMsg(role="assistant", content="noted")
        )
        (_, awaiting), state = asyncio.run(
            turn("churned")
        )  # a fresh process, same database
        assert awaiting is True, (
            "the open specialist task must still be known after the restart"
        )
    finally:
        a.litellm.completion = orig_completion


if __name__ == "__main__":
    test_call_remote_agent_clears_task_id_on_transport_failure()
    test_call_remote_agent_clears_task_id_on_json_rpc_error()
    test_call_remote_agent_resets_task_id_when_remote_task_completes()
    test_call_remote_agent_keeps_task_id_when_remote_awaits_input()
    test_call_remote_agent_handles_non_task_message_reply()
    test_run_turn_feeds_delegate_failure_back_to_llm_not_crash()
    test_run_turn_gates_delegate_call_until_skill_loaded()
    test_a_paused_specialist_survives_an_orchestrator_restart()
    print("delegation OK")


# ── HITL: the user's answer must reach the agent that asked ────────────────


def test_delegate_is_refused_while_another_agent_is_paused_on_a_question():
    """The failure: classification asks 'which column is the target?', the
    user answers, and the orchestrator delegates that answer to a different
    agent. The paused task is then stranded forever and the answer lands
    somewhere it makes no sense. Nothing forced the right choice — it was
    inference on a tool name."""
    remotes = {"classification": {"task_id": "paused-task-1", "context_id": "ctx-1"}}
    reply = asyncio.run(
        a._execute_tool_call(
            "delegate_to_visualization_agent",
            {"message": "the target is churn"},
            {"visualization-routing"},
            remotes,
        )
    )
    assert reply.startswith("REFUSED")
    assert "delegate_to_classification_agent" in reply, (
        "must name the tool that resumes the paused task"
    )
    assert remotes["classification"]["task_id"] == "paused-task-1", (
        "the paused task must be left intact"
    )


def test_delegate_to_the_paused_agent_itself_is_allowed_through():
    """The guard must not block the resume it exists to protect."""
    remotes = {"classification": {"task_id": "paused-task-1", "context_id": "ctx-1"}}
    calls = []

    async def _fake_call(agent_name, message, _remotes):
        calls.append((agent_name, message))
        return "resumed"

    orig = a._call_remote_agent
    a._call_remote_agent = _fake_call
    try:
        reply = asyncio.run(
            a._execute_tool_call(
                "delegate_to_classification_agent",
                {"message": "the target is churn"},
                {"classification-routing"},
                remotes,
            )
        )
    finally:
        a._call_remote_agent = orig
    assert reply == "resumed"
    assert calls == [("classification", "the target is churn")]


def test_no_paused_task_means_no_restriction():
    remotes = {"classification": {"task_id": None, "context_id": "ctx-1"}}
    calls = []

    async def _fake_call(agent_name, message, _remotes):
        calls.append(agent_name)
        return "ok"

    orig = a._call_remote_agent
    a._call_remote_agent = _fake_call
    try:
        reply = asyncio.run(
            a._execute_tool_call(
                "delegate_to_visualization_agent",
                {"message": "plot it"},
                {"visualization-routing"},
                remotes,
            )
        )
    finally:
        a._call_remote_agent = orig
    assert reply == "ok" and calls == ["visualization"]


# ── multi-user: identity forwarded, runs scoped to their owner ─────────────


def test_delegation_forwards_the_authenticated_user():
    """The specialist's tools stamp and check run ownership with this — a
    delegation that dropped it would make every run ownerless. What travels
    is the gateway-signed context, never a bare user_id anyone could write."""
    a._get_agent_card, a.A2AClient = _fake_get_agent_card, _FakeA2AClient
    _NEXT_OUTCOME[0] = _task_response(TaskState.completed, "ok")
    os.environ.setdefault("USER_CONTEXT_SECRET", "test-user-context-secret-32-bytes!!")
    token = a.runtime.sign_user_context("alice", project="billing")
    a.runtime.bind_user(SimpleNamespace(metadata={"user_context": token}))
    asyncio.run(
        a._call_remote_agent(
            "regression", "train", {"regression": {"task_id": None, "context_id": None}}
        )
    )
    sent = _SENT[-1].params.message.metadata
    assert sent.get("user_context") == token and "user_id" not in sent, sent
    a.runtime.bind_user(SimpleNamespace(metadata={}))


def test_a_forged_user_is_refused():
    os.environ.setdefault("USER_CONTEXT_SECRET", "test-user-context-secret-32-bytes!!")
    import jwt

    forged = jwt.encode(
        {"sub": "admin", "aud": "agentic-ml-agents", "exp": 9999999999},
        "not-the-secret",
    )
    for metadata in ({"user_context": forged}, {"user_context": "garbage"}):
        try:
            a.runtime.bind_user(SimpleNamespace(metadata=metadata))
            raise AssertionError(f"accepted {metadata}")
        except a.runtime.Unauthenticated:
            pass
    assert (
        a.runtime.bind_user(SimpleNamespace(metadata={"user_id": "admin"})) is None
    ), "a bare user_id is not identity"
    os.environ["AGENTIC_ML_REQUIRE_USER"] = "1"
    try:
        a.runtime.bind_user(SimpleNamespace(metadata={}))
        raise AssertionError("production must refuse a message with no user context")
    except a.runtime.Unauthenticated:
        pass
    finally:
        del os.environ["AGENTIC_ML_REQUIRE_USER"]


def test_inspect_runs_shows_only_the_asking_users_runs():
    """With many data scientists on one box, 'the newest run' is otherwise a
    colleague's — and routing would continue it."""
    for owner in ("alice", "bob"):
        run = a.RUNS_DIR / f"00000000-0000-4000-8000-00000000{owner[:4].encode().hex()}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "meta.json").write_text(json.dumps({"owner": owner, "target": "y"}))
    os.environ["AGENTIC_ML_USER"] = "alice"
    try:
        owners = {r["owner"] for r in json.loads(a._inspect_runs(limit=50))["runs"]}
        assert owners == {"alice"}, owners
        os.environ["AGENTIC_ML_ADMINS"] = "alice"
        owners = {r["owner"] for r in json.loads(a._inspect_runs(limit=50))["runs"]}
        assert {"alice", "bob"} <= owners, owners
    finally:
        os.environ.pop("AGENTIC_ML_USER"), os.environ.pop("AGENTIC_ML_ADMINS", None)
