"""One LangGraph host for every agent: the model/tool loop, human-in-the-loop
pauses, durable conversation state and the A2A server around it.

Each agent.py declares what is its own — name, port, skills, which skill gates
which tool, its A2A card, its hand-off footer — and calls `serve(spec)`.
Everything below used to be copied into ten agent.py files.

The loop is a LangGraph StateGraph:

    START → agent ─┬→ END                  (answer)
                   ├→ ask ──(interrupt)──→ agent   (ask_user: pause the thread)
                   └→ tools ─┬→ tools      (one tool call per step)
                             ├→ agent
                             └→ END        (round budget spent)

Durability: a checkpoint is written after every step — each tool call is its
own step — to SQLite (<data>/a2a/<agent>-threads.db) or Postgres
(AGENT_CHECKPOINT_URL=postgresql://…). A process that dies mid-pipeline loses
at most the tool call in flight; the next message continues from the last
completed one. An ask_user pause is a LangGraph interrupt, so a question asked
before a restart is still answerable after it. thread_id = A2A context_id.

Tools run in a per-turn MCP subprocess (python -m mcp_server.server) whose
arguments core/toolguard checks; each call has a timeout (AGENT_TOOL_TIMEOUT,
default 1800 s). The model is called through litellm, so the provider stays a
config choice (LLM_PROVIDER + <AGENT>_AGENT_MODEL).
"""

import asyncio
import contextlib
import json
import logging
import operator
import os
import secrets
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from dotenv import load_dotenv

from . import runtime, toolguard, tracing

load_dotenv()  # <AGENT>_AGENT_MODEL + whichever provider key it implies

import litellm  # noqa: E402

litellm.success_callback = [
    "langfuse_otel"
]  # v2 "langfuse" callback breaks on langfuse SDK 4.x

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[1]

AUTOPILOT_ANSWER = (
    "Autopilot is on: the user will not answer questions in this chat. Decide yourself — take the option "
    "you recommend (with no recommendation, the most conservative one) — and continue without asking again. "
    "In your reply, list each decision you made this way and why. Never pass force=true past a failed "
    "readiness gate: report the failure instead."
)


# ── spec ─────────────────────────────────────────────────────────────────────
@dataclass
class Agent:
    """What one agent is. Everything else is the host's."""

    name: str  # "regression": env prefix, trace name, stores
    port: int
    home: Path  # folder holding SKILL.md and skills/
    card: dict  # name, description, skill_id, skill_name, skill_description, tags, examples
    tool_to_skill: dict[str, str]
    ask_user: str = ""  # ask_user tool description; "" = this agent never pauses
    load_skill: str = "Load one phase's playbook before using its tools for the first time this conversation. Its tools are unusable until this is called."
    max_rounds: int = 30
    footer: Callable[[str, list, int], str] | None = (
        None  # (text, messages, turn_start) -> text
    )
    mcp: bool = True  # False: no MCP tool server (the orchestrator)
    tool_env: Callable[[], dict] = runtime.mcp_env
    hide_gated: bool = (
        True  # False: gated tools stay visible (routing must see them all)
    )
    # Tools the agent implements itself (the orchestrator's delegations):
    # schemas, and one handler (name, args, state) -> (reply, state update).
    local_tools: list[dict] = field(default_factory=list)
    call_local: Callable[[str, dict, dict], Awaitable[tuple[str, dict]]] | None = None
    recover_calls: Callable[[str | None], list[dict] | None] | None = (
        None  # text-written tool calls
    )
    awaiting: Callable[[dict], bool] | None = None  # paused elsewhere (relayed HITL)
    budget_text: Callable[[list, int], str] | None = (
        None  # (this turn's messages, rounds)
    )
    turn_attributes: Callable[[dict], dict] | None = None
    public_paths: tuple[str, ...] = ()  # served without the bearer key (e.g. /charts)
    mounts: dict = field(
        default_factory=dict
    )  # path -> ASGI app (e.g. StaticFiles for /charts)
    api_key_env: tuple[str, ...] = ()  # default (<NAME>_AGENT_API_KEY,)
    model_env: str = ""  # default <NAME>_AGENT_MODEL
    rounds_env: str = ""  # default <NAME>_MAX_TOOL_ROUNDS

    @property
    def env(self) -> str:
        return self.name.upper()

    @property
    def scripts_dir(self) -> Path:
        return self.home / "scripts"


def resolve_model(bare_name: str) -> str:
    """LLM_PROVIDER picks the API every agent's model resolves against; each
    *_AGENT_MODEL only needs the bare model name. litellm routes on a
    "<provider>/<model>" prefix for every provider it supports."""
    provider = os.environ.get("LLM_PROVIDER", "ollama")
    return (
        bare_name if bare_name.startswith(f"{provider}/") else f"{provider}/{bare_name}"
    )


def require_api_key(*env_vars: str) -> str:
    """No agent runs without auth: an unset key becomes a one-time random one
    (logged), so a misconfigured deployment fails closed, not open."""
    for var in env_vars:
        if key := os.environ.get(var):
            return key
    key = secrets.token_urlsafe(32)
    logger.warning(
        "%s not set — generated a one-time key for this process: %s", env_vars[0], key
    )
    return key


def read_skill(path: Path) -> tuple[dict, str]:
    """A SKILL.md's frontmatter (name/description) and body (the playbook)."""
    text = path.read_text()
    if text.startswith("---\n"):
        _, frontmatter, body = text.split("---\n", 2)
        meta = dict(
            line.split(": ", 1)
            for line in frontmatter.strip().splitlines()
            if ": " in line
        )
        return meta, body.strip()
    return {}, text


def discover_skills(skills_dir: Path) -> dict[str, dict]:
    """skills/<name>/SKILL.md, loaded into context on demand via load_skill.
    references/*.md are appended to the body; references/on_demand/*.md are
    fetched only via load_skill(skill, reference=<name>)."""
    skills = {}
    for skill_dir in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
        if not (skill_dir / "SKILL.md").exists():
            continue
        meta, body = read_skill(skill_dir / "SKILL.md")
        for ref in sorted((skill_dir / "references").glob("*.md")):
            body += f"\n\n---\n### Reference: {ref.name}\n\n{ref.read_text().strip()}"
        on_demand = {
            ref.stem: ref.read_text().strip()
            for ref in sorted((skill_dir / "references" / "on_demand").glob("*.md"))
        }
        skills[meta["name"]] = {
            "description": meta.get("description", ""),
            "body": body,
            "on_demand": on_demand,
        }
    return skills


def last_tool_json_value(messages: list[dict], key: str):
    """Most recent tool result carrying `key` — read back deterministically
    instead of trusting the model's prose to repeat it."""
    for m in reversed(messages):
        if m.get("role") != "tool":
            continue
        try:
            data = json.loads(m["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and data.get(key) is not None:
            return data[key]
    return None


def run_footer(text: str, messages: list[dict], turn_start: int) -> str:
    """The run_id (the run's one handle) and, if explain_model ran THIS turn,
    its feature_importance — what another agent needs to act on the run,
    appended even when the model's summary left it out."""
    footer = []
    if run_id := last_tool_json_value(messages, "run_id"):
        footer.append(f"Run ID: {run_id}")
    if importance := last_tool_json_value(messages[turn_start:], "feature_importance"):
        footer.append(
            f"feature_importance (pass this exact JSON to plot it, not the run_id): {json.dumps(importance)}"
        )
    return f"{text}\n\n---\n" + "\n".join(footer) if footer else text


# ── graph ────────────────────────────────────────────────────────────────────
class State(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    loaded_skills: list[str]
    rounds: int
    remotes: dict  # the orchestrator's open A2A tasks per agent


@dataclass
class Turn:
    """Per-invocation context — never checkpointed (a session is a live pipe)."""

    session: Any = None
    tools: list = field(default_factory=list)


def _calls(message: dict) -> list[dict]:
    return message.get("tool_calls") or []


def _open_calls(messages: list[dict]) -> list[dict]:
    """Tool calls of the last assistant message that have no result yet."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            answered = {
                m.get("tool_call_id")
                for m in messages[i + 1 :]
                if m.get("role") == "tool"
            }
            return [c for c in _calls(messages[i]) if c["id"] not in answered]
    return []


def _args(call: dict) -> dict:
    try:
        args = json.loads(call["function"].get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return args if isinstance(args, dict) else {}


class Host:
    def __init__(self, spec: Agent, checkpointer=None):
        self.spec = spec
        # An agent with no model of its own uses the team default, then the
        # orchestrator's — never a hard-coded name the provider may not have.
        self.model = resolve_model(
            os.environ.get(spec.model_env or f"{spec.env}_AGENT_MODEL")
            or os.environ.get("AGENT_MODEL")
            or os.environ.get("ORCHESTRATOR_AGENT_MODEL")
            or "llama3.1"
        )
        self.max_rounds = int(
            os.environ.get(
                spec.rounds_env or f"{spec.env}_MAX_TOOL_ROUNDS", spec.max_rounds
            )
        )
        self.tool_timeout = float(os.environ.get("AGENT_TOOL_TIMEOUT", "1800"))
        _, base = read_skill(spec.home / "SKILL.md")
        self.skills = (
            discover_skills(spec.home / "skills")
            if (spec.home / "skills").exists()
            else {}
        )
        index = "\n".join(
            f"- `{n}`: {s['description']}" for n, s in self.skills.items()
        )
        self.system_prompt = (
            f"{base}\n\n## Available skills\n\n{index}" if index else base
        )
        self.control_tools = self._control_tools()
        self._checkpointer = checkpointer
        self._graph = None
        self._lock = asyncio.Lock()
        self._threads: dict[
            str, asyncio.Lock
        ] = {}  # one turn at a time per conversation

    # control tools the host implements for every agent
    def _control_tools(self) -> list[dict]:
        props = {"skill": {"type": "string", "enum": list(self.skills)}}
        if any(s["on_demand"] for s in self.skills.values()):
            props["reference"] = {
                "type": "string",
                "description": "Name of an on-demand reference this skill's body pointed you to, if any.",
            }
        tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "load_skill",
                        "description": self.spec.load_skill,
                        "parameters": {
                            "type": "object",
                            "properties": props,
                            "required": ["skill"],
                        },
                    },
                }
            ]
            if self.skills
            else []
        )
        if self.spec.ask_user:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "description": self.spec.ask_user,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["question"],
                        },
                    },
                }
            )
        return tools

    def visible_tools(self, mcp_tools, loaded: set[str]) -> list[dict]:
        gate = self.spec.tool_to_skill
        shown = [
            t
            for t in mcp_tools
            if t.name not in toolguard.DEV_ONLY_TOOLS
            and (
                not self.spec.hide_gated or t.name not in gate or gate[t.name] in loaded
            )
        ]
        return (
            [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    },
                }
                for t in shown
            ]
            + self.spec.local_tools
            + self.control_tools
        )

    # nodes
    async def _agent(self, state: State, runtime: Runtime[Turn]) -> dict:
        rounds = state.get("rounds", 0) + 1
        tracing.set_attributes(**{"agent.rounds": rounds})
        # The CURRENT prompt, not one frozen into the thread: a skill fix then
        # reaches conversations already in flight.
        messages = [{"role": "system", "content": self.system_prompt}] + [
            m for m in state["messages"] if m.get("role") != "system"
        ]
        call = dict(
            model=self.model,
            messages=messages,
            tools=self.visible_tools(
                runtime.context.tools, set(state.get("loaded_skills") or [])
            ),
            num_retries=int(os.environ.get("LLM_NUM_RETRIES", "3")),
            timeout=float(os.environ.get("LLM_TIMEOUT", "120")),
            metadata=tracing.llm_metadata(self.spec.name),
        )
        # LLM_TIMEOUT is per READ: a provider that keeps the connection alive
        # with whitespace (OpenRouter does, while it waits on an upstream)
        # never trips it — one call held a turn for 58 minutes. A total
        # deadline per attempt, one fresh retry, then the turn fails loudly.
        # ponytail: the abandoned thread finishes in the background; a
        # cancellable async client if that ever piles up.
        deadline = float(os.environ.get("LLM_TOTAL_TIMEOUT", "300"))
        for attempt in (1, 2):
            try:
                resp = await asyncio.wait_for(asyncio.to_thread(litellm.completion, **call), deadline)
                break
            except asyncio.TimeoutError:
                tracing.set_error(None, message=f"LLM call exceeded {deadline:.0f}s (attempt {attempt})")
                if attempt == 2:
                    raise TimeoutError(f"the model did not answer within {deadline:.0f}s, twice")
        msg = resp.choices[0].message
        # Plain JSON: the checkpointer stores it, and every provider's tool
        # calls come back to the same {id, type, function} shape. Other fields
        # stay — reasoning models (DeepSeek, …) need reasoning_content back.
        dumped = {
            k: v
            for k, v in msg.model_dump(exclude_none=True).items()
            if k != "tool_calls"
        }
        message = {
            **json.loads(json.dumps(dumped, default=str)),
            "role": "assistant",
            "content": msg.content,
        }
        if msg.tool_calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments or "{}",
                    },
                }
                for c in msg.tool_calls
            ]
        if (
            not _calls(message)
            and self.spec.recover_calls
            and (recovered := self.spec.recover_calls(msg.content))
        ):
            # A model that wrote its tool call as text: rebuild it as a real one
            # so the history stays valid and the call actually runs.
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"fallback-{uuid.uuid4()}",
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c["arguments"]),
                        },
                    }
                    for c in recovered
                ],
            }
        return {"messages": [message], "rounds": rounds}

    def _after_agent(self, state: State) -> str:
        calls = _calls(state["messages"][-1])
        if not calls:
            return END
        return (
            "ask"
            if any(c["function"]["name"] == "ask_user" for c in calls)
            else "tools"
        )

    def _ask(self, state: State) -> dict:
        """ask_user anywhere in a batch pauses the whole batch — every sibling
        call still needs a result before the next model turn."""
        calls = _open_calls(state["messages"])
        ask = next(c for c in calls if c["function"]["name"] == "ask_user")
        args = _args(ask)
        question = str(args.get("question") or "")
        if options := args.get("options"):
            question += "\nOptions: " + ", ".join(map(str, options))
        # the thread pauses here until the user replies — unless they chose autopilot
        answer = AUTOPILOT_ANSWER if runtime.autopilot() else interrupt(question)
        skipped = [
            {
                "role": "tool",
                "tool_call_id": c["id"],
                "content": f"skipped — the user was asked a clarifying question first: {question!r}",
            }
            for c in calls
            if c["id"] != ask["id"]
        ]
        return {
            "messages": skipped
            + [{"role": "tool", "tool_call_id": ask["id"], "content": str(answer)}]
        }

    async def _tools(self, state: State, runtime: Runtime[Turn]) -> dict:
        """One tool call per step, so each result is checkpointed on its own."""
        call = _open_calls(state["messages"])[0]
        name, args = call["function"]["name"], _args(call)
        loaded = set(state.get("loaded_skills") or [])
        update: dict = {}
        if name == "load_skill":
            reply = self._load_skill(args, loaded)
            update["loaded_skills"] = sorted(loaded)
        elif (skill := self.spec.tool_to_skill.get(name)) and skill not in loaded:
            with tracing.tool_span(name, input=args) as span:
                tracing.set_error(None, span, message=f"skill gate: needs '{skill}'")
            reply = f"'{name}' requires the '{skill}' skill — call load_skill(\"{skill}\") first."
        elif self.spec.call_local and any(
            t["function"]["name"] == name for t in self.spec.local_tools
        ):
            reply, update = await self.spec.call_local(name, args, dict(state))
        else:
            reply = await self._call_mcp(name, args, runtime.context)
        return {
            "messages": [
                {"role": "tool", "tool_call_id": call["id"], "content": reply}
            ],
            **update,
        }

    def _after_tools(self, state: State) -> str:
        if _open_calls(state["messages"]):
            return "tools"
        return END if state.get("rounds", 0) >= self.max_rounds else "agent"

    def _load_skill(self, args: dict, loaded: set[str]) -> str:
        name, reference = args.get("skill"), args.get("reference")
        skill = self.skills.get(name)
        if not skill:
            return f"Unknown skill '{name}'. Available: {', '.join(self.skills)}"
        if reference:
            return skill["on_demand"].get(reference) or (
                f"Unknown on_demand reference '{reference}' for skill '{name}'. "
                f"Available: {', '.join(skill['on_demand']) or 'none'}"
            )
        loaded.add(name)
        return skill["body"]

    async def _call_mcp(self, name: str, args: dict, turn: Turn) -> str:
        if name in toolguard.DEV_ONLY_TOOLS or not any(
            t.name == name for t in turn.tools
        ):
            return f"Unknown tool: {name}"
        with tracing.tool_span(name, input=args) as span:
            try:
                result = await asyncio.wait_for(
                    turn.session.call_tool(name, args), self.tool_timeout
                )
                text = "".join(getattr(c, "text", "") for c in result.content) or "{}"
            except asyncio.TimeoutError:
                text = json.dumps(
                    {
                        "error": f"{name} did not finish within {self.tool_timeout:.0f}s — stopped"
                    }
                )
                tracing.set_error(None, span, message="tool timeout")
            tracing.set_output(text, span)
        return text

    # graph + checkpointer
    async def graph(self):
        async with self._lock:
            if self._graph is None:
                g = StateGraph(State, context_schema=Turn)
                g.add_node("agent", self._agent)
                g.add_node("ask", self._ask)
                g.add_node("tools", self._tools)
                g.add_edge(START, "agent")
                g.add_conditional_edges(
                    "agent", self._after_agent, ["ask", "tools", END]
                )
                g.add_edge("ask", "agent")
                g.add_conditional_edges(
                    "tools", self._after_tools, ["tools", "agent", END]
                )
                self._graph = g.compile(
                    checkpointer=self._checkpointer or await self._open_checkpointer()
                )
        return self._graph

    async def _open_checkpointer(self):
        url = os.environ.get("AGENT_CHECKPOINT_URL", "")
        if url.startswith("postgres"):
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool

            pool = AsyncConnectionPool(
                url, kwargs={"autocommit": True, "prepare_threshold": 0}, open=False
            )
            await pool.open()
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
            return saver
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        folder = toolguard.data_dir() / "a2a"
        folder.mkdir(parents=True, exist_ok=True)
        return AsyncSqliteSaver(
            await aiosqlite.connect(folder / f"{self.spec.name}-threads.db")
        )

    @contextlib.asynccontextmanager
    async def _turn_context(self):
        if not self.spec.mcp:
            yield Turn()
            return
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            cwd=str(self.spec.scripts_dir),
            env=self.spec.tool_env(),
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield Turn(session=session, tools=(await session.list_tools()).tools)

    async def turn(self, thread_id: str, user_text: str) -> tuple[str, bool]:
        """Run one user message on a thread. Returns (reply, awaiting_input)."""
        async with self._threads.setdefault(thread_id, asyncio.Lock()):
            return await self._turn(thread_id, user_text)

    async def _turn(self, thread_id: str, user_text: str) -> tuple[str, bool]:
        graph = await self.graph()
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.max_rounds * 50,
        }
        snapshot = await graph.aget_state(config)
        messages = list((snapshot.values or {}).get("messages") or [])
        turn_start = len(messages)
        if snapshot.interrupts:
            payload = Command(resume=user_text, update={"rounds": 0})
        else:
            # A turn cut off mid-batch (crash, cancel) left calls with no
            # result; providers reject that history, so close them first.
            new = [
                {
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "content": "interrupted before this ran — not executed",
                }
                for c in _open_calls(messages)
            ]
            payload = {
                "messages": new + [{"role": "user", "content": user_text}],
                "rounds": 0,
            }
        async with self._turn_context() as context:
            await graph.ainvoke(payload, config, context=context)
        snapshot = await graph.aget_state(config)
        state = snapshot.values
        if snapshot.interrupts:
            return str(snapshot.interrupts[0].value), True
        messages = state["messages"]
        last = messages[-1]
        if last.get("role") == "assistant" and not _calls(last):
            text = last.get("content") or ""
        else:
            tracing.set_error(None, message="tool-call budget exhausted")
            text = (
                self.spec.budget_text(messages[turn_start:], self.max_rounds)
                if self.spec.budget_text
                else f"Paused after {self.max_rounds} tool rounds — every step so far is saved. Ran this turn: "
                f"{', '.join(c['function']['name'] for m in messages[turn_start:] for c in _calls(m)) or 'nothing'}. "
                "Reply 'continue' to pick up where it stopped."
            )
        if self.spec.footer:
            text = self.spec.footer(text, messages, turn_start)
        return text, bool(self.spec.awaiting and self.spec.awaiting(state))

    async def state(self, thread_id: str) -> dict:
        snapshot = await (await self.graph()).aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        return snapshot.values or {}


# ── A2A server ───────────────────────────────────────────────────────────────
def _root_cause(exc: BaseException) -> str:
    """The first real error inside anyio/MCP exception groups."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"[:300]


def build_app(spec: Agent, host: Host | None = None):
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import TaskUpdater
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentSkill,
        HTTPAuthSecurityScheme,
        TaskState,
    )
    from a2a.utils import new_agent_text_message
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import JSONResponse

    host = host or Host(spec)
    api_key = require_api_key(*(spec.api_key_env or (f"{spec.env}_AGENT_API_KEY",)))

    class Executor(AgentExecutor):
        async def execute(self, context: RequestContext, event_queue) -> None:
            updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            try:
                runtime.bind_user(
                    context.message
                )  # who is asking — verified, never taken from the model
            except runtime.Unauthenticated as exc:
                await updater.update_status(
                    TaskState.rejected,
                    message=new_agent_text_message(str(exc)),
                    final=True,
                )
                return
            user_text = context.get_user_input()
            # "working" first: a caller that sent non-blocking gets the task id
            # now and polls, instead of holding a connection open for an hour.
            await updater.start_work()
            session_id, traceparent = tracing.inbound(
                context.message, context.context_id
            )
            with tracing.turn_span(
                spec.name,
                session_id=session_id,
                traceparent=traceparent,
                input=user_text,
                **{"gen_ai.request.model": host.model},
            ) as span:
                try:
                    text, awaiting = await runtime.limited(
                        spec.name, host.turn(context.context_id, user_text)
                    )
                except (
                    Exception
                ) as exc:  # the thread keeps every completed step; say so, don't 500
                    logger.exception("%s turn failed", spec.name)
                    tracing.set_error(exc, span)
                    await updater.update_status(
                        TaskState.failed,
                        final=True,
                        message=new_agent_text_message(
                            f"The {spec.name} agent hit an error ({_root_cause(exc)}). Every step that finished is "
                            "saved — reply 'continue' to pick up from there."
                        ),
                    )
                    return
                state = await host.state(context.context_id)
                tracing.set_output(text, span)
                tracing.set_attributes(
                    span,
                    **{
                        "agent.awaiting_input": awaiting,
                        "agent.loaded_skills": state.get("loaded_skills") or [],
                        **(spec.turn_attributes(state) if spec.turn_attributes else {}),
                    },
                )
            if awaiting:
                await updater.update_status(
                    TaskState.input_required, message=new_agent_text_message(text)
                )
            else:
                await updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(text),
                    final=True,
                )

        async def cancel(self, context: RequestContext, event_queue) -> None:
            # The request handler cancels the running turn (its MCP process
            # dies with it); the thread keeps every step that completed.
            await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()

    class BearerAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if not request.url.path.startswith(
                spec.public_paths or ("\0",)
            ) and not secrets.compare_digest(
                request.headers.get("Authorization", ""), f"Bearer {api_key}"
            ):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    card = spec.card
    agent_card = AgentCard(
        name=card["name"],
        description=card["description"],
        url=os.environ.get(
            f"{spec.env}_AGENT_URL", f"http://localhost:{spec.port}"
        ).rstrip("/")
        + "/",
        version=card.get("version", "0.2.0"),
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id=card["skill_id"],
                name=card["skill_name"],
                description=card["skill_description"],
                tags=card.get("tags", []),
                examples=card.get("examples", []),
            )
        ],
        security_schemes={"bearer": HTTPAuthSecurityScheme(scheme="bearer")},
        security=[{"bearer": []}],
    )
    handler = DefaultRequestHandler(
        agent_executor=Executor(), task_store=runtime.task_store(spec.name)
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler).build()
    for path, mounted in spec.mounts.items():
        app.mount(path, mounted)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=(
            os.environ.get("AGENT_CORS_ORIGINS", "").split(",")
            if os.environ.get("AGENT_CORS_ORIGINS")
            else []
        ),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(BearerAuth)
    app.add_event_handler("shutdown", tracing.flush)
    app.state.host = host
    return app


def serve(spec: Agent, app=None):
    import uvicorn

    uvicorn.run(
        app or build_app(spec),
        # Local only by default; AGENT_BIND=0.0.0.0 when agents span hosts.
        host=os.environ.get("AGENT_BIND", "127.0.0.1"),
        port=spec.port,
    )
