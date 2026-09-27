"""A2A orchestrator agent: no MCP tools, no local ML code. Routes each request
to the specialist agents as an A2A client — loading a routing skill on demand
via `load_skill` (all delegate_to_* tools stay visible, since routing must see
them all to choose, but a delegate call is refused until its routing skill is
loaded) — and relays their HITL questions straight to whoever is talking to
the orchestrator, pausing in the same input-required state until they answer.

The loop, durable (LangGraph-checkpointed) conversation state — including
which remote task is paused on a question — and the A2A server itself live
in core/agent_host.py; this file is the routing: which agents exist, how a
delegation is made, and the guards around it.

Env:
  CLASSIFICATION_AGENT_URL      default http://localhost:9000
  CLASSIFICATION_AGENT_API_KEY  bearer token sent to that agent's own BearerAuthMiddleware (see its agent.py)
  VISUALIZATION_AGENT_URL       default http://localhost:9200
  VISUALIZATION_AGENT_API_KEY   ditto
  REGRESSION_AGENT_URL          default http://localhost:9300
  REGRESSION_AGENT_API_KEY      ditto
  CLUSTERING_AGENT_URL          default http://localhost:9400
  CLUSTERING_AGENT_API_KEY      ditto
  ANOMALY_AGENT_URL             default http://localhost:9500
  ANOMALY_AGENT_API_KEY         ditto
  FORECASTING_AGENT_URL         default http://localhost:9600
  FORECASTING_AGENT_API_KEY     ditto
  DRIFT_AGENT_URL               default http://localhost:9700
  DRIFT_AGENT_API_KEY           ditto
  SERVING_AGENT_URL             default http://localhost:9800
  EXPLAIN_AGENT_URL             default http://localhost:9900
  EXPLAIN_AGENT_API_KEY         ditto
  SERVING_AGENT_API_KEY         ditto
  ORCHESTRATOR_API_KEY          if unset, a one-time key is generated and logged at startup; callers must send Authorization: Bearer <key>
  REMOTE_AGENT_TIMEOUT          httpx read timeout in seconds (default 600)

Each *_AGENT_API_KEY must match the value that specialist agent generated/was
given at its own startup — every remote agent's BearerAuthMiddleware is
always-on, so a call with no token or the wrong one gets a 401. If a remote
agent auto-generated its key (env var left unset there), copy the key from
its startup log into this process's matching env var.

Run: python agent.py  (serves on http://localhost:9100)
"""

import asyncio
import copy
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for core.*
from core import runtime, runs, tracing  # noqa: E402
from core.agent_host import (
    Agent,
    Host,
    build_app,
    discover_skills,
    litellm,
    resolve_model,
    serve,
)  # noqa: E402,F401

from a2a.client import (
    A2ACardResolver,
    A2AClient,
    create_text_message_object,
)  # noqa: E402
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    Task,
    TaskState,
)  # noqa: E402

logger = logging.getLogger(__name__)
HOME = Path(__file__).resolve().parent
REMOTE_TIMEOUT = float(os.environ.get("REMOTE_AGENT_TIMEOUT", "3600"))
SKILLS = discover_skills(HOME / "skills")
RUNS_DIR = runs.data_dir() / "runs"


def _inspect_runs(run_id: str = "", limit: int = 5) -> str:
    """The orchestrator's own view of what has actually happened, read from
    the artifacts every ML agent writes (core/runs.py) — not from what a
    remote agent said three turns ago, and not from what this model remembers.
    Chaining needs a real run_id; asking the LLM to carry one across four
    delegations is how you get a plausible uuid that belongs to nothing."""
    return json.dumps(runs.list_runs(run_id, limit))


INSPECT_RUNS_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_runs",
        "description": (
            "Read the actual state of ML runs on disk: run_id, how far each got (prepared / trained / tuned / "
            "threshold_tuned / exported), its target, model and domain, plus any exported monitoring profiles. "
            "Call this before continuing work on an existing run, whenever you need a run_id to pass to another "
            "agent, and whenever the user refers to a previous run ('the model from earlier'). It reads artifacts, "
            "so it is correct even when conversation history is not. Requires no skill to be loaded."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "one run to inspect; empty lists the most recent runs",
                },
                "limit": {
                    "type": "integer",
                    "description": "how many recent runs to list (default 5)",
                },
            },
        },
    },
}


REMOTE_AGENTS = {
    "classification": {
        "url": os.environ.get("CLASSIFICATION_AGENT_URL", "http://localhost:9000"),
        "api_key": os.environ.get("CLASSIFICATION_AGENT_API_KEY", ""),
        "tool": "delegate_to_classification_agent",
        "description": "Send a message to the remote classification agent (senior-data-scientist agent for tabular classification: EDA, imputation, imbalance, training, tuning, explainability, model export) and get its reply.",
    },
    "visualization": {
        "url": os.environ.get("VISUALIZATION_AGENT_URL", "http://localhost:9200"),
        "api_key": os.environ.get("VISUALIZATION_AGENT_API_KEY", ""),
        "tool": "delegate_to_visualization_agent",
        "description": "Send a message to the remote visualization agent (renders matplotlib/seaborn charts — exploratory (histograms, boxplots, scatter, correlation heatmaps, class distribution, missingness) and model evaluation for any training agent's run_id (classification ROC/PR/confusion/calibration/SHAP, regression predicted-vs-actual and residuals, forecast vs actual, cluster map, anomaly-score distribution), plus feature importance) and get its reply.",
    },
    "regression": {
        "url": os.environ.get("REGRESSION_AGENT_URL", "http://localhost:9300"),
        "api_key": os.environ.get("REGRESSION_AGENT_API_KEY", ""),
        "tool": "delegate_to_regression_agent",
        "description": "Send a message to the remote regression agent (tabular regression: EDA, imputation, training/tuning a numeric-target model, explainability, model export) and get its reply.",
    },
    "clustering": {
        "url": os.environ.get("CLUSTERING_AGENT_URL", "http://localhost:9400"),
        "api_key": os.environ.get("CLUSTERING_AGENT_API_KEY", ""),
        "tool": "delegate_to_clustering_agent",
        "description": "Send a message to the remote clustering agent (unsupervised segmentation of tabular data with no target column: choosing k, fitting, cluster profiling, model export) and get its reply.",
    },
    "anomaly": {
        "url": os.environ.get("ANOMALY_AGENT_URL", "http://localhost:9500"),
        "api_key": os.environ.get("ANOMALY_AGENT_API_KEY", ""),
        "tool": "delegate_to_anomaly_agent",
        "description": "Send a message to the remote anomaly-detection agent (PyOD-based unsupervised outlier/anomaly detection on tabular data, optionally evaluated against a rare label column, model export) and get its reply.",
    },
    "forecasting": {
        "url": os.environ.get("FORECASTING_AGENT_URL", "http://localhost:9600"),
        "api_key": os.environ.get("FORECASTING_AGENT_API_KEY", ""),
        "tool": "delegate_to_forecasting_agent",
        "description": "Send a message to the remote forecasting agent (time-series forecasting: naive/exponential-smoothing/XGBoost models, walk-forward backtesting, multi-step forecasting for a time-indexed metric) and get its reply.",
    },
    "drift": {
        "url": os.environ.get("DRIFT_AGENT_URL", "http://localhost:9700"),
        "api_key": os.environ.get("DRIFT_AGENT_API_KEY", ""),
        "tool": "delegate_to_drift_agent",
        "description": "Send a message to the remote drift agent (compares a reference/baseline dataset against a current one, reports per-column and overall drift severity plus a retrain recommendation) and get its reply.",
    },
    "explain": {
        "url": os.environ.get("EXPLAIN_AGENT_URL", "http://localhost:9900"),
        "api_key": os.environ.get("EXPLAIN_AGENT_API_KEY", ""),
        "tool": "delegate_to_explain_agent",
        "description": "Send a message to the remote explain agent (explains WHY an exported classification/regression model made a specific decision about a specific row — SHAP attribution against the model's own tuned threshold, a counterfactual for what would have changed it, plus a global driver ranking for a standalone .pkl) and get its reply.",
    },
    "serving": {
        "url": os.environ.get("SERVING_AGENT_URL", "http://localhost:9800"),
        "api_key": os.environ.get("SERVING_AGENT_API_KEY", ""),
        "tool": "delegate_to_serving_agent",
        "description": "Send a message to the remote serving agent (loads a .pkl already exported by classification/regression/clustering/anomaly/forecasting-agent and serves predictions/forecasts from it on new data) and get its reply.",
    },
}
TOOL_NAME_TO_AGENT = {v["tool"]: k for k, v in REMOTE_AGENTS.items()}

TOOL_TO_SKILL = {
    "delegate_to_classification_agent": "classification-routing",
    "delegate_to_visualization_agent": "visualization-routing",
    "delegate_to_regression_agent": "regression-routing",
    "delegate_to_clustering_agent": "clustering-routing",
    "delegate_to_anomaly_agent": "anomaly-routing",
    "delegate_to_forecasting_agent": "forecasting-routing",
    "delegate_to_drift_agent": "drift-routing",
    "delegate_to_serving_agent": "serving-routing",
    "delegate_to_explain_agent": "explain-routing",
}

DELEGATE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": info["tool"],
            "description": info["description"],
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    }
    for info in REMOTE_AGENTS.values()
]

_agent_card_cache: dict[str, AgentCard] = {}
_agent_card_locks: dict[str, asyncio.Lock] = {
    name: asyncio.Lock() for name in REMOTE_AGENTS
}


async def _get_agent_card(agent_name: str, url: str, httpx_client: httpx.AsyncClient):
    if agent_name not in _agent_card_cache:
        async with _agent_card_locks[agent_name]:
            if agent_name not in _agent_card_cache:
                _agent_card_cache[agent_name] = await A2ACardResolver(
                    httpx_client, url
                ).get_agent_card()
    return _agent_card_cache[agent_name]


async def _call_remote_agent(agent_name: str, message: str, remotes: dict) -> str:
    """remotes: {agent_name: {"task_id": ..., "context_id": ...}}, mutated in place.

    task_id is cleared on ANY failure (timeout, connection error, JSON-RPC
    error) so the next call opens a fresh task instead of referencing a dead
    one in the remote's task store.  context_id is kept so the remote
    agent remembers prior conversation turns.
    """
    url = REMOTE_AGENTS[agent_name]["url"]
    api_key = REMOTE_AGENTS[agent_name]["api_key"]
    state = remotes.setdefault(agent_name, {"task_id": None, "context_id": None})

    # connect=10 s — fast-fail when the remote is simply down.
    # read=REMOTE_TIMEOUT — LLM inference on large data can take 5-10 min.
    timeout = httpx.Timeout(REMOTE_TIMEOUT, connect=10.0)
    # Every remote agent's own BearerAuthMiddleware is always-on (its API_KEY
    # is auto-generated if unset — see that agent's agent.py), so the
    # orchestrator must send one too, on both the card fetch and the
    # message call, or every remote hop 401s.
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as httpx_client:
        card = await _get_agent_card(agent_name, url, httpx_client)
        client = A2AClient(httpx_client, agent_card=card)

        msg = create_text_message_object(content=message)
        msg.task_id = state["task_id"]
        msg.context_id = state["context_id"]
        # Carries the trace down the A2A hop: the remote agent re-parents its
        # own spans under this delegation span, and inherits the session so
        # its turn lands in the same Langfuse thread as the conversation.
        # And the user rides along: the remote's tools stamp and check run
        # ownership with it (core/runtime.py).
        msg.metadata = {**tracing.outbound_metadata(), **runtime.outbound_metadata()}

        request = SendMessageRequest(
            id=str(uuid.uuid4()), params=MessageSendParams(message=msg)
        )

        # A transport failure (timeout, connection reset) doesn't mean the
        # remote task is dead — it may still be paused server-side awaiting
        # input. Keep task_id so the next call can resume it instead of
        # silently starting a new task and losing the paused HITL context.
        try:
            response = await client.send_message(request)
        except Exception as exc:
            logger.warning(
                "Remote '%s' call failed (%s: %s) — keeping task_id '%s' for retry",
                agent_name,
                type(exc).__name__,
                exc,
                state["task_id"],
            )
            raise

        # FIX 2: remote returned a JSON-RPC error — clear stale task_id
        # before raising so the next attempt doesn't reuse a rejected ID.
        if hasattr(response.root, "error"):
            err = response.root.error
            logger.warning(
                "Remote '%s' returned JSON-RPC error (code=%s): %s — clearing stale task_id '%s'",
                agent_name,
                err.code,
                err.message,
                state["task_id"],
            )
            state["task_id"] = None
            raise RuntimeError(
                f"Remote '{agent_name}' agent returned a JSON-RPC error "
                f"(code={err.code}): {err.message}"
            )

        result = response.root.result

        if isinstance(result, Task):
            state["task_id"] = result.id
            state["context_id"] = result.context_id
            status_message = result.status.message
            text = "".join(
                p.root.text
                for p in (status_message.parts if status_message else [])
                if p.root.kind == "text"
            )
            tracing.set_attributes(
                **{
                    "a2a.remote_task_id": result.id,
                    "a2a.remote_context_id": result.context_id,
                    "a2a.state": str(result.status.state),
                }
            )
            if result.status.state == TaskState.completed:
                state["task_id"] = None
            return text or f"[{agent_name} agent task state: {result.status.state}]"

        state["context_id"] = result.context_id
        return "".join(p.root.text for p in result.parts if p.root.kind == "text")


_VALID_TOOL_NAMES = set(TOOL_NAME_TO_AGENT) | {"load_skill", "inspect_runs"}


def _normalize_fallback_call(item) -> dict | None:
    """item is either our own {"name", "arguments"} shape or an OpenAI-style
    tool-call object ({"type": "function", "function": {"name", "arguments"}})
    — models that fall back to writing tool calls as text tend to mimic the
    latter, since that's the shape they were shown in their own prior turn."""
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("function"), dict):
        fn = item["function"]
        name, args = fn.get("name"), fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
    else:
        name = item.get("name")
        args = item.get("arguments", item.get("parameters", {}))
    if name not in _VALID_TOOL_NAMES:
        return None
    return {"name": name, "arguments": args if isinstance(args, dict) else {}}


def _recover_fallback_tool_calls(content: str | None) -> list[dict] | None:
    """Some models (e.g. gemma via Ollama) don't reliably use real
    function-calling and instead sometimes write the tool call as plain
    text — a bare JSON object, one fenced in ```json, or a whole
    "Tool Calls: [...]" array of OpenAI-style call objects — which would
    otherwise leak straight to the user as the "final answer". Scan for the
    first valid JSON value anywhere in the text (skipping any label/prose
    around it) and recover it as a real call instead.
    """
    if not content:
        return None
    decoder = json.JSONDecoder()
    obj = None
    for i, ch in enumerate(content):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(content, i)
        except json.JSONDecodeError:
            continue
        break
    if obj is None:
        return None
    if isinstance(obj, list):
        calls = [c for c in (_normalize_fallback_call(item) for item in obj) if c]
        return calls or None
    call = _normalize_fallback_call(obj)
    return [call] if call else None


def _paused_agent(remotes: dict) -> str | None:
    """The agent holding an open task, i.e. one that asked the user something
    and is waiting server-side for the answer. At most one matters."""
    for agent_name, state in remotes.items():
        if state.get("task_id"):
            return agent_name
    return None


async def _execute_tool_call(
    name: str, args: dict, loaded_skills: set[str], remotes: dict
) -> str:
    with tracing.tool_span(
        name, input=args, delegated_to=TOOL_NAME_TO_AGENT.get(name)
    ) as span:
        reply = await _execute_tool_call_inner(name, args, loaded_skills, remotes)
        tracing.set_output(reply, span)
        return reply


async def _execute_tool_call_inner(
    name: str, args: dict, loaded_skills: set[str], remotes: dict
) -> str:
    if name == "inspect_runs":
        # No skill gate: knowing what exists is never the wrong move, and
        # gating it would mean the model has to guess which agent owns a run
        # before it is allowed to find out.
        return _inspect_runs(args.get("run_id", "") or "", int(args.get("limit") or 5))

    if name == "load_skill":
        skill_name = args.get("skill")
        skill = SKILLS.get(skill_name)
        if skill:
            loaded_skills.add(skill_name)
            return skill["body"]
        tracing.set_error(None, message=f"unknown skill '{skill_name}'")
        return f"Unknown skill '{skill_name}'. Available: {', '.join(SKILLS)}"

    required_skill = TOOL_TO_SKILL.get(name)
    if required_skill and required_skill not in loaded_skills:
        tracing.set_error(
            None, message=f"skill gate: '{name}' needs '{required_skill}'"
        )
        return f"'{name}' requires the '{required_skill}' skill — call load_skill(\"{required_skill}\") first."

    agent_name = TOOL_NAME_TO_AGENT.get(name)
    if agent_name is None:
        tracing.set_error(None, message=f"unknown tool: {name}")
        return f"Unknown tool: {name}"

    # A remote that asked a question is paused server-side holding that
    # question's task. The user's answer has to go back to THAT agent — its
    # task_id/context_id are what resume the paused conversation. Sending it
    # somewhere else strands the paused task forever AND delivers an answer
    # to an agent that never asked anything, which reads as a non-sequitur
    # and usually makes it ask its own question. Nothing forced the model to
    # pick the right one; it was inference on a tool name.
    paused = _paused_agent(remotes)
    if paused and paused != agent_name:
        tracing.set_error(
            None,
            message=f"refused: {paused} is paused awaiting an answer, not {agent_name}",
        )
        return (
            f"REFUSED: the {paused} agent is paused waiting for an answer to a question it asked the user. "
            f"That answer must go back to it via {REMOTE_AGENTS[paused]['tool']} — it is the only agent holding "
            f"the task this conversation is blocked on. Delegating to {agent_name} instead would strand that task. "
            f"Call {REMOTE_AGENTS[paused]['tool']} with the user's reply."
        )

    # FIX 3: catch delegate failures and feed the error back to the LLM as
    # a tool result instead of crashing the request.
    try:
        return await _call_remote_agent(agent_name, args.get("message", ""), remotes)
    except Exception as exc:
        tracing.set_error(exc)
        return f"[{agent_name} agent error — {type(exc).__name__}: {exc}. You may retry or inform the user.]"


# Why this is not 5.
#
# One delegation costs at least two hops: load_skill, then the delegate call
# itself. A real pipeline — train, chart the result, report, export, then
# serve or start monitoring — is four or five delegations, so eight to ten
# hops before a final answer exists. At a budget of 5 the loop ran out
# mid-chain every time and returned "Reached max hops without a final
# answer", which is why traces showed single-delegation turns and no
# chaining: the orchestrator was not choosing to stop, it was being cut off.
#
# The budget still exists to bound a model that loops on itself. It is now
# sized to the longest legitimate pipeline plus room for one retry, and
# exhausting it reports what was actually accomplished instead of discarding
# it — work already delegated has already happened on the remote agents;
# pretending otherwise strands real state.
MAX_HOPS = int(os.environ.get("ORCHESTRATOR_MAX_HOPS", "16"))


def _tool_call_name(call) -> str:
    """Tool calls reach the history in two shapes: nested dicts (what
    litellm's message.model_dump() produces, and what the fallback path
    below constructs by hand) and provider objects with .function.name.
    Which one you get depends on the litellm version and on whether the
    turn came through real function-calling or the text-recovery path.

    This runs while reporting a failed turn, so it must not raise — an
    error here would replace a useful "here's what already ran" message
    with a traceback at exactly the wrong moment."""
    try:
        if isinstance(call, dict):
            fn = call.get("function") or {}
            return (
                fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
            ) or ""
        return getattr(getattr(call, "function", None), "name", "") or ""
    except Exception:
        return ""


def _work_done(messages: list[dict]) -> list[str]:
    """Delegations that actually executed this turn, newest last — so an
    exhausted budget can say what happened rather than nothing."""
    done = []
    for m in messages:
        for call in m.get("tool_calls") or []:
            name = _tool_call_name(call)
            if name in TOOL_NAME_TO_AGENT:
                done.append(TOOL_NAME_TO_AGENT[name])
    return done


async def _call_local(name: str, args: dict, state: dict) -> tuple[str, dict]:
    """The host's hook for this agent's own tools: remotes (which agent holds
    an open task) live in the checkpointed graph state, so a paused question
    survives a restart of this process."""
    remotes = copy.deepcopy(state.get("remotes") or {})
    reply = await _execute_tool_call(
        name, args, set(state.get("loaded_skills") or []), remotes
    )
    return reply, {"remotes": remotes}


def _budget_text(turn_messages: list[dict], hops: int) -> str:
    # Budget exhausted. The delegations that ran are not hypothetical — those
    # agents did the work and hold the state. Name them, so the user can pick
    # up from there instead of re-running a pipeline that already half-ran.
    done = _work_done(turn_messages)
    return (
        f"Stopped after {hops} steps without reaching a final answer. "
        + (
            f"Already delegated to: {', '.join(dict.fromkeys(done))} — that work has happened and those agents hold "
            "the resulting state; call inspect_runs to see where it got to. "
            if done
            else "No delegation completed. "
        )
        + "Tell me which step to continue from, or raise ORCHESTRATOR_MAX_HOPS if this pipeline genuinely needs "
        "more steps."
    )


SPEC = Agent(
    name="orchestrator",
    port=9100,
    home=HOME,
    card={
        "name": "Orchestrator Agent",
        "description": "Front door for classification, regression, clustering, anomaly-detection, forecasting, drift-detection, model-serving, explainability and visualization requests — delegates to specialist agents over A2A, no local tools of its own.",
        "version": "0.4.0",
        "skill_id": "ml_orchestrator",
        "skill_name": "ML Orchestrator",
        "skill_description": "Routes classification, regression, clustering, anomaly-detection, forecasting, drift-detection, model-serving, explainability and chart requests to the matching specialist agent, relaying human-in-the-loop questions.",
        "tags": ["orchestrator", "ml", "visualization", "a2a"],
        "examples": [
            "Classify fraud in transactions.csv",
            "Predict house prices in housing.csv",
            "Segment customers.csv into behavioral clusters",
            "Find anomalous transactions in transactions.csv",
            "Forecast next month's demand in sales.csv",
            "Plot the class distribution of Survived in titanic.csv",
            "Has customers_2024.csv drifted from customers_2023.csv?",
            "Predict on new_customers.csv using fraud_model.pkl",
        ],
    },
    tool_to_skill=TOOL_TO_SKILL,
    load_skill="Load one remote agent's detailed routing playbook (e.g. classification-routing, regression-routing, visualization-routing) before delegating to it for the first time this conversation. Delegating is rejected until this is called.",
    max_rounds=MAX_HOPS,
    rounds_env="ORCHESTRATOR_MAX_HOPS",
    mcp=False,
    hide_gated=False,
    local_tools=DELEGATE_TOOLS + [INSPECT_RUNS_TOOL],
    call_local=_call_local,
    recover_calls=_recover_fallback_tool_calls,
    awaiting=lambda state: _paused_agent(state.get("remotes") or {}) is not None,
    budget_text=_budget_text,
    turn_attributes=lambda state: {
        "agent.delegated_to": sorted(
            {
                a
                for a in (
                    TOOL_NAME_TO_AGENT.get(_tool_call_name(c))
                    for m in state.get("messages") or []
                    for c in (m.get("tool_calls") or [])
                )
                if a
            }
        )
    },
    api_key_env=("ORCHESTRATOR_API_KEY", "ORCHESTRATOR_AGENT_API_KEY"),
)

HOST = Host(SPEC)
SYSTEM_PROMPT = HOST.system_prompt
TOOLS = SPEC.local_tools + HOST.control_tools
_resolve_model = resolve_model
app = build_app(SPEC, HOST)

if __name__ == "__main__":
    serve(SPEC, app)
