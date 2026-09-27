"""One Langfuse trace per turn, one Langfuse session per conversation.

Three things have to line up for the agent swarm to show up as a tree
instead of 200 loose `litellm_request` rows:

  * a parent span per turn — created here; litellm's `langfuse_otel`
    callback nests its generation under whatever span it finds in
    `metadata["litellm_parent_otel_span"]` (see `llm_metadata`),
  * `session.id` on that span — Langfuse groups traces carrying the same
    one into a single thread, which is the per-conversation view,
  * a `traceparent` across the A2A hop — the orchestrator sends one in the
    A2A message metadata, the remote agent passes it back in here, and the
    remote's spans land inside the orchestrator's trace instead of starting
    their own.

Everything else here exists so a trace answers optimization questions on its
own: what went in and out of every step (`input=`/`set_output`), which step
failed (`set_error` — the orchestrator swallows delegate failures into tool
results, so nothing else would mark the span), which build produced it
(environment + release), and which MLflow run a turn created (`ml.run_id`,
sniffed out of tool output).

langfuse.get_client() is what installs the global OTel TracerProvider that
exports to Langfuse; without it every span here is a no-op, which is also
exactly what we want when Langfuse isn't configured.
"""

import contextlib
import contextvars
import functools
import json
import logging
import os
import re
import subprocess

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from . import runtime

logger = logging.getLogger(__name__)

_PROPAGATOR = TraceContextTextMapPropagator()
_tracer = None
# Set by turn_span so llm_metadata() doesn't have to be threaded through
# every agent's turn signature (core/agent_host.py).
_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "langfuse_session_id", default=None
)

# Langfuse stores span attributes verbatim; a full training-tool payload can
# be megabytes. Long enough to read a report, short enough not to become the
# reason the exporter backs up.
MAX_VALUE_CHARS = 10_000

_RUN_ID_RE = re.compile(r'"run_id"\s*:\s*"([^"]+)"')
SCOPE = "agentic-ml"


def should_export(span) -> bool:
    """Langfuse 4's default filter keeps only its own spans, spans with gen_ai.*
    attributes and known LLM libraries — it silently dropped every tool call and
    every delegate hop (the spans the conversation is actually made of), which
    left each specialist's turn hanging off a parent Langfuse never received."""
    from langfuse.span_filter import is_default_export_span

    scope = getattr(span, "instrumentation_scope", None)
    return (scope is not None and scope.name == SCOPE) or is_default_export_span(span)


@functools.cache
def release() -> str:
    """Git SHA of the running code, so a latency/quality change is
    attributable to a commit rather than to "sometime last week". Resolved on
    first use, not at import: every agent imports this module before its own
    load_dotenv() runs, so at import time the .env values aren't there yet."""
    if os.getenv("LANGFUSE_RELEASE"):
        return os.environ["LANGFUSE_RELEASE"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


@functools.cache
def environment() -> str:
    return os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development")


def _get_tracer():
    global _tracer
    if _tracer is None:
        try:
            from langfuse import Langfuse

            Langfuse(
                should_export_span=should_export
            )  # installs the global TracerProvider exporting to Langfuse
        except (
            Exception
        ) as exc:  # no keys, bad host, langfuse missing — tracing is never fatal
            logger.warning(
                "Langfuse tracing unavailable (%s: %s) — spans will be no-ops",
                type(exc).__name__,
                exc,
            )
        _tracer = trace.get_tracer(SCOPE)
    return _tracer


def _stringify(value) -> str:
    if not isinstance(value, str):
        try:
            value = json.dumps(value, default=str)
        except Exception:
            value = str(value)
    if len(value) > MAX_VALUE_CHARS:
        return value[:MAX_VALUE_CHARS] + f"… [truncated, {len(value)} chars total]"
    return value


def _set(span, key: str, value) -> None:
    if span is not None and value is not None and span.is_recording():
        span.set_attribute(
            key, value if isinstance(value, (bool, int, float)) else _stringify(value)
        )


@contextlib.contextmanager
def turn_span(
    agent: str,
    session_id: str | None,
    traceparent: str | None = None,
    name: str | None = None,
    input=None,
    **attributes,
):
    """The per-turn root span. With a traceparent it is instead a child of
    the caller's span, so a delegated turn joins the orchestrator's trace."""
    parent_ctx = (
        _PROPAGATOR.extract({"traceparent": traceparent}) if traceparent else None
    )
    _session_id.set(session_id)
    # No caller trace = a new trace, deliberately. context=None would adopt
    # whatever span is ambient — the A2A SDK wraps its request handling in
    # spans of its own (not exported), and every turn after a chat's first
    # became a child of one: a trace with no root in Langfuse.
    start_in = parent_ctx if parent_ctx is not None else otel_context.Context()
    with _get_tracer().start_as_current_span(
        name or f"{agent} turn", context=start_in
    ) as span:
        _set(span, "langfuse.observation.type", "agent")
        _set(span, "langfuse.environment", environment())
        _set(span, "langfuse.release", release())
        _set(span, "session.id", session_id)
        # Who asked — the gateway-authenticated user (core/runtime.py). Langfuse
        # files the trace under this user, so each data scientist's work is one filter away.
        _set(span, "user.id", runtime.user())
        # langfuse.trace.* are TRACE-level in Langfuse's OTel mapping, not
        # span-level: whichever span sets them last wins for the whole trace.
        # A delegated turn is a child of the orchestrator's trace, so setting
        # them here renamed/retagged the orchestrator's trace after the fact —
        # the tree was nested correctly but showed up labelled "classification".
        # Only the span that actually roots the trace may set them.
        if parent_ctx is None:
            _set(span, "langfuse.trace.name", agent)
            _set(span, "langfuse.trace.tags", json.dumps([agent, environment()]))
            # The user's message as the TRACE's input — a session view lists
            # traces by input/output, so a session reads like the chat itself.
            _set(span, "langfuse.trace.input", input)
        _set(span, "agent.name", agent)
        _set(span, "langfuse.observation.input", input)
        for key, value in attributes.items():
            _set(span, key, value)
        yield span


@contextlib.contextmanager
def tool_span(name: str, input=None, observation_type: str = "tool", **attributes):
    with _get_tracer().start_as_current_span(name) as span:
        _set(span, "langfuse.observation.type", observation_type)
        _set(span, "langfuse.environment", environment())
        _set(span, "tool.name", name)
        _set(span, "langfuse.observation.input", input)
        for key, value in attributes.items():
            _set(span, key, value)
        yield span


def set_output(value, span=None) -> None:
    """Records a step's result, and lifts any run_id out of it — that id is
    the join between a trace and the MLflow run / artifacts it produced."""
    span = span or trace.get_current_span()
    _set(span, "langfuse.observation.output", value)
    if (
        getattr(span, "parent", "") is None
    ):  # the trace's root: its output is the reply the user read
        _set(span, "langfuse.trace.output", value)
    match = _RUN_ID_RE.search(value if isinstance(value, str) else _stringify(value))
    if match:
        _set(span, "ml.run_id", match.group(1))


def set_error(exc, span=None, message: str | None = None) -> None:
    """Marks a span failed. Needed because agents deliberately swallow
    failures into tool results for the model to recover from — without this
    an errored turn is indistinguishable from a clean one in the trace list."""
    span = span or trace.get_current_span()
    if span is None or not span.is_recording():
        return
    detail = message or (f"{type(exc).__name__}: {exc}" if exc is not None else "error")
    span.set_status(Status(StatusCode.ERROR, detail))
    _set(span, "langfuse.observation.level", "ERROR")
    _set(span, "langfuse.observation.status_message", detail)
    if isinstance(exc, BaseException):
        span.record_exception(exc)


def set_attributes(span=None, **attributes) -> None:
    span = span or trace.get_current_span()
    for key, value in attributes.items():
        _set(span, key, value)


def llm_metadata(agent: str) -> dict:
    """metadata= for litellm.completion(). The parent span is passed
    explicitly rather than relying on the ambient OTel context, because the
    completion runs in a worker thread (asyncio.to_thread) whose context the
    success callback may not inherit."""
    metadata = {
        "trace_name": agent,
        "litellm_parent_otel_span": trace.get_current_span(),
        "tags": [agent, environment()],
        "trace_release": release(),
    }
    session = _session_id.get()
    if session:
        metadata["session_id"] = session
    if runtime.user():
        metadata["trace_user_id"] = (
            runtime.user()
        )  # litellm's langfuse_otel maps it to user.id
    return metadata


def inbound(
    message, fallback_session_id: str | None = None
) -> tuple[str | None, str | None]:
    """(session_id, traceparent) out of an A2A message — what the orchestrator
    put in message.metadata when it delegated. Absent for a direct call, which
    then starts its own trace under its own conversation."""
    metadata = getattr(message, "metadata", None) or {}
    return metadata.get("langfuse_session_id") or fallback_session_id, metadata.get(
        "traceparent"
    )


def outbound_metadata() -> dict:
    """message.metadata for an A2A delegation — the other half of inbound()."""
    return {
        k: v
        for k, v in {
            "traceparent": outbound_traceparent(),
            "langfuse_session_id": session_id(),
        }.items()
        if v
    }


def session_id() -> str | None:
    """The conversation this turn belongs to, for passing to a remote agent."""
    return _session_id.get()


def outbound_traceparent() -> str | None:
    """W3C traceparent for the current span, to hand to a remote agent."""
    carrier: dict[str, str] = {}
    _PROPAGATOR.inject(carrier)
    return carrier.get("traceparent")


def flush() -> None:
    """Starlette shutdown hook: BatchSpanProcessor drops whatever it is still
    holding on SIGTERM, which is exactly the spans for the turn that was
    running when the agent went down — the ones worth having."""
    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
    except Exception as exc:
        logger.warning("Span flush failed (%s: %s)", type(exc).__name__, exc)
