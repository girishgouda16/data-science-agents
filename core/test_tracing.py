"""Self-check for core/tracing.py — the three things that have to hold for a
conversation to show up in Langfuse as one session containing one tree per
turn, instead of a flat list of litellm_request rows:

  1. litellm's generation span is a CHILD of the turn span (it honours
     metadata["litellm_parent_otel_span"]),
  2. the turn span carries session.id, which is what Langfuse threads on,
  3. a traceparent handed across the A2A hop puts the remote agent's turn
     inside the caller's trace,

plus the traceability detail every span is expected to carry: input/output,
environment + release, an ERROR status on failures the agents deliberately
swallow into tool results, and the run_id lifted out of a tool result so a
trace joins up with the MLflow run it produced.

No Langfuse and no LLM needed: spans go to an in-memory exporter and
litellm runs with mock_response.

Run: python test_tracing.py
"""

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-test")
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3000")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

import litellm

litellm.success_callback = ["langfuse_otel"]  # exactly what every agent sets

# langfuse_otel builds its own exporter from the LANGFUSE_* env vars on every
# request, so the only way to see its spans in-process is to hand it ours.
from litellm.integrations.opentelemetry import OpenTelemetry

OpenTelemetry.get_tracer_to_use_for_request = lambda self, kwargs: provider.get_tracer(
    "litellm"
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import tracing


async def main():
    with tracing.turn_span(
        "orchestrator",
        session_id="ctx-123",
        input="hi",
        **{"gen_ai.request.model": "gpt-4o-mini"},
    ) as turn_:
        # asyncio.to_thread is how every agent calls litellm — the parent span
        # has to survive the hop into the worker thread.
        await asyncio.to_thread(
            litellm.completion,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="hello",
            metadata=tracing.llm_metadata("orchestrator"),
        )
        tracing.set_output("answer", turn_)
        with tracing.tool_span(
            "delegate_to_classification", input={"message": "train it"}
        ) as tool:
            # Captured INSIDE the tool span, exactly where _call_remote_agent
            # builds msg.metadata — so the remote nests under the delegate
            # call, not merely somewhere in the trace.
            delegated = tracing.outbound_metadata()  # what the A2A message carries
            tracing.set_output('{"run_id": "run-42", "status": "ok"}', tool)
        # what a swallowed delegate failure has to look like
        with tracing.tool_span("delegate_to_regression") as failed:
            tracing.set_error(RuntimeError("remote down"), failed)

        # A later turn of the chat, arriving while the A2A SDK's own request span
    # is current: it must still root its own trace.
    with provider.get_tracer("a2a-python-sdk").start_as_current_span("on_message_send"):
        with tracing.turn_span(
            "orchestrator", session_id="ctx-123", name="orchestrator turn 2"
        ):
            pass

    # The remote agent's side: it only ever sees the A2A message metadata.
    session_id, traceparent = tracing.inbound(_FakeMessage(delegated), "its-own-ctx")
    with tracing.turn_span(
        "classification", session_id=session_id, traceparent=traceparent
    ):
        pass
    return session_id


class _FakeMessage:
    def __init__(self, metadata):
        self.metadata = metadata


os.environ["AGENTIC_ML_USER"] = (
    "ds-alice"  # what runtime.bind_user leaves for the request
)
remote_session = asyncio.run(main())
del os.environ["AGENTIC_ML_USER"]
spans = exporter.get_finished_spans()
by_name = {s.name: s for s in spans}

turn = by_name["orchestrator turn"]
assert turn.parent is None, "the turn span should be the trace root"
assert turn.attributes["session.id"] == "ctx-123"
assert turn.attributes["user.id"] == "ds-alice", (
    "Langfuse files the trace under the user who asked"
)
assert turn.attributes["langfuse.trace.name"] == "orchestrator"

generations = [
    s for s in spans if s.attributes.get("langfuse.observation.type") == "generation"
]
assert generations, "litellm produced no generation span"
assert any(
    s.parent and s.parent.span_id == turn.context.span_id for s in generations
), "the generation is not a child of the turn span"
assert all(s.context.trace_id == turn.context.trace_id for s in generations)
assert all(s.attributes.get("user.id") == "ds-alice" for s in generations), (
    "LLM calls not filed under the user"
)

tool = by_name["delegate_to_classification"]
assert tool.parent.span_id == turn.context.span_id
assert tool.attributes["langfuse.observation.type"] == "tool"

remote = by_name["classification turn"]
assert remote.context.trace_id == turn.context.trace_id, (
    "traceparent did not carry the trace across the hop"
)
assert remote.attributes["session.id"] == "ctx-123", (
    "remote agent did not inherit the conversation's session"
)
assert remote.attributes["user.id"] == "ds-alice", "the delegated turn lost the user"
assert remote.parent.span_id == tool.context.span_id, (
    "the delegated turn is not nested under the delegate tool span"
)
# langfuse.trace.* is trace-level, last writer wins: a delegated turn setting it
# renames the ORCHESTRATOR's trace to "classification".
assert "langfuse.trace.name" not in remote.attributes, (
    "delegated turn overwrote the trace name"
)
assert "langfuse.trace.tags" not in remote.attributes, (
    "delegated turn overwrote the trace tags"
)
assert remote_session == "ctx-123"

# A direct call (no orchestrator metadata) still traces, under its own session.
assert tracing.inbound(None, "direct-ctx") == ("direct-ctx", None)

# --- traceability detail -------------------------------------------------
assert turn.attributes["langfuse.observation.input"] == "hi"
assert turn.attributes["langfuse.observation.output"] == "answer"
assert (
    turn.attributes["langfuse.trace.input"] == "hi"
    and turn.attributes["langfuse.trace.output"] == "answer"
), "a session view lists traces by the user's message and the reply"
assert "langfuse.trace.output" not in tool.attributes, (
    "only the root span speaks for the trace"
)
assert "langfuse.trace.input" not in remote.attributes, (
    "a delegated turn must not relabel the trace"
)

second = by_name["orchestrator turn 2"]
assert second.parent is None, (
    "a later turn hung off the A2A SDK's span — a rootless trace in Langfuse"
)
assert tracing.should_export(tool) and tracing.should_export(turn), (
    "our tool/hop spans must reach Langfuse"
)
assert not tracing.should_export(by_name["on_message_send"]), (
    "the SDK's plumbing spans stay out"
)
assert turn.attributes["langfuse.environment"] == tracing.environment()
assert turn.attributes["langfuse.release"] == tracing.release()
assert turn.attributes["gen_ai.request.model"] == "gpt-4o-mini"

assert tool.attributes["langfuse.observation.input"] == '{"message": "train it"}'
assert tool.attributes["ml.run_id"] == "run-42", (
    "run_id not lifted out of the tool result"
)

failed = by_name["delegate_to_regression"]
from opentelemetry.trace import StatusCode

assert failed.status.status_code is StatusCode.ERROR, (
    "a swallowed failure must still mark its span"
)
assert "remote down" in failed.status.description
assert failed.attributes["langfuse.observation.level"] == "ERROR"

big = "x" * (tracing.MAX_VALUE_CHARS + 5000)
assert len(tracing._stringify(big)) < len(big), "oversized payloads must be truncated"

print(
    f"OK — {len(spans)} spans, one trace, session=ctx-123, release={tracing.release()}"
)
