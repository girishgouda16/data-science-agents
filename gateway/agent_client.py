"""Thin A2A client for the orchestrator agent — the gateway's only way in.

A turn is started NON-blocking: the orchestrator answers at once with the
task id (state "working"), and the gateway polls tasks/get until the task
finishes or asks the user something. No HTTP request stays open while a
model trains, so a reverse proxy's 60-second timeout never cuts a chat, and
tasks/cancel has a task id to cancel.

Talks to the orchestrator, never a specialist: routing is its job.
"""

import asyncio
import os
import uuid

import httpx
from a2a.client import A2ACardResolver, A2AClient, create_text_message_object
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    GetTaskRequest,
    MessageSendConfiguration,
    MessageSendParams,
    SendMessageRequest,
    Task,
    TaskIdParams,
    TaskQueryParams,
)
from dotenv import load_dotenv

# The gateway reads its orchestrator key from .env like every agent does.
load_dotenv()

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_AGENT_URL", "http://localhost:9100")
ORCHESTRATOR_API_KEY = os.environ.get("ORCHESTRATOR_AGENT_API_KEY") or os.environ.get(
    "ORCHESTRATOR_API_KEY", ""
)
# Every call is short now (start, poll, cancel); this only bounds a hung one.
CALL_TIMEOUT = float(os.environ.get("GATEWAY_AGENT_TIMEOUT", "60"))

_agent_card_cache: AgentCard | None = None
_agent_card_lock = asyncio.Lock()


def _http() -> httpx.AsyncClient:
    headers = (
        {"Authorization": f"Bearer {ORCHESTRATOR_API_KEY}"}
        if ORCHESTRATOR_API_KEY
        else {}
    )
    return httpx.AsyncClient(
        timeout=httpx.Timeout(CALL_TIMEOUT, connect=10.0), headers=headers
    )


async def _client(http: httpx.AsyncClient) -> A2AClient:
    global _agent_card_cache
    if _agent_card_cache is None:
        async with _agent_card_lock:
            if _agent_card_cache is None:
                _agent_card_cache = await A2ACardResolver(
                    http, ORCHESTRATOR_URL
                ).get_agent_card()
    return A2AClient(http, agent_card=_agent_card_cache)


def _turn(result) -> dict:
    """{"text", "task_id", "context_id", "state"} from a Task or a Message.
    state is an A2A TaskState value: working, input-required, completed,
    failed, canceled, rejected."""
    if isinstance(result, Task):
        message = result.status.message
        text = "".join(
            p.root.text
            for p in (message.parts if message else [])
            if p.root.kind == "text"
        )
        return {
            "text": text,
            "task_id": result.id,
            "context_id": result.context_id,
            "state": result.status.state.value,
        }
    text = "".join(p.root.text for p in result.parts if p.root.kind == "text")
    return {
        "text": text,
        "task_id": None,
        "context_id": result.context_id,
        "state": "completed",
    }


def _unwrap(response):
    if hasattr(response.root, "error"):
        raise RuntimeError(
            f"orchestrator error {response.root.error.code}: {response.root.error.message}"
        )
    return response.root.result


async def start_turn(
    message: str, task_id: str | None, context_id: str | None, metadata: dict
) -> dict:
    """Send one user message; returns as soon as the orchestrator has a task.
    task_id continues a task paused on a question; context_id keeps the
    conversation. metadata carries the signed user context and the Langfuse
    session id."""
    msg = create_text_message_object(content=message)
    msg.task_id, msg.context_id, msg.metadata = task_id, context_id, metadata or None
    params = MessageSendParams(
        message=msg, configuration=MessageSendConfiguration(blocking=False)
    )
    async with _http() as http:
        response = await (await _client(http)).send_message(
            SendMessageRequest(id=str(uuid.uuid4()), params=params)
        )
    return _turn(_unwrap(response))


async def get_turn(task_id: str) -> dict:
    async with _http() as http:
        response = await (await _client(http)).get_task(
            GetTaskRequest(
                id=str(uuid.uuid4()),
                params=TaskQueryParams(id=task_id, history_length=0),
            )
        )
    return _turn(_unwrap(response))


async def cancel_turn(task_id: str) -> dict:
    async with _http() as http:
        response = await (await _client(http)).cancel_task(
            CancelTaskRequest(id=str(uuid.uuid4()), params=TaskIdParams(id=task_id))
        )
    return _turn(_unwrap(response))
