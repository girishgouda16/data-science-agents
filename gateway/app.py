"""The one front door: REST + auth + session persistence over the
orchestrator's A2A chat, plus the built UI, charts, and read-only views of
runs and the model registry. No LLM code of its own.

Auth: Keycloak/OIDC in production, dev tokens locally (core/auth.py). The
user is then carried to the agents as a gateway-signed user context
(core/runtime.sign_user_context) — agents trust nothing else.

A chat turn never holds an HTTP request open: POST starts the turn on the
orchestrator (non-blocking A2A send) and returns at once with status
"working"; a background task polls the orchestrator's task until it
finishes or asks a question, and records the reply. The UI polls
GET /api/v1/sessions/{id}. POST …/cancel cancels the running turn. A gateway
restart resumes following every turn that was still working.

Ownership: every session row's session_id column holds the user (see
core/db.save_a2a_task), checked on every read/continue/cancel/delete.

Run: python -m gateway.app  (serves on http://localhost:9001, per core/config.py)
"""

import asyncio
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.status import HTTP_413_CONTENT_TOO_LARGE

from core import datasource, registry, runs, runtime, toolguard
from core.auth import verify_jwt
from core.config import settings
from core.db import (
    delete_a2a_task,
    get_a2a_task,
    init_db,
    list_a2a_tasks_for_user,
    list_a2a_tasks_in_state,
    save_a2a_task,
)
from gateway import agent_client

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
UPLOAD_DIR = (
    toolguard.data_dir() / "uploads"
)  # the folder the tool guard lets a user read
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# A session's file_path must resolve inside data/ — the agents are told
# "Dataset: <path>", and the tool guard would refuse anything else anyway.
DATA_ROOT = UPLOAD_DIR.parent.resolve()
CHARTS_DIR = REPO / "visualization-agent" / "charts"
POLL_SECONDS = float(os.environ.get("GATEWAY_POLL_SECONDS", "2"))
TURN_TIMEOUT = float(os.environ.get("GATEWAY_TURN_TIMEOUT", str(4 * 3600)))
ACTIVE = {"working", "submitted"}
_followers: dict[str, asyncio.Task] = {}


def _user_uploads(claims: dict) -> Path:
    """Each user's uploads in their own folder — a session may use its owner's
    uploads, never a colleague's."""
    return UPLOAD_DIR / toolguard.user_folder(claims.get("sub"))


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    init_db()
    if not runtime.sign_user_context("probe"):
        logger.warning(
            "no USER_CONTEXT_SECRET/JWT_SECRET in the environment — agents will see no user"
        )
    # Turns still running when the gateway last stopped: follow them again.
    for record in list_a2a_tasks_in_state("working"):
        artifacts = record["artifacts"] or {}
        if artifacts.get("running_task_id"):
            _follow_in_background(
                record["task_id"], record["session_id"], artifacts["running_task_id"]
            )
        else:
            _finish(
                record["task_id"],
                record["session_id"],
                artifacts,
                {
                    "state": "failed",
                    "text": "This turn was interrupted before it started. Send it again.",
                },
            )
    yield
    for follower in list(_followers.values()):
        follower.cancel()


app = FastAPI(title="Data Science Agents Gateway", lifespan=_lifespan)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
        allow_methods=["*"],
        allow_headers=["*"],
    )


class StartSessionRequest(BaseModel):
    message: str
    # Optional: "Dataset: <path>" is prefixed to the message for the agents.
    file_path: str | None = None
    # Optional team project the chat's models register under
    # (<agent>-<project>-<target>). Without one: <agent>-<user>.<dataset>-<target>.
    project: str | None = None
    # Agents answer their own questions (their recommended option) instead of pausing.
    autopilot: bool = False


class MessageRequest(BaseModel):
    message: str
    autopilot: bool | None = None  # None keeps the chat's current setting


def _record_to_response(record: dict) -> dict:
    artifacts = record["artifacts"] or {}
    history = artifacts.get("history", [])
    return {
        "session_id": record["task_id"],
        "status": record["state"],
        "title": artifacts.get("title", "New chat"),
        "project": artifacts.get("project"),
        "autopilot": bool(artifacts.get("autopilot")),
        "reply": next(
            (t["text"] for t in reversed(history) if t["role"] == "agent"), ""
        ),
        "history": history,
        "started_at": artifacts.get("started_at"),
        "updated_at": record["updated_at"].isoformat(),
    }


def _title_from(message: str, limit: int = 60) -> str:
    message = " ".join(message.split())  # a title is one line
    return message if len(message) <= limit else message[:limit].rstrip() + "…"


def _file_label(path: str) -> str:
    """An upload's own name: uploads are stored as <uuid>_<name>."""
    return re.sub(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_",
        "",
        Path(path).name,
    )


def _owned(session_id: str, claims: dict) -> dict:
    record = get_a2a_task(session_id)
    if record is None or record["session_id"] != claims.get("sub"):
        raise HTTPException(404, "Unknown session_id")
    return record


# ── turns ────────────────────────────────────────────────────────────────────
def _finish(session_id: str, owner: str, artifacts: dict, turn: dict) -> None:
    """Record a turn that is no longer running: its reply, and which task the
    next message continues (only a task paused on a question)."""
    text = turn.get("text") or {
        "canceled": "Stopped.",
        "failed": "The turn failed.",
    }.get(turn["state"], "")
    if text:
        artifacts.setdefault("history", []).append({"role": "agent", "text": text})
    artifacts["agent_task_id"] = (
        turn.get("task_id") if turn["state"] == "input-required" else None
    )
    if turn.get("context_id"):
        artifacts["agent_context_id"] = turn["context_id"]
    artifacts.pop("running_task_id", None)
    artifacts.pop("started_at", None)
    save_a2a_task(session_id, owner, turn["state"], artifacts)


async def _follow(session_id: str, owner: str, task_id: str) -> None:
    started, misses = time.monotonic(), 0
    try:
        while True:
            await asyncio.sleep(POLL_SECONDS)
            record = get_a2a_task(session_id)
            if (
                record is None
                or (record["artifacts"] or {}).get("running_task_id") != task_id
            ):
                return  # deleted, cancelled, or superseded
            try:
                turn = await agent_client.get_turn(task_id)
                misses = 0
            except Exception as exc:
                misses += 1
                logger.warning(
                    "polling %s for session %s failed (%s)", task_id, session_id, exc
                )
                turn = {"state": "working"}
                if misses * POLL_SECONDS > 600:
                    turn = {
                        "state": "failed",
                        "text": f"Lost contact with the orchestrator ({exc}). "
                        "Every step that finished is saved — say 'continue'.",
                    }
            if turn["state"] in ACTIVE and time.monotonic() - started > TURN_TIMEOUT:
                turn = {
                    "state": "failed",
                    "text": "This turn ran past the time limit and was stopped.",
                }
                await _cancel_quietly(task_id)
            if turn["state"] not in ACTIVE:
                _finish(session_id, owner, record["artifacts"], turn)
                return
    finally:
        _followers.pop(session_id, None)


def _follow_in_background(session_id: str, owner: str, task_id: str) -> None:
    if session_id not in _followers:
        _followers[session_id] = asyncio.create_task(
            _follow(session_id, owner, task_id)
        )


async def _cancel_quietly(task_id: str) -> None:
    try:
        await agent_client.cancel_turn(task_id)
    except (
        Exception
    ) as exc:  # already finished, or the orchestrator is gone — nothing left to stop
        logger.info("cancel of %s: %s", task_id, exc)


async def _start(
    session_id: str, claims: dict, artifacts: dict, text: str, shown: dict | None = None
) -> dict:
    """Hand one user message to the orchestrator and return the session as it
    now stands (usually "working")."""
    owner = claims.get("sub")
    metadata = {
        k: v
        for k, v in {
            # Who is asking, signed — the agents stamp ownership and approvals with it.
            runtime.CONTEXT_KEY: runtime.sign_user_context(
                owner,
                artifacts.get("project"),
                claims.get("roles") or [],
                autopilot=bool(artifacts.get("autopilot")),
            ),
            # The chat's own id as the Langfuse session: the session a user opens
            # in Langfuse is the chat they had.
            "langfuse_session_id": session_id,
        }.items()
        if v
    }
    try:
        turn = await agent_client.start_turn(
            text,
            artifacts.get("agent_task_id"),
            artifacts.get("agent_context_id"),
            metadata,
        )
    except Exception as exc:
        raise HTTPException(502, f"orchestrator unreachable: {exc}")
    # The chat shows what the user typed (and the file's own name), not the
    # server path the agents are handed.
    artifacts.setdefault("history", []).append(
        {"role": "user", "text": text, **(shown or {})}
    )
    if turn["state"] in ACTIVE:
        artifacts.update(
            running_task_id=turn["task_id"],
            agent_context_id=turn["context_id"],
            started_at=time.time(),
        )
        save_a2a_task(session_id, owner, "working", artifacts)
        _follow_in_background(session_id, owner, turn["task_id"])
    else:
        _finish(session_id, owner, artifacts, turn)
    return _record_to_response(get_a2a_task(session_id))


@app.post("/api/v1/sessions")
async def start_session(body: StartSessionRequest, claims: dict = Depends(verify_jwt)):
    prompt = body.message
    if body.file_path:
        try:
            resolved = Path(body.file_path).resolve()
            resolved.relative_to(DATA_ROOT)
        except ValueError:
            raise HTTPException(400, f"file_path must be under {DATA_ROOT}")
        if not resolved.is_file():
            raise HTTPException(400, f"no file at '{body.file_path}'")
        uploads, own = UPLOAD_DIR.resolve(), _user_uploads(claims).resolve()
        if (
            resolved.is_relative_to(uploads)
            and resolved.parent != uploads
            and not resolved.is_relative_to(own)
        ):
            raise HTTPException(403, "that upload belongs to another user")
        prompt = f"Dataset: {resolved}\n{body.message}"
    session_id = str(uuid.uuid4())
    artifacts = {
        "user_id": claims.get("sub"),
        "title": _title_from(body.message),
        "project": (body.project or "").strip() or None,
        "autopilot": body.autopilot,
        "history": [],
    }
    shown = (
        {"text": body.message, "attachment": _file_label(body.file_path)}
        if body.file_path
        else None
    )
    return await _start(session_id, claims, artifacts, prompt, shown)


@app.post("/api/v1/sessions/{session_id}/messages")
async def continue_session(
    session_id: str, body: MessageRequest, claims: dict = Depends(verify_jwt)
):
    record = _owned(session_id, claims)
    if record["state"] == "working":
        raise HTTPException(
            409,
            "the previous message is still being worked on — wait for it or cancel it",
        )
    artifacts = record["artifacts"] or {}
    if body.autopilot is not None:
        artifacts["autopilot"] = body.autopilot
    return await _start(session_id, claims, artifacts, body.message)


@app.post("/api/v1/sessions/{session_id}/cancel")
async def cancel_session(session_id: str, claims: dict = Depends(verify_jwt)):
    record = _owned(session_id, claims)
    artifacts = record["artifacts"] or {}
    task_id = artifacts.get("running_task_id")
    if record["state"] != "working" or not task_id:
        raise HTTPException(409, "nothing is running in this chat")
    await _cancel_quietly(task_id)
    if follower := _followers.pop(session_id, None):
        follower.cancel()
    _finish(
        session_id,
        claims.get("sub"),
        artifacts,
        {
            "state": "canceled",
            "text": (
                "Stopped at your request. Every step that finished is saved — say 'continue' to pick up from there."
            ),
        },
    )
    return _record_to_response(get_a2a_task(session_id))


@app.get("/api/v1/sessions")
async def list_sessions(claims: dict = Depends(verify_jwt)):
    return {
        "sessions": [
            _record_to_response(r) for r in list_a2a_tasks_for_user(claims.get("sub"))
        ]
    }


@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str, claims: dict = Depends(verify_jwt)):
    return _record_to_response(_owned(session_id, claims))


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str, claims: dict = Depends(verify_jwt)):
    record = _owned(session_id, claims)
    if task_id := (record["artifacts"] or {}).get("running_task_id"):
        await _cancel_quietly(task_id)
    delete_a2a_task(session_id)
    return {"deleted": session_id}


# ── uploads ──────────────────────────────────────────────────────────────────
@app.post("/api/v1/upload")
async def upload(file: UploadFile = File(...), claims: dict = Depends(verify_jwt)):
    # Authenticated: an open upload lets anyone fill the disk, and the file
    # lands in the uploader's own folder (see start_session).
    contents = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            HTTP_413_CONTENT_TOO_LARGE,
            f"File exceeds {settings.max_file_size_mb}MB limit",
        )
    # Data files only, and each must BE the format its name claims (magic
    # bytes for binary formats, no NUL bytes in text ones): a .pkl "dataset"
    # is code that runs when something loads it, and a renamed one must not
    # get in as "parquet". Tools only ever parse these with pandas/pyarrow.
    name = Path(file.filename or "upload.csv").name
    if not datasource.looks_like(name, contents[:8192]):
        raise HTTPException(415, f"only data files can be uploaded ({', '.join(sorted(datasource.DATA_SUFFIXES))}), "
                                 "and the content must match the extension")
    folder = _user_uploads(claims)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{uuid.uuid4()}_{name}"
    dest.write_bytes(contents)
    return {"file_path": str(dest.resolve()), "name": name, "size_bytes": len(contents)}


@app.get("/api/v1/download")
async def download(path: str, claims: dict = Depends(verify_jwt)):
    """One of the caller's own files — an upload, or an output written beside
    it (predictions). Anything else is 404, never a colleague's file."""
    file = Path(path).resolve()
    own = (
        _user_uploads(claims).resolve(),
        (
            toolguard.data_dir()
            / "predictions"
            / toolguard.user_folder(claims.get("sub"))
        ).resolve(),
    )
    if not file.is_file() or not any(file.is_relative_to(root) for root in own):
        raise HTTPException(404, "no such file of yours")
    return FileResponse(file, filename=_file_label(str(file)))


# ── who am I, runs, registry ─────────────────────────────────────────────────
@app.get("/api/v1/auth/config")
async def auth_config():
    """Public: how the UI should sign in."""
    if settings.oidc_issuer:
        return {
            "mode": "oidc",
            "issuer": settings.oidc_issuer,
            "client_id": settings.oidc_client_id,
        }
    return {"mode": "dev"}


@app.get("/api/v1/me")
async def me(claims: dict = Depends(verify_jwt)):
    runtime.bind(claims.get("sub"), roles=claims.get("roles") or [])
    return {
        "user": claims.get("sub"),
        "name": claims.get("name"),
        "email": claims.get("email"),
        "roles": claims.get("roles") or [],
        "admin": runtime.is_admin(),
    }


@app.get("/api/v1/runs")
async def list_runs(claims: dict = Depends(verify_jwt)):
    """The caller's runs on disk (all runs for an ml-admin)."""
    runtime.bind(claims.get("sub"), roles=claims.get("roles") or [])
    return await asyncio.to_thread(runs.list_runs, "", 100)


@app.delete("/api/v1/runs/{run_id}")
async def delete_run(run_id: str, claims: dict = Depends(verify_jwt)):
    runtime.bind(claims.get("sub"), roles=claims.get("roles") or [])
    result = await asyncio.to_thread(runs.delete_run, run_id)
    if "error" in result:
        raise HTTPException(result.get("status", 409), result["error"])
    return result


@app.delete("/api/v1/models/{name}")
async def delete_model(name: str, claims: dict = Depends(verify_jwt)):
    runtime.bind(claims.get("sub"), roles=claims.get("roles") or [])
    result = await asyncio.to_thread(registry.delete_model, name)
    if "error" in result:
        raise HTTPException(result.get("status", 409), result["error"])
    return result


@app.get("/api/v1/models")
async def list_models(_claims: dict = Depends(verify_jwt)):
    """The team's model registry: every model, its newest version and aliases."""
    return await asyncio.to_thread(registry.list_models)


@app.get("/api/v1/models/{name}")
async def model_versions(name: str, _claims: dict = Depends(verify_jwt)):
    return await asyncio.to_thread(registry.list_versions, name)


# Charts the visualization agent renders, same origin as the UI so remote
# browsers can load them (the agent's own :9200 is internal). Unguessable
# names; <img> tags cannot send a bearer token.
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/charts", StaticFiles(directory=CHARTS_DIR), name="charts")

# The built chat UI (`make ui-build` → ui/dist), same origin as the API.
# Mounted last: every route above matches first.
UI_DIST = REPO / "ui" / "dist"
if UI_DIST.is_dir():
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
