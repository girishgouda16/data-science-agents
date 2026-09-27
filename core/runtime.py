"""What every A2A agent needs per request, beyond its own ML: who is asking,
whether they may touch a run, how many turns may run at once, and task state
that survives a restart.

Identity: the gateway authenticates the user (Keycloak/OIDC, or a dev JWT)
and signs a short-lived USER CONTEXT token — user, project, roles — with
USER_CONTEXT_SECRET. It rides in A2A message metadata as `user_context`; the
orchestrator verifies it and forwards the same token on every delegation;
each agent verifies it (`bind_user`) and hands the user to that turn's MCP
tool process as AGENTIC_ML_USER (`mcp_env`). The LLM never sees or sets it —
a tool argument could be spoofed by a prompt, a signature cannot.

A bare `user_id` in metadata is never trusted: holding an agent's bearer key
is not the same as being any user you like. With AGENTIC_ML_REQUIRE_USER=1
(production) a message without a valid user context is refused; without it
(local dev, tests, direct A2A calls) it runs with no user and no ownership.
"""

import asyncio
import contextvars
import os
import time
from pathlib import Path

import jwt

USER_ENV = "AGENTIC_ML_USER"
PROJECT_ENV = "AGENTIC_ML_PROJECT"
ROLES_ENV = "AGENTIC_ML_ROLES"
AUTOPILOT_ENV = "AGENTIC_ML_AUTOPILOT"
CONTEXT_KEY = "user_context"
_AUDIENCE = "agentic-ml-agents"
_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_ml_user", default=None
)
_project: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_ml_project", default=None
)
_roles: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "agentic_ml_roles", default=()
)
_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_ml_user_context", default=None
)
_autopilot: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agentic_ml_autopilot", default=False
)


class Unauthenticated(Exception):
    """A request that must not run: bad, expired or (in production) missing user context."""


def _secret() -> str | None:
    return os.environ.get("USER_CONTEXT_SECRET") or os.environ.get("JWT_SECRET") or None


def sign_user_context(
    user: str,
    project: str | None = None,
    roles=(),
    ttl_seconds: int | None = None,
    autopilot: bool = False,
) -> str | None:
    """What the gateway attaches to each message: who, which project, which
    roles, whether the chat runs on autopilot — valid long enough to outlive
    the longest turn (USER_CONTEXT_TTL, default 4 h), short enough that a
    leaked one soon stops working."""
    secret = _secret()
    if not secret or not user:
        return None
    now = int(time.time())
    ttl = ttl_seconds or int(os.environ.get("USER_CONTEXT_TTL", 4 * 3600))
    claims = {
        "sub": user,
        "roles": sorted(roles),
        "aud": _AUDIENCE,
        "iat": now,
        "exp": now + ttl,
    }
    if project:
        claims["project"] = project
    if autopilot:
        claims["autopilot"] = True
    return jwt.encode(claims, secret, algorithm="HS256")


def _required() -> bool:
    return os.environ.get("AGENTIC_ML_REQUIRE_USER", "").lower() in ("1", "true", "yes")


def bind_user(message) -> str | None:
    """Bind the caller's verified identity (and the chat's project and the
    user's roles) from an inbound A2A message to this request."""
    metadata = getattr(message, "metadata", None) or {}
    token = metadata.get(CONTEXT_KEY)
    user, project, roles, autopilot_on = None, None, (), False
    if token:
        secret = _secret()
        if not secret:
            raise Unauthenticated(
                "this agent has no USER_CONTEXT_SECRET to verify the user context with"
            )
        try:
            claims = jwt.decode(token, secret, algorithms=["HS256"], audience=_AUDIENCE)
        except jwt.PyJWTError as exc:
            raise Unauthenticated(f"invalid user context: {exc}")
        user, project, roles = (
            claims["sub"],
            claims.get("project"),
            tuple(claims.get("roles") or ()),
        )
        autopilot_on = bool(claims.get("autopilot"))
    elif _required():
        raise Unauthenticated(
            "no signed user context — requests must come through the gateway"
        )
    return bind(user, project, roles, token, autopilot_on)


def bind(
    user: str | None,
    project: str | None = None,
    roles=(),
    token: str | None = None,
    autopilot: bool = False,
) -> str | None:
    """Set who this request acts for — bind_user after verifying, or the
    gateway for its own reads (runs list) on behalf of an authenticated user."""
    _user.set(user)
    _project.set(project)
    _roles.set(tuple(roles))
    _token.set(token)
    _autopilot.set(autopilot)
    return user


def user() -> str | None:
    """The requesting user: bound in an agent process, inherited in its tool process."""
    return _user.get() or os.environ.get(USER_ENV) or None


def project() -> str | None:
    """The project the user named for this chat — set by a person, never by the model."""
    return _project.get() or os.environ.get(PROJECT_ENV) or None


def roles() -> tuple:
    return _roles.get() or tuple(
        r for r in os.environ.get(ROLES_ENV, "").split(",") if r
    )


def autopilot() -> bool:
    """The user switched this chat to autopilot: agents decide instead of asking.
    Set by a person, never by the model."""
    return _autopilot.get() or os.environ.get(AUTOPILOT_ENV) == "1"


def outbound_metadata() -> dict:
    """A2A message metadata forwarding the verified user context to a delegated agent."""
    return {CONTEXT_KEY: _token.get()} if _token.get() else {}


# What a tool process may see of the agent's environment. Tools never call an
# LLM or another agent, so API keys, agent keys and the JWT secret stay out:
# a code-execution bug in a tool then has no credentials to take.
_TOOL_ENV = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "TMPDIR",
    "TZ",
    "VIRTUAL_ENV",
    "DATABASE_URL",
    "RUN_RETENTION_DAYS",
    "VISUALIZATION_AGENT_PORT",
    "CHARTS_PUBLIC_URL",
    "LANGFUSE_RELEASE",
    "LANGFUSE_TRACING_ENVIRONMENT",
    "AGENT_MAX_CONCURRENT_TURNS",
}
_TOOL_ENV_PREFIXES = (
    "AGENTIC_ML_",
    "MLFLOW_",
    "LC_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "CONDA",
    "PYTHON",
    "XDG_",
)


def mcp_env() -> dict:
    """Environment for this turn's MCP tool process: an allow-listed slice of
    the agent's own plus the user and project. Built per turn — concurrent
    turns each get their own."""
    env = {
        k: v
        for k, v in os.environ.items()
        if (k in _TOOL_ENV or k.startswith(_TOOL_ENV_PREFIXES))
        and k not in (USER_ENV, PROJECT_ENV, ROLES_ENV, AUTOPILOT_ENV)
    }
    for key, value in (
        (USER_ENV, user()),
        (PROJECT_ENV, project()),
        (ROLES_ENV, ",".join(roles())),
        (AUTOPILOT_ENV, "1" if autopilot() else ""),
    ):
        if value:
            env[key] = value
    return env


def is_admin(name: str | None = None) -> bool:
    """An admin by IdP role (AGENTIC_ML_ADMIN_ROLE, default ml-admin — the
    current user's) or by name in AGENTIC_ML_ADMINS (break-glass list)."""
    name = name or user()
    if not name:
        return False
    if (
        name == user()
        and os.environ.get("AGENTIC_ML_ADMIN_ROLE", "ml-admin") in roles()
    ):
        return True
    return name in {
        a.strip()
        for a in os.environ.get("AGENTIC_ML_ADMINS", "").split(",")
        if a.strip()
    }


def access_error(owner: str | None, what: str = "run") -> str | None:
    """Why the current user may not act on something `owner` created, or None.
    Unowned things (made before identity existed, or in local dev) and
    requests with no user stay open — identity is enforced where it exists."""
    current = user()
    if not owner or not current or current == owner or is_admin(current):
        return None
    return f"this {what} belongs to '{owner}' — '{current}' cannot use or change it (an ml-admin can)"


def stamp_owner(meta: dict) -> dict:
    """First writer owns the run, and the chat's project (if any) is the run's."""
    if user() and not meta.get("owner"):
        meta["owner"] = user()
    if project() and not meta.get("project"):
        meta["project"] = project()
    return meta


# ── concurrency ──────────────────────────────────────────────────────────────
# Each turn of a specialist runs its own MCP tool process, and a training one
# holds the dataset in memory — N users training at once is N copies. Past the
# cap, turns wait their turn instead of running the box out of memory.
_slots: dict[str, asyncio.Semaphore] = {}


def max_turns(agent: str) -> int:
    return int(
        os.environ.get(f"{agent.upper()}_MAX_CONCURRENT_TURNS")
        or os.environ.get("AGENT_MAX_CONCURRENT_TURNS")
        or 4
    )


async def limited(agent: str, turn):
    """Await the coroutine `turn` inside this agent's concurrency cap."""
    slot = _slots.setdefault(agent, asyncio.Semaphore(max_turns(agent)))
    async with slot:
        return await turn


# ── task store ───────────────────────────────────────────────────────────────
def task_store(agent: str):
    """A2A task state in a database, so a question the agent asked before a
    restart can still be answered after it. SQLite per agent by default
    (<data>/a2a/<agent>.db); A2A_TASK_DB_URL (e.g. postgresql+asyncpg://…)
    for a shared one. Falls back to in-memory if the SQL extra is missing."""
    from a2a.server.tasks import InMemoryTaskStore

    try:
        from a2a.server.tasks import DatabaseTaskStore
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError:
        return InMemoryTaskStore()
    url = os.environ.get("A2A_TASK_DB_URL")
    if not url:
        folder = (
            Path(
                os.environ.get("AGENTIC_ML_DATA_DIR")
                or Path(__file__).resolve().parents[1] / "data"
            )
            / "a2a"
        )
        folder.mkdir(parents=True, exist_ok=True)
        url = f"sqlite+aiosqlite:///{folder / f'{agent}.db'}"
    try:
        return DatabaseTaskStore(
            create_async_engine(url), table_name=f"a2a_tasks_{agent}"
        )
    except (
        Exception
    ):  # driver missing (aiosqlite / asyncpg) — still serve, just not durably
        return InMemoryTaskStore()
