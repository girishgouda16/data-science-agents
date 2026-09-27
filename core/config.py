"""
core/config.py

Config for the gateway (gateway/app.py, core/auth.py, core/db.py,
alembic/env.py) — the thin REST+auth layer, not the LLM agents (those read
their own env vars directly, see each agent's agent.py docstring).
"""

import logging
import secrets
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_INSECURE_JWT_DEFAULT = "dev-secret-change-me-in-.env-min-32-bytes-long"


class Settings(BaseSettings):
    # ── Persistence ───────────────────────────────────────────────────────
    # Defaults to a local SQLite file — zero setup for demos/dev.
    database_url: str = "sqlite:///./data/training_runs.db"

    # Fitted model pipelines (.pkl, via joblib) — one file per run, path
    # recorded in training_runs.artifact_path. Local disk only; swap for a
    # bucket path (s3://...) here if this ever runs on more than one box.
    artifacts_dir: str = "data/artifacts"

    # ── Auth (A2A/REST bearer JWT — one shared secret per environment) ────
    # CHANGEME in .env for anything beyond local dev — see get_settings(),
    # which refuses to actually run with this literal value.
    jwt_secret: str = _INSECURE_JWT_DEFAULT
    jwt_algorithm: str = "HS256"

    # ── Keycloak / OIDC (set oidc_issuer to turn it on; see core/auth.py) ─
    oidc_issuer: str = ""  # https://<keycloak>/realms/ml-agents
    oidc_audience: str = "ml-agents"  # the realm's audience mapper adds this
    oidc_client_id: str = "ml-agents-ui"  # public client the UI signs in with
    oidc_jwks_url: str = ""  # default <issuer>/protocol/openid-connect/certs
    oidc_user_claim: str = "preferred_username"
    cors_origins: str = ""  # comma-separated; empty = same origin only

    # ── App ───────────────────────────────────────────────────────────────
    app_host: str = "127.0.0.1"  # 0.0.0.0 only when the TLS proxy is on another host
    app_port: int = 9001
    max_file_size_mb: int = 500

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    """The literal default jwt_secret is public (it's in this file, in git
    history) — never let a process actually issue/verify tokens with it, the
    same way each agent's _require_api_key refuses to run auth-less."""
    s = Settings()
    if s.jwt_secret == _INSECURE_JWT_DEFAULT:
        s.jwt_secret = secrets.token_urlsafe(32)
        logger.warning(
            "JWT_SECRET not set in .env — generated a one-time secret for this "
            "process. Every previously issued token is now invalid. Set "
            "JWT_SECRET in .env (openssl rand -hex 32) to keep tokens valid "
            "across restarts."
        )
    return s


settings = get_settings()
