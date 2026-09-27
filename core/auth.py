"""
core/auth.py

Who is calling the gateway.

Production: Keycloak (any OIDC provider works). The UI signs the user in with
the authorization-code + PKCE flow and sends the access token; the gateway
verifies it against the realm's published keys (JWKS) and checks issuer,
audience and expiry. Set OIDC_ISSUER (e.g. https://sso.example.com/realms/ml-agents).

Local dev: no OIDC_ISSUER — HS256 tokens signed with JWT_SECRET, minted by
`python -m scripts.print_token <user>`.

Either way verify_jwt returns the claims with `sub` set to the user's stable
name (OIDC_USER_CLAIM, default preferred_username — what ownership, MLflow
tags and Langfuse show) and `roles` to their realm/client roles.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Header, HTTPException

from core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _jwks() -> jwt.PyJWKClient:
    url = (
        settings.oidc_jwks_url
        or settings.oidc_issuer.rstrip("/") + "/protocol/openid-connect/certs"
    )
    return jwt.PyJWKClient(url, cache_keys=True, lifespan=3600)


def _roles(claims: dict) -> list[str]:
    roles = set((claims.get("realm_access") or {}).get("roles") or [])
    for client in (claims.get("resource_access") or {}).values():
        roles.update(client.get("roles") or [])
    return sorted(roles)


def decode(token: str) -> dict[str, Any]:
    """Verified claims, or jwt.PyJWTError."""
    if settings.oidc_issuer:
        key = _jwks().get_signing_key_from_jwt(token).key
        # leeway: the IdP's clock and ours are different machines; seconds of
        # skew otherwise make a fresh token "not yet valid" at random.
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256", "ES256"],
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
            leeway=30,
            options={"require": ["exp", "iss", "sub"]},
        )
        user = claims.get(settings.oidc_user_claim) or claims["sub"]
        return {**claims, "sub": user, "roles": _roles(claims)}
    claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    return {**claims, "roles": claims.get("roles") or []}


def verify_jwt(authorization: str | None = Header(None)) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            401,
            "Missing or malformed Authorization header (expected 'Bearer <token>').",
        )
    try:
        return decode(authorization.removeprefix("Bearer "))
    except (jwt.PyJWTError, KeyError) as exc:
        logger.info("rejected token: %s: %s", type(exc).__name__, exc)
        raise HTTPException(401, f"Invalid token: {exc}")


def issue_token(
    subject: str, expires_hours: int = 24, roles: list[str] | None = None
) -> str:
    """Dev tokens only (no OIDC_ISSUER): local testing and scripts."""
    if settings.oidc_issuer:
        raise RuntimeError(
            "OIDC_ISSUER is set — sign in through Keycloak instead of minting tokens"
        )
    now = int(time.time())
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_hours * 3600,
        "roles": roles or [],
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
