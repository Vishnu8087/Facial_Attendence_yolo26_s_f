"""Authentication & authorization helpers.

The deployment ships with two interchangeable auth schemes:

* **API key** (default, demo-friendly) — requests carry
  ``Authorization: ApiKey <pre-shared-key>``. The expected key is read
  from ``settings.api_key``. If unset, auth is disabled (trusted-network
  deployments).
* **JWT bearer** — requests carry ``Authorization: Bearer <jwt>``. The
  token is verified with ``settings.jwt_secret`` using HS256. To enable,
  issue tokens from your identity provider and set ``jwt_secret``.

Both schemes resolve to a :class:`User` object that downstream code can
inspect (``user.is_admin``, ``user.subject``). Write endpoints
(``POST /api/employees``, ``DELETE /api/employees/{id}``) require an
admin user via :func:`require_admin`; read endpoints accept any user
(including anonymous when auth is disabled).

The WebSocket endpoint can't send the ``Authorization`` header (browser
limitation), so it accepts the token via a ``?token=`` query parameter
parsed in :mod:`app.api.routes.stream`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Security scheme definitions (also surfaced in OpenAPI)
# --------------------------------------------------------------------------- #
_bearer_scheme = HTTPBearer(auto_error=False)
_apikey_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


# --------------------------------------------------------------------------- #
# User principal
# --------------------------------------------------------------------------- #
@dataclass
class User:
    """Lightweight principal object passed to route handlers."""

    subject: str = "anonymous"
    is_admin: bool = False
    auth_method: str = "none"  # none | api_key | jwt
    scopes: list[str] = field(default_factory=list)
    raw_token: str | None = None
    claims: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _decode_jwt(token: str) -> dict[str, Any]:
    """Decode + verify a JWT. Raises ``HTTPException(401)`` on failure."""
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        ) from exc


def _user_from_jwt(token: str) -> User:
    claims = _decode_jwt(token)
    return User(
        subject=str(claims.get("sub", "unknown")),
        is_admin=bool(claims.get("is_admin", False)),
        auth_method="jwt",
        scopes=list(claims.get("scopes", [])),
        raw_token=token,
        claims=claims,
    )


def _user_from_api_key(key: str) -> User:
    return User(
        subject="api-client",
        is_admin=True,
        auth_method="api_key",
        scopes=["admin"],
        raw_token=key,
    )


# --------------------------------------------------------------------------- #
# FastAPI dependencies
# --------------------------------------------------------------------------- #
def get_current_user(
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
    api_key: str | None = Security(_apikey_scheme),
) -> User:
    """Resolve the current user from the request.

    Resolution order: API-key header first (cheapest), then Bearer JWT.
    If neither is present and ``settings.api_key`` is empty, we return an
    anonymous user (auth disabled — trusted-network mode).

    Raises ``401`` for malformed/expired tokens.
    """
    # API-key path
    if api_key:
        if settings.api_key and api_key != settings.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        if settings.api_key and api_key == settings.api_key:
            return _user_from_api_key(api_key)
        # api_key provided but settings.api_key unset — treat as anonymous

    # Bearer JWT path
    if bearer and bearer.credentials:
        return _user_from_jwt(bearer.credentials)

    # Anonymous mode
    if not settings.api_key:
        return User(subject="anonymous", auth_method="none")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Dependency — only admin users may pass.

    When auth is disabled (``settings.api_key`` empty AND no JWT
    ``api_key``/``jwt_secret`` enforcement), anonymous users are
    allowed through (treated as admin) — this is the trusted-network
    deployment mode. When auth is enabled, non-admin users get 403.
    """
    # Auth-disabled mode: api_key unset means we trust the network.
    if not settings.api_key:
        return User(
            subject="anonymous-admin",
            is_admin=True,
            auth_method="none",
        )
    if user.is_admin:
        return user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin privileges required",
    )


def user_from_token_param(token: str | None) -> User:
    """Resolve a user from a ``?token=`` query parameter.

    Used by the WebSocket endpoint (browsers can't set headers on
    ``new WebSocket()``). Same logic as :func:`get_current_user` but
    only the JWT path is exercised.
    """
    if not token:
        if not settings.api_key:
            return User(subject="anonymous", auth_method="none")
        raise HTTPException(status_code=401, detail="Authentication required")
    return _user_from_jwt(token)


# --------------------------------------------------------------------------- #
# Convenience aliases for OpenAPI grouping
# --------------------------------------------------------------------------- #
HTTPBearerDep = Depends(get_current_user)
APIKeyHeaderDep = Security(_apikey_scheme)
CurrentUser = User

__all__ = [
    "User",
    "CurrentUser",
    "get_current_user",
    "require_admin",
    "user_from_token_param",
    "HTTPBearerDep",
    "APIKeyHeaderDep",
]
