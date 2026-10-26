"""FastAPI dependencies shared by all route modules.

Centralising these here keeps routes thin and makes the wiring testable
(mocks can be injected via ``app.dependency_overrides``).
"""
from __future__ import annotations

from typing import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.enrollment import EnrollmentService
from app.services.stream_manager import stream_manager as default_stream_manager


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield a per-request :class:`AsyncSession`."""
    async for session in get_db():
        yield session


# --------------------------------------------------------------------------- #
# Services
# --------------------------------------------------------------------------- #
def get_enrollment_service() -> EnrollmentService:
    """Return a process-wide :class:`EnrollmentService` instance.

    Cached on ``app.state.enrollment_service`` by :mod:`app.main`'s
    lifespan; we resolve it from the request state to avoid import-time
    instantiation (the CV core may not be available yet).
    """
    service = _APP_STATE_DEFAULTS.get("enrollment_service")
    if service is not None:
        return service
    # Fallback (e.g. in unit tests before lifespan runs) — build a fresh
    # instance. Pre-computed embeddings remain usable if CV models are absent.
    return EnrollmentService()


def get_stream_manager():
    """Return the shared stream manager used by the app and WebSocket routes."""
    return default_stream_manager


def get_app_state(request: Request) -> Request:
    """Expose the FastAPI ``app.state`` for routes that need it."""
    return request


# --------------------------------------------------------------------------- #
# Module-level cache for service singletons (set by lifespan)
# --------------------------------------------------------------------------- #
_APP_STATE_DEFAULTS: dict[str, object] = {}


def set_app_state_defaults(**singletons: object) -> None:
    """Called by ``app.main`` lifespan to share singletons."""
    _APP_STATE_DEFAULTS.update(singletons)


__all__ = [
    "get_db_session",
    "get_enrollment_service",
    "get_stream_manager",
    "get_app_state",
    "set_app_state_defaults",
]
