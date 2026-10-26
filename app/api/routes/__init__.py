"""API routes (sub-routers registered by app.main)."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import attendance, employees, health, stream


def build_api_router() -> APIRouter:
    """Assemble the ``/api`` router with all sub-routers."""
    router = APIRouter()
    router.include_router(health.router, tags=["health"])
    router.include_router(employees.router, prefix="/employees", tags=["employees"])
    router.include_router(attendance.router, prefix="/attendance", tags=["attendance"])
    router.include_router(stream.rest_router, tags=["stream"])
    return router


def build_v1_router() -> APIRouter:
    """Assemble the ``/api/v1`` router (same endpoints, v1 prefix)."""
    router = APIRouter()
    router.include_router(health.router, tags=["health"])
    router.include_router(employees.router, prefix="/employees", tags=["employees"])
    router.include_router(attendance.router, prefix="/attendance", tags=["attendance"])
    router.include_router(stream.rest_router, tags=["stream"])
    return router


def build_root_router() -> APIRouter:
    """Root-level router — health/readyz at ``/`` (no ``/api`` prefix).

    Mounted directly on the app so that k8s/docker-compose probes that
    hit ``GET /health`` work without needing to know about the
    ``/api`` prefix.
    """
    router = APIRouter()
    router.include_router(health.router, tags=["health"])
    return router


__all__ = ["build_api_router", "build_v1_router", "build_root_router"]
