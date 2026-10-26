"""Health endpoints — liveness + readiness probes.

``GET /health``
    Liveness — returns 200 as long as the process is up.

``GET /readyz``
    Readiness — returns 200 only when the DB is reachable and CV
    models are loaded; 503 otherwise. Used by k8s readinessProbe /
    docker-compose ``healthcheck``.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app import __version__
from app.api.deps import get_db_session
from app.schemas.common import DependencyStatus, HealthResponse, ReadyzResponse
from app.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Module-level process start time for uptime reporting.
_PROCESS_START = time.monotonic()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Always returns 200 if the process is up. Use for k8s livenessProbe.",
)
async def health() -> HealthResponse:
    """Liveness — process is up."""
    return HealthResponse(
        status="ok",
        version=__version__,
        uptime_seconds=round(time.monotonic() - _PROCESS_START, 2),
    )


@router.get(
    "/readyz",
    response_model=ReadyzResponse,
    summary="Readiness probe",
    description=(
        "Returns 200 when the database is reachable and CV models are loaded. "
        "Returns 503 otherwise. Use for k8s readinessProbe."
    ),
    responses={
        200: {"model": ReadyzResponse, "description": "All dependencies healthy"},
        503: {"model": ReadyzResponse, "description": "One or more dependencies unavailable"},
    },
)
async def readyz(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> ReadyzResponse:
    """Readiness — DB + models + cache."""
    db_status = DependencyStatus(status="ok")
    models_status = DependencyStatus(status="ok")
    cache_status = DependencyStatus(status="skip")

    # DB
    try:
        t0 = time.monotonic()
        await session.execute(text("SELECT 1"))
        db_status.latency_ms = round((time.monotonic() - t0) * 1000.0, 2)
    except Exception as exc:
        db_status = DependencyStatus(status="down", error=str(exc)[:200])

    # Models (set by lifespan on app.state)
    models_loaded = getattr(request.app.state, "models_loaded", False)
    if not models_loaded:
        models_status = DependencyStatus(
            status="degraded",
            error="CV models not yet loaded (cold start or missing weights)",
        )
    else:
        models_status.detail = {
            "device": getattr(request.app.state, "detector_device", "unknown"),
            "backend": getattr(request.app.state, "detector_backend", "unknown"),
            "model": getattr(request.app.state, "detector_model", "unknown"),
            "pack": getattr(request.app.state, "model_name", "unknown"),
        }

    # Cache (Redis) — optional. We only mark it down if explicitly
    # configured and unreachable; otherwise we report "skip".
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is not None:
        try:
            t0 = time.monotonic()
            await redis_client.ping()
            cache_status = DependencyStatus(
                status="ok", latency_ms=round((time.monotonic() - t0) * 1000.0, 2)
            )
        except Exception as exc:
            cache_status = DependencyStatus(status="down", error=str(exc)[:200])
    else:
        cache_status = DependencyStatus(status="skip")

    overall = "ok" if (db_status.status == "ok" and models_status.status == "ok") else "unavailable"
    response = ReadyzResponse(
        status=overall,
        database=db_status,
        models=models_status,
        cache=cache_status,
        checked_at=datetime.now(timezone.utc),
    )
    if overall != "ok":
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(mode="json"),
        )
    return response


__all__ = ["router"]
