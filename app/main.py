"""FastAPI application entrypoint.

Wires up:

* Lifespan: init logging, DB pool, preload CV models, init stream manager.
* CORS middleware (configurable origins).
* Prometheus instrumentation at ``/metrics``.
* Static files from ``frontend/`` at ``/``.
* REST routers under ``/api`` (and ``/api/v1`` alias).
* WebSocket endpoints at ``/ws/stream`` and ``/ws/events``.
* ``/health`` and ``/readyz`` endpoints (registered both at root and under /api).
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import build_api_router, build_root_router, build_v1_router
from app.api.routes.stream import ws_router
from app.config import settings
from app.db.session import dispose_engine
from app.services.enrollment import EnrollmentService
from app.services.stream_manager import stream_manager
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup + shutdown lifecycle."""
    # --- Startup ---
    configure_logging()
    settings.ensure_dirs()
    logger.info(
        "app.startup",
        version=__version__,
        env=os.environ.get("APP_ENV", "dev"),
        database_url=settings.database_url.replace(settings.db_password, "***") if settings.db_password else settings.database_url,
    )
    logger.info(
        "app.cpu.cores",
        cores=os.cpu_count(),
        arcface_threading=settings.arcface_threading,
    )

    # Sentry (optional)
    if settings.sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.starlette import StarletteIntegration

            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                traces_sample_rate=settings.sentry_traces_sample_rate,
                integrations=[
                    StarletteIntegration(transaction_style="endpoint"),
                    FastApiIntegration(transaction_style="endpoint"),
                ],
            )
            logger.info("app.sentry.initialised")
        except Exception as exc:
            logger.warning("app.sentry.init_failed", error=str(exc))

    # Redis (optional)
    redis_client = None
    if settings.redis_url:
        try:
            import redis.asyncio as aioredis

            redis_client = aioredis.from_url(
                settings.redis_url, decode_responses=True, socket_timeout=2
            )
            await redis_client.ping()
            logger.info("app.redis.connected", url=settings.redis_url)
        except Exception as exc:
            logger.warning("app.redis.unavailable", error=str(exc))
            redis_client = None
    app.state.redis = redis_client

    # Enrollment service (lazy CV core).
    enrollment_service = EnrollmentService()
    app.state.enrollment_service = enrollment_service
    detector = enrollment_service.detector
    app.state.detector_backend = (
        "yolo26" if type(detector).__name__ == "YoloFaceDetector" else "insightface"
    )
    app.state.detector_model = (
        getattr(detector, "model_path", None)
        or getattr(detector, "model_pack_name", None)
        or settings.model_name
    )
    app.state.detector_device = getattr(detector, "device", None) or (
        "cuda" if settings.gpu_enabled else "cpu"
    )

    # Stream manager — pre-warm the shared pipeline.
    await stream_manager.startup()
    app.state.stream_manager = stream_manager

    # CV models "loaded" flag — for /readyz. We consider them loaded
    # if the EnrollmentService has a non-None extractor, OR if we're in
    # an environment where the CV core is intentionally absent (tests).
    app.state.models_loaded = (
        enrollment_service.extractor is not None
        and enrollment_service.detector is not None
    )
    app.state.gpu_enabled = settings.gpu_enabled
    app.state.model_name = settings.model_name
    app.state.started_at = time.time()

    # Auto-create tables for SQLite (test/CI convenience). In
    # production, Alembic migrations run before the app starts (see
    # deploy/docker/entrypoint.sh).
    if settings.database_url.startswith("sqlite"):
        try:
            from app.db.session import create_all as _create_all

            await _create_all()
            logger.info("app.startup.sqlite_tables_created")
        except Exception as exc:
            logger.warning("app.startup.sqlite_create_failed", error=str(exc))

    logger.info("app.startup.complete", models_loaded=app.state.models_loaded)

    yield

    # --- Shutdown ---
    logger.info("app.shutdown")
    try:
        await stream_manager.shutdown()
    except Exception as exc:
        logger.warning("app.shutdown.stream_manager_failed", error=str(exc))
    if redis_client is not None:
        try:
            await redis_client.close()
        except Exception:
            pass
    await dispose_engine()
    logger.info("app.shutdown.complete")


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    """Build the FastAPI application instance."""
    app = FastAPI(
        title="Facial Attendance & Motion Tracking System",
        description=(
            "Real-time facial attendance with InsightFace + ByteTrack + "
            "pgvector. Supports two strategy modes (Detection-only / "
            "Tracking-only / Both) and two input sources (Upload video / "
            "Live camera)."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus instrumentation
    if settings.prometheus_enabled:
        try:
            from prometheus_fastapi_instrumentator import Instrumentator

            Instrumentator(
                should_group_status_codes=True,
                should_ignore_untemplated=True,
                should_respect_env_var=False,
                excluded_handlers=["/metrics"],
            ).instrument(app).expose(
                app,
                endpoint=settings.prometheus_path,
                include_in_schema=False,
                tags=["metrics"],
            )
            logger.info("app.prometheus.enabled", path=settings.prometheus_path)
        except Exception as exc:
            logger.warning("app.prometheus.unavailable", error=str(exc))

    # Routers — register under both /api and /api/v1 so the spec's
    # "/api/v1" prefix and the tests'/frontend's "/api" prefix both work.
    app.include_router(build_root_router())  # /health, /readyz at root
    app.include_router(build_api_router(), prefix="/api")
    app.include_router(build_v1_router(), prefix="/api/v1")
    # WebSocket endpoints at root
    app.include_router(ws_router)

    # Static frontend
    frontend_dir = settings.frontend_dir
    if frontend_dir.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(frontend_dir)),
            name="static",
        )

        # Serve index.html at "/" — explicit route so it takes precedence
        # over the catch-all below.
        @app.get("/", include_in_schema=False)
        async def _index() -> FileResponse:
            return FileResponse(str(frontend_dir / "index.html"))

        # Catch-all for SPA-style routes (none in this app, but harmless).
        @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
        async def _spa(full_path: str, request: Request):  # type: ignore[no-untyped-def]
            candidate = frontend_dir / full_path
            if full_path and candidate.is_file():
                return FileResponse(str(candidate))
            return JSONResponse(
                {"detail": "Not found"}, status_code=404
            )

    # Request logging middleware
    @app.middleware("http")
    async def _log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000.0
        # Skip health/metrics noise from logs at INFO level
        path = request.url.path
        if path not in ("/health", "/metrics", "/readyz"):
            logger.info(
                "http.request",
                method=request.method,
                path=path,
                status=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
        return response

    return app


# --------------------------------------------------------------------------- #
# Module-level app (used by ``uvicorn app.main:app``)
# --------------------------------------------------------------------------- #
app = create_app()


__all__ = ["app", "create_app"]
