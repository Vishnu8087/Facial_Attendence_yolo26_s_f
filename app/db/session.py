"""Async SQLAlchemy engine + session factory.

A single :class:`AsyncEngine` is created lazily on first use; the
:class:`async_sessionmaker` provides per-request sessions. The FastAPI
dependency :func:`get_db` yields a session for a single request scope.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.engine import make_url

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Engine + sessionmaker (lazy module-level singletons)
# --------------------------------------------------------------------------- #
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> AsyncEngine:
    """Create the async engine based on ``settings.database_url``.

    For SQLite (used in unit tests) we disable pool size / recycle
    because SQLite ignores them and asyncpg-specific args would crash.
    For in-memory SQLite we use ``StaticPool`` so every connection
    shares the same DB (otherwise each connection gets a fresh empty
    DB and tables created in the lifespan are invisible to requests).
    """
    url = settings.database_url
    is_sqlite = url.startswith("sqlite")
    is_in_memory = is_sqlite and ":memory:" in url

    if is_sqlite:
        from sqlalchemy.pool import StaticPool

        return create_async_engine(
            url,
            echo=settings.db_echo,
            future=True,
            # StaticPool shares a single connection across the engine
            # so :memory: tables persist for the life of the engine.
            poolclass=StaticPool,
            connect_args={"check_same_thread": False} if is_in_memory else {},
        )

    # PostgreSQL (asyncpg) — pool sizing from settings.
    return create_async_engine(
        url,
        echo=settings.db_echo,
        future=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=True,
    )


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
        safe_url = make_url(settings.database_url).render_as_string(hide_password=True)
        logger.info("db.engine.created", url=safe_url)
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            expire_on_commit=False,
            autoflush=False,
            class_=AsyncSession,
        )
    return _session_factory


# Public aliases (mostly for backwards compat / external tooling).
async_engine = _get_engine  # callable returning the singleton engine
async_session_factory = _get_session_factory  # callable returning the singleton factory


# --------------------------------------------------------------------------- #
# FastAPI dependency
# --------------------------------------------------------------------------- #
_tables_created = False


async def _ensure_tables() -> None:
    """Lazily create tables on first DB use.

    Idempotent. In production, Alembic migrations run before the app
    starts (see ``deploy/docker/entrypoint.sh``) so this is a no-op
    because the tables already exist. In tests / dev with SQLite,
    this is what makes the API work without a manual ``create_all``
    call.
    """
    global _tables_created
    if _tables_created:
        return
    from app.db import models  # noqa: F401 — registers tables on Base.metadata
    from app.db.base import Base

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _tables_created = True


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield an :class:`AsyncSession` bound to the request scope.

    Commits on success, rolls back on exception, always closes. On the
    first call after process start, ensures tables exist (lazy
    migration for SQLite / dev; no-op in production where Alembic
    has already run).
    """
    await _ensure_tables()
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
        if session.in_transaction() and session.is_active:
            await session.commit()
    except Exception:
        if session.is_active:
            await session.rollback()
        raise
    finally:
        await session.close()


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Context-manager variant for use outside FastAPI (scripts, worker)."""
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# --------------------------------------------------------------------------- #
# Lifecycle helpers
# --------------------------------------------------------------------------- #
async def create_all() -> None:
    """Create every table registered on ``Base.metadata``.

    Convenience for tests / first-run bootstrap. Production migrations
    are managed by Alembic — see ``app/db/migrations/``.
    """
    from app.db import models  # noqa: F401 — registers tables on Base.metadata
    from app.db.base import Base

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Dispose of the engine — call on application shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("db.engine.disposed")
    _engine = None
    _session_factory = None


__all__ = [
    "async_engine",
    "async_session_factory",
    "create_all",
    "dispose_engine",
    "get_db",
    "get_session",
]
