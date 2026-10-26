"""Attendance logging service.

Wraps :func:`app.db.crud.log_attendance` with an in-process TTL cache
so the cooldown check is O(1) for hot employees and doesn't hit the
DB on every frame. The cache is best-effort — the DB remains the source
of truth; on a cold cache we fall through to the SQL query.

In multi-process deployments (uvicorn workers > 1) the cache is
per-process; misses just hit the DB and find the existing row. For
multi-node deduplication you can swap the cache for Redis by
implementing the same interface.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import crud
from app.utils.logging import get_logger
from app.utils.metrics import attendance_logged_total

logger = get_logger(__name__)


@dataclass
class _CacheEntry:
    last_logged_at: float  # epoch seconds
    log_id: str
    extras: dict[str, Any] = field(default_factory=dict)


class AttendanceLogger:
    """Cooldown-aware attendance logger with an in-memory TTL cache.

    Usage::

        logger = AttendanceLogger()
        async with get_session() as session:
            log, inserted = await logger.log_attendance(
                session,
                employee_id="EMP001",
                similarity=0.82,
                track_id=7,
                strategy="both",
                liveness_passed=True,
            )
    """

    def __init__(
        self,
        *,
        cooldown_seconds: int | None = None,
        cache_ttl_padding: float = 1.5,
    ) -> None:
        self.cooldown_seconds = (
            cooldown_seconds if cooldown_seconds is not None else settings.cooldown_seconds
        )
        # Padding so the cache entry expires slightly before the DB
        # cooldown — avoids a race where the cache says "still cooling
        # down" but the DB has already allowed a new log.
        self._cache_ttl_padding = cache_ttl_padding
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #
    async def log_attendance(
        self,
        session: AsyncSession,
        *,
        employee_id: str,
        employee_pk: Any | None = None,
        similarity: float | None = None,
        track_id: int | None = None,
        camera_id: str | None = None,
        strategy: str = "both",
        liveness_passed: bool | None = None,
        liveness_confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
        detected_at: datetime | None = None,
    ) -> tuple[Any, bool]:
        """Log an attendance event, respecting per-employee cooldown.

        Returns ``(log, inserted)`` where ``inserted=False`` means the
        event was deduplicated (an existing log within the cooldown was
        returned).
        """
        ts = detected_at or datetime.now(timezone.utc)
        # 1. Hot path — check in-memory cache.
        if self._is_cached_within_cooldown(employee_id, ts):
            attendance_logged_total.labels(strategy=strategy, result="deduplicated").inc()
            cached = self._cache.get(employee_id)
            # Re-fetch from DB to return a proper ORM object.
            existing = await self._fetch_most_recent(session, employee_id, ts)
            if existing is not None:
                return existing, False
            # Cache stale — fall through to DB check.

        # 2. Cold path — DB cooldown check + insert.
        log, inserted = await crud.log_attendance(
            session,
            employee_id=employee_id,
            employee_pk=employee_pk,
            detected_at=ts,
            similarity=similarity,
            track_id=track_id,
            camera_id=camera_id,
            strategy=strategy,
            liveness_passed=liveness_passed,
            liveness_confidence=liveness_confidence,
            metadata=metadata,
            cooldown_seconds=self.cooldown_seconds,
        )

        # 3. Update cache.
        if inserted:
            async with self._lock:
                self._cache[employee_id] = _CacheEntry(
                    last_logged_at=ts.timestamp(),
                    log_id=str(log.id),
                )
        return log, inserted

    async def recently_logged(
        self,
        session: AsyncSession,
        *,
        employee_id: str,
        now: datetime | None = None,
    ) -> bool:
        """True if a log exists for ``employee_id`` within the cooldown."""
        ts = (now or datetime.now(timezone.utc)).timestamp()
        if self._is_cached_within_cooldown(employee_id, datetime.fromtimestamp(ts, tz=timezone.utc)):
            return True
        return await crud.recently_logged(
            session,
            employee_id=employee_id,
            cooldown_seconds=self.cooldown_seconds,
            now=datetime.fromtimestamp(ts, tz=timezone.utc) if now is None else now,
        )

    def invalidate(self, employee_id: str | None = None) -> None:
        """Drop the cache entry for ``employee_id`` (or all entries if None)."""
        if employee_id is None:
            self._cache.clear()
        else:
            self._cache.pop(employee_id, None)

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #
    def _is_cached_within_cooldown(self, employee_id: str, now: datetime) -> bool:
        entry = self._cache.get(employee_id)
        if entry is None:
            return False
        elapsed = now.timestamp() - entry.last_logged_at
        return elapsed < (self.cooldown_seconds - self._cache_ttl_padding)

    async def _fetch_most_recent(
        self, session: AsyncSession, employee_id: str, now: datetime
    ) -> Any | None:
        from datetime import timedelta

        from sqlalchemy import and_, desc, select

        from app.db.models import AttendanceLog

        stmt = (
            select(AttendanceLog)
            .where(
                and_(
                    AttendanceLog.employee_id == employee_id,
                    AttendanceLog.detected_at
                    > now - timedelta(seconds=self.cooldown_seconds),
                )
            )
            .order_by(desc(AttendanceLog.detected_at))
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalars().first()


__all__ = ["AttendanceLogger"]
