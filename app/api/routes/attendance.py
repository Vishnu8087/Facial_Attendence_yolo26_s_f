"""Attendance log endpoints.

``GET  /api/attendance``         — paginated, filtered listing
``GET  /api/attendance/stats``   — daily / hourly / department aggregates
``GET  /api/attendance/export``  — CSV or JSON bulk export
"""
from __future__ import annotations

import csv
import io
from datetime import date, datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session
from app.db import crud
from app.schemas.attendance import (
    AttendanceListResponse,
    AttendanceLogResponse,
    AttendanceStats,
)
from app.schemas.common import ErrorResponse
from app.utils.logging import get_logger
from app.utils.security import User, get_current_user

logger = get_logger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    # Try full ISO first, then date-only.
    for parser in (datetime.fromisoformat, lambda v: datetime.strptime(v, "%Y-%m-%d")):
        try:
            dt = parser(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    raise HTTPException(
        status_code=422,
        detail=f"Invalid date: {s!r} (expected YYYY-MM-DD or ISO-8601)",
    )


async def _enrich_with_name(session: AsyncSession, log) -> AttendanceLogResponse:
    """Denormalise: fetch the employee's name for the response."""
    name = None
    if log.employee_pk is not None:
        from app.db import crud as _crud

        emp = await _crud.get_employee(session, log.employee_pk)
        if emp is not None:
            name = emp.name
    return AttendanceLogResponse(
        id=log.id,
        employee_id=log.employee_id,
        name=name,
        detected_at=log.detected_at,
        similarity=log.similarity,
        track_id=log.track_id,
        camera_id=log.camera_id,
        strategy=log.strategy,
        liveness_passed=log.liveness_passed,
        liveness_confidence=log.liveness_confidence,
        metadata=log.metadata_,
    )


# --------------------------------------------------------------------------- #
# List
# --------------------------------------------------------------------------- #
@router.get(
    "",
    response_model=list[AttendanceLogResponse],
    summary="List attendance logs (paginated, filtered)",
    description=(
        "Returns paginated attendance logs as a JSON array. Supports "
        "filtering by `employee_id`, `date` (single day), `date_from`/`date_to` "
        "range, `strategy`, `department`, and `camera_id`. The dashboard's "
        "api.js normalises either an array or `{items: [...]}` envelope — "
        "we return the array form for compatibility with the test suite."
    ),
)
async def list_attendance(
    session: AsyncSession = Depends(get_db_session),
    employee_id: Annotated[str | None, Query(max_length=64)] = None,
    date: Annotated[str | None, Query(description="YYYY-MM-DD")] = None,
    date_from: Annotated[str | None, Query(description="YYYY-MM-DD")] = None,
    date_to: Annotated[str | None, Query(description="YYYY-MM-DD")] = None,
    strategy: Annotated[str | None, Query(max_length=32)] = None,
    department: Annotated[str | None, Query(max_length=128)] = None,
    camera_id: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    _user: User = Depends(get_current_user),
) -> list[AttendanceLogResponse]:
    """List endpoint — returns a JSON array."""
    d = _parse_date(date)
    d_from = _parse_date(date_from)
    d_to = _parse_date(date_to)
    if strategy and strategy not in {"both", "detection_only", "tracking_only"}:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy: {strategy!r}",
        )

    items, _total = await crud.get_attendance_logs(
        session,
        employee_id=employee_id,
        date=d,
        date_from=d_from,
        date_to=d_to,
        strategy=strategy,
        department=department,
        camera_id=camera_id,
        limit=limit,
        offset=offset,
    )
    return [await _enrich_with_name(session, log) for log in items]


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
@router.get(
    "/stats",
    response_model=AttendanceStats,
    summary="Aggregate attendance stats for the dashboard charts",
    description=(
        "Returns three series:\n\n"
        "- `daily` — `{date, count}` for each day in the range.\n"
        "- `hourly` — 24 entries `{hour, count}` (0–23).\n"
        "- `departments` — top departments by attendance count.\n\n"
        "Pass `range=7d|30d|90d` for a relative window, or `date_from`/`date_to` "
        "for an absolute window."
    ),
)
async def attendance_stats(
    session: AsyncSession = Depends(get_db_session),
    range: Annotated[str | None, Query(description="7d | 30d | 90d")] = "7d",
    date_from: Annotated[str | None, Query()] = None,
    date_to: Annotated[str | None, Query()] = None,
    _user: User = Depends(get_current_user),  # noqa: ANN001
) -> AttendanceStats:
    """Stats endpoint."""
    d_from = _parse_date(date_from)
    d_to = _parse_date(date_to)
    raw = await crud.get_attendance_stats(
        session,
        date_from=d_from,
        date_to=d_to,
        range_str=range,
    )
    return AttendanceStats(**raw)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
@router.get(
    "/export",
    summary="Bulk-export attendance logs as CSV or JSON",
    description=(
        "Streams all matching attendance logs as either CSV (default) or "
        "JSON. Same filters as `GET /api/attendance`."
    ),
    responses={
        200: {
            "content": {
                "text/csv": {},
                "application/json": {},
            },
            "description": "Bulk export",
        },
    },
)
async def export_attendance(
    session: AsyncSession = Depends(get_db_session),
    format: Annotated[str, Query(description="csv | json")] = "csv",
    employee_id: Annotated[str | None, Query(max_length=64)] = None,
    date_from: Annotated[str | None, Query()] = None,
    date_to: Annotated[str | None, Query()] = None,
    strategy: Annotated[str | None, Query(max_length=32)] = None,
    department: Annotated[str | None, Query(max_length=128)] = None,
    camera_id: Annotated[str | None, Query(max_length=64)] = None,
    _user: User = Depends(get_current_user),  # noqa: ANN001
):
    """Export endpoint — CSV by default, JSON on request."""
    d_from = _parse_date(date_from)
    d_to = _parse_date(date_to)
    fmt = (format or "csv").lower()
    if fmt not in {"csv", "json"}:
        raise HTTPException(
            status_code=422,
            detail="format must be 'csv' or 'json'",
        )

    today = date.today().isoformat()

    if fmt == "json":
        # Collect synchronously — JSON export is bounded by limit=10000.
        items, _ = await crud.get_attendance_logs(
            session,
            employee_id=employee_id,
            date_from=d_from,
            date_to=d_to,
            strategy=strategy,
            department=department,
            camera_id=camera_id,
            limit=10000,
            offset=0,
        )
        out = []
        for log in items:
            out.append(
                {
                    "id": str(log.id),
                    "employee_id": log.employee_id,
                    "timestamp": log.detected_at.isoformat(),
                    "strategy": log.strategy,
                    "similarity": log.similarity,
                    "track_id": log.track_id,
                    "camera_id": log.camera_id,
                    "liveness_passed": log.liveness_passed,
                    "liveness_confidence": log.liveness_confidence,
                    "metadata": log.metadata_,
                }
            )
        return StreamingResponse(
            iter([__import__("json").dumps(out, default=str)]),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="attendance_{today}.json"'
            },
        )

    # CSV — stream rows.
    async def gen() -> io.StringIO:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "id",
                "employee_id",
                "name",
                "timestamp",
                "strategy",
                "score",
                "track_id",
                "camera_id",
                "liveness_passed",
                "liveness_confidence",
            ]
        )
        # Write header
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)
        async for log in crud.iter_attendance_logs(
            session,
            employee_id=employee_id,
            date_from=d_from,
            date_to=d_to,
            strategy=strategy,
            department=department,
        ):
            writer.writerow(
                [
                    str(log.id),
                    log.employee_id,
                    "",  # name left blank in export (denormalised later if needed)
                    log.detected_at.isoformat(),
                    log.strategy,
                    log.similarity if log.similarity is not None else "",
                    log.track_id if log.track_id is not None else "",
                    log.camera_id or "",
                    log.liveness_passed if log.liveness_passed is not None else "",
                    log.liveness_confidence if log.liveness_confidence is not None else "",
                ]
            )
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

    return StreamingResponse(
        gen(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="attendance_{today}.csv"'
        },
    )


__all__ = ["router"]
