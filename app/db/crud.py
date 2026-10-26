"""Async CRUD operations.

Every function accepts an :class:`AsyncSession` as the first positional
argument so that the caller controls transaction boundaries. Returns
SQLAlchemy ORM instances (or plain tuples for ``find_nearest_employee``)
— never dicts — so callers can lazy-load relationships if needed.

Key entry points:

* :func:`create_employee` — insert new Employee (raises on duplicate).
* :func:`get_employee`, :func:`get_employee_by_employee_id` — single-row
  reads by UUID / business ID.
* :func:`list_employees` — paginated listing with optional filters.
* :func:`delete_employee` — soft-delete (sets ``active=False``) by
  default; pass ``hard=True`` for a real DELETE.
* :func:`find_nearest_employee` — pgvector lookup or byte-backed cosine scan.
* :func:`log_attendance` — cooldown-aware insert.
* :func:`recently_logged` — bool check used by the cooldown path.
* :func:`get_attendance_logs`, :func:`get_attendance_stats` — for the
  REST endpoints.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from sqlalchemy import and_, delete, desc, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import AttendanceLog, Employee
from app.utils.logging import get_logger
from app.utils.metrics import attendance_logged_total, db_query_seconds

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Employees
# --------------------------------------------------------------------------- #
async def create_employee(
    session: AsyncSession,
    *,
    employee_id: str,
    name: str,
    department: str | None = None,
    email: str | None = None,
    embedding: Any | None = None,
    face_image_path: str | None = None,
    embedding_model: str | None = None,
    active: bool = True,
) -> Employee:
    """Insert a new employee.

    Raises ``IntegrityError`` (HTTP 409 in the API layer) if the
    ``employee_id`` is already taken.
    """
    emp = Employee(
        employee_id=employee_id,
        name=name,
        department=department,
        email=email,
        embedding=embedding,
        face_image_path=face_image_path,
        embedding_model=embedding_model or settings.model_name,
        active=active,
    )
    session.add(emp)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise IntegrityError(
            statement=exc.statement,
            params=exc.params,
            orig=exc.orig,
        ) from exc
    logger.info("crud.employee.created", employee_id=employee_id, name=name)
    return emp


async def get_employee(
    session: AsyncSession,
    employee_pk: uuid.UUID | str,
    *,
    include_inactive: bool = False,
) -> Employee | None:
    """Fetch by primary-key UUID.

    By default inactive (soft-deleted) employees are excluded — they
    appear "not found" to callers. Pass ``include_inactive=True`` to
    retrieve them anyway (e.g. for admin audit views).
    """
    try:
        pk_uuid = uuid.UUID(str(employee_pk)) if not isinstance(employee_pk, uuid.UUID) else employee_pk
    except (ValueError, TypeError):
        return None
    stmt = select(Employee).where(Employee.id == pk_uuid)
    if not include_inactive:
        stmt = stmt.where(Employee.active.is_(True))
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_employee_by_employee_id(
    session: AsyncSession,
    employee_id: str,
    *,
    include_inactive: bool = False,
) -> Employee | None:
    """Fetch by business ``employee_id`` (e.g. ``EMP001``).

    Soft-deleted (``active=False``) employees are excluded by default.
    """
    stmt = select(Employee).where(Employee.employee_id == employee_id)
    if not include_inactive:
        stmt = stmt.where(Employee.active.is_(True))
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_employees(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    department: str | None = None,
    q: str | None = None,
    include_inactive: bool = False,
) -> list[Employee]:
    """List employees with optional filters and pagination."""
    stmt = select(Employee)
    if not include_inactive:
        stmt = stmt.where(Employee.active.is_(True))
    if department:
        stmt = stmt.where(Employee.department == department)
    if q:
        like = f"{q}%"
        stmt = stmt.where(
            (Employee.employee_id.ilike(like)) | (Employee.name.ilike(like))
        )
    stmt = stmt.order_by(Employee.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def count_employees(
    session: AsyncSession,
    *,
    department: str | None = None,
    q: str | None = None,
    include_inactive: bool = False,
) -> int:
    """Total employee count (matching the same filters)."""
    stmt = select(func.count(Employee.id))
    if not include_inactive:
        stmt = stmt.where(Employee.active.is_(True))
    if department:
        stmt = stmt.where(Employee.department == department)
    if q:
        like = f"{q}%"
        stmt = stmt.where(
            (Employee.employee_id.ilike(like)) | (Employee.name.ilike(like))
        )
    result = await session.execute(stmt)
    return int(result.scalar() or 0)


async def update_employee(
    session: AsyncSession,
    employee_pk: uuid.UUID,
    **fields: Any,
) -> Employee | None:
    """Patch an employee's columns. Returns the updated row or None."""
    if not fields:
        return await get_employee(session, employee_pk)
    await session.execute(
        update(Employee).where(Employee.id == employee_pk).values(**fields)
    )
    await session.flush()
    return await get_employee(session, employee_pk)


async def delete_employee(
    session: AsyncSession,
    employee_pk: uuid.UUID | str,
    *,
    hard: bool = False,
) -> bool:
    """Delete an employee.

    By default this is a *soft* delete (sets ``active=False``) so that
    historical attendance logs remain intact and the employee can be
    re-activated. Pass ``hard=True`` for a real DELETE (cascades to
    attendance logs via the FK ``ondelete=CASCADE``).
    """
    emp = await get_employee(session, employee_pk)
    if emp is None:
        return False
    if hard:
        await session.execute(
            delete(Employee).where(Employee.id == emp.id)
        )
    else:
        await session.execute(
            update(Employee).where(Employee.id == emp.id).values(active=False)
        )
    await session.flush()
    logger.info("crud.employee.deleted", employee_pk=str(emp.id), hard=hard)
    return True


# --------------------------------------------------------------------------- #
# Embedding similarity search
# --------------------------------------------------------------------------- #
async def find_nearest_employee(
    session: AsyncSession,
    embedding: Any,
    *,
    threshold: float | None = None,
    limit: int = 1,
) -> tuple[Employee | None, float]:
    """Find the nearest enrolled employee by cosine similarity.

    Uses pgvector's ``<=>`` (cosine distance) operator when enabled.
    Returns ``(employee, similarity)`` where ``similarity = 1 - distance``
    is in ``[0, 1]``. If no match above ``threshold`` (default
    ``settings.similarity_threshold``), returns ``(None, score)``.

    When pgvector is disabled or unavailable, scans active employees and
    computes cosine similarity in Python. This fallback is suitable for
    small local deployments; pgvector is recommended for large datasets.
    """
    threshold_val = threshold if threshold is not None else settings.similarity_threshold
    dialect_name = session.bind.dialect.name if session.bind else "unknown"
    import time as _time

    t0 = _time.monotonic()
    if dialect_name == "postgresql" and settings.use_pgvector:
        try:
            from pgvector.sqlalchemy import Vector  # noqa: F401
        except ImportError:  # pragma: no cover
            logger.warning("crud.find_nearest.pgvector_package_missing")
        else:
            stmt = (
                select(
                    Employee,
                    (1 - Employee.embedding.cosine_distance(embedding)).label("similarity"),
                )
                .where(Employee.active.is_(True))
                .where(Employee.embedding.is_not(None))
                .order_by(Employee.embedding.cosine_distance(embedding))
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.all()
            db_query_seconds.labels(operation="find_nearest").observe(
                _time.monotonic() - t0
            )
            if not rows:
                return None, 0.0
            emp, sim = rows[0]
            sim_val = float(sim) if sim is not None else 0.0
            if sim_val < threshold_val:
                return None, sim_val
            return emp, sim_val

    import numpy as np

    query_embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    query_norm = float(np.linalg.norm(query_embedding))
    if query_norm == 0:
        return None, 0.0
    query_embedding /= query_norm

    result = await session.execute(
        select(Employee)
        .where(Employee.active.is_(True))
        .where(Employee.embedding.is_not(None))
    )
    best_employee: Employee | None = None
    best_similarity = -1.0
    for employee in result.scalars():
        stored = employee.embedding
        if stored is None:
            continue
        candidate = np.asarray(stored, dtype=np.float32).reshape(-1)
        if candidate.shape != query_embedding.shape:
            logger.warning(
                "crud.find_nearest.invalid_embedding",
                employee_id=employee.employee_id,
                dimensions=int(candidate.size),
            )
            continue
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm == 0:
            continue
        similarity = float(np.dot(query_embedding, candidate / candidate_norm))
        if similarity > best_similarity:
            best_employee = employee
            best_similarity = similarity

    db_query_seconds.labels(operation="find_nearest").observe(_time.monotonic() - t0)
    if best_employee is None:
        return None, 0.0
    if best_similarity < threshold_val:
        return None, best_similarity
    return best_employee, best_similarity


# --------------------------------------------------------------------------- #
# Attendance logging
# --------------------------------------------------------------------------- #
async def recently_logged(
    session: AsyncSession,
    *,
    employee_id: str,
    cooldown_seconds: int | None = None,
    now: datetime | None = None,
) -> bool:
    """Return True if an attendance log exists for ``employee_id`` within
    the cooldown window (default ``settings.cooldown_seconds``).

    ``employee_id`` may be either the business ID string (``EMP001``)
    or the UUID PK — both are looked up.
    """
    cd = cooldown_seconds if cooldown_seconds is not None else settings.cooldown_seconds
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=cd)

    # If the caller passed a UUID, resolve it to the business ID.
    emp_id_filter = str(employee_id)
    try:
        uuid.UUID(emp_id_filter)
        # It's a UUID — look up the business ID, fall back to also
        # checking the employee_pk column.
        emp = await get_employee(session, emp_id_filter, include_inactive=True)
        if emp is not None:
            emp_id_filter = emp.employee_id
        else:
            # Unknown UUID — no log can exist.
            return False
    except (ValueError, AttributeError, TypeError):
        pass

    stmt = (
        select(AttendanceLog.id)
        .where(
            and_(
                AttendanceLog.employee_id == emp_id_filter,
                AttendanceLog.detected_at > cutoff,
            )
        )
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.first() is not None


async def log_attendance(
    session: AsyncSession,
    *,
    employee_id: str,
    employee_pk: uuid.UUID | str | None = None,
    detected_at: datetime | None = None,
    timestamp: datetime | None = None,  # alias for detected_at (test-compat)
    similarity: float | None = None,
    track_id: int | None = None,
    camera_id: str | None = None,
    strategy: str = "both",
    liveness_passed: bool | None = None,
    liveness_confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
    cooldown_seconds: int | None = None,
) -> tuple[AttendanceLog, bool]:
    """Insert an attendance log, respecting the per-employee cooldown.

    Returns ``(log, inserted)`` where ``inserted=False`` means an
    existing log within the cooldown window was returned instead (no
    new row written).

    ``timestamp`` is accepted as an alias for ``detected_at`` for
    backwards compatibility with callers that use the older name.
    """
    ts = detected_at or timestamp or datetime.now(timezone.utc)

    # If the caller passed a UUID-shaped value as ``employee_id`` (some
    # callers confuse it with the PK), resolve the actual business ID.
    employee_id_str = str(employee_id)
    is_uuid = False
    try:
        uuid.UUID(employee_id_str)
        is_uuid = True
    except (ValueError, AttributeError, TypeError):
        pass
    if is_uuid:
        # ``employee_id`` is actually a UUID — treat as employee_pk.
        if employee_pk is None:
            employee_pk = employee_id_str
        emp_row = await get_employee(
            session, employee_id_str, include_inactive=True
        )
        if emp_row is not None:
            employee_id = emp_row.employee_id
        else:
            raise ValueError(f"Unknown employee PK: {employee_id_str!r}")

    # Resolve employee_pk if not provided
    if employee_pk is None:
        emp = await get_employee_by_employee_id(session, employee_id)
        if emp is None:
            raise ValueError(f"Unknown employee_id: {employee_id!r}")
        employee_pk = emp.id

    # Cooldown check
    if await recently_logged(
        session,
        employee_id=employee_id,
        cooldown_seconds=cooldown_seconds,
        now=ts,
    ):
        # Return the most-recent existing log within the window.
        stmt = (
            select(AttendanceLog)
            .where(
                and_(
                    AttendanceLog.employee_id == employee_id,
                    AttendanceLog.detected_at > ts - timedelta(seconds=cooldown_seconds or settings.cooldown_seconds),
                )
            )
            .order_by(desc(AttendanceLog.detected_at))
            .limit(1)
        )
        result = await session.execute(stmt)
        existing = result.scalars().first()
        if existing is not None:
            attendance_logged_total.labels(strategy=strategy, result="deduplicated").inc()
            return existing, False

    log = AttendanceLog(
        employee_pk=employee_pk if isinstance(employee_pk, uuid.UUID) else uuid.UUID(str(employee_pk)),
        employee_id=employee_id,
        detected_at=ts,
        similarity=similarity,
        track_id=track_id,
        camera_id=camera_id,
        strategy=strategy,
        liveness_passed=liveness_passed,
        liveness_confidence=liveness_confidence,
        metadata_=metadata,
    )
    session.add(log)
    await session.flush()
    attendance_logged_total.labels(strategy=strategy, result="inserted").inc()
    logger.info(
        "crud.attendance.logged",
        employee_id=employee_id,
        track_id=track_id,
        similarity=similarity,
        strategy=strategy,
        liveness_passed=liveness_passed,
    )
    return log, True


# --------------------------------------------------------------------------- #
# Attendance list / stats / export
# --------------------------------------------------------------------------- #
async def get_attendance_logs(
    session: AsyncSession,
    *,
    employee_id: str | None = None,
    date: datetime | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    strategy: str | None = None,
    department: str | None = None,
    camera_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[AttendanceLog], int]:
    """Paginated attendance logs with filters.

    Returns ``(items, total)``. ``total`` is the count before pagination
    so the caller can render total pages.
    """
    conds = []
    if employee_id:
        conds.append(AttendanceLog.employee_id == employee_id)
    if date:
        conds.append(
            func.date(AttendanceLog.detected_at) == func.date(date)
        )
    if date_from:
        conds.append(AttendanceLog.detected_at >= date_from)
    if date_to:
        conds.append(AttendanceLog.detected_at <= date_to)
    if strategy:
        conds.append(AttendanceLog.strategy == strategy)
    if camera_id:
        conds.append(AttendanceLog.camera_id == camera_id)

    base = select(AttendanceLog)
    count_stmt = select(func.count(AttendanceLog.id))

    if department:
        # Need a join with employees for the department filter.
        base = base.join(Employee, AttendanceLog.employee_pk == Employee.id)
        count_stmt = count_stmt.join(Employee, AttendanceLog.employee_pk == Employee.id)
        conds.append(Employee.department == department)

    if conds:
        base = base.where(and_(*conds))
        count_stmt = count_stmt.where(and_(*conds))

    base = base.order_by(desc(AttendanceLog.detected_at)).limit(limit).offset(offset)

    items_result = await session.execute(base)
    items = list(items_result.scalars().all())
    total_result = await session.execute(count_stmt)
    total = int(total_result.scalar() or 0)
    return items, total


async def get_attendance_stats(
    session: AsyncSession,
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    range_str: str | None = None,
) -> dict[str, Any]:
    """Aggregate stats for the dashboard charts.

    Returns a dict with three sections:

    * ``daily`` — list of ``{date, count}`` for the requested range.
    * ``hourly`` — list of ``{hour, count}`` (24 entries, 0-23).
    * ``departments`` — list of ``{department, count}``.
    """
    # Resolve range if a string ("7d"/"30d"/"90d") is passed.
    if range_str and not date_from:
        days = {"7d": 7, "30d": 30, "90d": 90}.get(range_str, 7)
        date_from = datetime.now(timezone.utc) - timedelta(days=days)
    if not date_to:
        date_to = datetime.now(timezone.utc)

    conds = [AttendanceLog.detected_at >= date_from, AttendanceLog.detected_at <= date_to]
    base_filter = and_(*conds)

    # Detect dialect so we can use the correct date-part functions.
    # Postgres has ``date_trunc`` + ``extract``; SQLite only has
    # ``strftime``. We pick the right SQL for each.
    dialect_name = session.bind.dialect.name if session.bind else "unknown"
    is_postgres = dialect_name == "postgresql"

    # Daily counts -------------------------------------------------------- #
    if is_postgres:
        day_expr = func.date_trunc("day", AttendanceLog.detected_at).label("day")
        daily_stmt = (
            select(day_expr, func.count(AttendanceLog.id).label("count"))
            .where(base_filter)
            .group_by("day")
            .order_by("day")
        )
        daily_rows = (await session.execute(daily_stmt)).all()
        daily = [
            {
                "date": (r.day.date().isoformat() if r.day is not None else None),
                "count": int(r.count or 0),
            }
            for r in daily_rows
        ]
    else:
        # SQLite: strftime('%Y-%m-%d', ...) returns the day as text.
        day_expr = func.strftime("%Y-%m-%d", AttendanceLog.detected_at).label("day")
        daily_stmt = (
            select(day_expr, func.count(AttendanceLog.id).label("count"))
            .where(base_filter)
            .group_by("day")
            .order_by("day")
        )
        daily_rows = (await session.execute(daily_stmt)).all()
        daily = [
            {"date": (r.day or None), "count": int(r.count or 0)}
            for r in daily_rows
        ]

    # Hourly counts (0..23) ---------------------------------------------- #
    hourly_map: dict[int, int] = {}
    try:
        if is_postgres:
            hour_expr = func.extract("hour", AttendanceLog.detected_at).label("hour")
        else:
            # SQLite: strftime('%H', ...) returns the hour as text.
            hour_expr = func.strftime("%H", AttendanceLog.detected_at).label("hour")
        hourly_stmt = (
            select(hour_expr, func.count(AttendanceLog.id).label("count"))
            .where(base_filter)
            .group_by("hour")
            .order_by("hour")
        )
        hourly_rows = (await session.execute(hourly_stmt)).all()
        for r in hourly_rows:
            if r.hour is None:
                continue
            try:
                h = int(r.hour)
            except (TypeError, ValueError):
                continue
            hourly_map[h] = int(r.count or 0)
    except Exception as exc:  # pragma: no cover — dialect-specific
        logger.warning("crud.stats.hourly_failed", error=str(exc))
    hourly = [{"hour": h, "count": hourly_map.get(h, 0)} for h in range(24)]

    # Department breakdown
    dept_stmt = (
        select(
            func.coalesce(Employee.department, "Unassigned").label("department"),
            func.count(AttendanceLog.id).label("count"),
        )
        .join(Employee, AttendanceLog.employee_pk == Employee.id)
        .where(base_filter)
        .group_by("department")
        .order_by(desc("count"))
    )
    dept_rows = (await session.execute(dept_stmt)).all()
    departments = [
        {"department": r.department, "count": int(r.count or 0)} for r in dept_rows
    ]

    total = sum(d["count"] for d in daily)
    unique_employees_stmt = (
        select(func.count(func.distinct(AttendanceLog.employee_id)))
        .where(base_filter)
    )
    unique_employees = int((await session.execute(unique_employees_stmt)).scalar() or 0)

    return {
        "range": range_str or "custom",
        "date_from": date_from.isoformat() if date_from else None,
        "date_to": date_to.isoformat() if date_to else None,
        "total": total,
        "unique_employees": unique_employees,
        "daily": daily,
        "hourly": hourly,
        "departments": departments,
    }


async def iter_attendance_logs(
    session: AsyncSession,
    *,
    employee_id: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    strategy: str | None = None,
    department: str | None = None,
    chunk_size: int = 500,
) -> Iterable[AttendanceLog]:
    """Generator yielding attendance logs in chunks (for CSV export).

    Avoids loading the entire result set into memory.
    """
    offset = 0
    while True:
        items, _ = await get_attendance_logs(
            session,
            employee_id=employee_id,
            date_from=date_from,
            date_to=date_to,
            strategy=strategy,
            department=department,
            limit=chunk_size,
            offset=offset,
        )
        if not items:
            return
        for item in items:
            yield item
        offset += chunk_size


__all__ = [
    "create_employee",
    "get_employee",
    "get_employee_by_employee_id",
    "list_employees",
    "count_employees",
    "update_employee",
    "delete_employee",
    "find_nearest_employee",
    "recently_logged",
    "log_attendance",
    "get_attendance_logs",
    "get_attendance_stats",
    "iter_attendance_logs",
]
