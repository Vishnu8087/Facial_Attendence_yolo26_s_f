"""SQLAlchemy ORM models.

Two tables:

* :class:`Employee` — one row per enrolled employee. The face embedding
  is stored as pgvector when enabled, or as float32 bytes otherwise.
  The byte-backed mode supports cosine matching without the PostgreSQL
  server extension.

* :class:`AttendanceLog` — one row per *deduplicated* attendance event.
  The ``(employee_id, detected_at)`` index makes the cooldown lookup
  ``WHERE employee_id=? AND detected_at > now() - INTERVAL '8h'`` cheap.

Both tables use a UUID primary key (``uuid-ossp`` extension on
Postgres; ``uuid4`` defaults in Python for other backends).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import LargeBinary

from app.config import settings
from app.db.base import Base

# --------------------------------------------------------------------------- #
# Embedding column type — pgvector when enabled, otherwise raw float32 bytes.
# --------------------------------------------------------------------------- #
EMBEDDING_DIM = 512


class EmbeddingVector(TypeDecorator):
    """A 512-d float32 embedding column.

    With pgvector enabled on Postgres this compiles to ``vector(512)`` and
    participates in ``<=>`` cosine-distance queries. Otherwise it stores
    raw float32 bytes, including on stock PostgreSQL installations without
    the pgvector server extension.
    """

    impl = LargeBinary
    cache_ok = True

    def load_dialect_impl(self, dialect):  # type: ignore[override]
        if dialect.name == "postgresql" and settings.use_pgvector:
            try:
                from pgvector.sqlalchemy import Vector  # type: ignore
            except ImportError:  # pragma: no cover — defensive
                return dialect.type_descriptor(LargeBinary())
            return dialect.type_descriptor(Vector(EMBEDDING_DIM))
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value, dialect):  # type: ignore[override]
        if value is None:
            return None
        # pgvector's Vector type accepts arrays directly; byte-backed
        # databases store a compact float32 representation instead.
        if dialect.name == "postgresql" and settings.use_pgvector:
            return value
        try:
            import numpy as np

            arr = np.asarray(value, dtype=np.float32).tobytes()
            return arr
        except Exception:
            # value may already be bytes
            return value

    def process_result_value(self, value, dialect):  # type: ignore[override]
        if value is None:
            return None
        if dialect.name == "postgresql" and settings.use_pgvector:
            # pgvector returns the embedding as a list-like; pass through.
            return value
        try:
            import numpy as np

            return np.frombuffer(value, dtype=np.float32).copy()
        except Exception:
            return value


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


# --------------------------------------------------------------------------- #
# Employee
# --------------------------------------------------------------------------- #
class Employee(Base):
    """An enrolled employee.

    The ``embedding`` column carries a 512-d L2-normalised ArcFace
    vector. ``face_image_path`` points at the saved face crop on disk
    (under ``settings.faces_dir``). ``active`` is a soft-delete flag —
    when False, the employee is excluded from matching but historical
    attendance logs are retained.
    """

    __tablename__ = "employees"

    if False:  # placeholder so static analysers see a UUID type
        id: Mapped[Any]

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    employee_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        doc="Business identifier (e.g. EMP001).",
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    department: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding: Mapped[Any] = mapped_column(
        EmbeddingVector(),
        nullable=True,
        doc="512-d L2-normalised face embedding.",
    )
    face_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc="Model name used to extract the embedding (e.g. buffalo_l).",
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )

    attendance_logs: Mapped[list["AttendanceLog"]] = relationship(
        "AttendanceLog",
        back_populates="employee",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("employee_id", name="uq_employees_employee_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return f"<Employee {self.employee_id} {self.name!r}>"


# --------------------------------------------------------------------------- #
# AttendanceLog
# --------------------------------------------------------------------------- #
class AttendanceLog(Base):
    """A single deduplicated attendance event.

    The cooldown logic (see :func:`app.db.crud.log_attendance`) ensures
    that two events for the same employee within
    ``settings.cooldown_seconds`` (default 8h) only produce one row —
    the second call returns the existing row.
    """

    __tablename__ = "attendance_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    employee_pk: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="FK to employees.id (the UUID PK, not employee_id string).",
    )
    employee_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Denormalised business employee_id for fast filtering.",
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        index=True,
    )
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    camera_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    strategy: Mapped[str] = mapped_column(String(32), default="both", nullable=False)
    liveness_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    liveness_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSONB().with_variant(LargeBinary(), "sqlite"),
        nullable=True,
        doc="Strategy-specific extras (blink count, head pose, etc.).",
    )

    employee: Mapped[Employee] = relationship("Employee", back_populates="attendance_logs")

    __table_args__ = (
        # Hot path: cooldown query scans WHERE employee_id=? AND detected_at > now() - cooldown
        Index(
            "ix_attendance_logs_employee_detected",
            "employee_id",
            "detected_at",
        ),
        Index("ix_attendance_logs_detected_at_strategy", "detected_at", "strategy"),
    )

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return f"<AttendanceLog {self.employee_id} @ {self.detected_at}>"


__all__ = ["Employee", "AttendanceLog", "EmbeddingVector", "EMBEDDING_DIM"]
