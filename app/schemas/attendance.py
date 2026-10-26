"""Attendance-related schemas (Pydantic v2)."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Attendance logs
# --------------------------------------------------------------------------- #
class AttendanceLogResponse(BaseModel):
    """Single attendance log entry.

    Denormalised — carries the employee's business ``employee_id`` and
    display ``name`` so the dashboard doesn't need a second round-trip.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    employee_id: str
    name: str | None = None
    detected_at: datetime
    similarity: float | None = None
    track_id: int | None = None
    camera_id: str | None = None
    strategy: str = "both"
    liveness_passed: bool | None = None
    liveness_confidence: float | None = None
    metadata: dict[str, Any] | None = None


class AttendanceListResponse(BaseModel):
    """Paginated attendance listing."""

    items: list[AttendanceLogResponse]
    total: int
    limit: int
    offset: int


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
class DailyCount(BaseModel):
    date: str
    count: int


class HourlyCount(BaseModel):
    hour: int
    count: int


class DepartmentCount(BaseModel):
    department: str
    count: int


class AttendanceStats(BaseModel):
    """Aggregate stats for the dashboard charts."""

    range: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    total: int = 0
    unique_employees: int = 0
    daily: list[DailyCount] = Field(default_factory=list)
    hourly: list[HourlyCount] = Field(default_factory=list)
    departments: list[DepartmentCount] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Recognize (REST single-image endpoint)
# --------------------------------------------------------------------------- #
class RecognizeRequest(BaseModel):
    """JSON body for ``POST /api/recognize``.

    Either ``image_base64`` (JPEG/PNG) for server-side extraction, or
    ``embedding_base64`` (512-d float32) for a pre-computed vector.
    """

    image_base64: str | None = Field(
        None, description="JPEG or PNG image, base64-encoded."
    )
    embedding_base64: str | None = Field(
        None, description="512-d L2-normalised embedding, float32 little-endian, base64-encoded."
    )
    threshold: float | None = Field(
        None, description="Override settings.similarity_threshold for this call."
    )
    strategy: str = Field(
        "both", description="Strategy used by the pipeline for this image."
    )

    model_config = ConfigDict(extra="forbid")


class RecognizedIdentity(BaseModel):
    """One recognized face within a RecognizeResponse."""

    track_id: int | None = None
    bbox: list[float] | None = None
    employee_id: str | None = None
    name: str | None = None
    similarity: float
    liveness_passed: bool | None = None
    liveness_confidence: float | None = None


class RecognizeResponse(BaseModel):
    """Response for ``POST /api/recognize``."""

    identities: list[RecognizedIdentity]
    strategy: str = "both"
    latency_ms: float = 0.0
    annotated_image_b64: str | None = None


__all__ = [
    "AttendanceLogResponse",
    "AttendanceListResponse",
    "AttendanceStats",
    "DailyCount",
    "HourlyCount",
    "DepartmentCount",
    "RecognizeRequest",
    "RecognizeResponse",
    "RecognizedIdentity",
]
