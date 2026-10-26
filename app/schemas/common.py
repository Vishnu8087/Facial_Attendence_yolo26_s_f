"""Common / shared Pydantic schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PaginatedMeta(BaseModel):
    """Pagination envelope metadata."""

    total: int = Field(..., description="Total items matching the query (pre-pagination).")
    limit: int = Field(..., description="Page size requested.")
    offset: int = Field(..., description="Offset applied.")
    page: int | None = Field(None, description="1-based page number (derived).")


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: dict[str, Any] = Field(
        ..., description="{code, message, details?} describing the failure."
    )
    request_id: str | None = Field(None, description="Correlation ID for tracing.")


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = "ok"
    version: str = "0.1.0"
    uptime_seconds: float = 0.0


class DependencyStatus(BaseModel):
    status: str = "ok"  # ok | degraded | down | skip
    latency_ms: float | None = None
    error: str | None = None
    detail: dict[str, Any] | None = None


class ReadyzResponse(BaseModel):
    """Readiness probe response — reports each dependency."""

    status: str = "ok"  # ok | unavailable
    database: DependencyStatus = Field(default_factory=DependencyStatus)
    models: DependencyStatus = Field(default_factory=DependencyStatus)
    cache: DependencyStatus = Field(default_factory=DependencyStatus)
    checked_at: datetime | None = None
