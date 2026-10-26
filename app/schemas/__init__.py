"""Pydantic v2 schemas (request/response models)."""
from __future__ import annotations

from app.schemas.attendance import (
    AttendanceListResponse,
    AttendanceLogResponse,
    AttendanceStats,
    RecognizeRequest,
    RecognizeResponse,
    RecognizedIdentity,
)
from app.schemas.common import ErrorResponse, HealthResponse, ReadyzResponse, PaginatedMeta
from app.schemas.employee import (
    EmployeeCreate,
    EmployeeCreateJSON,
    EmployeeListResponse,
    EmployeeResponse,
)

__all__ = [
    # employee
    "EmployeeCreate",
    "EmployeeCreateJSON",
    "EmployeeResponse",
    "EmployeeListResponse",
    # attendance
    "AttendanceLogResponse",
    "AttendanceListResponse",
    "AttendanceStats",
    "RecognizeRequest",
    "RecognizeResponse",
    "RecognizedIdentity",
    # common
    "ErrorResponse",
    "HealthResponse",
    "ReadyzResponse",
    "PaginatedMeta",
]
