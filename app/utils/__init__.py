"""Utility sub-package — logging, metrics, security helpers."""
from __future__ import annotations

from app.utils.logging import configure_logging, get_logger
from app.utils.metrics import (
    attendance_logged_total,
    faces_detected_total,
    frame_processing_seconds,
    frames_processed_total,
    recognitions_total,
    spoof_rejected_total,
)
from app.utils.security import (
    APIKeyHeaderDep,
    CurrentUser,
    HTTPBearerDep,
    get_current_user,
    require_admin,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "frames_processed_total",
    "faces_detected_total",
    "recognitions_total",
    "spoof_rejected_total",
    "attendance_logged_total",
    "frame_processing_seconds",
    "HTTPBearerDep",
    "APIKeyHeaderDep",
    "CurrentUser",
    "get_current_user",
    "require_admin",
]
