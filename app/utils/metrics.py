"""Prometheus metrics — counters + histograms.

The :mod:`prometheus_client` registry is global; we expose named metric
handles here that the API routes, WebSocket handler, worker, and the CV
pipeline (Task 1-a) update. The FastAPI instrumentation (HTTP request
latency / status codes) is wired up separately by
``prometheus_fastapi_instrumentator`` in :mod:`app.main`.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

# --------------------------------------------------------------------------- #
# Counters
# --------------------------------------------------------------------------- #
frames_processed_total = Counter(
    "frames_processed_total",
    "Total number of video frames processed by the pipeline.",
    ["strategy"],  # both | detection_only | tracking_only
)

faces_detected_total = Counter(
    "faces_detected_total",
    "Total number of faces detected (post-filter).",
    ["strategy"],
)

recognitions_total = Counter(
    "recognitions_total",
    "Total number of successful identity recognitions.",
    ["strategy", "outcome"],  # outcome: matched | no_match | below_threshold
)

spoof_rejected_total = Counter(
    "spoof_rejected_total",
    "Total number of detections rejected by the liveness verifier.",
    ["strategy"],
)

attendance_logged_total = Counter(
    "attendance_logged_total",
    "Total number of attendance logs actually written (post-cooldown).",
    ["strategy", "result"],  # result: inserted | deduplicated
)

ws_connections = Counter(
    "ws_connections_total",
    "Total WebSocket connections accepted on /ws/stream.",
)

ws_active = Counter(
    "ws_active_connections",
    "Currently active WebSocket connections (gauge-style via inc/dec).",
)

# --------------------------------------------------------------------------- #
# Histograms
# --------------------------------------------------------------------------- #
frame_processing_seconds = Histogram(
    "frame_processing_seconds",
    "Wall-clock time spent in AttendancePipeline.process_frame().",
    ["strategy"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0),
)

db_query_seconds = Histogram(
    "db_query_seconds",
    "Time spent in selected database queries.",
    ["operation"],  # operation: find_nearest | log_attendance | list_employees | …
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

__all__ = [
    "frames_processed_total",
    "faces_detected_total",
    "recognitions_total",
    "spoof_rejected_total",
    "attendance_logged_total",
    "ws_connections",
    "ws_active",
    "frame_processing_seconds",
    "db_query_seconds",
]
