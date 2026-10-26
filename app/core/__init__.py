"""CV core pipeline package.

Re-exports the public classes from the sub-modules so callers can do::

    from app.core import (
        AttendancePipeline,
        FaceAligner,
        FaceDetector,
        FeatureExtractor,
        LivenessVerifier,
        MotionTracker,
    )

All sub-modules are independently importable (they do not pull each
other in at import time), which keeps the test suite cheap and lets
Task 1-b's FastAPI handlers cherry-pick only what they need.
"""
from __future__ import annotations

from app.core.aligner import ARCFACE_REFERENCE_LANDMARKS_112, FaceAligner
from app.core.detector import DetectionResult, FaceDetector
from app.core.extractor import ExtractionResult, FeatureExtractor
from app.core.liveness import LivenessResult, LivenessVerifier
from app.core.pipeline import (
    STRATEGY_BOTH,
    STRATEGY_DETECTION_ONLY,
    STRATEGY_TRACKING_ONLY,
    AttendancePipeline,
    FrameResult,
    TrackResult,
)
from app.core.tracker import MotionTracker, Track
from app.core.utils import (
    TimedResult,
    compute_ear,
    cosine_similarity,
    crop_face,
    draw_boxes,
    draw_landmarks,
    euler_from_landmarks,
    get_logger,
    read_image_bytes,
    resize_keep_aspect,
)

__all__ = [
    # Detector
    "FaceDetector",
    "DetectionResult",
    # Aligner
    "FaceAligner",
    "ARCFACE_REFERENCE_LANDMARKS_112",
    # Extractor
    "FeatureExtractor",
    "ExtractionResult",
    # Tracker
    "MotionTracker",
    "Track",
    # Liveness
    "LivenessVerifier",
    "LivenessResult",
    # Pipeline
    "AttendancePipeline",
    "TrackResult",
    "FrameResult",
    "STRATEGY_BOTH",
    "STRATEGY_DETECTION_ONLY",
    "STRATEGY_TRACKING_ONLY",
    # Utils
    "cosine_similarity",
    "compute_ear",
    "euler_from_landmarks",
    "draw_boxes",
    "draw_landmarks",
    "read_image_bytes",
    "resize_keep_aspect",
    "crop_face",
    "get_logger",
    "TimedResult",
]
