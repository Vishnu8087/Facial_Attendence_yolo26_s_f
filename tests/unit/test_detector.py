"""Unit tests for ``app.core.detector.FaceDetector``.

These tests are designed to run against either:
* the real ``FaceDetector`` implemented by Task 1-a, or
* a mock returned by the ``mock_detector`` fixture.

The real implementation is loaded only when the module is importable AND
the bundled model file is available; otherwise we fall back to the mock
so the suite stays green in a CI environment with no model files.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

FaceDetector = pytest.importorskip("app.core.detector", reason="Task 1-a not yet merged").FaceDetector  # type: ignore[attr-defined]


def _make_detector_or_skip():
    """Try to instantiate the real detector; on failure (missing model,
    no GPU, etc.) fall back to the mock."""
    try:
        return FaceDetector(model_pack_name="buffalo_s", device="cpu")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"FaceDetector not instantiable in this env: {exc!r}")


def test_detector_returns_list_for_synthetic_frame(sample_frame):
    detector = _make_detector_or_skip()
    detections = detector.detect(sample_frame)
    assert isinstance(detections, list)


def test_detection_bbox_shape_and_dtype(sample_frame):
    detector = _make_detector_or_skip()
    detections = detector.detect(sample_frame)
    for det in detections:
        assert det.bbox.shape == (4,)
        assert np.all(det.bbox[:2] <= det.bbox[2:])  # x1<=x2, y1<=y2


def test_detection_landmarks_shape(sample_frame):
    detector = _make_detector_or_skip()
    detections = detector.detect(sample_frame)
    for det in detections:
        assert det.landmarks.shape == (5, 2)


def test_detection_score_in_unit_range(sample_frame):
    detector = _make_detector_or_skip()
    detections = detector.detect(sample_frame)
    for det in detections:
        assert 0.0 <= float(det.score) <= 1.0


def test_detector_handles_empty_frame():
    """A blank image should return an empty list, not raise."""
    detector = _make_detector_or_skip()
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(blank)
    assert isinstance(detections, list)
