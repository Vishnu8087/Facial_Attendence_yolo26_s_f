"""Unit tests for the YOLO26-face detector.

These tests use mocks — they don't require the ``ultralytics`` package or
the actual ``yolo26n-face.pt`` weights. They verify the contract that the
``AttendancePipeline`` relies on: the detector returns
``DetectionResult`` objects with the right shapes (bbox 4-vector,
landmarks (5, 2), score float).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_ultralytics_module():
    """Inject a fake ``ultralytics`` module so the import succeeds without pip install."""
    fake_mod = types.ModuleType("ultralytics")
    fake_yolo_cls = MagicMock(name="YOLO")
    fake_mod.YOLO = fake_yolo_cls
    # Patch into sys.modules *before* importing YoloFaceDetector.
    saved = sys.modules.get("ultralytics")
    sys.modules["ultralytics"] = fake_mod
    try:
        yield fake_mod, fake_yolo_cls
    finally:
        if saved is not None:
            sys.modules["ultralytics"] = saved
        else:
            sys.modules.pop("ultralytics", None)


@pytest.fixture
def yolo_detector_cls(fake_ultralytics_module):
    """Return the YoloFaceDetector class with ultralytics mocked out."""
    from app.core.detector_yolo26 import YoloFaceDetector
    return YoloFaceDetector


@pytest.fixture
def sample_frame():
    """640×480 BGR uint8 frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def _make_fake_result(boxes_xyxy, confs, kpts_xy):
    """Build a fake Ultralytics result object with the right attribute shape."""
    res = MagicMock()

    boxes = MagicMock()
    boxes.__len__ = lambda self: len(boxes_xyxy)
    boxes.xyxy = MagicMock()
    boxes.xyxy.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array(boxes_xyxy, dtype=np.float32)))
    boxes.conf = MagicMock()
    boxes.conf.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array(confs, dtype=np.float32)))
    res.boxes = boxes

    if kpts_xy is not None:
        kpts = MagicMock()
        kpts.xy = MagicMock()
        kpts.xy.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array(kpts_xy, dtype=np.float32)))
        res.keypoints = kpts
    else:
        res.keypoints = None

    return res


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestYoloFaceDetectorConstruction:
    def test_constructs_with_defaults(self, yolo_detector_cls):
        d = yolo_detector_cls(model_path="yolo26n-face.pt")
        assert d.conf_threshold == 0.5
        assert d.iou_threshold == 0.45
        assert d.imgsz == 640
        assert d.min_face_size == 60
        assert d.max_faces == 50

    def test_raises_if_ultralytics_missing(self, monkeypatch):
        # Force the import to fail
        monkeypatch.setitem(sys.modules, "ultralytics", None)
        from app.core.detector_yolo26 import YoloFaceDetector
        with pytest.raises(RuntimeError, match="ultralytics is not installed"):
            YoloFaceDetector()

    def test_resolves_model_path_from_env(self, yolo_detector_cls, tmp_path, monkeypatch):
        # Create a fake weights file in a temp dir
        weights = tmp_path / "yolo26n-face.pt"
        weights.write_bytes(b"fake-weights")
        monkeypatch.setenv("MODELS_DIR", str(tmp_path))

        d = yolo_detector_cls(model_path="yolo26n-face.pt")
        assert d._resolved_path == str(weights.resolve())


# --------------------------------------------------------------------------- #
# Detection contract
# --------------------------------------------------------------------------- #
class TestYoloFaceDetectorDetect:
    def test_returns_empty_for_empty_frame(self, yolo_detector_cls, sample_frame):
        d = yolo_detector_cls()
        assert d.detect(None) == []
        assert d.detect(np.array([])) == []

    def test_returns_empty_when_no_faces(self, yolo_detector_cls, sample_frame):
        d = yolo_detector_cls()
        # Inject a model whose predict() returns no boxes
        fake_model = MagicMock()
        fake_res = MagicMock()
        fake_res.boxes = MagicMock()
        fake_res.boxes.__len__ = lambda self: 0
        fake_res.boxes.xyxy = MagicMock()
        fake_res.boxes.conf = MagicMock()
        fake_res.keypoints = None
        fake_model.predict = MagicMock(return_value=[fake_res])
        d._model = fake_model
        d._loaded = True

        result = d.detect(sample_frame)
        assert result == []

    def test_returns_detection_with_correct_shapes(self, yolo_detector_cls, sample_frame):
        """Verify DetectionResult carries bbox(4,) + landmarks(5,2) + score float."""
        d = yolo_detector_cls(min_face_size=0)  # don't filter small boxes

        # Build a fake Ultralytics result with 1 face
        kpts = np.array([[100, 110], [120, 110], [110, 130], [105, 145], [115, 145]],
                        dtype=np.float32)  # (5, 2)
        fake_res = _make_fake_result(
            boxes_xyxy=[[80, 90, 140, 160]],
            confs=[0.92],
            kpts_xy=[kpts],
        )
        fake_model = MagicMock()
        fake_model.predict = MagicMock(return_value=[fake_res])
        d._model = fake_model
        d._loaded = True

        result = d.detect(sample_frame)
        assert len(result) == 1
        det = result[0]
        assert det.bbox.shape == (4,)
        assert det.landmarks.shape == (5, 2)
        assert det.score == pytest.approx(0.92)
        # Verify landmark order matches ArcFace convention (right_eye first)
        assert det.landmarks[0, 0] < det.landmarks[1, 0]  # right eye x < left eye x

    def test_filters_by_min_face_size(self, yolo_detector_cls, sample_frame):
        d = yolo_detector_cls(min_face_size=50)
        # A 20×20 box — should be filtered out
        kpts = np.array([[10, 11], [12, 11], [11, 13], [10, 14], [12, 14]], dtype=np.float32)
        fake_res = _make_fake_result(
            boxes_xyxy=[[0, 0, 20, 20]],
            confs=[0.95],
            kpts_xy=[kpts],
        )
        fake_model = MagicMock()
        fake_model.predict = MagicMock(return_value=[fake_res])
        d._model = fake_model
        d._loaded = True

        assert d.detect(sample_frame) == []

    def test_sorts_by_score_desc(self, yolo_detector_cls, sample_frame):
        d = yolo_detector_cls(min_face_size=0)
        kpts_a = np.array([[100, 110], [120, 110], [110, 130], [105, 145], [115, 145]], dtype=np.float32)
        kpts_b = np.array([[200, 210], [220, 210], [210, 230], [205, 245], [215, 245]], dtype=np.float32)
        fake_res = _make_fake_result(
            boxes_xyxy=[[80, 90, 140, 160], [180, 190, 240, 260]],
            confs=[0.7, 0.95],  # second face has higher score
            kpts_xy=[kpts_a, kpts_b],
        )
        fake_model = MagicMock()
        fake_model.predict = MagicMock(return_value=[fake_res])
        d._model = fake_model
        d._loaded = True

        result = d.detect(sample_frame)
        assert len(result) == 2
        assert result[0].score > result[1].score
        assert result[0].score == pytest.approx(0.95)

    def test_caps_to_max_faces(self, yolo_detector_cls, sample_frame):
        d = yolo_detector_cls(min_face_size=0, max_faces=2)
        # 5 detections all high-confidence
        boxes = [[i * 50, i * 50, i * 50 + 40, i * 50 + 40] for i in range(5)]
        confs = [0.9] * 5
        kpts = [[(i * 50 + 10, i * 50 + 10)] * 5 for i in range(5)]
        kpts_arr = np.array(kpts, dtype=np.float32)
        fake_res = _make_fake_result(boxes_xyxy=boxes, confs=confs, kpts_xy=kpts_arr)
        fake_model = MagicMock()
        fake_model.predict = MagicMock(return_value=[fake_res])
        d._model = fake_model
        d._loaded = True

        result = d.detect(sample_frame)
        assert len(result) == 2

    def test_synthesizes_landmarks_when_model_returns_none(self, yolo_detector_cls, sample_frame):
        """If keypoints is None, detector should still return 5 synthetic landmarks."""
        d = yolo_detector_cls(min_face_size=0)
        fake_res = _make_fake_result(
            boxes_xyxy=[[80, 90, 140, 160]],
            confs=[0.9],
            kpts_xy=None,
        )
        fake_model = MagicMock()
        fake_model.predict = MagicMock(return_value=[fake_res])
        d._model = fake_model
        d._loaded = True

        result = d.detect(sample_frame)
        assert len(result) == 1
        assert result[0].landmarks.shape == (5, 2)

    def test_predict_exception_returns_empty(self, yolo_detector_cls, sample_frame):
        """A model.predict() crash shouldn't bubble up — return [] instead."""
        d = yolo_detector_cls()
        fake_model = MagicMock()
        fake_model.predict = MagicMock(side_effect=RuntimeError("OOM"))
        d._model = fake_model
        d._loaded = True

        assert d.detect(sample_frame) == []


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
class TestYoloFaceDetectorLifecycle:
    def test_is_loaded_reflects_state(self, yolo_detector_cls):
        d = yolo_detector_cls()
        assert d.is_loaded() is False
        d._model = MagicMock()
        d._loaded = True
        assert d.is_loaded() is True

    def test_reset_clears_model(self, yolo_detector_cls):
        d = yolo_detector_cls()
        d._model = MagicMock()
        d._loaded = True
        d.reset()
        assert d._model is None
        assert d.is_loaded() is False

    def test_warmup_loads_model(self, yolo_detector_cls, fake_ultralytics_module):
        """warmup() should call _get_model() which loads via ultralytics.YOLO()."""
        _, fake_yolo_cls = fake_ultralytics_module
        fake_instance = MagicMock()
        fake_yolo_cls.return_value = fake_instance

        d = yolo_detector_cls(model_path="yolo26n-face.pt", device="cpu")
        d.warmup()
        assert d.is_loaded() is True
        fake_yolo_cls.assert_called_once_with(d._resolved_path)
        fake_instance.to.assert_called_once_with("cpu")
