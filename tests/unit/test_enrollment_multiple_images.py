"""Tests for enrollment from multiple face images."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.config import settings
from app.services.enrollment import EnrollmentError, EnrollmentService, _build_detector


@pytest.mark.asyncio
async def test_multiple_images_are_combined_into_one_normalized_embedding(monkeypatch):
    service = EnrollmentService(detector=object())
    monkeypatch.setattr(
        EnrollmentService,
        "_decode_image",
        staticmethod(lambda raw: np.asarray([len(raw)], dtype=np.uint8)),
    )

    saved_crops = []

    async def extract(frame):
        if frame[0] == 1:
            return np.asarray([1.0, 0.0] + [0.0] * 510, dtype=np.float32), 1, frame
        return np.asarray([0.0, 1.0] + [0.0] * 510, dtype=np.float32), 1, frame

    async def create_employee(session, **values):
        create_employee.values = values
        return SimpleNamespace(id="employee-id")

    service._extract_embedding = extract
    monkeypatch.setattr(
        EnrollmentService,
        "_save_face_image",
        staticmethod(lambda employee_id, crop: saved_crops.append(crop) or "face.jpg"),
    )
    monkeypatch.setattr("app.services.enrollment.crud.create_employee", create_employee)

    result = await service.enroll_from_image(
        object(),
        employee_id="EMP-MULTI",
        name="Multi Image",
        image_bytes_list=[b"a", b"bb"],
    )

    embedding = create_employee.values["embedding"]
    assert result.employee.id == "employee-id"
    assert result.num_faces_detected == 2
    assert np.isclose(np.linalg.norm(embedding), 1.0)
    assert np.allclose(embedding[:2], [2**-0.5, 2**-0.5])
    assert len(saved_crops) == 1


@pytest.mark.asyncio
async def test_image_enrollment_fails_when_cv_core_is_unavailable():
    service = object.__new__(EnrollmentService)
    service.detector = None
    service.extractor = None

    with pytest.raises(EnrollmentError, match="Face processing is unavailable"):
        await service._extract_embedding(np.zeros((32, 32, 3), dtype=np.uint8))


@pytest.mark.asyncio
async def test_image_enrollment_reports_detector_inference_failure():
    class BrokenDetector:
        def detect(self, _frame):
            raise RuntimeError("model inference failed")

    service = EnrollmentService(
        detector=BrokenDetector(),
        extractor=object(),
    )

    with pytest.raises(EnrollmentError, match="Face detection failed: model inference failed"):
        await service._extract_embedding(np.zeros((32, 32, 3), dtype=np.uint8))


@pytest.mark.asyncio
async def test_image_enrollment_reports_embedding_inference_failure():
    detection = SimpleNamespace(landmarks=np.zeros((5, 2), dtype=np.float32))

    class Detector:
        def detect(self, _frame):
            return [detection]

    class Extractor:
        def extract(self, _face):
            raise RuntimeError("ArcFace inference failed")

    service = EnrollmentService(
        detector=Detector(),
        aligner=None,
        extractor=Extractor(),
    )

    with pytest.raises(
        EnrollmentError,
        match="Face embedding extraction failed: ArcFace inference failed",
    ):
        await service._extract_embedding(np.zeros((32, 32, 3), dtype=np.uint8))


def test_yolo_detector_failure_is_reported_instead_of_falling_back(monkeypatch):
    class MissingYolo:
        DEFAULT_MODEL = "missing.pt"

        def __init__(self, **kwargs):
            raise RuntimeError("Ultralytics is unavailable")

    class InsightfaceDetector:
        def __init__(self, **kwargs):
            self.device = kwargs["device"]

    monkeypatch.setattr(settings, "detector_backend", "yolo26")
    with pytest.raises(RuntimeError, match="YOLO26 is configured but could not be initialized"):
        _build_detector(
            {
                "YoloFaceDetector": MissingYolo,
                "FaceDetector": InsightfaceDetector,
            }
        )
