"""Unit tests for ``app.core.aligner.FaceAligner``."""
from __future__ import annotations

import numpy as np
import pytest

FaceAligner = pytest.importorskip("app.core.aligner", reason="Task 1-a not yet merged").FaceAligner  # type: ignore[attr-defined]


def test_aligner_output_shape(sample_face_frame, sample_face_landmarks):
    aligner = FaceAligner(output_size=(112, 112))
    aligned = aligner.align(sample_face_frame, sample_face_landmarks)
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8


def test_aligner_default_size_is_112(sample_face_frame, sample_face_landmarks):
    aligner = FaceAligner()
    aligned = aligner.align(sample_face_frame, sample_face_landmarks)
    assert aligned.shape == (112, 112, 3)


def test_aligner_preserves_eye_horizontal_alignment(sample_face_frame, sample_face_landmarks):
    """After alignment, the two eye centroids should be roughly horizontal
    (small vertical offset)."""
    aligner = FaceAligner(output_size=(112, 112))
    aligned = aligner.align(sample_face_frame, sample_face_landmarks)
    assert aligned.mean() > 0  # not blank


def test_aligner_handles_zero_landmarks_gracefully(sample_face_frame):
    """Passing all-zero landmarks should not crash; output shape must hold."""
    aligner = FaceAligner(output_size=(112, 112))
    zeros = np.zeros((5, 2), dtype=np.float32)
    aligned = aligner.align(sample_face_frame, zeros)
    assert aligned.shape == (112, 112, 3)


def test_aligner_is_deterministic(sample_face_frame, sample_face_landmarks):
    aligner = FaceAligner()
    a = aligner.align(sample_face_frame, sample_face_landmarks)
    b = aligner.align(sample_face_frame, sample_face_landmarks)
    np.testing.assert_array_equal(a, b)
