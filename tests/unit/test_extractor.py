"""Unit tests for ``app.core.extractor.FeatureExtractor``."""
from __future__ import annotations

import numpy as np
import pytest

FeatureExtractor = pytest.importorskip("app.core.extractor", reason="Task 1-a not yet merged").FeatureExtractor  # type: ignore[attr-defined]


def _make_extractor_or_skip():
    try:
        return FeatureExtractor(model_pack_name="buffalo_s", device="cpu")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"FeatureExtractor not instantiable in this env: {exc!r}")


def test_extractor_returns_512d_vector(sample_face_frame):
    extractor = _make_extractor_or_skip()
    emb = extractor.extract(sample_face_frame)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)


def test_embedding_is_l2_normalised(sample_face_frame):
    extractor = _make_extractor_or_skip()
    emb = extractor.extract(sample_face_frame)
    np.testing.assert_allclose(np.linalg.norm(emb), 1.0, atol=1e-4)


def test_embedding_deterministic_for_same_input(sample_face_frame):
    extractor = _make_extractor_or_skip()
    a = extractor.extract(sample_face_frame)
    b = extractor.extract(sample_face_frame)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_embedding_differs_for_different_inputs(sample_face_frame, sample_frame):
    extractor = _make_extractor_or_skip()
    a = extractor.extract(sample_face_frame)
    b = extractor.extract(sample_frame[:112, :112])
    # Cosine similarity should be < 1.0
    cos = float(np.dot(a, b))
    assert cos < 0.999


def test_extractor_dtype_is_float32(sample_face_frame):
    extractor = _make_extractor_or_skip()
    emb = extractor.extract(sample_face_frame)
    assert emb.dtype == np.float32
