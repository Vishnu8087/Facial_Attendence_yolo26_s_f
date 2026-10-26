"""Unit tests for ``app.core.utils`` (or wherever cosine_similarity,
compute_ear, euler_from_landmarks live).

These helpers are pure numpy — no model files needed.
"""
from __future__ import annotations

import numpy as np
import pytest

# Try several likely module locations. Each candidate must expose all
# three helpers; otherwise we keep looking (an empty namespace package
# like `app.utils` will import but won't have the attributes).
_candidates = ["app.core.utils", "app.utils.utils", "app.utils"]
utils = None
for name in _candidates:
    try:
        mod = __import__(name, fromlist=["cosine_similarity", "compute_ear", "euler_from_landmarks"])
    except Exception:
        continue
    if all(hasattr(mod, attr) for attr in ("cosine_similarity", "compute_ear", "euler_from_landmarks")):
        utils = mod
        break
if utils is None:  # pragma: no cover
    pytest.skip("utils module not found — Task 1-a not yet merged", allow_module_level=True)


# --------------------------------------------------------------------------- #
# cosine_similarity
# --------------------------------------------------------------------------- #
def test_cosine_similarity_identical_vectors_is_one():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sim = float(utils.cosine_similarity(a, a))
    assert abs(sim - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    sim = float(utils.cosine_similarity(a, b))
    assert abs(sim) < 1e-6


def test_cosine_similarity_opposite_vectors_is_minus_one():
    a = np.array([1.0, 1.0], dtype=np.float32)
    b = np.array([-1.0, -1.0], dtype=np.float32)
    sim = float(utils.cosine_similarity(a, b))
    assert abs(sim - (-1.0)) < 1e-6


def test_cosine_similarity_handles_zero_vector():
    """Zero vector should not cause NaN — implementation may return 0.0."""
    a = np.zeros(3, dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sim = utils.cosine_similarity(a, b)
    assert not np.isnan(sim)


def test_cosine_similarity_batch_shapes():
    """cosine_similarity should accept (N, D) batches and return (N,)."""
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = utils.cosine_similarity(a, b)
    out = np.asarray(out)
    assert out.shape == (3,)


# --------------------------------------------------------------------------- #
# compute_ear
# --------------------------------------------------------------------------- #
def test_compute_ear_returns_positive_float():
    landmarks = np.array(
        [[0, 50], [25, 40], [50, 40], [100, 50], [50, 60], [25, 60]],
        dtype=np.float32,
    )
    ear = float(utils.compute_ear(landmarks))
    assert 0.0 < ear < 1.0


def test_compute_ear_closed_eye_lower_than_open():
    open_lm = np.array(
        [[0, 50], [25, 40], [50, 40], [100, 50], [50, 60], [25, 60]],
        dtype=np.float32,
    )
    closed_lm = open_lm.copy()
    closed_lm[1:3, 1] += 10
    closed_lm[4:6, 1] -= 10
    assert float(utils.compute_ear(closed_lm)) < float(utils.compute_ear(open_lm))


# --------------------------------------------------------------------------- #
# euler_from_landmarks
# --------------------------------------------------------------------------- #
def test_euler_from_landmarks_returns_three_floats():
    landmarks = np.array(
        [[70, 70], [130, 70], [100, 100], [70, 130], [130, 130]],
        dtype=np.float32,
    )
    out = utils.euler_from_landmarks(landmarks)
    out = tuple(np.atleast_1d(np.asarray(x)).ravel()[0] for x in out)
    assert len(out) == 3


def test_euler_symmetric_frontal_is_near_zero():
    landmarks = np.array(
        [[70, 70], [130, 70], [100, 100], [70, 130], [130, 130]],
        dtype=np.float32,
    )
    pitch, yaw, roll = (float(x) for x in utils.euler_from_landmarks(landmarks))
    assert abs(pitch) < 5.0 and abs(yaw) < 5.0 and abs(roll) < 5.0
