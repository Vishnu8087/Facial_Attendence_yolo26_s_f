"""Unit tests for ``app.core.liveness.LivenessVerifier``.

Tests cover:
* Eye-Aspect-Ratio (EAR) computation correctness
* Blink detection from a synthetic landmark sequence
* Head-pose (Euler angle) derivation from landmarks
* Static-frame rejection (no motion => not liveness-passing)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# Try to import both the verifier class and the helper utils.
_liveness_mod = pytest.importorskip("app.core.liveness", reason="Task 1-a not yet merged")
LivenessVerifier = _liveness_mod.LivenessVerifier  # type: ignore[attr-defined]

# utils may live either in app.core.utils or app.core.liveness
_utils_candidates = ["app.core.utils", "app.core.liveness"]
utils = None
for name in _utils_candidates:
    try:
        utils = __import__(name, fromlist=["compute_ear", "euler_from_landmarks"])
        break
    except Exception:
        continue
if utils is None:  # pragma: no cover
    pytest.skip("neither app.core.utils nor app.core.liveness exposes compute_ear", allow_module_level=True)


# --------------------------------------------------------------------------- #
# EAR computation
# --------------------------------------------------------------------------- #
def _eye_landmarks_open() -> np.ndarray:
    """6 eye landmarks forming a wide-open eye (tall aspect ratio)."""
    return np.array(
        [
            [0, 50],   # left corner
            [25, 40],
            [50, 40],
            [100, 50], # right corner
            [50, 60],
            [25, 60],
        ],
        dtype=np.float32,
    )


def _eye_landmarks_closed() -> np.ndarray:
    """6 eye landmarks forming an almost-closed eye."""
    return np.array(
        [
            [0, 50],
            [25, 49],
            [50, 49],
            [100, 50],
            [50, 51],
            [25, 51],
        ],
        dtype=np.float32,
    )


def test_compute_ear_open_eye_is_above_threshold():
    ear = float(utils.compute_ear(_eye_landmarks_open()))
    assert 0.20 < ear < 0.60


def test_compute_ear_closed_eye_is_below_threshold():
    ear = float(utils.compute_ear(_eye_landmarks_closed()))
    assert 0.0 <= ear < 0.10


def test_compute_ear_returns_scalar_float():
    ear = utils.compute_ear(_eye_landmarks_open())
    assert np.isscalar(ear) or ear.shape == ()


# --------------------------------------------------------------------------- #
# Euler angles from landmarks
# --------------------------------------------------------------------------- #
def test_euler_from_landmarks_returns_three_angles():
    landmarks = np.array(
        [
            [70, 70], [130, 70], [100, 100], [70, 130], [130, 130]
        ],
        dtype=np.float32,
    )
    pitch, yaw, roll = utils.euler_from_landmarks(landmarks)
    assert all(-90.0 <= float(a) <= 90.0 for a in (pitch, yaw, roll))


def test_euler_zero_for_symmetric_frontal_landmarks():
    landmarks = np.array(
        [
            [70, 70], [130, 70], [100, 100], [70, 130], [130, 130]
        ],
        dtype=np.float32,
    )
    pitch, yaw, roll = utils.euler_from_landmarks(landmarks)
    assert abs(float(pitch)) < 5.0
    assert abs(float(yaw)) < 5.0
    assert abs(float(roll)) < 5.0


# --------------------------------------------------------------------------- #
# Blink detection
# --------------------------------------------------------------------------- #
def _frame_seq_with_blink(base_landmarks: np.ndarray, n_open: int = 5, n_closed: int = 3):
    """Generate a sequence of landmark frames where eyes close in the middle."""
    closed = base_landmarks.copy()
    closed[:, 1] += 5  # squish vertically
    seq_open = [base_landmarks] * n_open
    seq_closed = [closed] * n_closed
    return seq_open + seq_closed + seq_open


def test_blink_detected_in_synthetic_sequence():
    verifier = LivenessVerifier(ear_threshold=0.20, min_blinks=1, window_seconds=5.0)
    base = np.array(
        [[70, 70], [130, 70], [100, 100], [70, 130], [130, 130]], dtype=np.float32
    )
    for lm in _frame_seq_with_blink(base):
        verifier.update(lm)
    result = verifier.verify()
    assert result["blink_count"] >= 1
    assert result["is_real"] is True


def test_static_sequence_does_not_pass_liveness():
    """The same landmarks repeated indefinitely should be flagged static."""
    verifier = LivenessVerifier(ear_threshold=0.20, min_blinks=1, window_seconds=5.0)
    base = np.array(
        [[70, 70], [130, 70], [100, 100], [70, 130], [130, 130]], dtype=np.float32
    )
    for _ in range(20):
        verifier.update(base)
    result = verifier.verify()
    assert result["is_real"] is False or result["score"] < 0.5


def test_liveness_returns_required_keys():
    verifier = LivenessVerifier()
    base = np.array(
        [[70, 70], [130, 70], [100, 100], [70, 130], [130, 130]], dtype=np.float32
    )
    verifier.update(base)
    result = verifier.verify()
    for key in ("is_real", "ear", "blink_count", "head_pose", "score"):
        assert key in result, f"missing key {key}"
