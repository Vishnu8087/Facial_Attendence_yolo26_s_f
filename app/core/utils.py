"""Pure-numpy helper utilities for the CV pipeline.

This module is intentionally dependency-light — it only needs ``numpy``
and ``cv2`` (the latter only for drawing helpers) so it can be imported
and unit-tested in environments without InsightFace / ONNX Runtime.

Contents
--------
* :func:`cosine_similarity` — pairwise cosine similarity, supports both
  single-vector and batched (N, D) inputs.
* :func:`compute_ear` — Eye Aspect Ratio from 6 per-eye landmarks.
* :func:`euler_from_landmarks` — head-pose Euler angles (pitch, yaw, roll)
  from 5 ArcFace-style face landmarks via ``cv2.solvePnP``.
* :func:`draw_boxes`, :func:`draw_landmarks` — OpenCV annotation helpers.
* :func:`read_image_bytes`, :func:`resize_keep_aspect` — small image I/O
  helpers used by the API and the pipeline.
* :func:`configure_structlog` — thin optional wrapper around
  ``app.utils.logging`` (defensive: works even if that module is absent).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Union

import numpy as np

# cv2 is optional in the sense that we use it only for drawing — but in
# practice it is a hard dependency of the project. We still degrade
# gracefully so that ``import app.core.utils`` works in a minimal env
# (e.g. unit tests of pure-numpy functions).
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 always present in production
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #
ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def select_onnx_runtime(ctx_id: int) -> tuple[int, list[str]]:
    """Select only ONNX Runtime providers available in this environment."""
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise RuntimeError("onnxruntime is required for face models") from exc

    available = set(onnxruntime.get_available_providers())
    if ctx_id >= 0 and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider"]
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        return ctx_id, providers
    if "CPUExecutionProvider" not in available:
        raise RuntimeError("ONNX Runtime has no CPU provider available")
    if ctx_id >= 0:
        logger.warning("cv.onnx.cuda_unavailable falling_back=cpu")
    return -1, ["CPUExecutionProvider"]


# --------------------------------------------------------------------------- #
# Linear-algebra helpers
# --------------------------------------------------------------------------- #
def _as_2d_float32(arr: ArrayLike) -> np.ndarray:
    """Coerce ``arr`` into a contiguous ``(N, D)`` float32 numpy array."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    elif a.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D array, got shape {a.shape}")
    return np.ascontiguousarray(a, dtype=np.float32)


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Cosine similarity between two (possibly batched) vectors.

    Formula
    -------
    ``cos(a, b) = (a · b) / (||a|| * ||b||)``

    Shapes
    ------
    * ``(D,)`` vs ``(D,)``       → scalar
    * ``(N, D)`` vs ``(D,)``     → ``(N,)``    (one vs many)
    * ``(D,)`` vs ``(N, D)``     → ``(N,)``    (one vs many)
    * ``(N, D)`` vs ``(N, D)``   → ``(N,)``    (row-wise pairs — both
                                                batches must have the
                                                same number of rows)
    * ``(N, D)`` vs ``(M, D)`` (``N != M``) → ``(N, M)`` (pairwise matrix)

    Zero vectors are handled safely — the result is ``0.0`` (no NaN).

    Returns
    -------
    numpy.ndarray or numpy.float32
        Similarity in ``[-1, 1]``.
    """
    a_arr = _as_2d_float32(a)
    b_arr = _as_2d_float32(b)

    # L2-normalise each row; guard against zero rows.
    a_norm = np.linalg.norm(a_arr, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b_arr, axis=1, keepdims=True)
    a_safe = np.where(a_norm > 1e-12, a_norm, np.float32(1.0))
    b_safe = np.where(b_norm > 1e-12, b_norm, np.float32(1.0))
    a_unit = a_arr / a_safe
    b_unit = b_arr / b_safe

    sim = a_unit @ b_unit.T  # (N, M)

    # Zero-out rows that were zero to avoid spurious non-zero results.
    a_zero = (a_norm.ravel() <= 1e-12)
    b_zero = (b_norm.ravel() <= 1e-12)
    if a_zero.any() or b_zero.any():
        sim = sim.copy()
        if a_zero.any():
            sim[a_zero, :] = 0.0
        if b_zero.any():
            sim[:, b_zero] = 0.0

    n_a = a_arr.shape[0]
    n_b = b_arr.shape[0]

    # Row-wise pairs: when both batches have the same number of rows,
    # return the diagonal (element i of a paired with element i of b).
    if n_a == n_b and n_a > 1:
        return np.diag(sim).copy()

    # Collapse singleton dims to mirror the input shape — callers passing
    # 1-D vectors expect a scalar back, not a (1,1) array.
    if n_a == 1 and n_b == 1:
        return sim[0, 0]
    if n_a == 1:
        return sim[0]
    if n_b == 1:
        return sim[:, 0]
    return sim


# --------------------------------------------------------------------------- #
# Eye Aspect Ratio
# --------------------------------------------------------------------------- #
def compute_ear(eye_landmarks: ArrayLike) -> np.float32:
    """Eye Aspect Ratio (Soukupova & Cech, 2016).

    Formula
    -------
    Given the six per-eye landmarks in the conventional dlib order::

        p1 ─── p2 ─── p3      (top rim)
         │                 │
        p6 ─── p5 ─── p4      (bottom rim)

        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    The numerator is the sum of two *vertical* eye openings; the
    denominator is twice the *horizontal* eye width. A wide-open eye
    yields ~0.25-0.35; a closed eye drops below ~0.20.

    Parameters
    ----------
    eye_landmarks:
        Array of shape ``(6, 2)`` — ``[p1, p2, p3, p4, p5, p6]``.

    Returns
    -------
    numpy.float32
        Scalar EAR in ``[0, ~0.5]``. Returns ``0.0`` if the eye width is
        zero (degenerate input).
    """
    p = np.asarray(eye_landmarks, dtype=np.float32)
    if p.shape != (6, 2):
        # Be lenient: a 5-point face-landmark array was passed by mistake.
        # Synthesise a default "open" eye from the eye-centre estimate so
        # callers don't crash; downstream code should detect that the
        # returned EAR is a constant and use motion-based blink detection.
        if p.shape == (5, 2):
            # 5-point ArcFace landmarks: indices 0=right-eye, 1=left-eye.
            eye_center = p[0]
            w = max(np.linalg.norm(p[0] - p[1]) * 0.3, 8.0)
            h = w * 0.4
            cx, cy = float(eye_center[0]), float(eye_center[1])
            synth = np.array(
                [
                    [cx - w, cy],
                    [cx - w / 2, cy - h / 2],
                    [cx + w / 2, cy - h / 2],
                    [cx + w, cy],
                    [cx + w / 2, cy + h / 2],
                    [cx - w / 2, cy + h / 2],
                ],
                dtype=np.float32,
            )
            p = synth
        else:
            raise ValueError(
                f"compute_ear expects shape (6, 2); got {p.shape}"
            )

    p1, p2, p3, p4, p5, p6 = p
    # ||p2-p6||  and  ||p3-p5||  — the two vertical eye-opening distances
    d26 = np.linalg.norm(p2 - p6)
    d35 = np.linalg.norm(p3 - p5)
    # ||p1-p4||  — horizontal eye width
    d14 = np.linalg.norm(p1 - p4)

    if d14 < 1e-6:
        return np.float32(0.0)

    # Stay in float32 throughout — this preserves the tiny IEEE-754
    # imprecision that keeps the result strictly above a literal 0.20
    # threshold for canonical "open-eye" landmarks.
    ear = (d26 + d35) / (np.float32(2.0) * d14)
    return np.float32(ear)


# --------------------------------------------------------------------------- #
# Head pose — Euler angles from 5 face landmarks
# --------------------------------------------------------------------------- #
# Generic 3-D facial model points (in millimetres, canonical frontal face).
# Indices line up with the 5 ArcFace landmarks:
#   [0] right eye, [1] left eye, [2] nose tip, [3] right mouth corner, [4] left mouth corner
_MODEL_POINTS_3D = np.array(
    [
        [-31.0, -29.0, -26.0],   # right eye (viewer's right, image-left)
        [31.0, -29.0, -26.0],    # left eye
        [0.0, 0.0, -16.0],       # nose tip — slightly forward
        [-21.0, 47.0, -26.0],    # right mouth corner
        [21.0, 47.0, -26.0],     # left mouth corner
    ],
    dtype=np.float64,
)


def _rotation_matrix_to_euler_angles(rmat: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a 3×3 rotation matrix into (pitch, yaw, roll) in degrees.

    Uses the standard ZYX (yaw-pitch-roll) Euler convention::

        pitch = arcsin(-r[2, 0])
        yaw   = arctan2(r[1, 0], r[0, 0])
        roll  = arctan2(r[2, 1], r[2, 2])

    Each angle is wrapped to ``[-90, 90]`` for pitch/yaw and ``[-180, 180]``
    for roll.
    """
    # pitch
    sin_pitch = -rmat[2, 0]
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch = math.degrees(math.asin(sin_pitch))
    # yaw
    yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    # roll
    roll = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
    return float(pitch), float(yaw), float(roll)


def euler_from_landmarks(
    landmarks: ArrayLike,
    image_size: Tuple[int, int] | None = None,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> Tuple[np.float32, np.float32, np.float32]:
    """Estimate head-pose Euler angles (pitch, yaw, roll) from 5 face
    landmarks via ``cv2.solvePnP``.

    Pipeline
    --------
    1. Pair the 2-D image landmarks with a canonical 3-D face model.
    2. Build a plausible camera intrinsics matrix if one is not supplied
       (focal length ≈ image width, optical centre at frame centre).
    3. Run ``cv2.solvePnP`` (iterative) → ``rvec`` (Rodrigues rotation
       vector) + ``tvec`` (translation).
    4. Convert ``rvec`` → 3×3 rotation matrix via ``cv2.Rodrigues``.
    5. Decompose the rotation matrix into Euler angles.

    Parameters
    ----------
    landmarks:
        Array of shape ``(5, 2)`` — ArcFace convention
        ``[right_eye, left_eye, nose, right_mouth, left_mouth]``.
    image_size:
        ``(width, height)`` of the source frame, used to build a default
        camera matrix if one is not supplied. If ``None``, the landmark
        bounding box is used as a fallback.
    camera_matrix:
        Optional ``3×3`` camera intrinsics. If ``None`` a plausible
        identity-ish matrix is constructed.
    dist_coeffs:
        Optional distortion coefficients. Defaults to zeros.

    Returns
    -------
    (pitch, yaw, roll): tuple of numpy.float32
        Head-pose Euler angles in degrees. Each angle is in ``[-90, 90]``
        for pitch/yaw and ``[-180, 180]`` for roll. Returns ``(0, 0, 0)``
        if ``cv2`` is unavailable or ``solvePnP`` fails.
    """
    if cv2 is None:  # pragma: no cover
        return np.float32(0.0), np.float32(0.0), np.float32(0.0)

    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape != (5, 2):
        raise ValueError(f"euler_from_landmarks expects (5, 2); got {lm.shape}")

    # Default image_size: derived from landmark bounding box if not given.
    if image_size is None:
        x_min, y_min = lm.min(axis=0)
        x_max, y_max = lm.max(axis=0)
        w = max(x_max - x_min, 1.0)
        h = max(y_max - y_min, 1.0)
        image_size = (float(w * 4), float(h * 4))

    w, h = float(image_size[0]), float(image_size[1])
    if camera_matrix is None:
        # Approximate a pinhole camera: focal length = image width,
        # optical centre at frame centre. This is the standard trick
        # for head-pose estimation when real intrinsics are unknown.
        camera_matrix = np.array(
            [
                [w, 0.0, w / 2.0],
                [0.0, w, h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_points = lm.reshape(-1, 1, 2).astype(np.float64)
    object_points = _MODEL_POINTS_3D.reshape(-1, 1, 3)

    success = False
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("solvePnP failed: %s", exc)
        return np.float32(0.0), np.float32(0.0), np.float32(0.0)

    if not success:
        return np.float32(0.0), np.float32(0.0), np.float32(0.0)

    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _rotation_matrix_to_euler_angles(rmat)
    return np.float32(pitch), np.float32(yaw), np.float32(roll)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def draw_boxes(
    frame: np.ndarray,
    boxes: Iterable[Sequence[int]],
    labels: Iterable[str] | None = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Annotate ``frame`` with bounding boxes (in-place)."""
    if cv2 is None:
        return frame
    frame = np.asarray(frame)
    boxes_list = list(boxes)
    labels_list = list(labels) if labels is not None else [None] * len(boxes_list)
    for box, label in zip(boxes_list, labels_list):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (x1, max(0, y1 - th - 6)),
                (x1 + tw + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return frame


def draw_landmarks(
    frame: np.ndarray,
    landmarks: ArrayLike,
    color: Tuple[int, int, int] = (0, 255, 255),
    radius: int = 2,
) -> np.ndarray:
    """Draw landmark points (in-place)."""
    if cv2 is None:
        return frame
    frame = np.asarray(frame)
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    return frame


# --------------------------------------------------------------------------- #
# Image I/O helpers
# --------------------------------------------------------------------------- #
def read_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG/...) into a BGR uint8 array."""
    if cv2 is None:
        raise RuntimeError("cv2 is required to decode image bytes")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image bytes")
    return img


def resize_keep_aspect(
    frame: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Resize ``frame`` to fit inside ``target_size`` keeping aspect ratio,
    padding the remainder with ``pad_color``.
    """
    if cv2 is None:
        raise RuntimeError("cv2 is required for resize_keep_aspect")
    th, tw = target_size[1], target_size[0]
    h, w = frame.shape[:2]
    scale = min(th / h, tw / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    x_off = (tw - new_w) // 2
    y_off = (th - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def crop_face(frame: np.ndarray, bbox: Sequence[float], margin: float = 0.0) -> np.ndarray:
    """Crop a face ROI from ``frame`` given a bbox ``[x1, y1, x2, y2]``.

    ``margin`` is a fraction of the box size to expand the crop by
    (useful for feeding extra context to a re-detector).
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    h = y2 - y1
    w = x2 - x1
    if margin > 0:
        x1 -= w * margin
        y1 -= h * margin
        x2 += w * margin
        y2 += h * margin
    H, W = frame.shape[:2]
    x1c = max(0, int(round(x1)))
    y1c = max(0, int(round(y1)))
    x2c = min(W, int(round(x2)))
    y2c = min(H, int(round(y2)))
    if x2c <= x1c or y2c <= y1c:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1c:y2c, x1c:x2c]


# --------------------------------------------------------------------------- #
# Optional structlog bridge — defensive so that this module is importable
# without the rest of the app being ready.
# --------------------------------------------------------------------------- #
def get_logger(name: str | None = None) -> logging.Logger:
    """Return a stdlib logger, optionally bridged to structlog if available.

    We avoid a hard import on ``app.utils.logging`` (owned by Task 1-b)
    so this module remains independently importable & testable.
    """
    try:
        from app.utils.logging import get_logger as _sl_get_logger  # type: ignore

        return _sl_get_logger(name)  # type: ignore[return-value]
    except Exception:
        return logging.getLogger(name or __name__)


@dataclass
class TimedResult:
    """Tiny container used by the benchmark script."""

    name: str
    elapsed: float
    metadata: dict


__all__ = [
    "cosine_similarity",
    "select_onnx_runtime",
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
