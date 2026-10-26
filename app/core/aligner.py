"""Face aligner module.

Implements the standard ArcFace 5-point affine alignment. Given a set of
5 face landmarks (right eye, left eye, nose, right mouth corner, left
mouth corner) we estimate a 2-D similarity transform that maps the
detected landmarks onto a canonical reference set, then warp the source
frame with that transform. The output is a ``112×112`` (or other
user-configurable size) BGR uint8 face chip ready to feed to ArcFace.

Reference
---------
The reference landmark positions for the canonical ``112×112`` ArcFace
template are taken from the InsightFace implementation and the original
ArcFace paper (Deng et al., 2019). The template can be scaled for other
output sizes — see :data:`ARCFACE_REFERENCE_LANDMARKS_112`.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Canonical ArcFace landmark template (for a 112×112 output).
#
# Coordinates are (x, y) in pixels. The ordering matches the InsightFace
# convention: [right_eye, left_eye, nose, right_mouth, left_mouth].
# These values are widely-used community constants — see e.g.
# https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
# --------------------------------------------------------------------------- #
ARCFACE_REFERENCE_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],   # right eye (image-left)
        [73.5318, 51.5014],   # left eye  (image-right)
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # right mouth corner
        [70.7299, 92.2041],   # left mouth corner
    ],
    dtype=np.float32,
)


class FaceAligner:
    """5-point affine aligner producing ArcFace-ready face chips.

    Parameters
    ----------
    output_size:
        ``(width, height)`` of the aligned output chip. Default
        ``(112, 112)`` — the size ArcFace was trained on.
    reference_landmarks:
        Optional ``(5, 2)`` reference landmark set for an output of size
        ``output_size``. If ``None`` the canonical 112-template is used
        and scaled to ``output_size`` if the latter differs from 112.
    border_mode:
        OpenCV border mode constant for warping (default
        ``cv2.BORDER_REPLICATE`` — avoids dark edges that confuse the
        recognizer).
    interpol_method:
        OpenCV interpolation method (default ``cv2.INTER_LINEAR``).
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (112, 112),
        reference_landmarks: np.ndarray | None = None,
        border_mode: int | None = None,
        interpol_method: int | None = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("cv2 is required for FaceAligner")

        out_w, out_h = int(output_size[0]), int(output_size[1])
        if out_w <= 0 or out_h <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        self.output_size = (out_w, out_h)

        if reference_landmarks is not None:
            ref = np.asarray(reference_landmarks, dtype=np.float32).reshape(5, 2)
        else:
            ref = ARCFACE_REFERENCE_LANDMARKS_112.copy()
            # Scale the canonical 112×112 template to the requested size.
            if (out_w, out_h) != (112, 112):
                sx = out_w / 112.0
                sy = out_h / 112.0
                ref = ref * np.array([sx, sy], dtype=np.float32)
        self.reference_landmarks = ref
        self.border_mode = border_mode if border_mode is not None else cv2.BORDER_REPLICATE
        self.interpol_method = interpol_method if interpol_method is not None else cv2.INTER_LINEAR

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def align(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        size: Tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Affine-warp ``frame`` so the given landmarks land on the
        canonical reference points.

        Math
        ----
        We use ``cv2.estimateAffinePartial2D`` to solve for a 2×3
        similarity transform matrix ``M`` (4 degrees of freedom:
        rotation, scale, translation x/y) that minimises the
        least-squares distance between the source landmarks and the
        reference landmarks::

            M = [[cos(θ)*s, -sin(θ)*s, tx],
                 [sin(θ)*s,  cos(θ)*s, ty]]

        Then the output chip is ``cv2.warpAffine(frame, M, (W, H))``.

        ``estimateAffinePartial2D`` is preferred over the full affine
        because ArcFace was trained on similarity-aligned chips —
        preserving aspect ratio and angles.

        Parameters
        ----------
        frame:
            ``(H, W, 3)`` uint8 BGR image.
        landmarks:
            ``(5, 2)`` array of source landmarks in the same coordinate
            system as ``frame``.
        size:
            Optional ``(width, height)`` override for this single call.
            If ``None``, ``self.output_size`` is used.

        Returns
        -------
        numpy.ndarray
            ``(H, W, 3)`` uint8 aligned face chip. If alignment fails
            (e.g. all-zero landmarks) a centred crop of the frame is
            returned as a graceful fallback — never raises.
        """
        if cv2 is None:
            raise RuntimeError("cv2 is required for FaceAligner")

        out_size = size if size is not None else self.output_size
        out_w, out_h = int(out_size[0]), int(out_size[1])

        # Determine the reference landmarks for this output size.
        if size is None or (out_w, out_h) == self.output_size:
            ref = self.reference_landmarks
        else:
            # Re-scale the canonical template on the fly.
            sx = out_w / 112.0
            sy = out_h / 112.0
            ref = ARCFACE_REFERENCE_LANDMARKS_112 * np.array([sx, sy], dtype=np.float32)

        src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        # Guard against degenerate (all-zero) input — return a centred
        # crop so callers don't crash.
        if not np.any(src) or np.linalg.norm(src) < 1e-6:
            return self._fallback_crop(frame, out_w, out_h)

        # Estimate the similarity transform: src → ref.
        # ``estimateAffinePartial2D`` returns (M, inliers).
        try:
            m, _ = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
        except Exception:
            m = None
        if m is None:
            # Degenerate configuration — fall back to a centred crop.
            return self._fallback_crop(frame, out_w, out_h)

        warped = cv2.warpAffine(
            frame,
            m,
            (out_w, out_h),
            flags=self.interpol_method,
            borderMode=self.border_mode,
        )
        if warped.ndim == 2:
            warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        elif warped.shape[2] == 4:
            warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(warped)

    def align_batch(
        self,
        frame: np.ndarray,
        landmarks_list: list[np.ndarray],
        size: Tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Align multiple sets of landmarks against the same frame.

        Returns
        -------
        numpy.ndarray
            ``(N, H, W, 3)`` uint8 array of aligned chips.
        """
        chips = [self.align(frame, lm, size=size) for lm in landmarks_list]
        if not chips:
            out_size = size if size is not None else self.output_size
            return np.zeros((0, out_size[1], out_size[0], 3), dtype=np.uint8)
        return np.stack(chips, axis=0)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fallback_crop(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        """Centred-resize fallback when alignment cannot be computed."""
        if cv2 is None:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        # Crop a centred square then resize.
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        crop = frame[y0 : y0 + side, x0 : x0 + side]
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


__all__ = ["FaceAligner", "ARCFACE_REFERENCE_LANDMARKS_112"]
