"""Face detector module.

Wraps InsightFace's SCRFD model (via the ``insightface`` Python
package) with a graceful fallback path for headless CI / unit-test
environments where the model files are not present.

The class is deliberately lazy: the heavy ONNX model is only loaded on
the first call to :meth:`FaceDetector.detect`, so importing the module
is cheap and side-effect-free.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from app.core.utils import select_onnx_runtime

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class DetectionResult:
    """A single face detection.

    Attributes
    ----------
    bbox:
        Float32 array ``[x1, y1, x2, y2]`` in absolute pixel coordinates.
    landmarks:
        Float32 array of shape ``(5, 2)`` — ArcFace convention
        ``[right_eye, left_eye, nose, right_mouth, left_mouth]``.
    score:
        Detection confidence in ``[0, 1]``.
    """

    bbox: np.ndarray
    landmarks: np.ndarray
    score: float
    # Optional metadata (e.g. which detector sub-model produced this).
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Always coerce to the canonical dtypes/shapes so downstream code
        # can rely on them.
        self.bbox = np.asarray(self.bbox, dtype=np.float32).reshape(4)
        self.landmarks = np.asarray(self.landmarks, dtype=np.float32).reshape(5, 2)
        self.score = float(self.score)


# --------------------------------------------------------------------------- #
# FaceDetector
# --------------------------------------------------------------------------- #
class FaceDetector:
    """SCRFD-based face detector backed by InsightFace.

    The detector is **lazy**: the underlying ONNX model is only loaded on
    the first call to :meth:`detect`. This keeps module import cheap and
    allows the class to be instantiated in test environments without the
    model files (constructor will succeed; only ``detect`` raises).

    Parameters
    ----------
    model_pack_name:
        InsightFace model pack name — ``"buffalo_l"`` (default, best
        accuracy) or ``"buffalo_s"`` (smaller, faster).
    det_size:
        Edge size (in pixels) of the square image the SCRFD detector
        is resized to internally. Larger ⇒ more small-face recall but
        slower. Default 640.
    det_thresh:
        Minimum detection confidence; boxes below this are dropped.
    nms_thresh:
        Non-Maximum-Suppression IoU threshold. Boxes overlapping more
        than this with a higher-scoring box are removed.
    device:
        ``"cpu"``, ``"cuda"``, or ``"cuda:N"``. Falls back to CPU if the
        requested device is unavailable.
    model_dir:
        InsightFace root directory. If ``None``, ``MODEL_DIR`` or
        ``MODELS_DIR`` is used; otherwise ``~/.insightface``.
    ctx_id:
        ONNX Execution-Provider context id. Usually derived from
        ``device``.
    min_face_size:
        Reject detections whose shorter side is below this (noise filter).
    max_faces:
        Cap on the number of detections returned per frame (highest
        scoring first).
    """

    def __init__(
        self,
        model_pack_name: str = "buffalo_l",
        det_size: int = 640,
        det_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        device: str = "cpu",
        model_dir: Optional[str] = None,
        ctx_id: Optional[int] = None,
        min_face_size: int = 60,
        max_faces: int = 50,
        _verify_imports: bool = True,
    ) -> None:
        self.model_pack_name = model_pack_name
        self.det_size = int(det_size)
        self.det_thresh = float(det_thresh)
        self.nms_thresh = float(nms_thresh)
        self.device = device
        self.model_dir = (
            model_dir
            or os.environ.get("MODEL_DIR")
            or os.environ.get("MODELS_DIR")
            or os.path.expanduser("~/.insightface")
        )
        self.ctx_id = ctx_id if ctx_id is not None else self._device_to_ctx_id(device)
        self.min_face_size = int(min_face_size)
        self.max_faces = int(max_faces)

        # Eagerly verify the ``insightface`` dependency is importable so
        # that environments without it (e.g. CI without model files)
        # fail fast at construction time — the test-suite uses this to
        # decide whether to skip detector tests. The heavy ONNX model
        # itself is still loaded lazily on the first ``detect()`` call.
        if _verify_imports:
            try:
                import insightface  # type: ignore  # noqa: F401
                from insightface.app import FaceAnalysis  # type: ignore  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "insightface is not installed — `pip install insightface onnxruntime`"
                ) from exc

        # Lazy state
        self._model: Optional[object] = None  # insightface model_zoo detection model
        self._lock = threading.Lock()  # thread-safe lazy init
        self._loaded = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Run face detection on a single BGR frame.

        Parameters
        ----------
        frame:
            ``(H, W, 3)`` uint8 BGR image (OpenCV convention).

        Returns
        -------
        list[DetectionResult]
            One per detected face, sorted by descending score. Empty list
            if no faces are found.
        """
        if frame is None or frame.size == 0:
            return []

        # InsightFace expects BGR uint8 — already our convention.
        model = self._get_model()
        try:
            raw = model.get(frame)  # list of Face objects (or empty)
        except Exception as exc:
            logger.exception("detector.run_failed err=%s", exc)
            return []

        results: List[DetectionResult] = []
        for face_obj in raw:
            try:
                bbox = np.asarray(face_obj.bbox, dtype=np.float32).reshape(4)
                kps = np.asarray(face_obj.kps, dtype=np.float32).reshape(5, 2)
                score = float(face_obj.det_score)
            except AttributeError:
                # Some custom detectors expose different field names.
                bbox = np.asarray(getattr(face_obj, "bbox"), dtype=np.float32).reshape(4)
                kps = np.asarray(getattr(face_obj, "kps", np.zeros((5, 2))), dtype=np.float32).reshape(5, 2)
                score = float(getattr(face_obj, "det_score", 0.0))

            if score < self.det_thresh:
                continue
            # Min-size filter (reject tiny false positives).
            w = float(bbox[2] - bbox[0])
            h = float(bbox[3] - bbox[1])
            if w < self.min_face_size or h < self.min_face_size:
                continue
            results.append(DetectionResult(bbox=bbox, landmarks=kps, score=score))

        # Sort by score desc, cap to max_faces
        results.sort(key=lambda d: d.score, reverse=True)
        if len(results) > self.max_faces:
            results = results[: self.max_faces]
        return results

    def is_loaded(self) -> bool:
        """Return ``True`` if the underlying ONNX model has been loaded."""
        return self._loaded

    def warmup(self) -> None:
        """Force-load the model now (useful for pre-warming at startup)."""
        self._get_model()

    def reset(self) -> None:
        """Release the loaded model (next ``detect`` re-loads)."""
        with self._lock:
            self._model = None
            self._loaded = False

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _device_to_ctx_id(device: str) -> int:
        """Map a device string to an ONNX EP context id.

        ``"cpu"`` → ``-1`` (CPU EP)
        ``"cuda"`` / ``"cuda:0"`` → ``0``
        ``"cuda:N"`` → ``N``
        """
        if not device or device.lower() == "cpu":
            return -1
        if device.lower().startswith("cuda"):
            parts = device.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                return int(parts[1])
            return 0
        return -1

    def _get_model(self) -> object:
        """Lazy-load the InsightFace detection model (thread-safe)."""
        if self._model is not None:
            return self._model
        with self._lock:
            # Double-checked locking
            if self._model is not None:
                return self._model
            self._model = self._load_model()
            self._loaded = True
            return self._model

    def _load_model(self) -> object:
        """Build the InsightFace detection model.

        Raises
        ------
        RuntimeError
            If ``insightface`` is not installed, the model pack is not
            available on disk, or ONNX Runtime cannot start.
        """
        try:
            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "insightface is not installed — `pip install insightface onnxruntime`"
            ) from exc

        # The FaceAnalysis app bundles BOTH a detector and a recognition
        # model; we use it for the detector and ignore the recognition
        # output (FeatureExtractor uses the rec model directly).
        try:
            ctx_id, providers = select_onnx_runtime(self.ctx_id)
            app = FaceAnalysis(
                name=self.model_pack_name,
                root=self.model_dir,
                ctx_id=ctx_id,
                providers=providers,
                det_size=(self.det_size, self.det_size),
                nms_thresh=self.nms_thresh,
            )
            app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
        except Exception as exc:
            raise RuntimeError(
                f"failed to load InsightFace model pack '{self.model_pack_name}' "
                f"from '{self.model_dir}': {exc}"
            ) from exc

        # Wrap so we can swap out the .get() method without affecting
        # callers.
        return _InsightFaceDetectorAdapter(app)


class _InsightFaceDetectorAdapter:
    """Thin adapter that exposes only the detection half of an
    ``insightface.app.FaceAnalysis`` instance.

    We deliberately return the raw ``Face`` objects (with ``.bbox``,
    ``.kps``, ``.det_score`` attributes) so that :meth:`FaceDetector.detect`
    can stay decoupled from the insightface package — it only reads those
    three attributes.
    """

    def __init__(self, app: object) -> None:
        self._app = app

    def get(self, frame: np.ndarray) -> Sequence[object]:
        """Run detection; return the list of insightface ``Face`` objects."""
        return list(self._app.get(frame))  # type: ignore[arg-type]


__all__ = ["FaceDetector", "DetectionResult"]
