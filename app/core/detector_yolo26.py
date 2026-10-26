"""YOLO26-face detector module.

Drop-in alternative to :class:`app.core.detector.FaceDetector` that uses
**Ultralytics YOLO26** (released January 2026) with the WIDERFace-pretrained
``yolo26n-face`` / ``yolo26s-face`` weights from
https://github.com/akanametov/yolo-face.

Why use YOLO26 instead of SCRFD?
-------------------------------------

* **NMS-free end-to-end head** — YOLO26's optional one-to-one detection
  head produces final predictions without non-maximum suppression, which
  removes a source of latency variability on dense scenes.
* **Lighter box regression** — YOLO26 removes Distribution Focal Loss
  (DFL), reducing detection-head complexity while preserving an
  unconstrained regression range.
* **43% faster CPU ONNX inference** vs YOLO11 on Intel Xeon
  (Ultralytics benchmark, Jan 2026).
* **Native keypoint output** — the ``-face`` variants output 5 facial
  landmarks (right_eye, left_eye, nose, right_mouth, left_mouth) in
  exactly the ArcFace convention, so the downstream
  :class:`app.core.aligner.FaceAligner` works unchanged.

The class implements the same ``detect(frame) -> List[DetectionResult]``
interface as the SCRFD detector, so the rest of the pipeline
(`FaceAligner`, `FeatureExtractor`, `MotionTracker`, `LivenessVerifier`,
`AttendancePipeline`) needs **no changes**.

Switching detectors
-------------------

In ``configs/config.yaml``::

    cv:
      detector_backend: yolo26       # was: insightface (default)
      detector_model: yolo26s-face.pt
      detector_device: cpu           # cpu | cuda | cuda:0

Or via env vars::

    DETECTOR_BACKEND=yolo26
    DETECTOR_MODEL=yolo26s-face.pt
    DETECTOR_DEVICE=cuda

References
----------
* Ultralytics YOLO26 docs: https://docs.ultralytics.com/models/yolo26
* YOLO-face repo: https://github.com/akanametov/yolo-face
* Paper: "YOLO26: An Analysis of NMS-Free End to End Framework"
  arXiv:2601.12882
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.core.detector import DetectionResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# YoloFaceDetector
# --------------------------------------------------------------------------- #
class YoloFaceDetector:
    """YOLO26-based face detector with 5-landmark output.

    Parameters
    ----------
    model_path:
        Path to a YOLO26-face ``.pt`` weights file (``yolo26n-face.pt``
        or ``yolo26s-face.pt``). If a bare name like
        ``"yolo26n-face.pt"`` is given, the constructor searches the
        configured models directory and the repository's ``models/``
        directory.
    device:
        ``"cpu"``, ``"cuda"``, or ``"cuda:N"``. Passed through to
        Ultralytics' ``model.to(device)``.
    conf_threshold:
        Minimum detection confidence; boxes below this are dropped.
    iou_threshold:
        IoU threshold for the (now-optional, kept for compatibility)
        NMS pass. YOLO26's end-to-end head is NMS-free, but we still
        run a light NMS to deduplicate any residual overlap.
    imgsz:
        Inference input size (pixels). YOLO26-face is trained at 640.
    min_face_size:
        Reject detections whose shorter side is below this (noise filter).
    max_faces:
        Cap on the number of detections returned per frame (highest
        scoring first).
    """

    # YOLO-face model weights are downloaded separately by
    # scripts/download_models.py.
    DEFAULT_MODEL = "yolo26s-face.pt"
    SUPPORTED_MODELS = {"yolo26n-face.pt", "yolo26s-face.pt"}

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        min_face_size: int = 60,
        max_faces: int = 50,
        _verify_imports: bool = True,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.min_face_size = int(min_face_size)
        self.max_faces = int(max_faces)

        # Resolve the model path: if a bare name was given, look in
        # $MODELS_DIR / $MODEL_DIR first.
        self._resolved_path = self._resolve_model_path(model_path)

        # Eagerly verify ultralytics is importable so missing-dependency
        # failures surface at construction time (mirrors the SCRFD
        # detector's behaviour).
        if _verify_imports:
            try:
                import ultralytics  # type: ignore  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "ultralytics is not installed — `pip install ultralytics>=8.3.0`"
                ) from exc

        # Lazy state
        self._model: Optional[object] = None
        self._lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------ #
    # Public API — identical to app.core.detector.FaceDetector
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
            if no faces are found. Each ``DetectionResult`` carries the
            5 ArcFace-convention landmarks in ``.landmarks`` so the
            aligner / extractor pipeline works unchanged.
        """
        if frame is None or frame.size == 0:
            return []

        model = self._get_model()
        try:
            # Ultralytics accepts BGR or RGB; it converts internally.
            # verbose=False suppresses per-frame log spam.
            results = model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("yolo26.detect_failed err=%s", exc)
            return []

        detections: List[DetectionResult] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            keypoints = getattr(r, "keypoints", None)
            if boxes is None or len(boxes) == 0:
                continue

            # boxes.xyxy: (N, 4) tensor in pixel coords
            # boxes.conf: (N,) tensor
            # keypoints.xy: (N, K, 2) tensor — K=5 for face models
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)

            kpts_xy = None
            if keypoints is not None and hasattr(keypoints, "xy"):
                kpts_xy = (
                    keypoints.xy.cpu().numpy()
                    if hasattr(keypoints.xy, "cpu")
                    else np.asarray(keypoints.xy)
                )

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i][:4]
                score = float(confs[i])

                # Min-size filter (reject tiny false positives).
                w = float(x2 - x1)
                h = float(y2 - y1)
                if w < self.min_face_size or h < self.min_face_size:
                    continue

                # Extract 5 landmarks in ArcFace convention:
                #   [right_eye, left_eye, nose, right_mouth, left_mouth]
                # YOLO-face trains with this exact ordering, so the
                # downstream aligner's reference landmarks map 1:1.
                if kpts_xy is not None and i < len(kpts_xy):
                    landmarks = np.asarray(kpts_xy[i], dtype=np.float32).reshape(-1, 2)
                    # If the model returned more than 5 (unlikely for
                    # face variant), take the first 5. If fewer, pad
                    # with the bbox centre so downstream code doesn't
                    # crash on shape mismatch.
                    if landmarks.shape[0] < 5:
                        pad = np.full((5 - landmarks.shape[0], 2),
                                      float((x1 + x2) / 2 + (y1 + y2) / 2) / 2,
                                      dtype=np.float32)
                        landmarks = np.vstack([landmarks, pad])
                    elif landmarks.shape[0] > 5:
                        landmarks = landmarks[:5]
                else:
                    # Fallback: synthesize 5 landmarks from the bbox.
                    # This degrades alignment quality but keeps the
                    # pipeline running if the model didn't emit keypoints.
                    cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
                    ew = w * 0.15
                    landmarks = np.array(
                        [
                            [cx - ew, cy - ew * 0.5],   # right_eye
                            [cx + ew, cy - ew * 0.5],   # left_eye
                            [cx,        cy],             # nose
                            [cx - ew * 0.6, cy + ew * 1.2],   # right_mouth
                            [cx + ew * 0.6, cy + ew * 1.2],   # left_mouth
                        ],
                        dtype=np.float32,
                    )

                detections.append(
                    DetectionResult(
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        landmarks=landmarks,
                        score=score,
                        meta={"detector": "yolo26", "model": Path(self._resolved_path).name},
                    )
                )

        # Sort by score desc, cap to max_faces
        detections.sort(key=lambda d: d.score, reverse=True)
        if len(detections) > self.max_faces:
            detections = detections[: self.max_faces]
        return detections

    def is_loaded(self) -> bool:
        """Return ``True`` if the underlying model has been loaded."""
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
    def _resolve_model_path(model_path: str) -> str:
        """Resolve a (possibly bare) model filename to a real path.

        Search order:
        1. If the path exists as given, return it.
        2. Look in ``$MODELS_DIR`` / ``$MODEL_DIR``.
        3. Look in the configured models directory and repository ``models/``.
        4. Return the bare name; loading will report a missing-model error.
        """
        p = Path(model_path)
        if p.exists():
            return str(p.resolve())

        for env_var in ("MODELS_DIR", "MODEL_DIR"):
            base = os.environ.get(env_var)
            if base:
                candidate = Path(base) / model_path
                if candidate.exists():
                    return str(candidate.resolve())

        from app.config import settings

        project_models = Path(__file__).resolve().parents[2] / "models"
        configured_models = Path(settings.models_dir)
        if not configured_models.is_absolute():
            configured_models = project_models.parent / configured_models
        for base in (configured_models, project_models):
            candidate = base / model_path
            if candidate.exists():
                return str(candidate.resolve())

        return model_path

    def _get_model(self) -> object:
        """Lazy-load the YOLO26 model (thread-safe)."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            self._model = self._load_model()
            self._loaded = True
            return self._model

    def _load_model(self) -> object:
        """Build the YOLO26-face model.

        Raises
        ------
        RuntimeError
            If ``ultralytics`` is not installed or the model file cannot
            be loaded (corrupt weights, OOM, etc.).
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed — `pip install ultralytics>=8.3.0`"
            ) from exc

        try:
            logger.info(
                "yolo26.load.start model=%s device=%s imgsz=%d",
                self._resolved_path, self.device, self.imgsz,
            )
            model = YOLO(self._resolved_path)
            # Move to the requested device. Ultralytics accepts torch device strings.
            try:
                model.to(self.device)
            except Exception as exc:  # device not available — fall back to CPU
                logger.warning("yolo26.device_fallback requested=%s err=%s", self.device, exc)
                self.device = "cpu"
                model.to("cpu")
            logger.info("yolo26.load.done model=%s", self._resolved_path)
            return model
        except Exception as exc:
            raise RuntimeError(
                f"failed to load YOLO26 model '{self._resolved_path}': {exc}. "
                f"If the file is missing, Ultralytics will try to auto-download it "
                f"from https://github.com/akanametov/yolo-face/releases on first use."
            ) from exc


__all__ = ["YoloFaceDetector"]
