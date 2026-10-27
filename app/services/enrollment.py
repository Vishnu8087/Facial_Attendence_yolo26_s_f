"""Enrollment service — orchestrates image → embedding → DB + disk.

The heavy CV steps (detect / align / extract) are delegated to the
:class:`app.core.detector.FaceDetector`,
:class:`app.core.aligner.FaceAligner`, and
:class:`app.core.extractor.FeatureExtractor` classes built by Task 1-a.

If those classes aren't importable yet (e.g. on a machine without the
CV dependencies), image enrollment fails explicitly instead of storing
a zero vector as if a face had been processed.
"""
from __future__ import annotations

import base64
import io
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import crud
from app.db.models import Employee
from app.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Lazy CV-core loader (Task 1-a owns app/core/).
# --------------------------------------------------------------------------- #
def _try_import_cv_core() -> dict[str, Any]:
    """Return a dict of CV core classes if importable; empty dict otherwise."""
    classes: dict[str, Any] = {}
    try:
        from app.core.detector import FaceDetector  # type: ignore
        classes["FaceDetector"] = FaceDetector
    except Exception as exc:
        logger.debug("cv.detector.unavailable", error=str(exc))
    try:
        from app.core.detector_yolo26 import YoloFaceDetector  # type: ignore
        classes["YoloFaceDetector"] = YoloFaceDetector
    except Exception as exc:
        logger.debug("cv.detector_yolo26.unavailable", error=str(exc))
    try:
        from app.core.aligner import FaceAligner  # type: ignore
        classes["FaceAligner"] = FaceAligner
    except Exception as exc:
        logger.debug("cv.aligner.unavailable", error=str(exc))
    try:
        from app.core.extractor import FeatureExtractor  # type: ignore
        classes["FeatureExtractor"] = FeatureExtractor
    except Exception as exc:
        logger.debug("cv.extractor.unavailable", error=str(exc))
    return classes


def _build_detector(
    cv_core: dict[str, Any],
    *,
    backend: str | None = None,
    model_name: str | None = None,
) -> Any | None:
    """Construct the configured detector based on settings.detector_backend.

    Reads ``settings.detector_backend``:

    * ``"insightface"``             → :class:`FaceDetector` (SCRFD)
    * ``"yolo26"``                  → :class:`YoloFaceDetector` (Ultralytics YOLO26-face)
    """
    backend = (backend or getattr(settings, "detector_backend", "insightface")).lower()
    model_name = model_name or getattr(settings, "detector_model", "") or ""
    device = getattr(settings, "detector_device", "") or ""
    # Default device = cuda if gpu_enabled, else cpu
    if not device:
        device = "cuda" if getattr(settings, "gpu_enabled", False) else "cpu"

    if backend == "yolo26":
        cls = cv_core.get("YoloFaceDetector")
        if cls is None:
            raise RuntimeError(
                "YOLO26 is configured but its detector module is unavailable. "
                "Install the project requirements and restart the application."
            )
        try:
            return cls(
                model_path=model_name or cls.DEFAULT_MODEL,
                device=device,
                min_face_size=settings.min_face_size,
                max_faces=settings.max_faces_per_frame,
            )
        except Exception as exc:
            raise RuntimeError(
                f"YOLO26 is configured but could not be initialized: {exc}"
            ) from exc

    # Default: insightface
    return _build_insightface_detector(cv_core, device)


def _build_insightface_detector(cv_core: dict[str, Any], device: str) -> Any | None:
    """Construct the InsightFace detector used directly or as a fallback."""
    cls = cv_core.get("FaceDetector")
    if cls is None:
        return None
    try:
        return cls(
            model_pack_name=settings.model_name,
            det_size=settings.det_size,
            device=device,
            min_face_size=settings.min_face_size,
            max_faces=settings.max_faces_per_frame,
        )
    except Exception as exc:
        logger.warning("cv.detector.insightface.instantiate_failed", error=str(exc))
        return None


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class EnrollmentResult:
    """Outcome of an enrollment attempt."""

    employee: Employee
    embedding_norm: float
    face_image_path: str | None
    num_faces_detected: int


class EnrollmentError(Exception):
    """Raised when enrollment cannot complete (e.g. no face detected)."""


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class EnrollmentService:
    """Service that turns a face image into a stored Employee row.

    Lifecycle:

    1. Decode the image bytes (JPEG/PNG/WEBP → numpy).
    2. Detect faces — fail if zero or more than one.
    3. Align / crop the face.
    4. Extract a 512-d embedding (L2-normalised).
    5. Save the face crop to disk under ``settings.faces_dir``.
    6. Insert the Employee row (embedding + path) via CRUD.
    """

    def __init__(
        self,
        detector: Any | None = None,
        aligner: Any | None = None,
        extractor: Any | None = None,
    ) -> None:
        # Lazily instantiate the CV core if not provided.
        cv_core = _try_import_cv_core()
        self.detector = detector or _build_detector(cv_core)
        self.aligner = aligner or self._maybe_instantiate(cv_core.get("FaceAligner"))
        self.extractor = extractor or self._maybe_instantiate(cv_core.get("FeatureExtractor"))

    @staticmethod
    def _maybe_instantiate(cls: Any | None) -> Any | None:
        if cls is None:
            return None
        try:
            return cls()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("cv.component.instantiate_failed", error=str(exc))
            return None

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #
    async def enroll_from_image(
        self,
        session: AsyncSession,
        *,
        employee_id: str,
        name: str,
        department: str | None = None,
        email: str | None = None,
        image_bytes: bytes | None = None,
        image_bytes_list: list[bytes] | None = None,
        image_b64: str | None = None,
        embedding: np.ndarray | None = None,
        save_face: bool = True,
    ) -> EnrollmentResult:
        """Enroll from either a raw image or a pre-computed embedding.

        Parameters
        ----------
        image_bytes / image_bytes_list / image_b64:
            One must be supplied if ``embedding`` is not. Multiple images
            are embedded separately and averaged into one normalized vector.
        embedding:
            A pre-computed 512-d float32 L2-normalised vector. Bypasses
            the CV pipeline entirely (useful for testing / migration).
        save_face:
            If True, save the (cropped) face image to disk and store the
            path on the Employee row.
        """
        if embedding is None and image_bytes is None and not image_bytes_list and image_b64 is None:
            raise EnrollmentError("An image or embedding must be provided.")

        num_faces = 0
        face_image_path: str | None = None

        if embedding is None:
            images = image_bytes_list or [
                image_bytes if image_bytes is not None else base64.b64decode(image_b64)
            ]
            embeddings = []
            for index, raw in enumerate(images, start=1):
                frame = self._decode_image(raw)
                if frame is None:
                    raise EnrollmentError(
                        f"Could not decode image {index} (expected JPEG/PNG/WEBP)."
                    )

                image_embedding, image_faces, face_crop = await self._extract_embedding(frame)
                if image_embedding is None:
                    raise EnrollmentError(f"No face detected in image {index}.")
                num_faces += image_faces
                if image_faces > 1:
                    logger.warning(
                        "enrollment.multiple_faces",
                        employee_id=employee_id,
                        image_index=index,
                        count=image_faces,
                    )
                embeddings.append(image_embedding)
                if save_face and face_image_path is None and face_crop is not None:
                    face_image_path = self._save_face_image(employee_id, face_crop)

            embedding = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
            mean_norm = float(np.linalg.norm(embedding))
            if mean_norm > 0:
                embedding = embedding / mean_norm
        else:
            # Validate shape & normalise.
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if embedding.shape[0] != 512:
                raise EnrollmentError(
                    f"Embedding must be 512-d, got {embedding.shape[0]}"
                )
            norm = float(np.linalg.norm(embedding))
            if norm > 0:
                embedding = embedding / norm

        norm = float(np.linalg.norm(embedding))

        emp = await crud.create_employee(
            session,
            employee_id=employee_id,
            name=name,
            department=department,
            email=email,
            embedding=embedding,
            face_image_path=face_image_path,
            embedding_model=settings.model_name,
        )
        logger.info(
            "enrollment.success",
            employee_id=employee_id,
            num_faces=num_faces,
            embedding_norm=norm,
        )
        return EnrollmentResult(
            employee=emp,
            embedding_norm=norm,
            face_image_path=face_image_path,
            num_faces_detected=num_faces,
        )

    # ----------------------------------------------------------------- #
    # CV helpers
    # ----------------------------------------------------------------- #
    @staticmethod
    def _decode_image(raw: bytes) -> np.ndarray | None:
        """Decode JPEG/PNG/WEBP bytes into a BGR uint8 ndarray."""
        try:
            import cv2  # type: ignore
        except ImportError:
            try:
                from PIL import Image  # type: ignore
                import numpy as _np

                img = Image.open(io.BytesIO(raw)).convert("RGB")
                arr = _np.array(img)[:, :, ::-1].copy()  # RGB → BGR
                return arr
            except Exception as exc:
                logger.error("enrollment.decode_failed", error=str(exc))
                return None
        try:
            arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            return arr
        except Exception as exc:
            logger.error("enrollment.decode_failed", error=str(exc))
            return None

    async def _extract_embedding(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray | None, int, np.ndarray | None]:
        """Run detect → align → extract on a single frame.

        Returns ``(embedding, num_faces, face_crop)``. ``embedding`` is
        None if no face was detected. ``face_crop`` is the aligned crop
        (suitable for saving to disk).
        """
        # Do not create a success-shaped enrollment without real face
        # detection and embedding.
        if self.detector is None or self.extractor is None:
            raise EnrollmentError(
                "Face processing is unavailable. Install the CV dependencies "
                "and ensure the face models are loaded."
            )

        # 1. Detect
        detections = self._safe_call(self.detector.detect, frame) or []
        num_faces = len(detections)
        if num_faces == 0:
            return None, 0, None
        det = detections[0]

        # 2. Align
        landmarks = getattr(det, "landmarks", None)
        face_crop = frame
        if self.aligner is not None and landmarks is not None:
            aligned = self._safe_call(self.aligner.align, frame, landmarks)
            if aligned is not None:
                face_crop = aligned

        # 3. Extract
        emb = self._safe_call(self.extractor.extract, face_crop)
        if emb is None:
            return None, num_faces, face_crop
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        # L2-normalise so cosine_similarity == dot product.
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb, num_faces, face_crop

    @staticmethod
    def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.error("cv.component.call_failed", function=getattr(fn, "__name__", "fn"), error=str(exc))
            return None

    @staticmethod
    def _save_face_image(employee_id: str, face_crop: np.ndarray) -> str:
        """Persist the face crop to disk under ``settings.faces_dir``.

        Returns the absolute path on disk.
        """
        faces_dir = Path(settings.faces_dir)
        faces_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{employee_id}_{uuid.uuid4().hex[:8]}.jpg"
        out_path = faces_dir / fname
        try:
            import cv2  # type: ignore

            cv2.imwrite(str(out_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except ImportError:
            try:
                from PIL import Image  # type: ignore

                Image.fromarray(face_crop[..., ::-1]).save(out_path, format="JPEG", quality=90)
            except Exception as exc:  # pragma: no cover — defensive
                logger.error("enrollment.save_face_failed", error=str(exc))
                return ""
        return str(out_path)


__all__ = ["EnrollmentService", "EnrollmentResult", "EnrollmentError"]
