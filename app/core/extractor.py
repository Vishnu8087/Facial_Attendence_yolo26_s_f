"""ArcFace feature extractor module.

Wraps InsightFace's ArcFace recognition model (from the ``buffalo_l`` /
``buffalo_s`` packs) and produces L2-normalised 512-D embeddings suitable
for cosine-similarity matching against a pgvector database.

As with :mod:`app.core.detector`, the model is loaded lazily on first
use so the module can be imported in test environments without the
ONNX weights.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Sequence

import numpy as np

from app.core.utils import cosine_similarity, select_onnx_runtime

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Public dataclass
# --------------------------------------------------------------------------- #
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    """One feature vector + optional metadata."""

    embedding: np.ndarray
    norm: float
    shape: tuple = (512,)


# --------------------------------------------------------------------------- #
# FeatureExtractor
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    """ArcFace 512-D feature extractor.

    Parameters
    ----------
    model_pack_name:
        InsightFace model pack name (``"buffalo_l"`` default).
    device:
        ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.
    model_dir:
        InsightFace root directory. Falls back to ``MODEL_DIR`` /
        ``MODELS_DIR`` env vars, then ``~/.insightface``.
    embedding_dim:
        Expected embedding dimensionality (``512`` for ArcFace-R100 in
        buffalo_l). Used for runtime validation.
    input_size:
        Edge size of the aligned input chip the model expects (``112``).
    """

    #: Dimensionality of the embedding the model produces.
    EMBEDDING_DIM: int = 512

    def __init__(
        self,
        model_pack_name: str = "buffalo_l",
        device: str = "cpu",
        model_dir: Optional[str] = None,
        embedding_dim: int = 512,
        input_size: int = 112,
        _verify_imports: bool = True,
    ) -> None:
        self.model_pack_name = model_pack_name
        self.device = device
        self.model_dir = (
            model_dir
            or os.environ.get("MODEL_DIR")
            or os.environ.get("MODELS_DIR")
            or os.path.expanduser("~/.insightface")
        )
        self.embedding_dim = int(embedding_dim)
        self.input_size = int(input_size)

        # Eagerly verify ``insightface`` is importable so environments
        # without it fail fast at construction time (matches the
        # detector's behaviour and lets the test-suite skip cleanly).
        if _verify_imports:
            try:
                import insightface  # type: ignore  # noqa: F401
                from insightface.app import FaceAnalysis  # type: ignore  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "insightface is not installed — `pip install insightface onnxruntime`"
                ) from exc

        self._model: Optional[object] = None
        self._lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def embedding_dim_value(self) -> int:
        """Backwards-compatible alias for ``self.embedding_dim``."""
        return self.embedding_dim

    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        """Extract a single L2-normalised 512-D embedding.

        Parameters
        ----------
        aligned_face:
            ``(112, 112, 3)`` BGR uint8 face chip (output of
            :class:`app.core.aligner.FaceAligner`).

        Returns
        -------
        numpy.ndarray
            ``(512,)`` float32 vector with L2 norm == 1.
        """
        batch = self.extract_batch([aligned_face])
        if batch.shape[0] == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return batch[0]

    def extract_batch(self, faces: Sequence[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a batch of aligned face chips.

        Parameters
        ----------
        faces:
            Sequence of ``(H, W, 3)`` uint8 arrays. They need NOT all be
            the same size — the extractor resizes each to
            ``input_size × input_size`` before stacking.

        Returns
        -------
        numpy.ndarray
            ``(N, 512)`` float32 L2-normalised embeddings.
        """
        if not faces:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # InsightFace's forward method accepts RGB float32 data in NCHW
        # layout and applies the model's mean/std normalization itself.
        blob = self._preprocess(faces)
        model = self._get_model()
        try:
            embeddings = model.get_feat(blob)
        except Exception as exc:
            raise RuntimeError("ArcFace embedding inference failed") from exc

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        # Validate dim
        if embeddings.shape[1] != self.embedding_dim:
            # Some models output a different dim — accept it but warn.
            logger.warning(
                "extractor.dim_mismatch expected=%d got=%d", self.embedding_dim, embeddings.shape[1]
            )
            self.embedding_dim = int(embeddings.shape[1])

        # L2-normalise each row. Guard against zero rows.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms_safe = np.where(norms > 1e-12, norms, np.float32(1.0))
        embeddings = embeddings / norms_safe
        return embeddings.astype(np.float32, copy=False)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between two (already-normalised) embeddings.

        Thin wrapper around :func:`app.core.utils.cosine_similarity` —
        kept here so callers have a single import for "extract + match".
        """
        return cosine_similarity(a, b)

    def is_loaded(self) -> bool:
        return self._loaded

    def warmup(self) -> None:
        self._get_model()

    def reset(self) -> None:
        with self._lock:
            self._model = None
            self._loaded = False

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _preprocess(self, faces: Sequence[np.ndarray]) -> np.ndarray:
        """Resize, convert BGR to RGB, transpose to NCHW, and cast to float32.

        InsightFace ArcFace expects ``(N, 3, 112, 112)`` float32 input
        in RGB channel order. Its ``forward`` method applies mean/std
        normalization.
        """
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("cv2 is required for FeatureExtractor") from exc

        size = self.input_size
        chips: List[np.ndarray] = []
        for face in faces:
            f = np.asarray(face)
            if f.ndim != 3 or f.shape[2] != 3:
                # Promote grayscale / RGBA to BGR.
                if f.ndim == 2:
                    f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                elif f.shape[2] == 4:
                    f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
                else:
                    f = np.stack([f[..., 0]] * 3, axis=-1)
            if f.shape[0] != size or f.shape[1] != size:
                f = cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
            chips.append(f)

        # Stack into (N, H, W, 3) then transpose to NCHW.
        batch = np.stack(chips, axis=0).astype(np.float32)
        batch = batch[..., ::-1]  # BGR → RGB
        batch = np.transpose(batch, (0, 3, 1, 2))  # → NCHW
        return np.ascontiguousarray(batch, dtype=np.float32)

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            self._model = self._load_model()
            self._loaded = True
            return self._model

    def _load_model(self) -> object:
        """Load the ArcFace recognition model.

        Strategy
        --------
        InsightFace bundles the rec model inside the same pack as the
        detector. We instantiate a ``FaceAnalysis`` app and pluck out
        the ``models['recognition']`` entry to skip the detection step.

        Raises
        ------
        RuntimeError
            If insightface is not installed or the model cannot be loaded.
        """
        try:
            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "insightface is not installed — `pip install insightface onnxruntime`"
            ) from exc

        ctx_id = -1 if (not self.device or self.device.lower() == "cpu") else 0
        if self.device and self.device.lower().startswith("cuda:"):
            try:
                ctx_id = int(self.device.split(":")[1])
            except (ValueError, IndexError):
                ctx_id = 0

        try:
            ctx_id, providers = select_onnx_runtime(ctx_id)
            app = FaceAnalysis(
                name=self.model_pack_name,
                root=self.model_dir,
                ctx_id=ctx_id,
                providers=providers,
                allowed_modules=["detection", "recognition"],
                prepare_db=False,
            )
            app.prepare(ctx_id=ctx_id, det_size=(self.input_size, self.input_size))
        except Exception as exc:
            raise RuntimeError(
                f"failed to load ArcFace model pack '{self.model_pack_name}' "
                f"from '{self.model_dir}': {exc}"
            ) from exc

        return _InsightFaceRecAdapter(app)


class _InsightFaceRecAdapter:
    """Adapter that exposes only the recognition (ArcFace) part of an
    InsightFace ``FaceAnalysis`` app.

    InsightFace's standard API expects ``app.get(frame)`` which both
    detects AND embeds faces. We need a *batch* embedder that takes
    pre-aligned chips — so we pluck out the underlying ``rec_model``
    and call it directly.
    """

    def __init__(self, app: object) -> None:
        self._app = app
        # Pluck the recognition model out of the app's internal list.
        self._rec = None
        models = getattr(app, "models", {}) or {}
        rec = models.get("recognition")
        if rec is None and isinstance(models, list):  # older API
            for m in models:
                if getattr(m, "name", "") == "recognition":
                    rec = m
                    break
        if rec is None:
            raise RuntimeError(
                "ArcFace recognition model not found in the loaded pack"
            )
        self._rec = rec

    def get_feat(self, blob: np.ndarray) -> np.ndarray:
        """Run the recognition model on a pre-aligned NCHW blob.

        The model's ``forward`` method applies its configured normalization
        and returns an ``(N, 512)`` array of raw embeddings.
        """
        forward = getattr(self._rec, "forward", None)
        if forward is None:
            raise RuntimeError("InsightFace recognition model does not expose forward()")
        return forward(blob)


__all__ = ["FeatureExtractor", "ExtractionResult"]
