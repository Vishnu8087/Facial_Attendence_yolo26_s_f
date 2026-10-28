"""End-to-end attendance pipeline orchestrating the CV modules.

The pipeline wires together the five CV components — detector, aligner,
extractor, tracker, liveness — and exposes a single entry point
:meth:`AttendancePipeline.process_frame`. It supports three runtime
strategies (the toggle required by orchestrator task ``0-amend``):

* ``"both"`` (default) — full pipeline:
  detect → align → track → liveness → extract → match → log.
* ``"detection_only"`` — Strategy 1 standalone. Skip tracker + liveness;
  re-detect every frame, match against DB immediately. ``track_id``
  in the result is ``None`` and ``liveness_passed`` is always ``False``.
* ``"tracking_only"`` — Strategy 2 standalone. Skip embedding
  extraction + DB match; just track + liveness. ``identity`` is ``None``
  and ``similarity`` is ``0.0``. No attendance logging.

The pipeline is thread-safe — a single instance can be shared across
WebSocket workers. Per-track embedding / identity caches are guarded by
a re-entrant lock.

Attendance logging is delegated to an injected ``logger`` object (any
object with a ``log_attendance(...)`` method) and only fires when:

1. the strategy is ``"both"`` (logging requires an identity match), AND
2. identity is not ``None`` (matched above ``similarity_threshold``), AND
3. ``liveness_passed`` is ``True``, AND
4. the same identity has not been logged within ``cooldown_seconds``.

Identity matching is delegated to an optional ``matcher`` object (any
object with a ``find_nearest(embedding, threshold) -> (identity, sim)``
method). When ``matcher`` is ``None`` (e.g. in tests) no identity is
ever returned and no attendance is logged.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.core.detector import DetectionResult
from app.core.tracker import Track

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Strategy enum (kept as plain strings for WebSocket-friendliness)
# --------------------------------------------------------------------------- #
STRATEGY_BOTH = "both"
STRATEGY_DETECTION_ONLY = "detection_only"
STRATEGY_TRACKING_ONLY = "tracking_only"
_VALID_STRATEGIES = (STRATEGY_BOTH, STRATEGY_DETECTION_ONLY, STRATEGY_TRACKING_ONLY)


# --------------------------------------------------------------------------- #
# Public dataclasses — these are the wire format consumed by Task 1-b's
# WebSocket / REST handlers.
# --------------------------------------------------------------------------- #
@dataclass
class TrackResult:
    """One row of the per-frame result.

    Attributes
    ----------
    track_id:
        Persistent track ID, or ``None`` in ``detection_only`` mode.
    bbox:
        Pixel bbox ``(x1, y1, x2, y2)``.
    identity:
        ``{"employee_id": str, "name": str}`` if matched, else ``None``.
        Always ``None`` in ``tracking_only`` mode.
    similarity:
        Cosine similarity to the matched identity (``0.0`` if no match
        or ``tracking_only``).
    liveness_passed:
        ``True`` if the liveness verifier approved this track.
        ``None`` when liveness is not run in ``detection_only`` mode.
    liveness_confidence:
        Confidence score in ``[0, 1]`` from the liveness verifier, or
        ``None`` when liveness is not run.
    strategy:
        Which strategy produced this row.
    """

    track_id: Optional[int]
    bbox: Tuple[int, int, int, int]
    identity: Optional[Dict[str, Any]]
    similarity: float
    liveness_passed: Optional[bool]
    liveness_confidence: Optional[float]
    strategy: str

    def __post_init__(self) -> None:
        # Coerce bbox to a clean tuple of ints for JSON-friendliness.
        self.bbox = tuple(int(v) for v in self.bbox)  # type: ignore[assignment]
        self.similarity = float(self.similarity)
        if self.liveness_confidence is not None:
            self.liveness_confidence = float(self.liveness_confidence)


@dataclass
class FrameResult:
    """Top-level result of :meth:`AttendancePipeline.process_frame`."""

    annotated_frame: np.ndarray
    tracks: List[TrackResult]
    fps: float
    strategy: str
    timestamp: str
    # Optional stats (populated even in error paths).
    num_detections: int = 0
    num_logged: int = 0
    error: Optional[str] = None

    def __post_init__(self) -> None:
        self.fps = float(self.fps)
        self.num_detections = int(self.num_detections)
        self.num_logged = int(self.num_logged)

    # ``detections`` is kept as an alias of ``tracks`` so callers (and
    # tests) can use either name.
    @property
    def detections(self) -> List[TrackResult]:
        return self.tracks


# --------------------------------------------------------------------------- #
# AttendancePipeline
# --------------------------------------------------------------------------- #
class AttendancePipeline:
    """Orchestrates detect → align → track → liveness → extract → match → log.

    Parameters
    ----------
    detector:
        Object with a ``detect(frame) -> List[DetectionResult]`` method.
    aligner:
        Object with an ``align(frame, landmarks, size=...) -> np.ndarray``
        method (typically :class:`app.core.aligner.FaceAligner`).
    extractor:
        Object with an ``extract(aligned_face) -> np.ndarray`` method
        (typically :class:`app.core.extractor.FeatureExtractor`).
    tracker:
        Object with an ``update(detections, frame) -> List[Track]``
        method (typically :class:`app.core.tracker.MotionTracker`).
    liveness:
        Object with a ``verify(frame, landmarks, track_id)`` method
        returning either a :class:`LivenessResult` or a dict with at
        least ``is_real`` and ``score`` keys.
    logger:
        Optional attendance logger (any object exposing
        ``log_attendance(...)``). If ``None``, no logging is performed.
    matcher:
        Optional identity matcher. Any object exposing
        ``find_nearest(embedding: np.ndarray, threshold: float)
        -> (identity: dict | None, similarity: float)``.
        If ``None``, identities are never resolved.
    cooldown_seconds:
        Min wall-clock seconds between two logs for the same identity.
        Default ``30.0``; the orchestrator's spec also mentions an
        8-hour default for production deployments — set this from env.
    similarity_threshold:
        Cosine-similarity cutoff above which an embedding is considered
        a match. Default ``0.68`` per the orchestrator's spec.
    reverify_every_n_frames:
        Re-extract + re-match a track's identity every N frames. Between
        re-verifications the cached identity is reused. Default ``30``.
    min_face_size:
        Faces smaller than this (in pixels) are skipped.
    draw_annotations:
        If ``True`` (default), the returned ``annotated_frame`` has
        bboxes / labels drawn on it.
    """

    def __init__(
        self,
        detector: Any,
        aligner: Any,
        extractor: Any,
        tracker: Any,
        liveness: Any,
        logger: Any = None,
        matcher: Any = None,
        cooldown_seconds: float = 30.0,
        similarity_threshold: float = 0.68,
        reverify_every_n_frames: int = 30,
        min_face_size: int = 60,
        draw_annotations: bool = True,
        default_strategy: str = STRATEGY_BOTH,
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.extractor = extractor
        self.tracker = tracker
        self.liveness = liveness
        self.logger = logger
        self.matcher = matcher
        self.cooldown_seconds = float(cooldown_seconds)
        self.similarity_threshold = float(similarity_threshold)
        self.reverify_every_n_frames = int(reverify_every_n_frames)
        self.min_face_size = int(min_face_size)
        self.draw_annotations = bool(draw_annotations)
        self.default_strategy = default_strategy if default_strategy in _VALID_STRATEGIES else STRATEGY_BOTH

        # Per-track caches — guarded by ``_lock`` for thread safety.
        self._lock = threading.RLock()
        self._track_embeddings: Dict[Any, np.ndarray] = {}
        self._track_identities: Dict[Any, Optional[Dict[str, Any]]] = {}
        self._track_last_extract_frame: Dict[Any, int] = {}
        self._track_liveness: Dict[Any, Any] = {}
        # Cooldown: identity-key -> last log timestamp.
        self._logged_at: Dict[str, float] = {}
        # FPS smoothing.
        self._last_frame_time: Optional[float] = None
        self._fps_ema: float = 0.0
        # Frame counter (monotonic per-instance).
        self._frame_index: int = 0
        # Total logs emitted (for stats / metrics).
        self._total_logged: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_frame(
        self,
        frame: np.ndarray,
        strategy: Optional[str] = None,
        detections: Optional[List[DetectionResult]] = None,
    ) -> FrameResult:
        """Run one frame through the pipeline.

        Parameters
        ----------
        frame:
            ``(H, W, 3)`` uint8 BGR image.
        strategy:
            Pipeline strategy — ``"both"`` (default), ``"detection_only"``,
            or ``"tracking_only"``. Invalid values raise ``ValueError``.
        detections:
            Optional precomputed detector output used by the streaming
            pipeline's parallel detection stage.

        Returns
        -------
        FrameResult
            See module docstring for the schema.
        """
        strat = strategy or self.default_strategy
        if strat not in _VALID_STRATEGIES:
            raise ValueError(
                f"invalid strategy '{strat}'; expected one of {_VALID_STRATEGIES}"
            )

        start = time.monotonic()
        with self._lock:
            self._frame_index += 1
            frame_idx = self._frame_index

        # ---- 1. Detect (always) ----------------------------------------
        try:
            detections = (
                self._safe_detect(frame)
                if detections is None
                else list(detections)
            )
        except Exception as exc:
            logger.exception("pipeline.detect_failed err=%s", exc)
            return self._empty_result(frame, strat, error=f"detect_failed: {exc!r}")

        # ---- 2. Dispatch on strategy ----------------------------------
        try:
            if strat == STRATEGY_DETECTION_ONLY:
                result = self._run_detection_only(frame, detections, strat, frame_idx)
            elif strat == STRATEGY_TRACKING_ONLY:
                result = self._run_tracking_only(frame, detections, strat, frame_idx)
            else:  # STRATEGY_BOTH
                result = self._run_both(frame, detections, strat, frame_idx)
        except Exception as exc:
            logger.exception("pipeline.run_failed strategy=%s err=%s", strat, exc)
            return self._empty_result(frame, strat, error=f"run_failed: {exc!r}")

        # ---- 3. FPS smoothing -----------------------------------------
        result.fps = self._update_fps(start)
        return result

    def reset(self) -> None:
        """Clear all per-track caches and cooldown state."""
        with self._lock:
            self._track_embeddings.clear()
            self._track_identities.clear()
            self._track_last_extract_frame.clear()
            self._track_liveness.clear()
            self._logged_at.clear()
            self._frame_index = 0
            self._total_logged = 0
            self._last_frame_time = None
            self._fps_ema = 0.0
        # Reset tracker if it supports it.
        if hasattr(self.tracker, "reset"):
            try:
                self.tracker.reset()
            except Exception:
                pass
        if hasattr(self.liveness, "reset"):
            try:
                self.liveness.reset()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #
    def _run_detection_only(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        strategy: str,
        frame_idx: int,
    ) -> FrameResult:
        """Strategy 1 — re-detect every frame, align+extract+match each."""
        tracks: List[TrackResult] = []
        annotated = np.asarray(frame).copy() if self.draw_annotations else np.asarray(frame)

        aligned_detections: list[tuple[DetectionResult, Optional[np.ndarray]]] = []
        chips: list[np.ndarray] = []
        for det in detections:
            if not self._passes_min_face(det):
                continue
            try:
                chip = self.aligner.align(frame, det.landmarks)
            except Exception as exc:
                logger.debug("aligner.align raised: %s", exc)
                chip = None
            aligned_detections.append((det, chip))
            if chip is not None:
                chips.append(chip)

        embeddings: list[Optional[np.ndarray]] = []
        if chips:
            try:
                if callable(
                    getattr(type(self.extractor), "extract_multithreaded", None)
                ):
                    batch = self.extractor.extract_multithreaded(chips)
                else:
                    batch = np.stack(
                        [self.extractor.extract(chip) for chip in chips],
                        axis=0,
                    )
                if len(batch) != len(chips):
                    raise RuntimeError(
                        "ArcFace returned a different number of embeddings than input chips"
                    )
                embeddings = [np.asarray(embedding) for embedding in batch]
            except Exception as exc:
                logger.warning("batch ArcFace extraction failed: %s", exc)
                embeddings = []
                for chip in chips:
                    try:
                        embeddings.append(self.extractor.extract(chip))
                    except Exception as fallback_exc:
                        logger.debug("extractor.extract raised: %s", fallback_exc)
                        embeddings.append(None)

        embedding_index = 0
        for det, chip in aligned_detections:
            if chip is None:
                identity, sim = None, 0.0
            else:
                emb = embeddings[embedding_index] if embedding_index < len(embeddings) else None
                embedding_index += 1
                identity, sim = self._match(emb) if emb is not None else (None, 0.0)
            tr = TrackResult(
                track_id=None,
                bbox=self._bbox_tuple(det.bbox),
                identity=identity,
                similarity=sim,
                liveness_passed=None,
                liveness_confidence=None,
                strategy=strategy,
            )
            tracks.append(tr)

            # In detection_only we still try to log (no liveness gate).
            # The cooldown still applies.
            if identity is not None:
                self._try_log(
                    identity=identity,
                    similarity=sim,
                    liveness_passed=None,
                    track_id=None,
                )

        if self.draw_annotations:
            annotated = self._annotate(annotated, tracks, strategy)
        return FrameResult(
            annotated_frame=annotated,
            tracks=tracks,
            fps=0.0,  # filled in by caller
            strategy=strategy,
            timestamp=_iso_now(),
            num_detections=len(detections),
            num_logged=self._total_logged,
        )

    def _maybe_reextract_tracks_and_match(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        frame_idx: int,
    ) -> List[Tuple[Optional[Dict[str, Any]], float]]:
        """Batch align and extract uncached tracks without changing cache timing."""
        if len(tracks) <= 2 or not callable(
            getattr(type(self.extractor), "extract_multithreaded", None)
        ):
            return [
                self._maybe_reextract_and_match(frame, track, frame_idx)
                for track in tracks
            ]

        needs_extract: list[Track] = []
        with self._lock:
            for track in tracks:
                last_extract = self._track_last_extract_frame.get(track.track_id)
                if (
                    last_extract is None
                    or (frame_idx - last_extract) >= self.reverify_every_n_frames
                ):
                    needs_extract.append(track)
        if len(needs_extract) <= 2:
            return [
                self._maybe_reextract_and_match(frame, track, frame_idx)
                for track in tracks
            ]

        aligned_tracks: list[Track] = []
        chips: list[np.ndarray] = []
        for track in needs_extract:
            try:
                chips.append(self.aligner.align(frame, track.landmarks))
                aligned_tracks.append(track)
            except Exception as exc:
                logger.debug("aligner.align raised for track %s: %s", track.track_id, exc)

        try:
            embeddings = self.extractor.extract_multithreaded(chips)
            if len(embeddings) != len(aligned_tracks):
                raise RuntimeError(
                    "ArcFace returned a different number of embeddings than input chips"
                )
        except Exception as exc:
            logger.warning("extractor.extract_multithreaded failed: %s", exc)
            return [
                self._maybe_reextract_and_match(frame, track, frame_idx)
                for track in tracks
            ]

        precomputed = {
            track.track_id: embedding
            for track, embedding in zip(aligned_tracks, embeddings)
        }
        return [
            self._maybe_reextract_and_match(
                frame,
                track,
                frame_idx,
                precomputed_embedding=precomputed.get(track.track_id),
            )
            for track in tracks
        ]

    def _run_tracking_only(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        strategy: str,
        frame_idx: int,
    ) -> FrameResult:
        """Strategy 2 — track + liveness only, no extraction / match / log."""
        tracked: List[Track] = self._safe_tracker_update(detections, frame)
        tracks: List[TrackResult] = []
        annotated = np.asarray(frame).copy() if self.draw_annotations else np.asarray(frame)

        for tr in tracked:
            liveness_result = self._verify_liveness(frame, tr.landmarks, tr.track_id)
            is_live = self._extract_is_live(liveness_result)
            conf = self._extract_liveness_confidence(liveness_result)
            tracks.append(
                TrackResult(
                    track_id=int(tr.track_id),
                    bbox=self._bbox_tuple(tr.bbox),
                    identity=None,  # No identity in tracking_only
                    similarity=0.0,
                    liveness_passed=bool(is_live),
                    liveness_confidence=float(conf),
                    strategy=strategy,
                )
            )

        if self.draw_annotations:
            annotated = self._annotate(annotated, tracks, strategy)
        return FrameResult(
            annotated_frame=annotated,
            tracks=tracks,
            fps=0.0,
            strategy=strategy,
            timestamp=_iso_now(),
            num_detections=len(detections),
            num_logged=self._total_logged,
        )

    def _run_both(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        strategy: str,
        frame_idx: int,
    ) -> FrameResult:
        """Full pipeline — detect → align → track → liveness → extract → match → log."""
        tracked: List[Track] = self._safe_tracker_update(detections, frame)
        tracks: List[TrackResult] = []
        annotated = np.asarray(frame).copy() if self.draw_annotations else np.asarray(frame)

        identities = self._maybe_reextract_tracks_and_match(frame, tracked, frame_idx)

        for tr, (identity, sim) in zip(tracked, identities):
            # ---- Liveness ----
            liveness_result = self._verify_liveness(frame, tr.landmarks, tr.track_id)
            is_live = self._extract_is_live(liveness_result)
            conf = self._extract_liveness_confidence(liveness_result)

            tracks.append(
                TrackResult(
                    track_id=int(tr.track_id),
                    bbox=self._bbox_tuple(tr.bbox),
                    identity=identity,
                    similarity=float(sim),
                    liveness_passed=bool(is_live),
                    liveness_confidence=float(conf),
                    strategy=strategy,
                )
            )

            # ---- Log if all conditions met ----
            if identity is not None and is_live:
                self._try_log(
                    identity=identity,
                    similarity=sim,
                    liveness_passed=True,
                    track_id=int(tr.track_id),
                )

        if self.draw_annotations:
            annotated = self._annotate(annotated, tracks, strategy)
        return FrameResult(
            annotated_frame=annotated,
            tracks=tracks,
            fps=0.0,
            strategy=strategy,
            timestamp=_iso_now(),
            num_detections=len(detections),
            num_logged=self._total_logged,
        )

    # ------------------------------------------------------------------ #
    # CV helpers
    # ------------------------------------------------------------------ #
    def _safe_detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Run the detector, returning ``[]`` on failure (never raises)."""
        if frame is None or not hasattr(frame, "shape"):
            return []
        try:
            return list(self.detector.detect(frame))
        except Exception as exc:
            logger.warning("detector.detect raised: %s", exc)
            return []

    def _safe_tracker_update(
        self,
        detections: List[DetectionResult],
        frame: np.ndarray,
    ) -> List[Track]:
        """Run ``tracker.update`` defensively.

        The spec calls for ``update(detections, frame)`` but some
        lightweight trackers (and several test mocks) only accept
        ``update(detections)`` — we try the 2-arg form first and fall
        back to the 1-arg form on ``TypeError``.
        """
        try:
            return list(self.tracker.update(detections, frame))
        except TypeError:
            try:
                return list(self.tracker.update(detections))
            except Exception as exc:
                logger.warning("tracker.update raised: %s", exc)
                return []
        except Exception as exc:
            logger.warning("tracker.update raised: %s", exc)
            return []

    def _passes_min_face(self, det: DetectionResult) -> bool:
        """Apply the min-face-size filter to a detection."""
        try:
            x1, y1, x2, y2 = (float(v) for v in det.bbox)
            return (x2 - x1) >= self.min_face_size and (y2 - y1) >= self.min_face_size
        except Exception:
            return True

    def _extract_and_match(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Align + extract + match (no caching). Used by detection_only."""
        try:
            aligned = self.aligner.align(frame, landmarks)
        except Exception as exc:
            logger.debug("aligner.align raised: %s", exc)
            return None, 0.0
        try:
            emb = self.extractor.extract(aligned)
        except Exception as exc:
            logger.debug("extractor.extract raised: %s", exc)
            return None, 0.0
        return self._match(emb)

    def _maybe_reextract_and_match(
        self,
        frame: np.ndarray,
        track: Track,
        frame_idx: int,
        *,
        precomputed_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Return cached identity for the track, re-extracting every N frames.

        The very first time a track is seen we always extract. After that
        we re-extract when ``frame_idx - last_extract_frame >= reverify_every_n_frames``.
        """
        with self._lock:
            tid = track.track_id
            last_extract = self._track_last_extract_frame.get(tid)
            cached_id = self._track_identities.get(tid)
            # Need to extract if: never extracted OR reverify window elapsed
            need_extract = (
                last_extract is None
                or (frame_idx - last_extract) >= self.reverify_every_n_frames
            )

        if not need_extract:
            # Re-use cached identity + similarity. Re-compute similarity
            # from the cached embedding so callers always get a number.
            with self._lock:
                cached_emb = self._track_embeddings.get(tid)
                cached_id = self._track_identities.get(tid)
            sim = 0.0
            if cached_emb is not None and cached_id is not None and self.matcher is not None:
                try:
                    _, sim = self.matcher.find_nearest(cached_emb, self.similarity_threshold)
                except Exception:
                    sim = 0.0
            return cached_id, float(sim)

        # Extract + match, then cache.
        identity, sim, emb = None, 0.0, None
        try:
            if precomputed_embedding is None:
                aligned = self.aligner.align(frame, track.landmarks)
                emb = self.extractor.extract(aligned)
            else:
                emb = precomputed_embedding
            identity, sim = self._match(emb)
        except Exception as exc:
            logger.debug("extract+match raised for track %s: %s", tid, exc)

        with self._lock:
            self._track_embeddings[tid] = emb
            self._track_identities[tid] = identity
            self._track_last_extract_frame[tid] = frame_idx
        return identity, float(sim)

    def _match(self, emb: np.ndarray) -> Tuple[Optional[Dict[str, Any]], float]:
        """Look up the closest identity in the matcher (if any)."""
        if emb is None or self.matcher is None:
            return None, 0.0
        try:
            identity, sim = self.matcher.find_nearest(emb, self.similarity_threshold)
        except Exception as exc:
            logger.debug("matcher.find_nearest raised: %s", exc)
            return None, 0.0
        if identity is None:
            return None, float(sim)
        return identity, float(sim)

    def _verify_liveness(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray],
        track_id: Any,
    ) -> Any:
        """Run the liveness verifier; never raises."""
        if self.liveness is None or landmarks is None:
            return None
        try:
            result = self.liveness.verify(frame, landmarks, track_id)
        except Exception as exc:
            logger.debug("liveness.verify raised: %s", exc)
            return None
        with self._lock:
            self._track_liveness[track_id] = result
        return result

    @staticmethod
    def _extract_is_live(liveness_result: Any) -> bool:
        """Pull ``is_real`` / ``is_live`` from either a dict or a dataclass."""
        if liveness_result is None:
            return False
        if isinstance(liveness_result, dict):
            return bool(liveness_result.get("is_real", liveness_result.get("is_live", False)))
        return bool(
            getattr(liveness_result, "is_real", getattr(liveness_result, "is_live", False))
        )

    @staticmethod
    def _extract_liveness_confidence(liveness_result: Any) -> float:
        """Pull the confidence / score from either a dict or a dataclass."""
        if liveness_result is None:
            return 0.0
        if isinstance(liveness_result, dict):
            return float(liveness_result.get("score", liveness_result.get("confidence", 0.0)))
        return float(
            getattr(
                liveness_result,
                "confidence",
                getattr(liveness_result, "score", 0.0),
            )
        )

    # ------------------------------------------------------------------ #
    # Logging + cooldown
    # ------------------------------------------------------------------ #
    def _try_log(
        self,
        identity: Dict[str, Any],
        similarity: float,
        liveness_passed: Optional[bool],
        track_id: Optional[int],
    ) -> bool:
        """Call ``logger.log_attendance`` if cooldown allows. Returns True if logged."""
        if self.logger is None:
            return False
        # Build a stable identity key (employee_id preferred).
        key = str(identity.get("employee_id") or identity.get("name") or id(identity))
        now = time.monotonic()
        with self._lock:
            last = self._logged_at.get(key)
            if last is not None and (now - last) < self.cooldown_seconds:
                return False
            self._logged_at[key] = now
            self._total_logged += 1

        # Build the log payload (defensive — accept any logger shape).
        payload = {
            "identity": identity,
            "similarity": float(similarity),
            "liveness_passed": liveness_passed,
            "track_id": track_id,
            "timestamp": _iso_now(),
        }
        try:
            # Prefer a structured method; fall back to __call__.
            if hasattr(self.logger, "log_attendance"):
                self.logger.log_attendance(payload)
            elif callable(self.logger):
                self.logger(payload)
            else:
                logger.warning("logger has no log_attendance method and is not callable")
                return False
        except Exception as exc:
            logger.warning("logger.log_attendance raised: %s", exc)
            return False
        return True

    # ------------------------------------------------------------------ #
    # Annotation + utility
    # ------------------------------------------------------------------ #
    def _annotate(
        self,
        frame: np.ndarray,
        tracks: List[TrackResult],
        strategy: str,
    ) -> np.ndarray:
        """Draw bboxes + labels on the frame in-place."""
        try:
            from app.core.utils import draw_boxes  # local import keeps utils optional
        except Exception:
            return frame
        boxes = [t.bbox for t in tracks]
        labels = []
        for t in tracks:
            name = (t.identity or {}).get("name") or f"track={t.track_id}"
            tag = (
                "N/A"
                if t.liveness_passed is None
                else ("LIVE" if t.liveness_passed else "spoof?")
            )
            labels.append(f"{name} sim={t.similarity:.2f} [{tag}]")
        return draw_boxes(frame, boxes, labels=labels)

    @staticmethod
    def _bbox_tuple(bbox: Any) -> Tuple[int, int, int, int]:
        try:
            arr = np.asarray(bbox, dtype=np.float32).ravel()
            return (int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]))
        except Exception:
            return (0, 0, 0, 0)

    def _update_fps(self, start: float) -> float:
        """Update the EMA-smoothed FPS estimate."""
        now = time.monotonic()
        if self._last_frame_time is None:
            self._last_frame_time = start
            return 0.0
        elapsed = max(now - start, 1e-6)
        instant_fps = 1.0 / elapsed
        # EMA smoothing (α=0.1).
        if self._fps_ema == 0.0:
            self._fps_ema = instant_fps
        else:
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * instant_fps
        self._last_frame_time = now
        return float(self._fps_ema)

    def _empty_result(
        self,
        frame: np.ndarray,
        strategy: str,
        error: Optional[str] = None,
    ) -> FrameResult:
        return FrameResult(
            annotated_frame=np.asarray(frame).copy() if frame is not None else np.zeros((1, 1, 3), dtype=np.uint8),
            tracks=[],
            fps=0.0,
            strategy=strategy,
            timestamp=_iso_now(),
            num_detections=0,
            num_logged=self._total_logged,
            error=error,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _iso_now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "AttendancePipeline",
    "TrackResult",
    "FrameResult",
    "STRATEGY_BOTH",
    "STRATEGY_DETECTION_ONLY",
    "STRATEGY_TRACKING_ONLY",
]
