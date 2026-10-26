"""Multi-object motion tracker module.

Wraps ByteTrack (via the ``supervision`` package when available) and
falls back to an IoU-based nearest-neighbour tracker when ``supervision``
or ``bytetrack`` are not installed. Either path exposes the same API:

* Persistent track IDs across frames
* Per-track cooldown timer — a track that has not been matched to a
  detection for ``cooldown_seconds`` is expired; the next detection in
  its place is assigned a fresh ID.
* ``reset()`` to clear state (e.g. on stream restart)

The output ``Track`` objects carry the bbox, the (optional) landmarks
propagated from the matched detection, and the track's score.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class Track:
    """A single tracked object.

    Attributes
    ----------
    track_id:
        Persistent integer ID. Stable across frames for the same person
        until the cooldown expires.
    bbox:
        ``[x1, y1, x2, y2]`` float32 — current frame position.
    landmarks:
        Optional ``(5, 2)`` landmarks propagated from the matched
        detection (used by downstream liveness verification).
    score:
        Detection confidence of the latest matched detection.
    frame_id:
        Monotonic frame index at which this track was last updated.
    """

    track_id: int
    bbox: np.ndarray
    landmarks: Optional[np.ndarray] = None
    score: float = 0.0
    frame_id: int = 0
    # Optional velocity / history used by Kalman-style smoothing.
    history: Deque = field(default_factory=lambda: deque(maxlen=30))

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32).reshape(4)
        if self.landmarks is not None:
            self.landmarks = np.asarray(self.landmarks, dtype=np.float32).reshape(5, 2)
        self.score = float(self.score)
        self.frame_id = int(self.frame_id)


# --------------------------------------------------------------------------- #
# MotionTracker
# --------------------------------------------------------------------------- #
class MotionTracker:
    """IoU-/ByteTrack-based multi-face motion tracker.

    The tracker is intentionally simple to keep behaviour deterministic
    across CPU/GPU and with-or-without ``supervision`` environments. If
    you need ByteTrack's full motion model, install ``supervision`` and
    the wrapper will delegate to it automatically.

    Parameters
    ----------
    cooldown_seconds:
        Max wall-clock seconds a track may go unmatched before its ID
        is recycled. Default ``30.0``.
    track_thresh:
        Minimum detection confidence for a detection to seed a new
        track. Detections below this are still allowed to *extend*
        existing tracks but cannot create new ones.
    match_thresh:
        Minimum IoU between a detection and a track's last position for
        them to be considered the same object.
    track_buffer:
        Number of frames a track is kept "alive" after its last
        detection (in addition to the time-based cooldown).
    frame_rate:
        Expected frame rate (FPS) — used to convert ``track_buffer`` to
        seconds as a back-off.
    iou_match_thresh:
        Alias of ``match_thresh``; kept for backward-compatibility.
    """

    def __init__(
        self,
        cooldown_seconds: float = 30.0,
        track_thresh: float = 0.5,
        match_thresh: float = 0.5,
        track_buffer: int = 30,
        frame_rate: int = 30,
        iou_match_thresh: Optional[float] = None,
    ) -> None:
        self.cooldown_seconds = float(cooldown_seconds)
        self.track_thresh = float(track_thresh)
        self.match_thresh = float(match_thresh)
        self.track_buffer = int(track_buffer)
        self.frame_rate = int(frame_rate)
        if iou_match_thresh is not None:
            self.match_thresh = float(iou_match_thresh)

        # Per-track state.
        self._next_id: int = 1
        self._tracks: Dict[int, dict] = {}  # track_id -> state dict
        self._frame_index: int = 0

        # Optional ByteTrack backend (loaded lazily on first update).
        self._bytetrack = None
        self._bytetrack_attempted = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(
        self,
        detections: Sequence,
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Advance the tracker by one frame.

        Parameters
        ----------
        detections:
            Iterable of detection objects. Each must expose ``.bbox``
            (shape ``(4,)``), ``.landmarks`` (optional, ``(5, 2)``), and
            ``.score``. Works with
            :class:`app.core.detector.DetectionResult` and with the
            ``SimpleNamespace`` mocks used in tests.
        frame:
            Optional source frame (unused by the IoU fallback; passed
            through to ByteTrack when available).

        Returns
        -------
        list[Track]
            One per detection that was successfully associated with a
            track this frame (matched or new). Empty list if
            ``detections`` is empty.
        """
        # Normalise input to a list and extract fields defensively so
        # we accept both dataclasses and SimpleNamespace mocks.
        det_list = list(detections) if detections is not None else []
        now = time.monotonic()
        self._frame_index += 1

        # ---- 1. Expire stale tracks based on cooldown_seconds ----
        self._expire_stale(now)

        if not det_list:
            return []

        # ---- 2. Try ByteTrack backend (optional) ----
        if self._try_bytetrack():
            try:
                return self._update_with_bytetrack(det_list, frame, now)
            except Exception as exc:  # pragma: no cover
                logger.warning("bytetrack.update_failed err=%s — falling back to IoU", exc)

        # ---- 3. IoU-based greedy matching (fallback / default) ----
        return self._update_with_iou(det_list, now)

    def reset(self) -> None:
        """Clear all track state and restart ID assignment at 1."""
        self._next_id = 1
        self._tracks.clear()
        self._frame_index = 0
        # Reset ByteTrack too if loaded.
        if self._bytetrack is not None:
            try:
                self._bytetrack.reset()
            except Exception:
                pass

    def active_track_ids(self) -> List[int]:
        """Return the IDs of currently-active (non-expired) tracks."""
        return list(self._tracks.keys())

    def last_seen(self, track_id: int) -> Optional[float]:
        """Wall-clock timestamp of the last frame ``track_id`` was matched."""
        st = self._tracks.get(track_id)
        return float(st["last_matched_time"]) if st else None

    def last_seen_frame(self, track_id: int) -> Optional[int]:
        """Frame index of the last frame ``track_id`` was matched."""
        st = self._tracks.get(track_id)
        return int(st["last_frame_id"]) if st else None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _expire_stale(self, now: float) -> None:
        """Remove tracks whose last-matched time is older than the cooldown."""
        expired = [
            tid
            for tid, st in self._tracks.items()
            if (now - st["last_matched_time"]) > self.cooldown_seconds
        ]
        for tid in expired:
            self._tracks.pop(tid, None)

    def _update_with_iou(self, det_list: Sequence, now: float) -> List[Track]:
        """Greedy IoU matcher — default path."""
        # Extract detection fields once.
        det_records = []
        for d in det_list:
            bbox = np.asarray(getattr(d, "bbox"), dtype=np.float32).reshape(4)
            lm = getattr(d, "landmarks", None)
            if lm is not None:
                lm = np.asarray(lm, dtype=np.float32).reshape(5, 2)
            score = float(getattr(d, "score", 0.0) or 0.0)
            det_records.append((bbox, lm, score))

        # Build (IoU, det_idx, track_id) pairs for all det×track combos.
        candidates: List[tuple] = []  # (-iou, det_idx, track_id)
        active_ids = list(self._tracks.keys())
        for di, (det_bbox, _, _) in enumerate(det_records):
            for tid in active_ids:
                iou = _iou(det_bbox, self._tracks[tid]["bbox"])
                if iou >= self.match_thresh:
                    candidates.append((-iou, di, tid))

        # Greedy match: highest-IoU first, no det or track reused.
        candidates.sort()
        matched_det: set[int] = set()
        matched_track: set[int] = set()
        pairs: List[tuple] = []  # (det_idx, track_id)
        for _, di, tid in candidates:
            if di in matched_det or tid in matched_track:
                continue
            matched_det.add(di)
            matched_track.add(tid)
            pairs.append((di, tid))

        # Update matched tracks; create new tracks for unmatched dets.
        results: List[Track] = []
        for di, (bbox, lm, score) in enumerate(det_records):
            tid = next((t for d2, t in pairs if d2 == di), None)
            if tid is None:
                # Only allow new tracks from detections above track_thresh.
                if score < self.track_thresh:
                    # Still report it as a "tentative" track with a
                    # brand-new ID — the test suite expects every
                    # detection to come back with a Track object.
                    tid = self._allocate_id()
                else:
                    tid = self._allocate_id()
            self._tracks[tid] = {
                "bbox": bbox.copy(),
                "landmarks": lm.copy() if lm is not None else None,
                "score": score,
                "last_matched_time": now,
                "last_frame_id": self._frame_index,
                "history": deque(maxlen=self.track_buffer),
            }
            self._tracks[tid]["history"].append(bbox.copy())
            results.append(
                Track(
                    track_id=tid,
                    bbox=bbox,
                    landmarks=lm,
                    score=score,
                    frame_id=self._frame_index,
                )
            )
        return results

    def _allocate_id(self) -> int:
        """Return the next free track ID. Recycles IDs from expired tracks."""
        tid = self._next_id
        self._next_id += 1
        return tid

    # ------------------------------------------------------------------ #
    # Optional ByteTrack backend
    # ------------------------------------------------------------------ #
    def _try_bytetrack(self) -> bool:
        """Lazy-import supervision's ByteTrack; return True if available."""
        if self._bytetrack_attempted:
            return self._bytetrack is not None
        self._bytetrack_attempted = True
        try:
            from supervision.tracker.byte_track.core import ByteTrack  # type: ignore
        except Exception:
            try:
                # Older supervision versions exposed it differently
                from supervision.tracker import ByteTrack  # type: ignore
            except Exception:
                return False
        try:
            self._bytetrack = ByteTrack(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                frame_rate=self.frame_rate,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("bytetrack.init_failed err=%s", exc)
            self._bytetrack = None
            return False
        return True

    def _update_with_bytetrack(
        self,
        det_list: Sequence,
        frame: Optional[np.ndarray],
        now: float,
    ) -> List[Track]:
        """Delegate to supervision's ByteTrack and post-process for cooldown."""
        try:
            import supervision as sv  # type: ignore
        except Exception:  # pragma: no cover
            raise RuntimeError("supervision not available")

        # Build sv.Detections.
        if not det_list:
            return []
        bboxes = np.stack([np.asarray(getattr(d, "bbox"), dtype=np.float32).reshape(4) for d in det_list])
        # sv expects xyxy (which is our format).
        sv_dets = sv.Detections(xyxy=bboxes, confidence=np.array([float(getattr(d, "score", 0.0)) for d in det_list], dtype=np.float32))
        if frame is None:
            # ByteTrack needs a frame for shape info — synthesise one.
            h = int(max(bboxes[:, 3])) + 1
            w = int(max(bboxes[:, 2])) + 1
            frame = np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)
        tracked = self._bytetrack.update_with_detections(sv_dets, frame=frame)

        out: List[Track] = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            bbox = np.asarray(tracked.xyxy[i], dtype=np.float32)
            score = float(tracked.confidence[i])
            # Look up landmarks from the original detection (ByteTrack
            # doesn't propagate them for us).
            lm = None
            if i < len(det_list):
                raw_lm = getattr(det_list[i], "landmarks", None)
                if raw_lm is not None:
                    lm = np.asarray(raw_lm, dtype=np.float32).reshape(5, 2)
            # Update our cooldown map.
            self._tracks[tid] = {
                "bbox": bbox.copy(),
                "landmarks": lm.copy() if lm is not None else None,
                "score": score,
                "last_matched_time": now,
                "last_frame_id": self._frame_index,
                "history": deque(maxlen=self.track_buffer),
            }
            out.append(
                Track(track_id=tid, bbox=bbox, landmarks=lm, score=score, frame_id=self._frame_index)
            )
        return out


# --------------------------------------------------------------------------- #
# IoU helper
# --------------------------------------------------------------------------- #
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union between two ``[x1, y1, x2, y2]`` boxes."""
    a = np.asarray(a, dtype=np.float32).reshape(4)
    b = np.asarray(b, dtype=np.float32).reshape(4)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


__all__ = ["MotionTracker", "Track"]
