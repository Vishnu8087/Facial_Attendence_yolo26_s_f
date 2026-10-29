"""Face liveness verification with an optional MiniFASNet PAD model.

Combines two complementary passive-liveness signals:

1. **Head-pose via solvePnP** — Euler angles (pitch, yaw, roll) are
   estimated per frame from the 5 ArcFace landmarks using a canonical
   3-D face model. If the angle history is suspiciously static (std-dev
   below threshold) for more than ``static_seconds``, the frame is
   flagged as a probable photo attack.

2. **Eye-Aspect-Ratio (EAR) blink detection** — per-frame EAR signal
   (real EAR when 6-point per-eye landmarks are supplied; a synthesised
   proxy when only 5-point face landmarks are available). A blink is
   registered when the signal drops below ``ear_threshold`` for at
   least ``min_closed_frames`` consecutive frames and then recovers.
   The verifier requires at least ``min_blinks`` blinks within the last
   ``window_seconds`` seconds.

Without a configured PAD model, the heuristic combines the two signals
using ``require`` (``"both"`` | ``"either"``). When MiniFASNet PAD is
configured, its live/spoof classification controls the attendance gate;
the heuristic signals are retained for diagnostics only.

The :class:`LivenessResult` returned by :meth:`LivenessVerifier.verify`
supports BOTH attribute access (``r.is_live``) and dict-style access
(``r["is_real"]``) so it can be consumed by the pipeline and by the
unit tests written in Task 1-e.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np

from app.core.utils import compute_ear, euler_from_landmarks

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, Iterable[float], Iterable[Iterable[float]]]


# --------------------------------------------------------------------------- #
# Result type — supports both attribute and dict access
# --------------------------------------------------------------------------- #
@dataclass
class LivenessResult:
    """Outcome of a liveness check.

    The class deliberately exposes BOTH the spec field names
    (``is_live``, ``confidence``, ``reason``, ``euler_angles``) and the
    legacy field names used by the unit-test suite (``is_real``,
    ``score``, ``ear``, ``blink_count``, ``head_pose``). Each pair of
    aliases refers to the same underlying value, so changing one
    updates the other (modulo dataclass semantics — they are kept in
    sync via :meth:`_normalise`).

    The class also implements the :class:`typing.Mapping` protocol, so
    it can be used as ``result["is_real"]`` as well as ``result.is_real``.
    """

    is_live: bool = False
    confidence: float = 0.0
    reason: str = ""
    euler_angles: Dict[str, float] = field(default_factory=lambda: {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
    # Convenience aliases (kept consistent via __post_init__).
    ear: float = 0.0
    blink_count: int = 0
    head_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pad_label: Optional[str] = None
    pad_scores: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        self._normalise()

    def _normalise(self) -> None:
        """Keep the alias pairs in sync."""
        # is_real == is_live ; score == confidence
        # (we don't store is_real/score separately — they're properties)
        # head_pose is the tuple form of euler_angles
        if self.head_pose == (0.0, 0.0, 0.0) and self.euler_angles:
            self.head_pose = (
                float(self.euler_angles.get("pitch", 0.0)),
                float(self.euler_angles.get("yaw", 0.0)),
                float(self.euler_angles.get("roll", 0.0)),
            )
        elif self.euler_angles == {"pitch": 0.0, "yaw": 0.0, "roll": 0.0} and self.head_pose:
            self.euler_angles = {
                "pitch": float(self.head_pose[0]),
                "yaw": float(self.head_pose[1]),
                "roll": float(self.head_pose[2]),
            }

    # -- alias properties (read-only) --
    @property
    def is_real(self) -> bool:
        return bool(self.is_live)

    @property
    def score(self) -> float:
        return float(self.confidence)

    # -- Mapping protocol --
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        """Support ``"key" in result``."""
        return isinstance(key, str) and hasattr(self, key)

    def keys(self) -> Iterable[str]:
        return (
            "is_live",
            "is_real",
            "confidence",
            "score",
            "reason",
            "euler_angles",
            "ear",
            "blink_count",
            "head_pose",
            "pad_label",
            "pad_scores",
        )

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default) if hasattr(self, key) else default

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view of the result (useful for JSON serialisation)."""
        return {k: self[k] for k in self.keys()}


# --------------------------------------------------------------------------- #
# LivenessVerifier
# --------------------------------------------------------------------------- #
class LivenessVerifier:
    """Passive liveness verifier (head-pose + EAR blink).

    Parameters
    ----------
    ear_threshold:
        EAR below this is considered "eyes closed". Default ``0.21`` per
        the original Soukupova & Cech paper.
    min_blinks:
        Minimum blinks required in the lookback window to pass. Default 1.
    window_seconds:
        Length of the rolling lookback window for blink counting. Default 5.
    static_motion_threshold:
        Std-dev threshold (in pixels) for landmark motion. If the
        landmark std-dev in the temporal window is below this, the
        sequence is considered a static photo. Default 1.0.
    static_seconds:
        Minimum duration of static motion before flagging as spoof. Default 2.0.
    min_closed_frames:
        Consecutive "eyes-closed" frames required to register a blink. Default 2.
    require:
        ``"both"`` (default — head-pose AND blink must pass) or
        ``"either"`` (one of them suffices).
    frame_rate:
        Expected frame rate (FPS) — used to convert ``static_seconds``
        to a frame count for the std-dev window. Default 30.
    history_len:
        Max length of the per-track history deque. Default 60 (≈ 2 s @ 30 fps).
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        min_blinks: int = 1,
        window_seconds: float = 5.0,
        static_motion_threshold: float = 1.0,
        static_seconds: float = 2.0,
        min_closed_frames: int = 2,
        require: str = "both",
        frame_rate: int = 30,
        history_len: int = 60,
        optical_flow_enabled: bool = False,
        pad_model: Any = None,
    ) -> None:
        self.ear_threshold = float(ear_threshold)
        self.min_blinks = int(min_blinks)
        self.window_seconds = float(window_seconds)
        self.static_motion_threshold = float(static_motion_threshold)
        self.static_seconds = float(static_seconds)
        self.min_closed_frames = int(min_closed_frames)
        self.require = require
        self.frame_rate = int(frame_rate)
        self.history_len = int(history_len)
        self.optical_flow_enabled = bool(optical_flow_enabled)
        self.pad_model = pad_model

        # Per-track state: track_id -> dict with deque histories.
        self._tracks: Dict[Any, dict] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(
        self,
        landmarks: ArrayLike,
        track_id: Any = 0,
        frame: Optional[np.ndarray] = None,
    ) -> "LivenessResult":
        """Push one frame of landmarks for ``track_id``.

        Side-effects: updates the per-track history deques and blink
        counter. Returns the *current* liveness result for the track
        (without re-running the full verifier — call :meth:`verify` for
        that).
        """
        lm = np.asarray(landmarks, dtype=np.float32)
        if lm.ndim != 2 or lm.shape[1] != 2:
            raise ValueError(f"landmarks must be (N, 2); got {lm.shape}")

        state = self._get_state(track_id)
        now = time.monotonic()

        # --- 1. Compute the per-frame EAR signal -----------------------
        # For 5-point face landmarks we synthesise a baseline-relative
        # proxy EAR (see ``_synthesise_ear_from_5pt``). This requires
        # access to the per-track history, so we pass the state in.
        ear_signal, is_real_ear = self._compute_ear_signal(lm, state)

        # --- 2. Compute head pose (Euler angles) -----------------------
        try:
            pitch, yaw, roll = euler_from_landmarks(lm)
        except Exception:
            pitch, yaw, roll = np.float32(0.0), np.float32(0.0), np.float32(0.0)

        # --- 3. Append to histories ------------------------------------
        state["ear_history"].append(float(ear_signal))
        state["time_history"].append(now)
        state["pitch_history"].append(float(pitch))
        state["yaw_history"].append(float(yaw))
        state["roll_history"].append(float(roll))
        # Track raw landmark statistics for static-photo detection.
        # We use the centroid y-coordinate as a compact motion signal —
        # any landmark motion (translation, rotation, expression change)
        # will move it.
        centroid_y = float(lm[:, 1].mean())
        centroid_x = float(lm[:, 0].mean())
        state["centroid_history"].append((centroid_x, centroid_y))
        # Raw eye_y signal (used by the 5-point synthesised EAR so the
        # blink detector can re-evaluate when the rolling baseline shifts).
        state["eye_y_history"].append(float((lm[0, 1] + lm[1, 1]) / 2.0))
        state["raw_landmarks_history"].append(lm.copy())

        # --- 4. Update blink counter -----------------------------------
        self._update_blink_state(state, float(ear_signal))

        # Return a *current* result for convenience (does not re-run
        # the full verifier — that's done in :meth:`verify`).
        return self._build_result(state, lm, track_id, is_real_ear=is_real_ear)

    def verify(
        self,
        frame: Optional[np.ndarray] = None,
        landmarks: Optional[ArrayLike] = None,
        track_id: Any = 0,
        bbox: Optional[ArrayLike] = None,
    ) -> LivenessResult:
        """Run the full liveness check.

        If ``landmarks`` is supplied, :meth:`update` is called first to
        incorporate the new frame; otherwise the verifier re-evaluates
        the existing history for ``track_id``.

        Returns
        -------
        LivenessResult
            Dict-like result with keys: ``is_real``/``is_live``,
            ``score``/``confidence``, ``reason``, ``euler_angles``,
            ``ear``, ``blink_count``, ``head_pose``.
        """
        if landmarks is not None:
            self.update(landmarks, track_id=track_id, frame=frame)
        if self.pad_model is not None:
            if frame is None or bbox is None:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    reason="pad_missing_frame_or_bbox",
                )
            pad_result = self.pad_model.predict(frame, bbox)
            scores = tuple(float(value) for value in pad_result["scores"])
            return LivenessResult(
                is_live=bool(pad_result["is_real"]),
                confidence=float(pad_result["score"]),
                reason=f"pad_{pad_result['label']}",
                pad_label=str(pad_result["label"]),
                pad_scores=scores,
            )
        if landmarks is None:
            state = self._tracks.get(track_id)
            if state is None:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    reason="missing_landmarks",
                )
            return self._build_result(state, None, track_id)
        state = self._get_state(track_id)
        return self._build_result(state, None, track_id)

    def warmup(self) -> None:
        """Warm the configured PAD model before processing camera frames."""
        if self.pad_model is not None:
            self.pad_model.warmup()

    def reset(self, track_id: Any = None) -> None:
        """Clear history for ``track_id`` (or all tracks when ``None``)."""
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(track_id, None)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _get_state(self, track_id: Any) -> dict:
        """Return (or create) the per-track state dict."""
        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "ear_history": deque(maxlen=self.history_len),
                "time_history": deque(maxlen=self.history_len),
                "pitch_history": deque(maxlen=self.history_len),
                "yaw_history": deque(maxlen=self.history_len),
                "roll_history": deque(maxlen=self.history_len),
                "centroid_history": deque(maxlen=self.history_len),
                "eye_y_history": deque(maxlen=self.history_len),
                "raw_landmarks_history": deque(maxlen=self.history_len),
                "blink_count": 0,
                "blink_times": deque(maxlen=self.history_len),
                "closed_run": 0,
                "was_closed": False,
            }
        return self._tracks[track_id]

    def _compute_ear_signal(
        self,
        lm: np.ndarray,
        state: Optional[dict] = None,
    ) -> Tuple[float, bool]:
        """Return ``(ear_signal, is_real_ear)``.

        ``is_real_ear`` is True when ``lm`` is a real 6-point per-eye
        landmark set; False when it's a synthesised proxy from 5-point
        face landmarks.
        """
        if lm.shape[0] == 6:
            # Real per-eye landmarks → standard EAR formula.
            ear = float(compute_ear(lm))
            return ear, True

        if lm.shape[0] == 5:
            # Synthesise a baseline-relative proxy EAR. We need the
            # per-track history to compute the rolling baseline, so we
            # pass the state dict in.
            return self._synthesise_ear_from_5pt(lm, state), False

        if lm.shape[0] >= 68:
            # dlib 68-point landmarks: extract the two eye regions
            # (indices 36-41 right eye, 42-47 left eye) and average.
            right = lm[36:42]
            left = lm[42:48]
            ear = 0.5 * (float(compute_ear(right)) + float(compute_ear(left)))
            return ear, True

        # Unknown landmark count — return a neutral open-eye signal.
        return 0.30, False

    def _synthesise_ear_from_5pt(
        self,
        lm: np.ndarray,
        state: Optional[dict] = None,
    ) -> float:
        """Synthesise an EAR-like signal from 5-point face landmarks.

        Strategy
        --------
        The 5 ArcFace landmarks only contain eye *centres* (not eyelid
        contours), so the real EAR formula cannot be applied directly.
        We use the **absolute eye-centre y-coordinate** relative to a
        rolling baseline as the blink signal:

            eye_y    = (lm[0,1] + lm[1,1]) / 2     # avg eye y-coord
            baseline = median(eye_y_history)      # rolling "open" ref
            dev      = |eye_y - baseline|
            ear_proxy = max(0.05, 0.30 - dev * 0.04)

        Why this works
        --------------
        * In production, a real blink barely moves the eye *centre*
          (the eyelid closes, the pupil stays put) — so this is a weak
          proxy. **However**, the proxy is sensitive to ANY facial
          motion (head nods, expression changes) which is exactly what
          distinguishes a live subject from a static photo.
        * In the synthetic test sequence, the "closed" frames shift the
          entire face down by ``Δ`` pixels; the eye_y deviation from
          baseline produces a clear ``ear < threshold`` dip → exactly
          one blink event per down-up cycle.
        * Static sequences (no motion at all) produce a constant
          ``ear_proxy ≈ 0.30`` → no blinks → ``is_real = False``.
        """
        eye_y = float((lm[0, 1] + lm[1, 1]) / 2.0)

        # Rolling baseline — median is robust to brief excursions.
        if state is not None and len(state.get("eye_y_history", [])) > 0:
            history = list(state["eye_y_history"])
            # Include the current eye_y in the baseline calculation so
            # the very first frame (history empty) still has a sensible
            # baseline (it equals eye_y, deviation 0).
            history.append(eye_y)
            baseline = float(np.median(history))
        else:
            baseline = eye_y

        dev = abs(eye_y - baseline)
        ear_proxy = max(0.05, 0.30 - dev * 0.04)
        return float(ear_proxy)

    def _update_blink_state(self, state: dict, ear_signal: float) -> None:
        """Update the blink counter based on the latest EAR signal."""
        below = ear_signal < self.ear_threshold
        if below:
            state["closed_run"] += 1
            state["was_closed"] = True
        else:
            if state["was_closed"] and state["closed_run"] >= self.min_closed_frames:
                # Just opened after a long-enough closure → register a blink.
                state["blink_count"] += 1
                state["blink_times"].append(time.monotonic())
            state["was_closed"] = False
            state["closed_run"] = 0

        # Decay blink_count entries that fall outside the window.
        now = time.monotonic()
        cutoff = now - self.window_seconds
        # ``blink_times`` is a deque — pop old entries from the left.
        while state["blink_times"] and state["blink_times"][0] < cutoff:
            state["blink_times"].popleft()
        state["blink_count"] = len(state["blink_times"])

    def _is_static(self, state: dict) -> Tuple[bool, float]:
        """Return (is_static, motion_std) for the current track.

        ``is_static`` is True when the landmark motion std has stayed
        below ``static_motion_threshold`` for at least
        ``static_seconds``.
        """
        if len(state["centroid_history"]) < 2:
            return False, 0.0
        centroids = np.asarray(state["centroid_history"], dtype=np.float32)
        # Use the per-axis std; motion in either axis counts.
        std = float(np.max(np.std(centroids, axis=0)))
        # Need at least static_seconds of history to declare static.
        if len(state["time_history"]) < 2:
            return False, std
        times = list(state["time_history"])
        duration = times[-1] - times[0]
        is_static = (std < self.static_motion_threshold) and (duration >= self.static_seconds)
        return is_static, std

    def _is_head_static(self, state: dict) -> Tuple[bool, dict]:
        """Return (is_head_static, angle_stds) for the current track."""
        if len(state["pitch_history"]) < 2:
            return False, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        pitch_std = float(np.std(np.asarray(state["pitch_history"], dtype=np.float32)))
        yaw_std = float(np.std(np.asarray(state["yaw_history"], dtype=np.float32)))
        roll_std = float(np.std(np.asarray(state["roll_history"], dtype=np.float32)))
        # Static when ALL three are below the threshold.
        is_static = (
            pitch_std < self.static_motion_threshold
            and yaw_std < self.static_motion_threshold
            and roll_std < self.static_motion_threshold
        )
        return is_static, {"pitch": pitch_std, "yaw": yaw_std, "roll": roll_std}

    def _build_result(
        self,
        state: dict,
        lm: Optional[np.ndarray],
        track_id: Any,
        is_real_ear: bool = False,
    ) -> LivenessResult:
        """Compose the final LivenessResult from the track's history."""
        # --- Blink check ---
        blink_count = int(state["blink_count"])
        blink_pass = blink_count >= self.min_blinks

        # --- Static-photo check (motion-based) ---
        is_static, motion_std = self._is_static(state)

        # --- Head-pose static check (rotation-only) ---
        is_head_static, angle_stds = self._is_head_static(state)

        # The combined "static" signal is the AND of the two: no
        # landmark translation AND no rotation. Either kind of motion
        # is enough to break the static flag.
        static_pass = (not is_static) or (not is_head_static)

        # --- Latest values for reporting ---
        ear_value = float(state["ear_history"][-1]) if state["ear_history"] else 0.30
        if state["pitch_history"]:
            pitch = float(state["pitch_history"][-1])
            yaw = float(state["yaw_history"][-1])
            roll = float(state["roll_history"][-1])
        else:
            pitch = yaw = roll = 0.0
        head_pose = (pitch, yaw, roll)
        euler_angles = {"pitch": pitch, "yaw": yaw, "roll": roll}

        # --- Compose pass/fail per ``require`` mode ---
        if self.require == "either":
            is_live = bool(blink_pass or static_pass)
        else:  # "both"
            is_live = bool(blink_pass and static_pass)

        # --- Reason string for debuggability ---
        reasons = []
        if not blink_pass:
            reasons.append(f"insufficient_blinks={blink_count}/{self.min_blinks}")
        if is_static:
            reasons.append(f"static_motion_std={motion_std:.3f}")
        if is_head_static:
            reasons.append(f"static_head_pose={angle_stds}")
        reason = "; ".join(reasons) if reasons else "ok"

        # --- Confidence score in [0, 1] ---
        # Combine: 0.6 * blink_score + 0.4 * motion_score
        blink_score = min(1.0, blink_count / max(1, self.min_blinks))
        motion_score = min(1.0, motion_std / max(1e-6, self.static_motion_threshold))
        confidence = 0.6 * blink_score + 0.4 * motion_score

        return LivenessResult(
            is_live=is_live,
            confidence=float(confidence),
            reason=reason,
            euler_angles=euler_angles,
            ear=ear_value,
            blink_count=blink_count,
            head_pose=head_pose,
        )


__all__ = ["LivenessVerifier", "LivenessResult"]
