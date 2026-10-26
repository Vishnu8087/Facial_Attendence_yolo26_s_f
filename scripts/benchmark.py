#!/usr/bin/env python3
"""CLI benchmark for the facial-attendance CV pipeline.

Feeds a video file (or a synthetic test pattern) through
:class:`app.core.pipeline.AttendancePipeline` and reports:

* End-to-end FPS (EMA-smoothed).
* Per-stage mean / p50 / p95 / p99 latencies:
    - detection
    - alignment
    - extraction
    - tracking
    - liveness
    - total (end-to-end)
* Counts: faces detected, tracks assigned, identities matched,
  liveness passes, attendance logs emitted.

Outputs a JSON report on stdout (and optionally writes it to a file
with ``--out``). The report schema is documented inline.

Usage
-----

    # Real video:
    python scripts/benchmark.py --video /path/to/face.mp4

    # Synthetic pattern (no file needed — useful for CI smoke tests):
    python scripts/benchmark.py --synthetic --frames 200

    # Override strategy:
    python scripts/benchmark.py --video face.mp4 --strategy detection_only

    # Write JSON report to disk:
    python scripts/benchmark.py --video face.mp4 --out report.json

Notes
-----
* The pipeline components are constructed via the real
  :class:`FaceDetector`, :class:`FaceAligner`, :class:`FeatureExtractor`,
  :class:`MotionTracker`, :class:`LivenessVerifier`. If InsightFace is
  not installed the script fails fast with a clear message — use
  ``--mock`` to run against synthetic mocks instead (handy for CI).
* The script is intentionally synchronous — benchmark numbers reflect
  single-threaded throughput, which is what the WebSocket worker sees.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Make `app` importable from a checkout.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core import (  # noqa: E402
    AttendancePipeline,
    FaceAligner,
    FaceDetector,
    FeatureExtractor,
    FrameResult,
    LivenessVerifier,
    MotionTracker,
)
from app.core.utils import cosine_similarity  # noqa: E402

logger = logging.getLogger("benchmark")


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class LatencyStats:
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, samples_ms: List[float]) -> "LatencyStats":
        if not samples_ms:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s = sorted(samples_ms)
        n = len(s)

        def _pct(p: float) -> float:
            if n == 1:
                return s[0]
            k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
            return s[k]

        return cls(
            n=n,
            mean_ms=float(statistics.fmean(s)),
            p50_ms=float(_pct(50)),
            p95_ms=float(_pct(95)),
            p99_ms=float(_pct(99)),
            max_ms=float(s[-1]),
        )


@dataclass
class BenchmarkReport:
    config: Dict[str, Any]
    summary: Dict[str, Any]
    latencies: Dict[str, Any]
    counts: Dict[str, int]
    fps_series: List[float] = field(default_factory=list)
    timestamp: str = ""
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Frame source — video file or synthetic pattern
# --------------------------------------------------------------------------- #
class VideoFrameSource:
    """Yield BGR frames from a video file via OpenCV."""

    def __init__(self, path: Path) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("opencv-python is required to read video files") from exc
        self._cv2 = cv2
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video file: {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            raise StopIteration
        return frame

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


class SyntheticFrameSource:
    """Generate ``n_frames`` synthetic BGR frames with a moving "face".

    The pattern is intentionally simple — a coloured ellipse + two
    "eyes" that drift across the frame — so the benchmark can run in
    environments without a camera or video file. When paired with
    ``--mock`` it exercises the pipeline without InsightFace.
    """

    def __init__(self, n_frames: int = 200, width: int = 640, height: int = 480) -> None:
        self.n_frames = int(n_frames)
        self.width = int(width)
        self.height = int(height)
        try:
            import cv2  # type: ignore
            self._cv2 = cv2
        except ImportError:
            self._cv2 = None
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self._i >= self.n_frames:
            raise StopIteration
        if self._cv2 is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            frame = np.full((self.height, self.width, 3), (40, 40, 40), dtype=np.uint8)
            # Drifting "face" — moves diagonally to exercise the tracker.
            cx = 80 + (self._i * 3) % (self.width - 160)
            cy = 80 + (self._i * 2) % (self.height - 160)
            self._cv2.ellipse(frame, (cx, cy), (60, 80), 0, 0, 360, (180, 160, 120), -1)
            self._cv2.circle(frame, (cx - 20, cy - 10), 8, (40, 40, 40), -1)
            self._cv2.circle(frame, (cx + 20, cy - 10), 8, (40, 40, 40), -1)
            self._cv2.ellipse(frame, (cx, cy + 20), (25, 8), 0, 0, 180, (40, 20, 20), 2)
        self._i += 1
        return frame


# --------------------------------------------------------------------------- #
# Instrumented pipeline wrapper
# --------------------------------------------------------------------------- #
class InstrumentedPipeline:
    """Wrap an :class:`AttendancePipeline` and record per-stage latencies.

    The wrapping is done by monkey-patching the wrapped pipeline's
    internal helper methods (``_safe_detect``, ``_safe_tracker_update``,
    ``_verify_liveness``, ``_extract_and_match``) so we can time each
    stage without modifying the production code path.
    """

    def __init__(self, pipeline: AttendancePipeline) -> None:
        self.pipe = pipeline
        self.detector_samples: List[float] = []
        self.tracker_samples: List[float] = []
        self.liveness_samples: List[float] = []
        self.extract_samples: List[float] = []
        self.total_samples: List[float] = []
        self.fps_samples: List[float] = []
        self.counts = {
            "frames": 0,
            "detections": 0,
            "tracks": 0,
            "identities_matched": 0,
            "liveness_passed": 0,
            "logs_emitted": 0,
        }

        # Monkey-patch the wrapped pipeline's helpers to record timing.
        # We patch BOTH ``_extract_and_match`` (used by detection_only)
        # and ``_maybe_reextract_and_match`` (used by both) so the
        # extraction latency is captured regardless of strategy.
        self._wrap("_safe_detect", self.detector_samples)
        self._wrap("_safe_tracker_update", self.tracker_samples)
        self._wrap("_verify_liveness", self.liveness_samples)
        self._wrap("_extract_and_match", self.extract_samples)
        self._wrap("_maybe_reextract_and_match", self.extract_samples)

    def _wrap(self, method_name: str, samples: List[float]) -> None:
        original = getattr(self.pipe, method_name)

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                samples.append((time.perf_counter() - t0) * 1000.0)

        setattr(self.pipe, method_name, wrapper)

    def process(self, frame: np.ndarray, strategy: str) -> FrameResult:
        t0 = time.perf_counter()
        try:
            result = self.pipe.process_frame(frame, strategy=strategy)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.total_samples.append(elapsed_ms)
        # Update counts.
        self.counts["frames"] += 1
        self.counts["detections"] += int(result.num_detections)
        self.counts["tracks"] += len(result.tracks)
        for tr in result.tracks:
            if tr.identity is not None:
                self.counts["identities_matched"] += 1
            if tr.liveness_passed:
                self.counts["liveness_passed"] += 1
        self.counts["logs_emitted"] = int(result.num_logged)
        self.fps_samples.append(float(result.fps))
        return result


# --------------------------------------------------------------------------- #
# Mock pipeline (for CI / no-model environments)
# --------------------------------------------------------------------------- #
def build_mock_pipeline() -> AttendancePipeline:
    """Construct an :class:`AttendancePipeline` with mocked CV components.

    The mocks return canned results so the benchmark can run without
    InsightFace / ONNX weights. Useful for CI smoke-testing the
    benchmark itself.
    """
    from unittest.mock import MagicMock
    from types import SimpleNamespace

    # Mutable state container — keep it OUT of the MagicMock so the
    # counter stays a plain int (MagicMock auto-creates attributes as
    # MagicMocks, which would break numpy later on).
    state = {"call_count": 0, "next_track_id": 1, "active": {}}

    detector = MagicMock(name="FaceDetector")
    def _detect(frame):
        h, w = frame.shape[:2]
        state["call_count"] += 1
        cx = 80 + (state["call_count"] * 3) % max(1, w - 160)
        cy = 120
        return [
            SimpleNamespace(
                bbox=np.array([cx - 60, cy - 80, cx + 60, cy + 80], dtype=np.float32),
                landmarks=np.array(
                    [
                        [cx - 20, cy - 10],
                        [cx + 20, cy - 10],
                        [cx, cy + 10],
                        [cx - 20, cy + 30],
                        [cx + 20, cy + 30],
                    ],
                    dtype=np.float32,
                ),
                score=0.95,
            )
        ]
    detector.detect.side_effect = _detect

    aligner = MagicMock(name="FaceAligner")
    aligner.align.side_effect = lambda frame, lm, size=None: (
        np.ascontiguousarray(frame[:112, :112]) if frame.shape[0] >= 112 and frame.shape[1] >= 112
        else np.zeros((112, 112, 3), dtype=np.uint8)
    )

    extractor = MagicMock(name="FeatureExtractor")
    extractor.embedding_dim = 512
    def _embed(face_img):
        arr = np.asarray(face_img, dtype=np.float32)
        seed = int(arr.sum()) % (2**32 - 1)
        rng = np.random.default_rng(seed=seed)
        emb = rng.standard_normal(512).astype(np.float32)
        emb /= max(float(np.linalg.norm(emb)), 1e-9)
        return emb
    extractor.extract.side_effect = _embed
    extractor.extract_batch.side_effect = lambda faces: (
        np.stack([_embed(f) for f in faces]) if faces else np.zeros((0, 512), dtype=np.float32)
    )

    tracker = MotionTracker(cooldown_seconds=30.0)
    liveness = LivenessVerifier()
    logger_mock = MagicMock(name="AttendanceLogger")

    return AttendancePipeline(
        detector=detector,
        aligner=aligner,
        extractor=extractor,
        tracker=tracker,
        liveness=liveness,
        logger=logger_mock,
        cooldown_seconds=30.0,
    )


def build_real_pipeline(model_pack: str, device: str, det_size: int) -> AttendancePipeline:
    """Construct a pipeline backed by real InsightFace models."""
    detector = FaceDetector(model_pack_name=model_pack, device=device, det_size=det_size)
    aligner = FaceAligner(output_size=(112, 112))
    extractor = FeatureExtractor(model_pack_name=model_pack, device=device)
    tracker = MotionTracker(cooldown_seconds=30.0)
    liveness = LivenessVerifier()
    return AttendancePipeline(
        detector=detector,
        aligner=aligner,
        extractor=extractor,
        tracker=tracker,
        liveness=liveness,
        logger=None,  # No real logger wired — benchmark only measures CV stages.
        cooldown_seconds=30.0,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_benchmark(
    *,
    source,
    pipeline: AttendancePipeline,
    strategy: str,
    max_frames: Optional[int],
    warmup_frames: int,
) -> BenchmarkReport:
    """Run the benchmark loop and return a populated :class:`BenchmarkReport`."""
    instrumented = InstrumentedPipeline(pipeline)

    logger.info("benchmark.warmup frames=%d", warmup_frames)
    for i, frame in enumerate(source):
        if i >= warmup_frames:
            break
        instrumented.process(frame, strategy=strategy)
    # Reset counters after warmup.
    instrumented.counts = {k: 0 for k in instrumented.counts}
    instrumented.fps_samples.clear()

    logger.info("benchmark.start strategy=%s max_frames=%s", strategy, max_frames)
    frame_idx = 0
    for frame in source:
        instrumented.process(frame, strategy=strategy)
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    # Compose the report.
    latencies = {
        "detection": asdict(LatencyStats.from_samples(instrumented.detector_samples)),
        "tracking": asdict(LatencyStats.from_samples(instrumented.tracker_samples)),
        "liveness": asdict(LatencyStats.from_samples(instrumented.liveness_samples)),
        "extraction": asdict(LatencyStats.from_samples(instrumented.extract_samples)),
        "total": asdict(LatencyStats.from_samples(instrumented.total_samples)),
    }

    fps = instrumented.fps_samples
    summary = {
        "frames_processed": frame_idx,
        "mean_fps": float(statistics.fmean(fps)) if fps else 0.0,
        "p50_fps": float(statistics.median(fps)) if fps else 0.0,
        "max_fps": float(max(fps)) if fps else 0.0,
        "min_fps": float(min(fps)) if fps else 0.0,
        "warmup_frames": int(warmup_frames),
        "strategy": strategy,
    }
    if hasattr(source, "width"):
        summary["source_width"] = int(source.width)
        summary["source_height"] = int(source.height)
        summary["source_fps"] = float(getattr(source, "fps", 0.0))

    import datetime as _dt

    return BenchmarkReport(
        config={
            "strategy": strategy,
            "max_frames": max_frames,
            "warmup_frames": warmup_frames,
            "pipeline_class": type(pipeline).__name__,
            "device": getattr(pipeline, "device", "n/a"),
        },
        summary=summary,
        latencies=latencies,
        counts=instrumented.counts,
        fps_series=fps[-200:],  # last 200 samples for sparkline charts
        timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the facial-attendance CV pipeline.",
    )
    src = parser.add_argument_group("source")
    src.add_argument("--video", type=Path, default=None, help="Path to a video file.")
    src.add_argument("--synthetic", action="store_true", help="Use a synthetic frame source (no file needed).")
    src.add_argument("--frames", type=int, default=200, help="Number of synthetic frames (default 200).")
    src.add_argument("--width", type=int, default=640, help="Synthetic frame width.")
    src.add_argument("--height", type=int, default=480, help="Synthetic frame height.")

    pl = parser.add_argument_group("pipeline")
    pl.add_argument("--strategy", default="both", choices=["both", "detection_only", "tracking_only"])
    pl.add_argument("--mock", action="store_true", help="Use mocked CV components (no InsightFace).")
    pl.add_argument("--model-pack", default="buffalo_l", help="InsightFace model pack name.")
    pl.add_argument("--device", default="cpu", help="Compute device (cpu / cuda / cuda:0).")
    pl.add_argument("--det-size", type=int, default=640, help="RetinaFace det_size.")
    pl.add_argument("--max-frames", type=int, default=None, help="Cap the number of frames processed.")
    pl.add_argument("--warmup", type=int, default=10, help="Warmup frames excluded from stats.")

    out = parser.add_argument_group("output")
    out.add_argument("--out", type=Path, default=None, help="Write JSON report to this path.")
    out.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    out.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    # ---- Build frame source ----
    if args.video is not None:
        if not args.video.exists():
            logger.error("benchmark.video.missing path=%s", args.video)
            return 2
        source = VideoFrameSource(args.video)
    elif args.synthetic:
        source = SyntheticFrameSource(n_frames=args.frames, width=args.width, height=args.height)
    else:
        parser.error("specify --video or --synthetic")

    # ---- Build pipeline ----
    if args.mock:
        pipeline = build_mock_pipeline()
    else:
        try:
            pipeline = build_real_pipeline(args.model_pack, args.device, args.det_size)
        except RuntimeError as exc:
            logger.error("benchmark.pipeline.build_failed err=%s", exc)
            logger.error("benchmark.hint consider --mock for a no-model smoke test")
            return 3

    # ---- Run ----
    report = run_benchmark(
        source=source,
        pipeline=pipeline,
        strategy=args.strategy,
        max_frames=args.max_frames,
        warmup_frames=args.warmup,
    )
    if hasattr(source, "release"):
        source.release()

    # ---- Emit report ----
    report_json = json.dumps(
        asdict(report),
        indent=2 if args.pretty else None,
        default=str,
    )
    print(report_json)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report_json + "\n")
        logger.info("benchmark.report.written path=%s", args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
