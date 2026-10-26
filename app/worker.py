"""Standalone worker entrypoint.

Run with::

    python -m app.worker

The worker consumes frames from a Redis queue (or an in-process queue
when Redis is unavailable) and processes them via the shared
:class:`AttendancePipeline`. It broadcasts attendance events back to a
Redis pub/sub channel which the API's ``/ws/events`` WebSocket forwards
to dashboard clients.

Used by:

* docker-compose ``worker`` service (``command: ["worker"]``)
* k8s ``worker-deployment.yaml`` (``command: ["python", "-m", "app.worker"]``)
"""
from __future__ import annotations

import asyncio
import base64
import json
import signal
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.config import settings
from app.db.session import dispose_engine, get_session
from app.services.attendance_logger import AttendanceLogger
from app.services.stream_manager import _build_pipeline
from app.utils.logging import configure_logging, get_logger
from app.utils.metrics import (
    attendance_logged_total,
    faces_detected_total,
    frame_processing_seconds,
    frames_processed_total,
    recognitions_total,
    spoof_rejected_total,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class Worker:
    """Background frame processor.

    Pulls base64-encoded JPEG frames off a Redis list (BLPOP) and runs
    them through the pipeline. Falls back to a no-op loop when Redis is
    unavailable so the worker still boots for development / smoke tests.
    """

    def __init__(self) -> None:
        self.attendance_logger = AttendanceLogger()
        self.pipeline: Any | None = None
        self.redis: Any | None = None
        self._stop = asyncio.Event()

    # ----------------------------------------------------------------- #
    # Lifecycle
    # ----------------------------------------------------------------- #
    async def start(self) -> None:
        configure_logging()
        logger.info("worker.startup", queue=settings.worker_queue_name)
        settings.ensure_dirs()

        # Redis
        self.redis = await self._connect_redis()
        if self.redis is None:
            logger.warning("worker.redis.unavailable.using_noop_loop")

        # Pipeline (lazy CV core)
        self.pipeline = _build_pipeline(self.attendance_logger)
        if self.pipeline is None:
            logger.warning("worker.pipeline.unavailable")

        # Run loop
        tasks = [
            asyncio.create_task(self._consume_loop()),
        ]
        if self.redis is not None:
            tasks.append(asyncio.create_task(self._publish_loop()))

        # Wait for stop signal
        await self._stop.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._cleanup()
        logger.info("worker.shutdown.complete")

    async def stop(self) -> None:
        self._stop.set()

    # ----------------------------------------------------------------- #
    # Internal
    # ----------------------------------------------------------------- #
    async def _connect_redis(self) -> Any | None:
        if not settings.redis_url:
            return None
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(settings.redis_url, decode_responses=False)
            await client.ping()
            logger.info("worker.redis.connected", url=settings.redis_url)
            return client
        except Exception as exc:
            logger.warning("worker.redis.connect_failed", error=str(exc))
            return None

    async def _consume_loop(self) -> None:
        """Main consumer loop — BLPOP from the Redis queue."""
        if self.redis is None:
            # Noop loop — keeps the worker process alive for dev / smoke.
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
            return

        queue_name = settings.worker_queue_name
        logger.info("worker.consume_loop.start", queue=queue_name)
        while not self._stop.is_set():
            try:
                # BLPOP returns (queue_name, payload) or None on timeout.
                result = await self.redis.blpop(queue_name, timeout=1)
                if result is None:
                    continue
                _, payload = result
                await self._handle_payload(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("worker.consume_loop.error", error=str(exc))
                await asyncio.sleep(1.0)

    async def _publish_loop(self) -> None:
        """Optional: keepalive / metrics publisher (no-op for now)."""
        while not self._stop.is_set():
            await asyncio.sleep(10.0)

    async def _handle_payload(self, payload: bytes | str) -> None:
        """Decode a queue message and process it.

        Message format (JSON):

            {
              "frame_b64": "<base64 JPEG>",
              "strategy": "both",
              "camera_id": "cam-1",
              "submitted_at": "2024-..."
            }
        """
        try:
            if isinstance(payload, (bytes, bytearray)):
                payload_str = payload.decode("utf-8")
            else:
                payload_str = payload
            msg = json.loads(payload_str)
        except Exception as exc:
            logger.error("worker.payload.decode_failed", error=str(exc))
            return

        frame_b64 = msg.get("frame_b64")
        strategy = msg.get("strategy") or settings.default_strategy
        camera_id = msg.get("camera_id")

        if not frame_b64:
            logger.warning("worker.payload.missing_frame_b64")
            return

        try:
            raw = base64.b64decode(frame_b64)
        except Exception as exc:
            logger.error("worker.payload.frame_b64_decode_failed", error=str(exc))
            return

        frame = self._decode_frame(raw)
        if frame is None:
            logger.warning("worker.payload.frame_decode_failed")
            return

        await self._process_frame(frame, strategy, camera_id)

    async def _process_frame(
        self, frame: np.ndarray, strategy: str, camera_id: str | None
    ) -> None:
        """Run the pipeline and emit attendance events."""
        if self.pipeline is None:
            logger.debug("worker.pipeline.missing.skip")
            return
        t0 = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: self.pipeline.process_frame(frame, strategy=strategy)
            )
        except ValueError as exc:
            logger.warning("worker.process.invalid_strategy", error=str(exc))
            return
        except Exception as exc:
            logger.exception("worker.process.error", error=str(exc))
            return

        latency_ms = (time.monotonic() - t0) * 1000.0
        frames_processed_total.labels(strategy=strategy).inc()
        frame_processing_seconds.labels(strategy=strategy).observe(latency_ms / 1000.0)

        tracks = getattr(result, "tracks", None) or []
        if tracks:
            faces_detected_total.labels(strategy=strategy).inc(len(tracks))

        for tr in tracks:
            identity = getattr(tr, "identity", None)
            similarity = getattr(tr, "similarity", None)
            liveness_passed = getattr(tr, "liveness_passed", None)
            track_id = getattr(tr, "track_id", None)

            if liveness_passed is False:
                spoof_rejected_total.labels(strategy=strategy).inc()
                continue

            if identity is None or similarity is None:
                continue
            if similarity < settings.similarity_threshold:
                recognitions_total.labels(strategy=strategy, outcome="below_threshold").inc()
                continue

            emp_id = (
                identity.get("employee_id")
                if isinstance(identity, dict)
                else getattr(identity, "employee_id", None)
            )
            emp_name = (
                identity.get("name")
                if isinstance(identity, dict)
                else getattr(identity, "name", None)
            )
            emp_pk = (
                identity.get("employee_pk")
                if isinstance(identity, dict)
                else getattr(identity, "employee_pk", None)
            )
            if not emp_id:
                continue

            recognitions_total.labels(strategy=strategy, outcome="matched").inc()
            inserted = await self._log_attendance(
                employee_id=str(emp_id),
                employee_pk=emp_pk,
                similarity=float(similarity),
                track_id=track_id,
                camera_id=camera_id,
                strategy=strategy,
                liveness_passed=liveness_passed,
                liveness_confidence=getattr(tr, "liveness_confidence", None),
            )
            if inserted:
                await self._publish_event(
                    {
                        "type": "attendance",
                        "employee_id": str(emp_id),
                        "name": emp_name or "",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "strategy": strategy,
                        "score": round(float(similarity), 4),
                        "track_id": track_id,
                        "camera_id": camera_id,
                        "liveness_passed": liveness_passed,
                    }
                )

    async def _log_attendance(
        self,
        *,
        employee_id: str,
        employee_pk: Any | None,
        similarity: float,
        track_id: int | None,
        camera_id: str | None,
        strategy: str,
        liveness_passed: bool | None,
        liveness_confidence: float | None,
    ) -> bool:
        """Insert a cooldown-aware attendance log. Returns True if inserted."""
        try:
            async with get_session() as session:
                _, inserted = await self.attendance_logger.log_attendance(
                    session,
                    employee_id=employee_id,
                    employee_pk=employee_pk,
                    similarity=similarity,
                    track_id=track_id,
                    camera_id=camera_id,
                    strategy=strategy,
                    liveness_passed=liveness_passed,
                    liveness_confidence=liveness_confidence,
                )
                return inserted
        except Exception as exc:
            logger.exception("worker.log_attendance.error", error=str(exc))
            attendance_logged_total.labels(strategy=strategy, result="error").inc()
            return False

    async def _publish_event(self, event: dict[str, Any]) -> None:
        """Publish an attendance event to the Redis pub/sub channel."""
        if self.redis is None:
            return
        try:
            await self.redis.publish(
                settings.worker_events_channel,
                json.dumps(event, default=str),
            )
        except Exception as exc:
            logger.warning("worker.publish.error", error=str(exc))

    @staticmethod
    def _decode_frame(data: bytes) -> np.ndarray | None:
        try:
            import cv2  # type: ignore

            return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except ImportError:
            try:
                from PIL import Image  # type: ignore
                import io

                img = Image.open(io.BytesIO(data)).convert("RGB")
                return np.array(img)[:, :, ::-1].copy()
            except Exception:
                return None
        except Exception:
            return None

    async def _cleanup(self) -> None:
        if self.redis is not None:
            try:
                await self.redis.close()
            except Exception:
                pass
        await dispose_engine()


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
async def _main() -> None:
    worker = Worker()

    def _signal_handler(*_: Any) -> None:  # noqa: ANN001
        logger.info("worker.signal.received")
        asyncio.create_task(worker.stop())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows
            signal.signal(sig, lambda *_: None)

    await worker.start()


def main() -> None:
    """Synchronous entrypoint for ``python -m app.worker``."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
