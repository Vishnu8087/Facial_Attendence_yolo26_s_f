"""StreamManager — owns WebSocket connections and per-stream pipelines.

Responsibilities:

* Track every active ``/ws/stream`` connection.
* Maintain per-connection state: current strategy, threshold, liveness
  flag, frame queue, pipeline instance.
* Spin up a background asyncio task per connection that drains the
  frame queue, runs ``AttendancePipeline.process_frame(frame, strategy)``,
  and pushes results back over the WebSocket.
* Broadcast attendance events to a separate ``/ws/events`` channel for
  the dashboard's live feed.
* Gracefully clean up tracker / pipeline state on disconnect.

The CV pipeline is imported lazily so the app still imports when
``app.core`` isn't present yet.
"""
from __future__ import annotations

import asyncio
import base64
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.config import settings
from app.db.session import get_session
from app.services.attendance_logger import AttendanceLogger
from app.utils.logging import get_logger
from app.utils.metrics import (
    faces_detected_total,
    frame_processing_seconds,
    frames_processed_total,
    recognitions_total,
    spoof_rejected_total,
    ws_active,
    ws_connections,
)

logger = get_logger(__name__)


class _EmployeeMatcher:
    """Thread-safe in-memory snapshot of enrolled employee embeddings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._employees: list[tuple[dict[str, Any], np.ndarray]] = []

    def replace(self, employees: list[Any]) -> int:
        snapshot: list[tuple[dict[str, Any], np.ndarray]] = []
        for employee in employees:
            if employee.embedding is None:
                continue
            embedding = np.asarray(employee.embedding, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(embedding))
            if embedding.size == 0 or norm <= 1e-12 or not np.isfinite(norm):
                continue
            identity = {
                "employee_id": employee.employee_id,
                "employee_pk": employee.id,
                "name": employee.name,
            }
            snapshot.append((identity, embedding / norm))
        with self._lock:
            self._employees = snapshot
        return len(snapshot)

    def find_nearest(
        self, embedding: np.ndarray, threshold: float
    ) -> tuple[dict[str, Any] | None, float]:
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if query.size == 0 or norm <= 1e-12 or not np.isfinite(norm):
            return None, 0.0
        query /= norm

        with self._lock:
            employees = self._employees
        best_identity: dict[str, Any] | None = None
        best_similarity = -1.0
        for identity, candidate in employees:
            if candidate.shape != query.shape:
                continue
            similarity = float(np.dot(query, candidate))
            if similarity > best_similarity:
                best_identity = identity
                best_similarity = similarity
        if best_identity is None:
            return None, 0.0
        if best_similarity < threshold:
            return None, best_similarity
        return best_identity, best_similarity


# --------------------------------------------------------------------------- #
# CV pipeline lazy loader
# --------------------------------------------------------------------------- #
def _try_import_pipeline() -> Any | None:
    try:
        from app.core.pipeline import AttendancePipeline  # type: ignore

        return AttendancePipeline
    except Exception as exc:
        logger.debug("cv.pipeline.unavailable", error=str(exc))
        return None


def _build_pipeline(
    matcher: Any | None = None,
    cv_service: Any | None = None,
) -> Any | None:
    """Instantiate an AttendancePipeline with all CV components.

    Best-effort: returns None if the CV core isn't importable. In that
    case the StreamManager emits a stub ``frame`` event so the
    WebSocket still functions (used in tests / cold-start).
    """
    PipelineCls = _try_import_pipeline()
    if PipelineCls is None:
        return None
    try:
        from app.core.tracker import MotionTracker  # type: ignore
        from app.core.liveness import LivenessVerifier  # type: ignore
        from app.services.enrollment import EnrollmentService

        cv_service = cv_service or EnrollmentService()
        if any(
            component is None
            for component in (cv_service.detector, cv_service.aligner, cv_service.extractor)
        ):
            raise RuntimeError("One or more face-processing components could not be initialized")
        return PipelineCls(
            detector=cv_service.detector,
            aligner=cv_service.aligner,
            extractor=cv_service.extractor,
            tracker=MotionTracker(),
            liveness=LivenessVerifier(frame_rate=settings.max_stream_fps),
            matcher=matcher,
            cooldown_seconds=settings.cooldown_seconds,
            similarity_threshold=settings.similarity_threshold,
            min_face_size=settings.min_face_size,
            default_strategy=settings.default_strategy,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("stream.pipeline.build_failed", error=str(exc))
        return None


# --------------------------------------------------------------------------- #
# Per-connection state
# --------------------------------------------------------------------------- #
@dataclass
class ConnectionState:
    """Per-WebSocket state — one per active connection."""

    ws: WebSocket
    connection_id: str
    pipeline: Any | None
    attendance_logger: AttendanceLogger
    current_strategy: str = "both"
    current_threshold: float | None = None
    liveness_enabled: bool = True
    source_mode: str = "camera"  # upload | camera
    # Detector hot-swap state — updated by `{"type":"config","detector":{...}}` messages
    detector_backend: str = settings.detector_backend
    detector_model: str = settings.detector_model
    frame_queue: asyncio.Queue[bytes] = field(
        default_factory=lambda: asyncio.Queue(maxsize=2)
    )
    extract_queue: asyncio.Queue[tuple[np.ndarray, list[Any]]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=2)
    )
    last_frame_ts: float = 0.0
    last_result_ts: float = 0.0
    frame_counter: int = 0
    task: asyncio.Task[None] | None = None
    closed: bool = False
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def effective_threshold(self) -> float:
        return self.current_threshold if self.current_threshold is not None else settings.similarity_threshold


# --------------------------------------------------------------------------- #
# StreamManager
# --------------------------------------------------------------------------- #
class StreamManager:
    """Manages WebSocket connections, per-stream pipelines, broadcasting."""

    def __init__(self) -> None:
        self._connections: dict[str, ConnectionState] = {}
        self._event_subscribers: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._employee_matcher = _EmployeeMatcher()
        self._matcher_refresh_lock = asyncio.Lock()
        self._matcher_refreshed_at = 0.0
        self._detector_lock = threading.Lock()
        # Single shared attendance logger (cooldown cache) across streams.
        self.attendance_logger = AttendanceLogger()
        # Single shared pipeline (one pipeline can serve many streams in
        # sequence; for higher throughput, instantiate one per stream).
        self._default_pipeline: Any | None = None
        self._cv_service: Any | None = None

    # ----------------------------------------------------------------- #
    # Lifecycle
    # ----------------------------------------------------------------- #
    async def startup(self, cv_service: Any | None = None) -> None:
        """Build the default pipeline and pre-warm its shared CV components."""
        self._cv_service = cv_service
        self._default_pipeline = _build_pipeline(
            self._employee_matcher,
            cv_service=cv_service,
        )
        if self._default_pipeline is None:
            logger.warning("stream.startup.pipeline_unavailable")
        else:
            for name in ("detector", "extractor"):
                component = getattr(self._default_pipeline, name, None)
                warmup = getattr(component, "warmup", None)
                if callable(warmup):
                    try:
                        await asyncio.to_thread(warmup)
                    except Exception as exc:
                        logger.exception(
                            "stream.startup.component_warmup_failed",
                            component=name,
                            error=str(exc),
                        )
                        raise RuntimeError(
                            f"Required stream component '{name}' failed warmup"
                        ) from exc
            logger.info("stream.startup.pipeline_ready")

    async def shutdown(self) -> None:
        """Close all connections and tear down pipelines."""
        async with self._lock:
            for state in list(self._connections.values()):
                await self._close_state(state, code=1001, reason="server shutdown")
            for ws in list(self._event_subscribers):
                try:
                    await ws.close(code=1001)
                except Exception:
                    pass
            self._event_subscribers.clear()
            self._connections.clear()

    # ----------------------------------------------------------------- #
    # Connection registration
    # ----------------------------------------------------------------- #
    async def accept(self, ws: WebSocket) -> ConnectionState:
        await ws.accept()
        ws_connections.inc()
        ws_active.inc()
        connection_id = uuid.uuid4().hex
        pipeline = self._default_pipeline
        state = ConnectionState(
            ws=ws,
            connection_id=connection_id,
            pipeline=pipeline,
            attendance_logger=self.attendance_logger,
            current_strategy=settings.default_strategy,
        )
        async with self._lock:
            self._connections[connection_id] = state
        logger.info("stream.connected", connection_id=connection_id, total=len(self._connections))
        # Note: We do NOT auto-send a `welcome` message here because the
        # frontend's app.js doesn't expect one and some defensive tests
        # assert the first received type is in {ack, frame, attendance, error}.
        # Callers that want to greet the client can do so explicitly.
        # Start the per-connection frame processor.
        state.task = asyncio.create_task(self._run(state))
        return state

    async def disconnect(self, state: ConnectionState, code: int = 1000, reason: str = "") -> None:
        await self._close_state(state, code=code, reason=reason)

    async def _close_state(self, state: ConnectionState, *, code: int, reason: str) -> None:
        if state.closed:
            return
        state.closed = True
        ws_active.dec()
        # Cancel the worker task.
        if state.task is not None and not state.task.done():
            state.task.cancel()
            try:
                await state.task
            except (asyncio.CancelledError, Exception):
                pass
        # Reset pipeline tracker state if supported.
        if state.pipeline is not None and hasattr(state.pipeline, "reset"):
            try:
                state.pipeline.reset()
            except Exception:  # pragma: no cover
                pass
        # Close the WebSocket.
        try:
            await state.ws.close(code=code, reason=reason)
        except Exception:
            pass
        async with self._lock:
            self._connections.pop(state.connection_id, None)
        logger.info("stream.disconnected", connection_id=state.connection_id)

    # ----------------------------------------------------------------- #
    # Message handling
    # ----------------------------------------------------------------- #
    async def handle_message(self, state: ConnectionState, message: Any) -> None:
        """Dispatch a single inbound text JSON message."""
        if isinstance(message, (bytes, bytearray)):
            await self.handle_binary(state, bytes(message))
            return

        if not isinstance(message, dict):
            await self._send_error(state, "invalid_message", "Expected JSON object")
            return

        mtype = message.get("type")
        if mtype == "config":
            await self._handle_config(state, message)
        elif mtype == "source":
            await self._handle_source(state, message)
        elif mtype == "ping":
            await self._send_json(state, {"type": "pong", "t": message.get("t")})
        else:
            await self._send_error(state, "invalid_message", f"Unknown type: {mtype!r}")

    async def handle_binary(self, state: ConnectionState, data: bytes) -> None:
        """Enqueue a binary JPEG/PNG frame for processing."""
        if len(data) > settings.max_frame_bytes:
            await self._send_error(
                state,
                "frame_too_large",
                f"Frame exceeds {settings.max_frame_bytes} bytes",
            )
            return
        # Rate limit (per-connection FPS).
        now = time.monotonic()
        min_interval = 1.0 / max(1, settings.max_stream_fps)
        if now - state.last_frame_ts < min_interval:
            logger.debug(
                "stream.frame.rate_limited",
                connection_id=state.connection_id,
                max_fps=settings.max_stream_fps,
            )
            await self._send_json(state, {"type": "frame_dropped"})
            return
        state.last_frame_ts = now
        # Drop oldest if queue is full (backpressure).
        queue_limit = state.frame_queue.maxsize or settings.ws_max_queue
        if state.frame_queue.qsize() >= queue_limit:
            try:
                state.frame_queue.get_nowait()
                await self._send_json(state, {"type": "frame_dropped"})
            except asyncio.QueueEmpty:
                pass
        await state.frame_queue.put(data)

    async def _handle_config(self, state: ConnectionState, message: dict[str, Any]) -> None:
        strategy = message.get("strategy")
        if strategy is not None:
            allowed = {"both", "detection_only", "tracking_only"}
            if strategy not in allowed:
                await self._send_error(
                    state,
                    "invalid_strategy",
                    f"strategy must be one of {sorted(allowed)}",
                )
                return
            state.current_strategy = strategy
            await self._send_json(
                state, {"type": "ack", "for": "config", "strategy": strategy}
            )
            return

        threshold = message.get("threshold")
        if threshold is not None:
            try:
                state.current_threshold = float(threshold)
                await self._send_json(
                    state,
                    {"type": "ack", "for": "config", "threshold": state.current_threshold},
                )
            except (TypeError, ValueError):
                await self._send_error(state, "invalid_message", "threshold must be a number")
                return

        liveness = message.get("liveness")
        if liveness is not None:
            state.liveness_enabled = bool(liveness)
            await self._send_json(
                state,
                {"type": "ack", "for": "config", "liveness": state.liveness_enabled},
            )

        # Detector switch (live hot-swap of the face detector model).
        # Sent by the dashboard's "Detector Model" dropdown.
        detector = message.get("detector")
        if detector is not None:
            if not isinstance(detector, dict):
                await self._send_error(
                    state, "invalid_detector", "detector must be an object {backend, model}"
                )
                return
            backend = str(detector.get("backend", "")).lower()
            model = str(detector.get("model", ""))
            allowed_backends = {"insightface", "yolo26"}
            if backend not in allowed_backends:
                await self._send_error(
                    state,
                    "invalid_detector",
                    f"detector.backend must be one of {sorted(allowed_backends)}",
                )
                return
            if not model:
                await self._send_error(state, "invalid_detector", "detector.model is required")
                return
            # The frontend resends its selected detector when a stream starts.
            # Keep the warmed detector instance when the selection is unchanged.
            if (
                backend == state.detector_backend
                and model == state.detector_model
            ):
                await self._send_json(
                    state,
                    {
                        "type": "ack",
                        "for": "config",
                        "detector": {"backend": backend, "model": model},
                    },
                )
                return

            try:
                await self._rebuild_pipeline_detector(state, backend, model)
                state.detector_backend = backend
                state.detector_model = model
                await self._send_json(
                    state,
                    {"type": "ack", "for": "config",
                     "detector": {"backend": backend, "model": model}},
                )
                logger.info(
                    "stream.detector.switched",
                    connection_id=state.connection_id,
                    backend=backend,
                    model=model,
                )
            except Exception as exc:
                await self._send_error(
                    state, "detector_switch_failed", f"failed to switch detector: {exc}"
                )

    async def _rebuild_pipeline_detector(
        self,
        state: ConnectionState,
        backend: str,
        model: str,
    ) -> None:
        """Hot-swap the detector on the active pipeline.

        Tracker state, liveness history (blink counters, head-pose deques),
        and identity cache are all preserved — only the detector instance is
        replaced, so the next frame uses the new model without losing track
        of any in-progress subjects.
        """
        if state.pipeline is None:
            raise RuntimeError(
                "cannot change detector because the stream pipeline is unavailable"
            )
        # Import lazily so the module is importable in test envs without CV core
        try:
            from app.services.enrollment import _try_import_cv_core, _build_detector
            cv_core = _try_import_cv_core()
            new_detector = _build_detector(
                cv_core,
                backend=backend,
                model_name=model,
            )
            if new_detector is None:
                raise RuntimeError("could not build new detector (CV core not available)")
            warmup = getattr(new_detector, "warmup", None)
            if callable(warmup):
                await asyncio.to_thread(warmup)
            # Hot-swap detector on the existing pipeline
            if hasattr(state.pipeline, "detector"):
                state.pipeline.detector = new_detector
            elif hasattr(state.pipeline, "_detector"):
                state.pipeline._detector = new_detector
            else:
                # Pipeline doesn't expose detector attribute; rebuild from scratch
                state.pipeline = _build_pipeline(self._employee_matcher)
            if self._cv_service is not None:
                self._cv_service.detector = new_detector
        except ImportError as exc:
            raise RuntimeError(
                "CV core is unavailable; detector cannot be changed"
            ) from exc

    async def _handle_source(self, state: ConnectionState, message: dict[str, Any]) -> None:
        mode = message.get("mode")
        if mode not in {"upload", "camera", "cctv"}:
            await self._send_error(
                state, "invalid_source", "mode must be 'upload', 'camera', or 'cctv'"
            )
            return
        state.source_mode = mode

        # CCTV / RTSP handling — connect or disconnect
        if mode == "cctv":
            action = message.get("action", "connect")
            if action == "disconnect":
                # Stop any active RTSP reader
                if hasattr(state, "rtsp_cap") and state.rtsp_cap is not None:
                    try:
                        state.rtsp_cap.release()
                    except Exception:
                        pass
                    state.rtsp_cap = None
                if hasattr(state, "rtsp_task") and state.rtsp_task is not None:
                    state.rtsp_task.cancel()
                    state.rtsp_task = None
                state.cctv_camera_id = None
                state.cctv_camera_name = None
                await self._send_json(state, {
                    "type": "ack", "for": "source", "mode": "cctv",
                    "action": "disconnected",
                })
                logger.info("stream.cctv.disconnected", connection_id=state.connection_id)
                return

            # action == "connect"
            camera = message.get("camera") or {}
            cam_id = str(camera.get("id", "")).strip()
            cam_name = str(camera.get("name", "")).strip()
            cam_url = str(camera.get("url", "")).strip()
            if not cam_id or not cam_name or not cam_url:
                await self._send_error(
                    state, "invalid_cctv_config",
                    "camera.id, camera.name, and camera.url are all required",
                )
                return

            # Release any previous RTSP capture
            if hasattr(state, "rtsp_cap") and state.rtsp_cap is not None:
                try:
                    state.rtsp_cap.release()
                except Exception:
                    pass

            # Open the RTSP/HTTP stream via OpenCV (runs server-side)
            try:
                import cv2
                cap = cv2.VideoCapture(cam_url)
                if not cap.isOpened():
                    await self._send_error(
                        state, "cctv_connect_failed",
                        f"OpenCV could not open stream: {cam_url.split('@')[-1] if '@' in cam_url else cam_url}",
                    )
                    return
            except Exception as exc:
                await self._send_error(
                    state, "cctv_connect_failed", f"failed to open stream: {exc}"
                )
                return

            state.rtsp_cap = cap
            state.cctv_camera_id = cam_id
            state.cctv_camera_name = cam_name

            # Spawn a background task that reads frames from the RTSP stream
            # and pushes them into the same frame_queue the WebSocket client uses.
            if hasattr(state, "rtsp_task") and state.rtsp_task is not None:
                state.rtsp_task.cancel()
            state.rtsp_task = asyncio.create_task(self._rtsp_reader(state, cap, cam_id, cam_name))

            await self._send_json(state, {
                "type": "ack", "for": "source", "mode": "cctv",
                "action": "connected",
                "camera": {"id": cam_id, "name": cam_name},
            })
            logger.info(
                "stream.cctv.connected",
                connection_id=state.connection_id,
                camera_id=cam_id,
                camera_name=cam_name,
            )
            return

        # For upload/camera modes, just ack
        await self._send_json(state, {"type": "ack", "for": "source", "mode": mode})

    async def _rtsp_reader(self, state: ConnectionState, cap: Any,
                           camera_id: str, camera_name: str) -> None:
        """Background task: read frames from RTSP/HTTP stream and enqueue them.

        The frames go into the same frame_queue that WebSocket binary frames
        use — so the same pipeline processes them (detect → track → liveness
        → match → log). Each frame is JPEG-encoded to match the wire format.

        Auto-reconnects on stream hiccups (RTSP streams can drop periodically).
        """
        import cv2
        frame_interval = 1.0 / max(1, settings.max_stream_fps)
        last_frame_ts = 0.0
        try:
            while not state.closed and cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    logger.warning(
                        "stream.cctv.read_failed — reconnecting",
                        connection_id=state.connection_id,
                        camera_id=camera_id,
                    )
                    # Try to reconnect once
                    await asyncio.sleep(1.0)
                    try:
                        cap.release()
                    except Exception:
                        pass
                    # Note: we don't have the original URL here — caller must
                    # reconnect manually if this fails repeatedly. Send error.
                    await self._send_error(
                        state, "cctv_stream_dropped",
                        f"Stream from {camera_name} dropped — please reconnect",
                    )
                    break

                # Throttle to the configured stream FPS cap.
                now = time.time()
                if now - last_frame_ts < frame_interval:
                    await asyncio.sleep(frame_interval - (now - last_frame_ts))
                last_frame_ts = now

                # JPEG-encode and push to the same queue as WebSocket frames
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    continue
                # Set the camera_id so attendance logs reference it
                state.active_camera_id = camera_id
                # Drop oldest if queue full
                queue_limit = state.frame_queue.maxsize or settings.ws_max_queue
                if state.frame_queue.qsize() >= queue_limit:
                    try:
                        state.frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await state.frame_queue.put(buf.tobytes())
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("stream.cctv.reader_crashed: %s", exc)
        finally:
            try:
                cap.release()
            except Exception:
                pass

    # ----------------------------------------------------------------- #
    # Per-connection frame processor
    # ----------------------------------------------------------------- #
    async def _run(self, state: ConnectionState) -> None:
        """Run frame detection and downstream processing as bounded stages."""
        try:
            if state.pipeline is None or not hasattr(state.pipeline, "detector"):
                await self._run_serial(state)
                return

            detect_task = asyncio.create_task(self._detect_frames(state))
            process_task = asyncio.create_task(self._process_detected_frames(state))
            try:
                await asyncio.gather(detect_task, process_task)
            finally:
                for task in (detect_task, process_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    detect_task, process_task, return_exceptions=True
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("stream.runner.crashed", connection_id=state.connection_id, error=str(exc))
            await self._send_error(state, "internal_error", str(exc), fatal=True)

    async def _run_serial(self, state: ConnectionState) -> None:
        """Process frames serially when a detector pipeline is unavailable."""
        while not state.closed:
            try:
                data = await asyncio.wait_for(state.frame_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await self._process_one(state, data)

    async def _detect_frames(self, state: ConnectionState) -> None:
        """Run detector inference in a worker thread and enqueue bounded results."""
        loop = asyncio.get_running_loop()
        while not state.closed:
            try:
                data = await asyncio.wait_for(state.frame_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            frame = self._decode_frame(data)
            if frame is None:
                await self._send_error(
                    state, "unsupported_format", "Frame is neither JPEG nor PNG"
                )
                continue
            try:
                detections = await loop.run_in_executor(
                    None, self._detect_frame, state.pipeline, frame
                )
            except Exception as exc:
                logger.exception(
                    "stream.detect_frame.failed",
                    connection_id=state.connection_id,
                    error=str(exc),
                )
                await self._send_error(
                    state,
                    "detection_failed",
                    f"Face detection failed: {exc}",
                )
                continue
            if state.extract_queue.full():
                try:
                    state.extract_queue.get_nowait()
                    await self._send_json(state, {"type": "frame_dropped"})
                except asyncio.QueueEmpty:
                    pass
            await state.extract_queue.put((frame, detections))

    def _detect_frame(self, pipeline: Any, frame: np.ndarray) -> list[Any]:
        """Serialize access to the shared detector while allowing downstream overlap."""
        with self._detector_lock:
            return list(pipeline.detector.detect(frame))

    async def _process_detected_frames(self, state: ConnectionState) -> None:
        """Process frames after detection, preserving tracker and result ordering."""
        while not state.closed:
            try:
                frame, detections = await asyncio.wait_for(
                    state.extract_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            await self._process_one(
                state,
                b"",
                decoded_frame=frame,
                precomputed_detections=detections,
            )

    async def _process_one(
        self,
        state: ConnectionState,
        data: bytes,
        *,
        decoded_frame: np.ndarray | None = None,
        precomputed_detections: list[Any] | None = None,
    ) -> None:
        """Decode, process, and emit results for a single frame."""
        t0 = time.monotonic()
        frame = decoded_frame if decoded_frame is not None else self._decode_frame(data)
        if frame is None:
            await self._send_error(state, "unsupported_format", "Frame is neither JPEG nor PNG")
            return

        strategy = state.current_strategy
        annotated_frame: np.ndarray | None = None
        detections: list[dict[str, Any]] = []
        attendance_events: list[dict[str, Any]] = []

        if state.pipeline is None:
            annotated_frame = frame
        else:
            try:
                await self._refresh_employee_matcher()
                result = await self._run_pipeline(
                    state.pipeline,
                    frame,
                    strategy,
                    detections=precomputed_detections,
                )
                annotated_frame, detections, attendance_events = self._unpack_result(
                    result, state
                )
            except ValueError as exc:
                await self._send_error(state, "invalid_strategy", str(exc))
                return
            except Exception as exc:
                logger.exception("stream.process_frame.failed", error=str(exc))
                await self._send_error(state, "pipeline_error", str(exc))
                return

        image_b64 = self._encode_frame(
            annotated_frame if annotated_frame is not None else frame
        )
        state.frame_counter += 1
        completed_at = time.monotonic()
        latency_ms = (completed_at - t0) * 1000.0
        last_result_ts = getattr(state, "last_result_ts", 0.0)
        delivered_fps = (
            1.0 / (completed_at - last_result_ts)
            if last_result_ts > 0
            else 0.0
        )
        state.last_result_ts = completed_at

        frames_processed_total.labels(strategy=strategy).inc()
        frame_processing_seconds.labels(strategy=strategy).observe(latency_ms / 1000.0)
        if detections:
            faces_detected_total.labels(strategy=strategy).inc(len(detections))

        await self._send_json(
            state,
            {
                "type": "frame",
                "frame_id": state.frame_counter,
                "image": image_b64,
                "strategy": strategy,
                "detections": detections,
                "latency_ms": round(latency_ms, 2),
                "fps": round(delivered_fps, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        for evt in attendance_events:
            await self._send_json(state, {"type": "attendance", **evt})
            await self.broadcast_event({"type": "attendance", **evt})

    async def _refresh_employee_matcher(self) -> None:
        """Refresh active employee embeddings at most once every five seconds."""
        if time.monotonic() - self._matcher_refreshed_at < 5.0:
            return
        async with self._matcher_refresh_lock:
            if time.monotonic() - self._matcher_refreshed_at < 5.0:
                return
            try:
                from app.db import crud

                employees = []
                async with get_session() as session:
                    offset = 0
                    while True:
                        batch = await crud.list_employees(
                            session, limit=1000, offset=offset
                        )
                        employees.extend(batch)
                        if len(batch) < 1000:
                            break
                        offset += len(batch)
                matchable_employees = self._employee_matcher.replace(employees)
                self._matcher_refreshed_at = time.monotonic()
                logger.info(
                    "stream.matcher.refreshed",
                    active_employees=len(employees),
                    matchable_employees=matchable_employees,
                )
            except Exception:
                logger.exception("stream.matcher.refresh_failed")

    async def _run_pipeline(
        self,
        pipeline: Any,
        frame: np.ndarray,
        strategy: str,
        *,
        detections: list[Any] | None = None,
    ) -> Any:
        """Run ``pipeline.process_frame`` in a thread executor (it's sync)."""
        loop = asyncio.get_running_loop()
        if detections is not None:
            return await loop.run_in_executor(
                None,
                lambda: pipeline.process_frame(
                    frame, strategy=strategy, detections=detections
                ),
            )
        return await loop.run_in_executor(
            None, lambda: pipeline.process_frame(frame, strategy=strategy)
        )

    def _unpack_result(
        self, result: Any, state: ConnectionState
    ) -> tuple[np.ndarray | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """Translate a ``FrameResult`` into ``(annotated_frame, detections, events)``.

        FrameResult shape (per Task 1-a interface contract):
            annotated_frame (np.ndarray), tracks (list[TrackResult]),
            fps (float), strategy (str), timestamp (str)
        TrackResult shape:
            track_id, bbox, identity (dict|None), similarity,
            liveness_passed, liveness_confidence, strategy
        """
        annotated_frame = getattr(result, "annotated_frame", None)
        tracks = getattr(result, "tracks", None) or []
        detections: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []

        for tr in tracks:
            bbox = getattr(tr, "bbox", None)
            score = getattr(tr, "score", None) or getattr(tr, "similarity", None)
            track_id = getattr(tr, "track_id", None)
            identity = getattr(tr, "identity", None)
            similarity = getattr(tr, "similarity", None)
            liveness_passed = getattr(tr, "liveness_passed", None)
            liveness_confidence = getattr(tr, "liveness_confidence", None)
            strategy = getattr(tr, "strategy", state.current_strategy)

            det: dict[str, Any] = {
                "bbox": [float(x) for x in bbox] if bbox is not None else None,
                "score": float(score) if score is not None else None,
                "track_id": int(track_id) if track_id is not None else None,
                "is_real": liveness_passed if liveness_passed is not None else None,
            }
            if identity is not None:
                emp_id = (
                    identity.get("employee_id")
                    if isinstance(identity, dict)
                    else getattr(identity, "employee_id", None)
                )
                if emp_id is not None:
                    det["identity"] = str(emp_id)
            detections.append(det)

            # Attendance must never be logged unless liveness explicitly passed.
            if liveness_passed is not True:
                if liveness_passed is False:
                    spoof_rejected_total.labels(strategy=strategy).inc()
                continue

            # If we have a matched identity, log attendance.
            if identity is not None and similarity is not None and similarity >= state.effective_threshold:
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
                if emp_id:
                    recognitions_total.labels(
                        strategy=strategy, outcome="matched"
                    ).inc()
                    # Fire-and-forget DB write in the background.
                    asyncio.create_task(
                        self._log_attendance_async(
                            state,
                            employee_id=str(emp_id),
                            employee_pk=emp_pk,
                            similarity=float(similarity),
                            track_id=track_id,
                            strategy=strategy,
                            liveness_passed=liveness_passed,
                            liveness_confidence=liveness_confidence,
                        )
                    )
                    events.append(
                        {
                            "employee_id": str(emp_id),
                            "name": emp_name or "",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "strategy": strategy,
                            "score": round(float(similarity), 4),
                            "track_id": int(track_id) if track_id is not None else None,
                            "liveness_passed": liveness_passed,
                        }
                    )
        return annotated_frame, detections, events

    async def _log_attendance_async(
        self,
        state: ConnectionState,
        *,
        employee_id: str,
        employee_pk: Any | None,
        similarity: float,
        track_id: int | None,
        strategy: str,
        liveness_passed: bool | None,
        liveness_confidence: float | None,
    ) -> None:
        """Persist attendance in a fresh DB session (background task)."""
        try:
            async with get_session() as session:
                await state.attendance_logger.log_attendance(
                    session,
                    employee_id=employee_id,
                    employee_pk=employee_pk,
                    similarity=similarity,
                    track_id=track_id,
                    camera_id=state.connection_id,
                    strategy=strategy,
                    liveness_passed=liveness_passed,
                    liveness_confidence=liveness_confidence,
                )
        except Exception as exc:  # pragma: no cover
            logger.exception("stream.attendance.log_failed", error=str(exc))

    # ----------------------------------------------------------------- #
    # Events broadcast (for /ws/events dashboard subscribers)
    # ----------------------------------------------------------------- #
    async def add_event_subscriber(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._event_subscribers.add(ws)
        logger.info("stream.events.subscribed", total=len(self._event_subscribers))

    async def remove_event_subscriber(self, ws: WebSocket) -> None:
        async with self._lock:
            self._event_subscribers.discard(ws)
        try:
            await ws.close()
        except Exception:
            pass

    async def broadcast_event(self, event: dict[str, Any]) -> None:
        """Push an event to every /ws/events subscriber."""
        if not self._event_subscribers:
            return
        dead: list[WebSocket] = []
        async with self._lock:
            subscribers = list(self._event_subscribers)
        for ws in subscribers:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._event_subscribers.discard(ws)

    # ----------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------- #
    def _decode_frame(self, data: bytes) -> np.ndarray | None:
        """Decode a JPEG/PNG byte blob into a BGR uint8 ndarray."""
        try:
            import cv2  # type: ignore

            arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            return arr
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

    def _encode_frame(self, frame: np.ndarray) -> str:
        """JPEG-encode a frame and return base64."""
        try:
            import cv2  # type: ignore

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return ""
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except ImportError:
            try:
                from PIL import Image  # type: ignore
                import io

                img = Image.fromarray(frame[..., ::-1])
                bio = io.BytesIO()
                img.save(bio, format="JPEG", quality=80)
                return base64.b64encode(bio.getvalue()).decode("ascii")
            except Exception:
                return ""
        except Exception:
            return ""

    async def _send_json(self, state: ConnectionState, payload: dict[str, Any]) -> None:
        if state.closed:
            return
        send_lock = getattr(state, "send_lock", None)
        if send_lock is None:
            await self._send_json_unlocked(state, payload)
            return
        async with send_lock:
            if not state.closed:
                await self._send_json_unlocked(state, payload)

    async def _send_json_unlocked(
        self, state: ConnectionState, payload: dict[str, Any]
    ) -> None:
        try:
            await state.ws.send_json(payload)
        except Exception as exc:
            logger.debug("stream.send_failed", error=str(exc))

    async def _send_error(
        self,
        state: ConnectionState,
        code: str,
        message: str,
        *,
        fatal: bool = False,
    ) -> None:
        await self._send_json(
            state,
            {"type": "error", "code": code, "message": message, "fatal": fatal},
        )

    # ----------------------------------------------------------------- #
    # Introspection
    # ----------------------------------------------------------------- #
    @property
    def active_connections(self) -> int:
        return len(self._connections)


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #
stream_manager = StreamManager()

__all__ = ["StreamManager", "stream_manager", "ConnectionState"]
