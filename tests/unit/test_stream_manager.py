"""Tests for WebSocket stream frame handling."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.stream_manager import (
    ConnectionState,
    StreamManager,
    _EmployeeMatcher,
    _build_pipeline,
)
from app.utils.metrics import ws_active


@pytest.mark.asyncio
async def test_disconnect_decrements_active_connections_and_cleans_state():
    manager = StreamManager()
    before = ws_active._value.get()
    ws_active.inc()

    class WebSocket:
        async def close(self, code, reason):
            return None

    state = ConnectionState(
        ws=WebSocket(),
        connection_id="disconnect-test",
        pipeline=None,
        attendance_logger=SimpleNamespace(),
    )
    manager._connections[state.connection_id] = state

    await manager.disconnect(state)

    assert ws_active._value.get() == before
    assert state.connection_id not in manager._connections
    assert state.closed is True


def test_default_stream_pipeline_builds_with_configured_detector():
    matcher = _EmployeeMatcher()
    pipeline = _build_pipeline(matcher)

    assert pipeline is not None
    assert pipeline.detector is not None
    assert pipeline.liveness is not None
    assert pipeline.matcher is matcher


@pytest.mark.asyncio
async def test_startup_warms_detector_and_extractor(monkeypatch):
    warmed = []
    pipeline = SimpleNamespace(
        detector=SimpleNamespace(warmup=lambda: warmed.append("detector")),
        extractor=SimpleNamespace(warmup=lambda: warmed.append("extractor")),
    )
    monkeypatch.setattr(
        "app.services.stream_manager._build_pipeline",
        lambda _matcher, **_kwargs: pipeline,
    )

    await StreamManager().startup()

    assert warmed == ["detector", "extractor"]


@pytest.mark.asyncio
async def test_startup_uses_enrollment_service_for_stream_pipeline(monkeypatch):
    service = SimpleNamespace(detector=object(), aligner=object(), extractor=object())
    manager = StreamManager()
    received = {}
    pipeline = SimpleNamespace(
        detector=SimpleNamespace(),
        extractor=SimpleNamespace(),
    )

    def build_pipeline(_matcher, *, cv_service=None):
        received["cv_service"] = cv_service
        return pipeline

    monkeypatch.setattr(
        "app.services.stream_manager._build_pipeline",
        build_pipeline,
    )

    await manager.startup(service)

    assert received["cv_service"] is service


@pytest.mark.asyncio
async def test_startup_fails_if_required_component_warmup_fails(monkeypatch):
    def fail_warmup():
        raise RuntimeError("model failed")

    pipeline = SimpleNamespace(
        detector=SimpleNamespace(warmup=fail_warmup),
        extractor=SimpleNamespace(warmup=lambda: None),
    )
    monkeypatch.setattr(
        "app.services.stream_manager._build_pipeline",
        lambda _matcher, **_kwargs: pipeline,
    )

    with pytest.raises(RuntimeError, match="detector.*failed warmup"):
        await StreamManager().startup()


@pytest.mark.asyncio
async def test_startup_detector_config_keeps_preloaded_model(monkeypatch):
    manager = StreamManager()
    messages = []

    class WebSocket:
        async def send_json(self, payload):
            messages.append(payload)

    state = SimpleNamespace(
        detector_backend="yolo26",
        detector_model="yolo26s-face.pt",
        ws=WebSocket(),
        closed=False,
    )
    monkeypatch.setattr(
        manager,
        "_rebuild_pipeline_detector",
        lambda *_args: pytest.fail("unchanged detector should not be rebuilt"),
    )

    await manager._handle_config(
        state,
        {"detector": {"backend": "yolo26", "model": "yolo26s-face.pt"}},
    )

    assert messages == [
        {
            "type": "ack",
            "for": "config",
            "detector": {"backend": "yolo26", "model": "yolo26s-face.pt"},
        }
    ]


@pytest.mark.asyncio
async def test_detector_change_updates_state_after_rebuild(monkeypatch):
    manager = StreamManager()
    messages = []

    class WebSocket:
        async def send_json(self, payload):
            messages.append(payload)

    state = SimpleNamespace(
        detector_backend="yolo26",
        detector_model="yolo26s-face.pt",
        ws=WebSocket(),
        closed=False,
    )
    rebuilt = []

    async def rebuild(_state, backend, model):
        rebuilt.append((backend, model))

    monkeypatch.setattr(manager, "_rebuild_pipeline_detector", rebuild)

    await manager._handle_config(
        state,
        {"detector": {"backend": "yolo26", "model": "yolo26n-face.pt"}},
    )

    assert rebuilt == [("yolo26", "yolo26n-face.pt")]
    assert state.detector_model == "yolo26n-face.pt"
    assert messages[0]["type"] == "ack"


@pytest.mark.asyncio
async def test_detector_hot_swap_updates_shared_enrollment_service(monkeypatch):
    manager = StreamManager()
    service = SimpleNamespace(detector=object())
    manager._cv_service = service
    old_detector = object()
    state = SimpleNamespace(
        pipeline=SimpleNamespace(detector=old_detector),
    )
    new_detector = SimpleNamespace(warmup=lambda: None)
    monkeypatch.setattr(
        "app.services.enrollment._try_import_cv_core",
        lambda: {"YoloFaceDetector": object},
    )
    monkeypatch.setattr(
        "app.services.enrollment._build_detector",
        lambda _core, **_kwargs: new_detector,
    )

    await manager._rebuild_pipeline_detector(
        state,
        "yolo26",
        "yolo26n-face.pt",
    )

    assert state.pipeline.detector is new_detector
    assert service.detector is new_detector


def test_employee_matcher_matches_enrolled_embeddings():
    matcher = _EmployeeMatcher()
    employee = SimpleNamespace(
        employee_id="EMP001",
        id="employee-pk",
        name="Test Employee",
        embedding=np.array([3.0, 4.0], dtype=np.float32),
    )
    matcher.replace([employee])

    identity, similarity = matcher.find_nearest(
        np.array([3.0, 4.0], dtype=np.float32), threshold=0.8
    )

    assert identity == {
        "employee_id": "EMP001",
        "employee_pk": "employee-pk",
        "name": "Test Employee",
    }
    assert similarity == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_rate_limited_frame_is_dropped_without_backend_error():
    manager = StreamManager()
    messages = []

    class WebSocket:
        async def send_json(self, message):
            messages.append(message)

    state = SimpleNamespace(
        connection_id="test-connection",
        last_frame_ts=time.monotonic(),
        frame_queue=asyncio.Queue(),
        closed=False,
        ws=WebSocket(),
    )

    await manager.handle_binary(state, b"frame")

    assert state.frame_queue.empty()
    assert messages == [{"type": "frame_dropped"}]


@pytest.mark.asyncio
async def test_rate_limit_rejects_frames_faster_than_30_fps(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_stream_fps", 30)
    manager = StreamManager()
    messages = []

    class WebSocket:
        async def send_json(self, message):
            messages.append(message)

    state = SimpleNamespace(
        connection_id="test-connection",
        last_frame_ts=time.monotonic() - (1.0 / 60.0),
        frame_queue=asyncio.Queue(),
        closed=False,
        ws=WebSocket(),
    )

    await manager.handle_binary(state, b"frame")

    assert state.frame_queue.empty()
    assert messages == [{"type": "frame_dropped"}]


@pytest.mark.asyncio
async def test_frame_below_rate_limit_is_queued():
    manager = StreamManager()
    state = SimpleNamespace(
        connection_id="test-connection",
        last_frame_ts=0.0,
        frame_queue=asyncio.Queue(),
    )

    await manager.handle_binary(state, b"frame")

    assert await state.frame_queue.get() == b"frame"


@pytest.mark.asyncio
async def test_process_one_serializes_numpy_frame_without_boolean_coercion(monkeypatch):
    manager = StreamManager()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    messages = []

    class WebSocket:
        async def send_json(self, message):
            messages.append(message)

    state = SimpleNamespace(
        pipeline=None,
        current_strategy="both",
        frame_counter=0,
        closed=False,
        ws=WebSocket(),
    )
    monkeypatch.setattr(manager, "_decode_frame", lambda _data: frame)
    monkeypatch.setattr(manager, "_encode_frame", lambda image: "encoded")

    await manager._process_one(state, b"frame")

    assert messages[0]["type"] == "frame"
    assert messages[0]["image"] == "encoded"
    assert messages[0]["fps"] == 0.0


@pytest.mark.asyncio
async def test_reported_fps_uses_time_between_completed_frames(monkeypatch):
    manager = StreamManager()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    messages = []
    tick = iter([1.0, 1.0, 1.05, 1.05])

    class WebSocket:
        async def send_json(self, message):
            messages.append(message)

    state = SimpleNamespace(
        pipeline=None,
        current_strategy="both",
        frame_counter=0,
        closed=False,
        ws=WebSocket(),
        last_result_ts=0.0,
    )
    monkeypatch.setattr(manager, "_decode_frame", lambda _data: frame)
    monkeypatch.setattr(manager, "_encode_frame", lambda image: "encoded")
    monkeypatch.setattr(
        "app.services.stream_manager.time",
        SimpleNamespace(monotonic=lambda: next(tick)),
    )

    await manager._process_one(state, b"frame")
    await manager._process_one(state, b"frame")

    assert messages[0]["fps"] == 0.0
    assert messages[1]["fps"] == pytest.approx(20.0)


def test_connection_uses_bounded_stage_queues():
    state = ConnectionState(
        ws=SimpleNamespace(),
        connection_id="test-connection",
        pipeline=None,
        attendance_logger=SimpleNamespace(),
    )

    assert state.frame_queue.maxsize == 2
    assert state.extract_queue.maxsize == 2


@pytest.mark.asyncio
async def test_process_one_passes_precomputed_detections_to_pipeline(monkeypatch):
    manager = StreamManager()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    marker = object()
    received = {}
    messages = []

    class Pipeline:
        def process_frame(self, frame, strategy, *, detections=None):
            received["detections"] = detections
            return SimpleNamespace(
                annotated_frame=frame,
                tracks=[],
                detections=[],
                attendance_events=[],
            )

    class WebSocket:
        async def send_json(self, message):
            messages.append(message)

    state = SimpleNamespace(
        pipeline=Pipeline(),
        current_strategy="both",
        frame_counter=0,
        closed=False,
        ws=WebSocket(),
    )
    monkeypatch.setattr(manager, "_encode_frame", lambda image: "encoded")
    monkeypatch.setattr(manager, "_refresh_employee_matcher", _noop_async)
    monkeypatch.setattr(
        manager,
        "_unpack_result",
        lambda result, _state: (result.annotated_frame, [], []),
    )

    await manager._process_one(
        state,
        b"",
        decoded_frame=frame,
        precomputed_detections=[marker],
    )

    assert received["detections"] == [marker]
    assert messages[0]["type"] == "frame"


@pytest.mark.asyncio
async def test_detection_only_identity_is_logged_without_failing_liveness_gate(monkeypatch):
    manager = StreamManager()
    identity = {
        "employee_id": "employee-1",
        "name": "Test Employee",
        "employee_pk": 1,
    }
    logged = []

    async def record_attendance(_state, **kwargs):
        logged.append(kwargs)

    monkeypatch.setattr(manager, "_log_attendance_async", record_attendance)
    state = SimpleNamespace(current_strategy="detection_only", effective_threshold=0.5)
    result = SimpleNamespace(
        annotated_frame=np.zeros((2, 2, 3), dtype=np.uint8),
        tracks=[
            SimpleNamespace(
                bbox=(0, 0, 1, 1),
                score=0.95,
                track_id=None,
                identity=identity,
                similarity=0.95,
                liveness_passed=None,
                liveness_confidence=None,
                strategy="detection_only",
            )
        ],
    )

    _, detections, events = manager._unpack_result(result, state)
    await asyncio.sleep(0)

    assert detections[0]["is_real"] is None
    assert events[0]["employee_id"] == "employee-1"
    assert events[0]["liveness_passed"] is None
    assert logged[0]["liveness_passed"] is None


@pytest.mark.asyncio
async def test_detector_inference_failure_is_reported_to_stream_client(monkeypatch):
    manager = StreamManager()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    messages = []
    state = SimpleNamespace(
        closed=False,
        connection_id="test-connection",
        frame_queue=asyncio.Queue(),
        pipeline=SimpleNamespace(
            detector=SimpleNamespace(
                detect=lambda _frame: (_ for _ in ()).throw(
                    RuntimeError("model inference failed")
                )
            )
        ),
    )
    await state.frame_queue.put(b"frame")

    async def send_error(_state, code, message, **_kwargs):
        messages.append({"code": code, "message": message})
        state.closed = True

    monkeypatch.setattr(manager, "_decode_frame", lambda _data: frame)
    monkeypatch.setattr(manager, "_send_error", send_error)

    await manager._detect_frames(state)

    assert messages == [
        {
            "code": "detection_failed",
            "message": "Face detection failed: model inference failed",
        }
    ]


async def _noop_async():
    return None
