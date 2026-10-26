"""Tests for WebSocket stream frame handling."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.stream_manager import StreamManager, _EmployeeMatcher, _build_pipeline


def test_default_stream_pipeline_builds_with_configured_detector():
    matcher = _EmployeeMatcher()
    pipeline = _build_pipeline(matcher)

    assert pipeline is not None
    assert pipeline.detector is not None
    assert pipeline.liveness is not None
    assert pipeline.matcher is matcher


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
    state = SimpleNamespace(
        connection_id="test-connection",
        last_frame_ts=time.monotonic(),
        frame_queue=asyncio.Queue(),
    )

    await manager.handle_binary(state, b"frame")

    assert state.frame_queue.empty()


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
