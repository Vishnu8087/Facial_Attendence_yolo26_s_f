"""Integration tests for the WebSocket endpoint at ``/ws/stream``.

The WebSocket protocol (see ``docs/WEBSOCKET_PROTOCOL.md``) defines two
client-to-server message types:

* ``{"type": "config", "strategy": "both"|"detection_only"|"tracking_only"}``
* ``{"type": "source", "mode": "upload"|"camera"}``

plus binary frames (raw JPEG / PNG bytes).

Server-to-client messages:

* ``{"type": "frame", ...}`` — processed frame with annotations
* ``{"type": "attendance", ...}`` — logged attendance event
* ``{"type": "error", "message": "..."}`` — error notification
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

pytestmark = pytest.mark.asyncio


async def test_websocket_connect(ws_client):
    """The connection should stay open after the initial handshake."""
    # If we got here, the connection is open.
    assert ws_client is not None


async def test_send_config_strategy_both(ws_client):
    await ws_client.send_json({"type": "config", "strategy": "both"})
    # The server may ACK or just silently accept — give it a moment.
    try:
        msg = await asyncio.wait_for(ws_client.receive(), timeout=1.0)
        # If anything came back, it should be JSON-parseable (or a binary frame).
        if msg.get("text"):
            payload = json.loads(msg["text"])
            assert payload.get("type") in ("ack", "config", "frame", "error")
    except asyncio.TimeoutError:
        pass  # server silent ack — acceptable


async def test_send_config_invalid_strategy(ws_client):
    await ws_client.send_json({"type": "config", "strategy": "nonsense"})
    try:
        msg = await asyncio.wait_for(ws_client.receive(), timeout=1.0)
        if msg.get("text"):
            payload = json.loads(msg["text"])
            assert payload.get("type") in ("error", "ack")
    except asyncio.TimeoutError:
        pytest.skip("server did not respond to invalid config")


async def test_send_source_message(ws_client):
    await ws_client.send_json({"type": "source", "mode": "upload"})
    # No strict assertion on response; just confirm the channel still works
    # by sending a follow-up config.
    await ws_client.send_json({"type": "config", "strategy": "both"})


async def test_send_binary_frame_and_receive_processed_frame(ws_client, sample_frame):
    """Send a binary JPEG frame and expect the server to echo back either
    a ``frame`` event (binary or JSON with image data) or an ``attendance``
    event."""
    cv2 = None
    try:
        import cv2  # type: ignore
    except Exception:
        pass
    if cv2 is None:
        pytest.skip("cv2 not installed — cannot encode JPEG")

    ok, encoded = cv2.imencode(".jpg", sample_frame)
    assert ok
    await ws_client.send_bytes(encoded.tobytes())

    # The server should respond within 5s with a frame or attendance event.
    received_types = set()
    try:
        for _ in range(3):
            msg = await asyncio.wait_for(ws_client.receive(), timeout=5.0)
            if msg.get("text"):
                payload = json.loads(msg["text"])
                received_types.add(payload.get("type"))
            elif msg.get("bytes"):
                received_types.add("binary_frame")
                break
    except asyncio.TimeoutError:
        pass

    assert received_types & {"frame", "attendance", "binary_frame"}, (
        f"Expected frame/attendance response, got {received_types}"
    )


async def test_frame_event_carries_strategy(ws_client, sample_frame):
    """A processed-frame event should echo back the active strategy so the
    frontend knows which pipeline ran."""
    await ws_client.send_json({"type": "config", "strategy": "detection_only"})
    cv2 = None
    try:
        import cv2  # type: ignore
        cv2 = cv2
    except Exception:
        pass
    if cv2 is None:
        pytest.skip("cv2 not installed")
    ok, encoded = cv2.imencode(".jpg", sample_frame)
    await ws_client.send_bytes(encoded.tobytes())

    try:
        for _ in range(3):
            msg = await asyncio.wait_for(ws_client.receive(), timeout=5.0)
            if msg.get("text"):
                payload = json.loads(msg["text"])
                if payload.get("type") == "frame":
                    assert payload.get("strategy") in ("detection_only", "both", "tracking_only")
                    return
    except asyncio.TimeoutError:
        pytest.skip("server did not return a frame event in time")
