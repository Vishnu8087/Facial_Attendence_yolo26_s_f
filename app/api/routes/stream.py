"""WebSocket `/ws/stream` + REST `/api/recognize`.

WebSocket protocol (see ``docs/WEBSOCKET_PROTOCOL.md``):

Client → Server:
* ``{"type": "config", "strategy": "both" | "detection_only" | "tracking_only"}``
* ``{"type": "config", "threshold": 0.7}``
* ``{"type": "config", "liveness": true}``
* ``{"type": "source", "mode": "upload" | "camera"}``
* ``{"type": "ping", "t": <epoch_ms>}``
* Binary frame: JPEG or PNG bytes

Server → Client:
* ``{"type": "ack", "for": "config" | "source", ...}``
* ``{"type": "frame", "frame_id": int, "image": <b64>, "strategy": ..., "detections": [...], "latency_ms": ...}``
* ``{"type": "attendance", "employee_id": ..., "name": ..., "timestamp": ..., "strategy": ..., "score": ..., "track_id": ..., "liveness_passed": ...}``
* ``{"type": "pong", "t": <epoch_ms>}``
* ``{"type": "error", "code": ..., "message": ..., "fatal": false}``

REST:
* ``POST /api/recognize`` — single-image recognition (no streaming).
"""
from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
from typing import Annotated

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session, get_stream_manager
from app.config import settings
from app.schemas.attendance import RecognizeRequest, RecognizeResponse, RecognizedIdentity
from app.schemas.common import ErrorResponse
from app.services.enrollment import EnrollmentService
from app.services.stream_manager import StreamManager
from app.utils.logging import get_logger
from app.utils.security import User, user_from_token_param

logger = get_logger(__name__)

# Two routers: one for the WebSocket (no prefix — mounted at root),
# one for the REST /recognize endpoint (mounted under /api).
ws_router = APIRouter()
rest_router = APIRouter()


# --------------------------------------------------------------------------- #
# WebSocket endpoint
# --------------------------------------------------------------------------- #
@ws_router.websocket("/ws/stream")
async def stream_endpoint(
    ws: WebSocket,
    token: Annotated[str | None, Query(description="JWT for browsers (cannot set Authorization header)")] = None,
    stream_manager: StreamManager = Depends(get_stream_manager),
) -> None:
    """Live frame ingestion WebSocket.

    Accepts binary JPEG/PNG frames and JSON control messages
    (``config``/``source``/``ping``). Pushes back ``frame``, ``attendance``,
    ``ack``, ``pong``, and ``error`` events.

    Auth: the ``token`` query parameter is validated via
    :func:`app.utils.security.user_from_token_param`. When
    ``settings.api_key`` is empty, auth is disabled (trusted-network mode).
    """
    # Validate token (if auth enabled). This raises HTTPException(401) →
    # Starlette converts to a WebSocket close with code 1008.
    try:
        user_from_token_param(token)
    except HTTPException:
        await ws.close(code=1008)  # policy violation
        return

    state = await stream_manager.accept(ws)
    logger.info("ws.stream.accepted", connection_id=state.connection_id)
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text"):
                try:
                    import json

                    payload = json.loads(message["text"])
                except Exception as exc:
                    await stream_manager._send_error(
                        state, "invalid_message", f"JSON parse failed: {exc}"
                    )
                    continue
                await stream_manager.handle_message(state, payload)
            elif message.get("bytes"):
                await stream_manager.handle_binary(state, message["bytes"])
    except WebSocketDisconnect:
        logger.info("ws.stream.disconnected", connection_id=state.connection_id)
    except Exception as exc:
        logger.exception("ws.stream.error", connection_id=state.connection_id, error=str(exc))
    finally:
        await stream_manager.disconnect(state, code=1000, reason="client disconnected")


# --------------------------------------------------------------------------- #
# Optional: events broadcast channel for the dashboard live feed
# --------------------------------------------------------------------------- #
@ws_router.websocket("/ws/events")
async def events_endpoint(
    ws: WebSocket,
    stream_manager: StreamManager = Depends(get_stream_manager),
) -> None:
    """Subscribe to live attendance events from all streams.

    Used by dashboard widgets that don't need to send frames but want
    to react to new attendance logs (e.g. the "recent activity" panel).
    """
    await stream_manager.add_event_subscriber(ws)
    try:
        while True:
            # We don't expect inbound messages — just keep the socket open.
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        await stream_manager.remove_event_subscriber(ws)


# --------------------------------------------------------------------------- #
# REST: POST /api/recognize
# --------------------------------------------------------------------------- #
@rest_router.post(
    "/recognize",
    response_model=RecognizeResponse,
    summary="Recognise faces in a single image (REST, non-streaming)",
    description=(
        "Accepts a JSON body with either `image_base64` (the backend runs "
        "the full pipeline) or `embedding_base64` (the backend skips "
        "detection/extraction and only does the DB lookup). Returns the "
        "identities found, one per face."
    ),
    responses={
        200: {"description": "OK"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "CV models not loaded"},
    },
)
async def recognize(
    body: RecognizeRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _user: User = Depends(lambda u: u),  # noqa: ANN001 — any user
) -> RecognizeResponse:
    """Recognize endpoint — single-image REST variant."""
    import time as _time

    from app.db import crud

    t0 = _time.monotonic()
    threshold = body.threshold if body.threshold is not None else settings.similarity_threshold

    # Embedding-only path (no CV core needed).
    if body.embedding_base64 and not body.image_base64:
        from app.schemas.employee import decode_embedding_b64

        try:
            embedding = decode_embedding_b64(body.embedding_base64)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid embedding_base64: {exc}",
            ) from exc
        emp, sim = await crud.find_nearest_employee(session, embedding, threshold=threshold)
        latency_ms = (_time.monotonic() - t0) * 1000.0
        return RecognizeResponse(
            identities=[
                RecognizedIdentity(
                    employee_id=emp.employee_id if emp else None,
                    name=emp.name if emp else None,
                    similarity=float(sim),
                )
            ],
            strategy=body.strategy,
            latency_ms=round(latency_ms, 2),
        )

    # Image path — needs CV core.
    if not body.image_base64:
        raise HTTPException(
            status_code=422,
            detail="Either image_base64 or embedding_base64 must be provided.",
        )

    if not getattr(request.app.state, "models_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CV models not loaded",
        )

    try:
        raw = base64.b64decode(body.image_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid image_base64: {exc}",
        ) from exc

    # Run the CV pipeline.
    enrollment = EnrollmentService()
    frame = enrollment._decode_image(raw)
    if frame is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image (expected JPEG/PNG)",
        )
    embedding, num_faces, _ = await enrollment._extract_embedding(frame)
    if embedding is None:
        latency_ms = (_time.monotonic() - t0) * 1000.0
        return RecognizeResponse(
            identities=[],
            strategy=body.strategy,
            latency_ms=round(latency_ms, 2),
        )

    emp, sim = await crud.find_nearest_employee(session, embedding, threshold=threshold)
    latency_ms = (_time.monotonic() - t0) * 1000.0
    identities: list[RecognizedIdentity] = []
    if emp is not None:
        identities.append(
            RecognizedIdentity(
                employee_id=emp.employee_id,
                name=emp.name,
                similarity=float(sim),
            )
        )
    else:
        identities.append(RecognizedIdentity(similarity=float(sim)))
    return RecognizeResponse(
        identities=identities,
        strategy=body.strategy,
        latency_ms=round(latency_ms, 2),
    )


__all__ = ["ws_router", "rest_router"]
