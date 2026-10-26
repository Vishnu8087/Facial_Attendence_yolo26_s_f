# WebSocket Protocol — `/ws/stream`

> **See also**: [README.md](../README.md) ·
> [ARCHITECTURE.md#websocket-protocol-summary](ARCHITECTURE.md#websocket-protocol-summary) ·
> [API.md](API.md) · [DEVELOPMENT.md](DEVELOPMENT.md)

The WebSocket endpoint at `/ws/stream` carries the live video stream
between the browser dashboard and the FastAPI backend. This document is
the authoritative protocol spec — every message type is listed with a
JSON example.

- **Protocol version**: `1`
- **Subprotocol**: none (we use raw text frames for JSON, binary frames
  for image data).
- **Encoding**: UTF-8 JSON for text frames; binary JPEG/PNG for image
  frames.
- **Default port**: `8000` (same as the REST API).
- **URL**: `ws(s)://<host>/ws/stream?token=<jwt>`

## Table of contents

- [Connection lifecycle](#connection-lifecycle)
- [Authentication](#authentication)
- [Message types — client → server](#message-types--client--server)
  - [`config`](#config)
  - [`source`](#source)
  - [`ping`](#ping)
  - Binary frame](#binary-frame)
- [Message types — server → client](#message-types--server--client)
  - [`frame`](#frame)
  - [`attendance`](#attendance)
  - [`ack`](#ack)
  - [`pong`](#pong)
  - [`error`](#error)
  - [`welcome`](#welcome)
- [Error codes](#error-codes)
- [Rate limiting & backpressure](#rate-limiting--backpressure)
- [Heartbeat](#heartbeat)
- [Versioning](#versioning)
- [Examples](#examples)

## Connection lifecycle

```
Browser                                Server
  │                                      │
  │ WS Upgrade: GET /ws/stream?token=…   │
  ├─────────────────────────────────────▶│
  │                                      │ verify JWT
  │ 101 Switching Protocols              │
  │◀─────────────────────────────────────┤
  │                                      │
  │ {type: welcome, ...}                 │
  │◀─────────────────────────────────────┤
  │                                      │
  │ {type: config, strategy: both}       │
  ├─────────────────────────────────────▶│
  │ {type: ack, for: config, ...}        │
  │◀─────────────────────────────────────┤
  │                                      │
  │ binary JPEG frame                    │
  ├─────────────────────────────────────▶│
  │                                      │ process_frame(...)
  │ {type: frame, ...}                   │
  │◀─────────────────────────────────────┤
  │ {type: attendance, ...}              │ (only if logged)
  │◀─────────────────────────────────────┤
  │                                      │
  │ {type: ping}            (or WS ping) │
  ├─────────────────────────────────────▶│
  │ {type: pong}                          │
  │◀─────────────────────────────────────┤
  │                                      │
  │ Close (1000)                         │
  ├─────────────────────────────────────▶│
```

## Authentication

Pass a JWT as a query parameter (cannot set the `Authorization` header
on `new WebSocket()`):

```
wss://host/ws/stream?token=eyJhbGciOiJIUzI1NiIs...
```

The server validates the JWT before accepting the upgrade. On failure
the server returns HTTP 401 (the WebSocket handshake fails). On success
the connection is established and a `welcome` message is sent.

For trusted-network deployments where auth is disabled, the `token`
parameter may be omitted.

## Message types — client → server

### `config`

Hot-swap the pipeline strategy **mid-stream**. The change takes effect
on the **next** frame — the in-flight frame finishes with the old
strategy. No reconnect needed.

```json
{
  "type": "config",
  "strategy": "both"
}
```

| Field      | Type   | Required | Allowed values                              |
|------------|--------|----------|---------------------------------------------|
| `type`     | string | yes      | `"config"`                                  |
| `strategy` | string | yes      | `"both"`, `"detection_only"`, `"tracking_only"` |

**Server response**: an [`ack`](#ack) message, then the next `frame`
event carries the new `strategy` value.

**Error**: invalid `strategy` ⇒ [`error`](#error) with code
`invalid_strategy`.

### `source`

Notify the backend of an input-source change. Three modes are supported:

#### `camera` and `upload` modes

Frames flow browser→backend via WebSocket binary blobs. Backend just
acknowledges the mode change.

```json
{ "type": "source", "mode": "camera" }
{ "type": "source", "mode": "upload" }
```

#### `cctv` mode (NEW — server-side RTSP/HTTP stream)

Backend opens the stream via `cv2.VideoCapture(url)` and reads frames
server-side. Frames flow server→pipeline directly (no browser round-trip
for raw video). Annotated frames still sent back to dashboard.

**Connect:**
```json
{
  "type": "source",
  "mode": "cctv",
  "action": "connect",
  "camera": {
    "id": "cam_entrance_1",
    "name": "Main Entrance",
    "url": "rtsp://user:pass@192.168.1.100:554/stream"
  }
}
```

**Disconnect:**
```json
{ "type": "source", "mode": "cctv", "action": "disconnect" }
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | `"source"` |
| `mode` | string | yes | `"camera"`, `"upload"`, or `"cctv"` |
| `action` | string | only for cctv | `"connect"` or `"disconnect"` |
| `camera.id` | string | cctv connect | Unique camera ID — stored in `attendance_logs.camera_id` |
| `camera.name` | string | cctv connect | Human-readable label |
| `camera.url` | string | cctv connect | RTSP/HTTP URL — credentials embedded as `rtsp://user:pass@host/path` |

**Server response**: `{"type":"ack","for":"source","mode":"cctv","action":"connected","camera":{"id":"...","name":"..."}}`

**Error codes**: `invalid_cctv_config` (missing fields), `cctv_connect_failed` (OpenCV couldn't open stream), `cctv_stream_dropped` (stream dropped — user must reconnect).

### `ping`

Application-level keepalive. Useful when the WebSocket library doesn't
expose WS-layer pings. The server replies with [`pong`](#pong).

```json
{ "type": "ping", "t": 1705315200000 }
```

| Field  | Type   | Required | Notes                                |
|--------|--------|----------|--------------------------------------|
| `type` | string | yes      | `"ping"`                             |
| `t`    | number | no       | Client epoch-ms; echoed back in pong |

### Binary frame

A single video frame, JPEG or PNG encoded. The server processes it with
the currently active strategy and replies with a [`frame`](#frame)
event (and optionally an [`attendance`](#attendance) event).

- Max size: **2 MB** (≈ a 1280×720 JPEG at quality 80).
- Max FPS: `MAX_STREAM_FPS` (default 15). Frames arriving faster are
  silently dropped and logged at debug level. The dashboard targets up to
  30 FPS but sends only one frame at a time, waiting for each processed-frame
  response; actual throughput is bounded by server processing and its FPS cap.
- Format detection by magic bytes (`\xFF\xD8\xFF` for JPEG,
  `\x89PNG\r\n\x1a\n` for PNG).

## Message types — server → client

### `frame`

An annotated processed frame, returned for every binary frame received
from the client. The image is JPEG-encoded, base64-wrapped (so the whole
message stays a single text frame).

```json
{
  "type": "frame",
  "frame_id": 42,
  "image": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "strategy": "both",
  "detections": [
    {
      "track_id": 7,
      "bbox": [120, 80, 280, 260],
      "score": 0.96,
      "identity": "EMP001",
      "is_real": true
    }
  ],
  "latency_ms": 38
}
```

| Field         | Type    | Always present | Notes                                |
|---------------|---------|----------------|--------------------------------------|
| `type`        | string  | yes            | `"frame"`                            |
| `frame_id`    | int     | yes            | Monotonic per connection.            |
| `image`       | string  | yes            | Base64-encoded JPEG.                 |
| `strategy`    | string  | yes            | Echoes the active strategy.          |
| `detections`  | array   | yes            | One object per detected face.        |
| `latency_ms`  | number  | yes            | Server-side processing time.         |

`detections[*]`:

| Field       | Type    | Always present | Notes                                  |
|-------------|---------|----------------|----------------------------------------|
| `track_id`  | int     | in `both`/`tracking_only` | Persisted motion ID.        |
| `bbox`      | array   | yes            | `[x1, y1, x2, y2]` in source-pixel coords. |
| `score`     | number  | yes            | Detector confidence 0..1.              |
| `identity`  | string  | no             | `employee_id` if matched, else absent. |
| `is_real`   | bool    | no             | Liveness verdict (only in strategies that run liveness). |

### `attendance`

Emitted **once per logged attendance** (i.e. when the cooldown allows a
new log). Multiple `frame` events between two `attendance` events is
normal.

```json
{
  "type": "attendance",
  "employee_id": "EMP001",
  "name": "Alice Alpha",
  "timestamp": "2024-01-15T10:30:00Z",
  "strategy": "both",
  "score": 0.83,
  "track_id": 7,
  "metadata": { "blink_count": 2, "head_pose": [0.1, -1.2, 0.5] }
}
```

| Field         | Type   | Always present | Notes                                  |
|---------------|--------|----------------|----------------------------------------|
| `type`        | string | yes            | `"attendance"`                         |
| `employee_id` | string | yes            | Business ID of the matched employee.   |
| `name`        | string | yes            | Display name (for the dashboard feed). |
| `timestamp`   | string | yes            | ISO-8601 UTC.                          |
| `strategy`    | string | yes            | Which strategy produced this log.      |
| `score`       | number | yes            | Cosine similarity (0..1, higher=better). |
| `track_id`    | int    | no             | The motion track that triggered this.  |
| `metadata`    | object | no             | Strategy-specific extras.              |

### `ack`

Acknowledges a client control message (`config` or `source`).

```json
{
  "type": "ack",
  "for": "config",
  "strategy": "both"
}
```

| Field      | Type   | Always present | Notes                                |
|------------|--------|----------------|--------------------------------------|
| `type`     | string | yes            | `"ack"`                              |
| `for`      | string | yes            | The message type being acked.        |
| `strategy` | string | when `for=config` | Echoes the new strategy.         |
| `mode`     | string | when `for=source` | Echoes the new source mode.      |

### `pong`

Reply to a client [`ping`](#ping).

```json
{ "type": "pong", "t": 1705315200000 }
```

### `error`

Non-fatal error notification. The connection stays open unless the code
is in the `fatal` category (see [Error codes](#error-codes)).

```json
{
  "type": "error",
  "code": "frame_too_large",
  "message": "Frame exceeds 2097152 bytes",
  "fatal": false
}
```

Frames dropped by the FPS cap do not produce an error event.

| Field     | Type    | Always present | Notes                                  |
|-----------|---------|----------------|----------------------------------------|
| `type`    | string  | yes            | `"error"`                              |
| `code`    | string  | yes            | See [Error codes](#error-codes).       |
| `message` | string  | yes            | Human-readable.                        |
| `fatal`   | boolean | yes            | If true, the server will close after.  |

### `welcome`

First message after the WS handshake. Identifies the server and
protocol version.

```json
{
  "type": "welcome",
  "server": "facial-attendance/0.1.0",
  "protocol": 1,
  "capabilities": ["config", "source", "binary_frame"],
  "defaults": {
    "strategy": "both",
    "source": "camera",
    "max_fps": 15
  }
}
```

## Error codes

| Code                  | Fatal | Meaning                                                          |
|-----------------------|-------|------------------------------------------------------------------|
| `invalid_strategy`    | no    | `config.strategy` not in allowed set.                            |
| `invalid_source`      | no    | `source.mode` not in allowed set.                                |
| `invalid_message`     | no    | JSON parse error or unknown `type`.                              |
| `frame_too_large`     | no    | Binary frame > 2 MB. Dropped.                                    |
| `unsupported_format`  | no    | Binary frame is neither JPEG nor PNG.                            |
| `pipeline_error`      | no    | Exception inside `process_frame`. Frame skipped.                 |
| `db_unavailable`      | no    | Postgres write failed; attendance log dropped.                   |
| `auth_expired`        | yes   | JWT expired mid-connection. Client must reconnect with a new token. |
| `internal_error`      | yes   | Unrecoverable server error. Connection closed (1011).            |

When `fatal: true`, the server sends the `error` message and then closes
the WebSocket with code 1011 (internal error) or 1008 (policy
violation).

## Rate limiting & backpressure

- **Per-connection FPS**: capped at `MAX_STREAM_FPS` (default 15).
  Excess frames are dropped silently and recorded at debug level. The
  dashboard targets up to 30 FPS and allows only one unacknowledged frame
  per connection, so model latency naturally applies backpressure.
- **Per-IP connections**: capped at 10 concurrent WS connections.
- **Per-frame size**: capped at 2 MB.
- **Slow client**: if the client's TCP send buffer fills (i.e. the
  client isn't reading server messages fast enough), the server drops
  the oldest queued `frame` events to keep latency bounded.

## Heartbeat

Two heartbeat mechanisms:

1. **WS-layer ping/pong** — the server sends WS pings every 20s; if no
   pong within 30s the connection is closed.
2. **App-layer `ping`/`pong`** — for clients whose WS library doesn't
   expose WS-layer pings (e.g. some browsers). See [`ping`](#ping).

## Versioning

The protocol is versioned via the `protocol` field in the `welcome`
message. The current version is **1**. Backward-compatible additions
(bumping to 1.1, 1.2) do not require a client upgrade.

Breaking changes (e.g. renaming a field) bump to version 2 and are
negotiated via the `Sec-WebSocket-Protocol` header — clients request
`v1` or `v2`, the server picks the highest mutually supported.

## Examples

### Example 1: switch strategy mid-stream

```jsonc
// Client → Server
{ "type": "config", "strategy": "detection_only" }

// Server → Client
{ "type": "ack", "for": "config", "strategy": "detection_only" }

// (client sends binary JPEG)
// Server → Client
{
  "type": "frame",
  "frame_id": 13,
  "image": "/9j/4AAQ...",
  "strategy": "detection_only",
  "detections": [
    { "bbox": [120, 80, 280, 260], "score": 0.96, "identity": "EMP001" }
  ],
  "latency_ms": 172
}
```

### Example 2: liveness rejection in `both` mode

```jsonc
// Server → Client
{
  "type": "frame",
  "frame_id": 7,
  "image": "/9j/4AAQ...",
  "strategy": "both",
  "detections": [
    { "track_id": 3, "bbox": [120, 80, 280, 260], "score": 0.95, "is_real": false }
  ],
  "latency_ms": 41
}
// Note: no `attendance` event because liveness failed → no log.
```

### Example 3: switch input source

```jsonc
// Client → Server
{ "type": "source", "mode": "upload" }

// Server → Client
{ "type": "ack", "for": "source", "mode": "upload" }
```

### Example 4: rate-limit drop

When a binary frame arrives before the per-connection interval has elapsed,
the server drops it without sending an error. The client avoids this by
waiting for the previous frame response before sending another.

### Example 5: JWT expiry

```jsonc
// Server → Client (mid-stream, after JWT expired)
{
  "type": "error",
  "code": "auth_expired",
  "message": "JWT expired at 2024-01-15T11:30:00Z. Please reconnect with a fresh token.",
  "fatal": true
}
// (server closes with code 1008)
```
