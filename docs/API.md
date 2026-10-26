# API Reference

> **See also**: [README.md](../README.md) · [ARCHITECTURE.md](ARCHITECTURE.md) ·
> [WEBSOCKET_PROTOCOL.md](WEBSOCKET_PROTOCOL.md) · [SECURITY.md](SECURITY.md)

The FastAPI app exposes a self-describing OpenAPI 3.1 schema. This
document gives a stable endpoint reference; for live, always-up-to-date
docs, use the interactive UIs shipped with the server.

## Table of contents

- [Interactive docs](#interactive-docs)
- [Exporting the OpenAPI spec](#exporting-the-openapi-spec)
- [Authentication](#authentication)
- [Base URL & content types](#base-url--content-types)
- [Health](#health)
- [Employees](#employees)
- [Attendance](#attendance)
- [WebSocket](#websocket)
- [Error model](#error-model)
- [Pagination & filtering conventions](#pagination--filtering-conventions)

## Interactive docs

| UI           | URL                    | Notes                                            |
|--------------|------------------------|--------------------------------------------------|
| Swagger UI   | `/docs`                | Interactive — try endpoints in the browser.      |
| ReDoc        | `/redoc`               | Read-only, better for browsing large schemas.    |
| Raw JSON     | `/openapi.json`        | Machine-readable OpenAPI 3.1 schema.             |

## Exporting the OpenAPI spec

To produce a static `openapi.json` and an HTML bundle (e.g. to ship in
release artifacts):

```bash
# 1. Start the server
uvicorn app.main:app --port 8000 &

# 2. Dump the schema
curl -s http://localhost:8000/openapi.json > docs/openapi.json

# 3. (optional) Build a static ReDoc HTML
npx @redocly/cli build-docs docs/openapi.json --output docs/api.html

# 4. (optional) Generate a typed client
npx openapi-typescript-codegen --input docs/openapi.json --output docs/client
```

The `release.yml` workflow uploads `docs/openapi.json` as a release
artifact for every tagged release.

## Authentication

Most read endpoints are public (intended for trusted-network deployments).
Write endpoints (`POST /api/employees`, `DELETE /api/employees/{id}`)
require a `Bearer` JWT signed with `JWT_SECRET`:

```http
POST /api/employees HTTP/1.1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: multipart/form-data; boundary=----xyz
```

Issue a token:

```bash
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "..."}'
# {"access_token": "...", "token_type": "bearer", "expires_in": 3600}
```

WebSocket connections authenticate via a `?token=...` query parameter
(cannot use `Authorization` header on `new WebSocket()`).

## Base URL & content types

- Base URL: `http://<host>:<APP_PORT>` (default `:8000`)
- Request body: `application/json` unless noted (multipart for image uploads)
- Response body: `application/json; charset=utf-8`
- All timestamps are ISO-8601 UTC (`2024-01-15T10:30:00Z`).

## Health

### `GET /health`

Liveness probe. Always returns 200 if the process is up.

**Response 200**
```json
{ "status": "ok", "version": "0.1.0", "uptime_seconds": 1234 }
```

### `GET /readyz`

Readiness probe. Returns 200 only when DB + models are loaded; 503
otherwise.

**Response 200**
```json
{
  "status": "ok",
  "database": { "status": "ok", "latency_ms": 2 },
  "models":   { "status": "ok", "device": "cuda", "pack": "buffalo_s" },
  "cache":    { "status": "skip" }
}
```

**Response 503**
```json
{ "status": "unavailable", "database": { "status": "down", "error": "..." } }
```

## Employees

### `POST /api/employees`

Enroll a new employee. Accepts either a multipart image upload (the
backend extracts the embedding) or a JSON body with a pre-computed
embedding.

**Multipart form**

| Field         | Type   | Required | Notes                          |
|---------------|--------|----------|--------------------------------|
| `employee_id` | string | yes      | Unique business identifier.    |
| `name`        | string | yes      | Display name.                  |
| `department`  | string | no       | Free-form department label.    |
| `image`       | file   | yes*     | JPEG/PNG face crop (≥ 80×80).  |

\* Mutually exclusive with the JSON `embedding_base64` variant.

**JSON body**
```json
{
  "employee_id": "EMP001",
  "name": "Alice Alpha",
  "department": "Engineering",
  "embedding_base64": "<base64-encoded float32[512]>"
}
```

**Response 201**
```json
{
  "id": 42,
  "employee_id": "EMP001",
  "name": "Alice Alpha",
  "department": "Engineering",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Errors**
- `409` — `employee_id` already exists.
- `422` — Missing required fields / invalid image.
- `400` — No face detected in the uploaded image.

### `GET /api/employees`

List employees, paginated.

| Query param | Type | Default | Description              |
|-------------|------|---------|--------------------------|
| `limit`     | int  | 50      | Max 200.                 |
| `offset`    | int  | 0       |                          |
| `department`| str  | —       | Optional filter.         |
| `q`         | str  | —       | Name/employee_id prefix. |

**Response 200**
```json
{
  "items": [ { "id": 42, "employee_id": "EMP001", ... } ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### `GET /api/employees/{id}`

**Response 200** — single employee object (same shape as POST response).
**Response 404** — not found.

### `DELETE /api/employees/{id}`

Soft-deletes the employee and revokes their embedding from the pgvector
index. Attendance logs are **retained** for audit.

**Response 204** — deleted.
**Response 404** — not found.

## Attendance

### `GET /api/attendance`

List attendance logs.

| Query param    | Type   | Default | Description                              |
|----------------|--------|---------|------------------------------------------|
| `employee_id`  | str    | —       | Filter by business `employee_id`.        |
| `date`         | date   | —       | ISO date (`2024-01-15`).                 |
| `date_from`    | date   | —       | Inclusive lower bound.                   |
| `date_to`      | date   | —       | Inclusive upper bound.                   |
| `strategy`     | str    | —       | `detection_only` / `tracking_only` / `both`. |
| `limit`        | int    | 100     | Max 1000.                                |
| `offset`       | int    | 0       |                                          |

**Response 200**
```json
{
  "items": [
    {
      "id": 1234,
      "employee_id": "EMP001",
      "name": "Alice Alpha",
      "timestamp": "2024-01-15T10:30:00Z",
      "strategy": "both",
      "score": 0.83,
      "metadata": { "track_id": 7, "blink_count": 2 }
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

### `GET /api/attendance/export`

Bulk-export attendance logs as CSV or JSON.

| Query param | Type | Default | Description                          |
|-------------|------|---------|--------------------------------------|
| `format`    | str  | `csv`   | `csv` or `json`.                     |
| `date_from` | date | —       | Same filters as GET /api/attendance. |
| `date_to`   | date | —       |                                      |
| `employee_id` | str | —       |                                    |

**Response 200 (CSV)**
```
HTTP/1.1 200 OK
Content-Type: text/csv; charset=utf-8
Content-Disposition: attachment; filename="attendance_2024-01-15.csv"

id,employee_id,name,timestamp,strategy,score
1234,EMP001,Alice Alpha,2024-01-15T10:30:00Z,both,0.83
```

**Response 200 (JSON)** — a JSON array of the same item shape as
`GET /api/attendance`.

## WebSocket

See [`WEBSOCKET_PROTOCOL.md`](WEBSOCKET_PROTOCOL.md) for the full
bidirectional message spec. Endpoint: `WS /ws/stream?token=<jwt>`.

## Error model

All non-2xx responses use a consistent envelope:

```json
{
  "error": {
    "code": "not_found",
    "message": "Employee 99 does not exist.",
    "details": { "resource": "Employee", "id": 99 }
  },
  "request_id": "req_01HM2J3..."
}
```

Common codes: `bad_request`, `unauthorized`, `forbidden`, `not_found`,
`conflict`, `rate_limited`, `internal_error`.

## Pagination & filtering conventions

- All list endpoints accept `limit` + `offset` and return an envelope
  with `items`, `total`, `limit`, `offset`.
- Date filters accept ISO-8601 dates (`2024-01-15`) or full timestamps
  (`2024-01-15T10:30:00Z`).
- String filters are exact-match unless they start with `q_` (prefix).
