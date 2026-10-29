# Architecture

> **See also**: [README.md](../README.md) · [API.md](API.md) ·
> [DEPLOYMENT.md](DEPLOYMENT.md) · [DEVELOPMENT.md](DEVELOPMENT.md) ·
> [WEBSOCKET_PROTOCOL.md](WEBSOCKET_PROTOCOL.md) · [SECURITY.md](SECURITY.md)

This document is the authoritative technical reference for the Facial
Attendance & Motion Tracking System. It covers the system diagram,
component breakdown, data flow, WebSocket protocol summary, all three
strategy modes with sequence diagrams, anti-spoofing deep-dive, and
scaling considerations.

## Table of contents

- [System diagram](#system-diagram)
- [Component breakdown](#component-breakdown)
- [Data flow diagram](#data-flow-diagram)
- [WebSocket protocol summary](#websocket-protocol-summary)
- [Strategy modes](#strategy-modes)
- [Anti-spoofing deep-dive](#anti-spoofing-deep-dive)
- [Scaling considerations](#scaling-considerations)

## System diagram

```
                           ┌──────────────────────────────────────┐
                           │           Browser dashboard           │
                           │       frontend/index.html             │
                           │                                       │
                           │   ┌──────────────┐ ┌──────────────┐  │
                           │   │ Input toggle │ │Strategy toggle│  │
                           │   │ upload|live  │ │ det|trk|both │  │
                           │   └──────┬───────┘ └──────┬───────┘  │
                           └──────────┼─────────────────┼──────────┘
                                      │  WebSocket /ws/stream (JSON + binary)
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          FastAPI app (app/)                            │
│                                                                        │
│  ┌───────────────────── routes/ ──────────────┐  ┌── services/ ───┐   │
│  │ health.py    /health                       │  │ StreamManager   │  │
│  │              /readyz                       │  │ EnrollmentSvc   │  │
│  │ employees.py /api/employees  CRUD          │  │ AttendanceLogger│  │
│  │ attendance.py/api/attendance list+export   │  └─────────────────┘  │
│  │ stream.py    /ws/stream  (WebSocket)       │                       │
│  └────────────────────────────────────────────┘                       │
│                                                                        │
│  ┌─────────────────── core/ AttendancePipeline ────────────────────┐  │
│  │  FaceDetector ─▶ FaceAligner ─▶ MotionTracker ─▶ LivenessVerifier│  │
│  │                                          │                       │  │
│  │                                          ▼                       │  │
│  │                              FeatureExtractor (ArcFace)           │  │
│  │                                          │                       │  │
│  │                                          ▼                       │  │
│  │                              find_nearest_employee (pgvector)     │  │
│  │                                          │                       │  │
│  │                                          ▼                       │  │
│  │                              AttendanceLogger (cooldown)          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└───────┬────────────────────────────────────────────────┬───────────────┘
        │                                                │
        ▼                                                ▼
┌─────────────────────┐                       ┌────────────────────────┐
│  PostgreSQL 15      │                       │   ONNX Runtime         │
│  + pgvector         │                       │   CPU or CUDA          │
│  - employees        │                       │   RetinaFace / YOLO26  │
│  - attendance_log   │                       │   ArcFace recognizer   │
│  - alembic_history  │                       └────────────────────────┘
└─────────────────────┘
```

## Component breakdown

### CV pipeline — `app/core/`

| Module                | Responsibility                                              | Inputs → Outputs                                  |
|-----------------------|-------------------------------------------------------------|---------------------------------------------------|
| `detector.py`         | RetinaFace face detection + 5-pt landmarks                  | `np.ndarray HxWx3` → `List[Detection(bbox, landmarks, score)]` |
| `detector_yolo26.py`  | Ultralytics YOLO26-face alternative detector (NMS-free)     | Same interface as `detector.py` — swap via `DETECTOR_BACKEND=yolo26` |
| `aligner.py`          | Affine warp to 112×112 canonical crop                       | `frame + landmarks` → `np.ndarray 112x112x3`      |
| `extractor.py`        | ArcFace 512-d embedding, L2-normalised                      | `112x112x3` → `np.ndarray (512,) float32`         |
| `tracker.py`          | ByteTrack-style tracking with cooldown-based ID reset       | `List[Detection]` → `List[Track(track_id, bbox, landmarks)]` |
| `liveness.py`         | Landmark heuristics, or optional MiniFASNet PAD gate        | `frame + bbox` → `{is_real, score, pad_label, pad_scores}`            |
| `pipeline.py`         | Orchestrates all of the above per `strategy`                | `frame + strategy` → `FrameResult`                |
| `utils.py`            | `cosine_similarity`, `compute_ear`, `euler_from_landmarks`  | pure numpy helpers                                |

### API — `app/api/`

| Route file      | Endpoint(s)                                  | Notes                                       |
|-----------------|----------------------------------------------|---------------------------------------------|
| `health.py`     | `GET /health`, `GET /readyz`                 | Liveness + readiness probes                 |
| `employees.py`  | `POST/GET/DELETE /api/employees[/{id}]`      | Multipart image upload on POST              |
| `attendance.py` | `GET /api/attendance`, `/api/attendance/export` | CSV / JSON export via `?format=`         |
| `stream.py`     | `WS /ws/stream`                              | Bidirectional, see WEBSOCKET_PROTOCOL.md    |

### DB — `app/db/`

- **ORM**: SQLAlchemy 2.0 async (`asyncpg` driver).
- **Migrations**: Alembic. Apply with `alembic upgrade head`.
- **Models**:
  - `Employee` (`id`, `employee_id`, `name`, `department`, `embedding (vector(512))`, `created_at`)
  - `AttendanceLog` (`id`, `employee_id`, `timestamp`, `strategy`, `score`, `metadata`)
- **CRUD** (`crud.py`): `create_employee`, `get_employee_by_employee_id`,
  `list_employees`, `delete_employee`, `find_nearest_employee` (pgvector
  cosine), `log_attendance`, `recently_logged` (cooldown check),
  `list_attendance` (filtered + paginated).

### Frontend — `frontend/`

A single-page vanilla-JS dashboard with **no build step**:

- `frontend/index.html` — shell with the two toggle controls + video preview.
- `frontend/css/` — minimal styling.
- `frontend/js/` — WebSocket client, camera/upload handling, Canvas
  frame-grabbing, attendance feed rendering.

The two toggles are described in [`README.md#-two-dashboard-toggles`](../README.md#-two-dashboard-toggles).

### Deploy — `deploy/`

- `deploy/docker/` — Dockerfile.cpu, Dockerfile.gpu, entrypoint.sh.
- `deploy/k8s/` — Plain Kubernetes release YAML manifests (namespace, deployments, services, HPA, ingress, network policies, prometheus, grafana, redis). No Helm dependency.
  GPU, TLS, monitoring.

## Data flow diagram

```
[Camera / Video file]
        │
        │ (browser Canvas → JPEG bytes)
        ▼
┌──────────────────┐     config msg        ┌──────────────────┐
│  StreamManager    │◀──────────────────── │  Dashboard JS    │
│  (per-conn state) │                       │  (toggle UI)     │
└────────┬──────────┘                       └──────────────────┘
         │ frame + strategy
         ▼
┌──────────────────────────────────────────────────┐
│ AttendancePipeline.process_frame(frame, strategy)│
└────────┬─────────────────────────────────────────┘
         │ detections
         ▼
   [FaceDetector] ── [FaceAligner] ── [MotionTracker] ── [LivenessVerifier]
                                                              │
                                                              │ (track birth or strategy=detection_only)
                                                              ▼
                                                   [FeatureExtractor]
                                                              │ embedding (512,)
                                                              ▼
                                          [find_nearest_employee (pgvector)]
                                                              │ match? + cooldown OK?
                                                              ▼
                                                  [AttendanceLogger.log]
                                                              │
                                                              ▼
                                                  Postgres attendance_log
                                                              │
                                                              │ (async event)
                                                              ▼
                                          StreamManager broadcasts {type: attendance, ...}
                                                              │
                                                              ▼
                                                       Dashboard feed
```

## WebSocket protocol summary

The full spec lives in [`WEBSOCKET_PROTOCOL.md`](WEBSOCKET_PROTOCOL.md).
Summary:

**Client → Server**

| Message                                            | Purpose                                  |
|----------------------------------------------------|------------------------------------------|
| `{"type": "config", "strategy": "both"}`           | Hot-swap pipeline strategy mid-stream    |
| `{"type": "source", "mode": "upload"}`             | Notify backend of input source change    |
| `{"type": "ping"}`                                 | App-level keepalive                      |
| Binary JPEG/PNG frame                              | A video frame to process                 |

**Server → Client**

| Message                                            | Purpose                                  |
|----------------------------------------------------|------------------------------------------|
| `{"type": "frame", "image": "<base64>", "strategy": "both", "detections": [...]}` | Annotated processed frame |
| `{"type": "attendance", "employee_id": "...", "timestamp": "...", "score": 0.83}`  | A logged attendance event  |
| `{"type": "ack", "for": "config", "strategy": "both"}`                              | Acknowledgement of control msg |
| `{"type": "error", "message": "...", "code": "..."}`                                | Error notification           |
| `{"type": "pong"}`                                                                  | Keepalive response           |

## Strategy modes

### `strategy="both"` (default) — full pipeline

```
 Browser                StreamManager              Pipeline                  DB
   │   binary frame         │                        │                      │
   ├───────────────────────▶│                        │                      │
   │                        │  process_frame(s=both) │                      │
   │                        ├───────────────────────▶│                      │
   │                        │                        │ detect → align       │
   │                        │                        │ track (existing IDs) │
   │                        │                        │ liveness             │
   │                        │                        │─ if track birth ─────▶│ extract → match → log
   │                        │                        │─ else (alive) ───────│ skip extract
   │                        │                        │                      │
   │                        │◀───────────────────────┤ FrameResult          │
   │                        │  emit {type: frame}    │                      │
   │◀───────────────────────┤                        │                      │
   │                        │  emit {type: attendance} (if logged)          │
   │◀───────────────────────┤                                                │
```

**When to use**: production. Best accuracy with reasonable cost — the
512-d ArcFace embedding is computed only once per track birth (and again
after the cooldown expires), not every frame.

### `strategy="detection_only"` — Strategy 1 with per-frame recognition

```
 Browser            StreamManager           Pipeline                    DB
   │  frame              │                      │                       │
   ├────────────────────▶│                      │                       │
   │                     │ process_frame(s=det) │                       │
   │                     ├─────────────────────▶│                       │
   │                     │                      │ detect → track IDs    │
   │                     │                      │ align → extract       │
   │                     │                      │ (every frame) → match│
   │                     │                      │ liveness → log if live│
   │                     │                      ├──────────────────────▶│
   │                     │                      │                       │
   │                     │◀─────────────────────┤ FrameResult           │
   │◀────────────────────┤ emit {type: frame, type: attendance}        │
```

**When to use**: high-accuracy scenarios with few faces per frame (e.g.
single-door check-in). Detection, ArcFace extraction, and matching run for
every processed frame. Temporary track IDs retain liveness history; attendance
is logged only when a matched identity passes liveness. This is more expensive
than `both` because it does not reuse cached embeddings.

### `strategy="tracking_only"` — Strategy 2 standalone

```
 Browser            StreamManager           Pipeline                    DB
   │  frame              │                      │                       │
   ├────────────────────▶│                      │                       │
   │                     │ process_frame(s=trk) │                       │
   │                     ├─────────────────────▶│                       │
   │                     │                      │ detect → track        │
   │                     │                      │ liveness              │
   │                     │                      │ (NO extract, NO match)│
   │                     │                      │ (NO DB write)         │
   │                     │◀─────────────────────┤ FrameResult           │
   │◀────────────────────┤ emit {type: frame}   │                       │
   │                     │ (no attendance events)                       │
```

**When to use**: pure motion / crowd-flow analytics where identity is
irrelevant. Cheapest mode — no embedding extraction, no DB writes.

### Strategy hot-swap

A `{"type": "config", "strategy": "..."}` message takes effect on the
**next** frame processed by the StreamManager. The current frame
finishes with the old strategy; no reconnect is needed. Tracker state is preserved across switches; Strategy 1 also uses track IDs for
its liveness history while continuing to re-extract embeddings every frame.

## Anti-spoofing deep-dive

The default `LivenessVerifier` uses landmark-derived blink and motion
heuristics. Five-point ArcFace landmarks contain eye centers rather than
eyelid contours, so the current blink estimate is only a weak proxy; Euler
angles and motion are also affected by camera movement. Do not treat this
default heuristic as dependable protection against printed photos or screen
replays.

An optional **MiniFASNet PAD** model is available for evaluation. It is
disabled by default. Enable it only after downloading the pinned model files:

```powershell
python scripts\download_pad_models.py
```

Then configure `PAD_ENABLED=true` and `PAD_MODEL_DIR=models/pad`. The two
80×80 ONNX models classify a face crop at two context scales (2.7× and 4×).
Their softmax probabilities are averaged; class 1 is treated as bona fide,
while classes 0 and 2 are treated as spoof. The fused bona-fide score must
also meet `PAD_THRESHOLD`. The application uses the detector's face bounding
box to create the model crops. When PAD is enabled, this model result controls
the attendance liveness gate; Euler/blink heuristics do not override it.

If PAD is enabled but a model is missing, invalid, or inference fails, the
frame fails closed and cannot produce attendance. The model sessions are
warmed during application startup. Without PAD enabled, the application
continues using the existing heuristic behavior.

The pretrained models are from [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
and an Apache-2.0 ONNX conversion pinned in
[`scripts/download_pad_models.py`](../scripts/download_pad_models.py). This
is a practical baseline, not a security guarantee: performance depends on
camera, lighting, face crop, and attack medium. Evaluate with separate,
representative live, printed-photo, and screen-replay samples from the
deployment camera. The notebook
[`notebooks/pad_minifasnet_evaluation.ipynb`](../notebooks/pad_minifasnet_evaluation.ipynb)
reports false accepts, false rejects, balanced accuracy, and latency. It does
not certify the system or select a production threshold for you.

## Scaling considerations

### Vertical

- **CPU**: `buffalo_s` model pack runs at ~18 FPS on a modern i7 with
  `strategy="both"`. Switch to `buffalo_l` only if you have ≥ 8 cores
  and need the extra recall.
- **GPU**: a single NVIDIA A10 handles 60+ FPS with `strategy="both"`.
  Pin `INFERENCE_DEVICE=cuda` and use the GPU Dockerfile.

### Horizontal

- The API is **stateless** — scale horizontally behind a load balancer.
- The WebSocket `/ws/stream` is **stateful per connection** — use sticky
  sessions or a dedicated StreamManager pool with consistent hashing on
  `employee_id`.
- Postgres: pgvector HNSW index keeps nearest-neighbour at <5 ms even at
  1M employees. For >10M, shard by `department` or use a dedicated
  vector DB (Milvus / Qdrant) and replicate.

### Backpressure

- `MAX_STREAM_FPS` (default 30) caps the server-side processing rate.
  Frames arriving faster are dropped at the WebSocket layer with a
  `{type: error, code: "rate_limited"}` notification.
- The `AttendanceLogger` is fire-and-forget; it pushes to a queue and a
  background task batches DB writes to avoid contention.

### Multi-tenancy

- The schema has no built-in `tenant_id` column yet. For multi-tenant
  deployments, add `tenant_id` to `Employee` and `AttendanceLog` and
  filter every CRUD call. The pgvector index should be partitioned by
  `tenant_id` to keep per-tenant search fast.
