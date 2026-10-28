# Facial Attendance & Motion Tracking System

[![CI](https://github.com/your-org/facial-attendance-system/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/facial-attendance-system/actions/workflows/ci.yml)
[![Docker](https://github.com/your-org/facial-attendance-system/actions/workflows/docker.yml/badge.svg)](https://github.com/your-org/facial-attendance-system/actions/workflows/docker.yml)
[![Security](https://github.com/your-org/facial-attendance-system/actions/workflows/security.yml/badge.svg)](https://github.com/your-org/facial-attendance-system/actions/workflows/security.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

A production-grade, GPU-friendly facial attendance and motion tracking
system. Built around **InsightFace** (SCRFD detector + ArcFace
recognizer), **ByteTrack** motion tracking, an **anti-spoofing liveness
verifier**, and a real-time **FastAPI + WebSocket** backend with a
browser-based dashboard.

> **Two live dashboard toggles** (see [§ Toggles](#-two-dashboard-toggles))
> let the operator switch — without restarting the stream — between
> *upload a video* vs. *live camera* and between the three pipeline
> strategies (Detection-only / Tracking-only / Both combined).

---

## Table of contents

- [✨ Features](#-features)
- [🧱 Architecture overview](#-architecture-overview)
- [🔀 The two core strategies](#-the-two-core-strategies)
- [🎛 Two dashboard toggles](#-two-dashboard-toggles)
- [🚀 Quick start](#-quick-start)
- [📦 Prerequisites](#-prerequisites)
- [⚙ Configuration](#-configuration)
- [📡 API reference](#-api-reference)
- [🛰 Deployment](#-deployment)
- [🛠 Development setup](#-development-setup)
- [🧪 Tests](#-tests)
- [📊 Performance benchmarks](#-performance-benchmarks)
- [🩺 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

- **Face detection** with SCRFD (InsightFace, 5-point landmarks, 640×640 input) — or **YOLO26-face** selectable at runtime via `DETECTOR_BACKEND`.
- **Face alignment** to a canonical 112×112 crop for stable embeddings.
- **512-d ArcFace embeddings** L2-normalised; cosine similarity match
  using pgvector when enabled, or a byte-backed Python scan for local
  PostgreSQL installations without the server extension.
- **ByteTrack motion tracking** — persistent IDs across frames, no
  re-embedding needed while a track is alive.
- **Anti-spoofing liveness** — Eye-Aspect-Ratio blink detection +
  head-pose estimation; rejects static photos and screen-replay attacks.
- **Cooldown duplicate-suppression** — the same identity is logged at
  most once per `ATTENDANCE_COOLDOWN_SECONDS` (default 30s).
- **WebSocket streaming** — push JPEG frames, get back annotated frames
  + `attendance` events in real time. Strategy and source toggles
  hot-swappable mid-stream.
- **REST API** for employee enrollment, attendance queries, CSV export,
  health checks.
- **Dashboard** (`frontend/index.html`) — single-page, no build step;
  toggle between **upload video** and **live camera**, and between
  **detection-only / tracking-only / both**.
- **Deploy** — docker-compose (CPU & GPU), plain Kubernetes release YAML manifests (no Helm)
  chart, Prometheus + Grafana monitoring.

---

## 🧱 Architecture overview

```
                            ┌────────────────────────────────────────┐
                            │             Browser dashboard          │
                            │   (frontend/index.html, vanilla JS)    │
                            │                                        │
                            │   ┌──────────────┐  ┌──────────────┐  │
                            │   │ Input toggle │  │Strategy toggle│ │
                            │   │ upload|live  │  │ det|trk|both │  │
                            │   └──────┬───────┘  └──────┬───────┘  │
                            └──────────┼─────────────────┼──────────┘
                                       │  WebSocket /ws/stream
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         FastAPI app (app/)                           │
│                                                                      │
│   routes/        services/             core/                db/      │
│   ├ health.py    ├ StreamManager      ├ FaceDetector       ├ models │
│   ├ employees.py ├ EnrollmentService  ├ FaceAligner        ├ crud   │
│   ├ attendance.py├ AttendanceLogger   ├ FeatureExtractor   └ alembic
│   └ stream.py    └ ...                ├ MotionTracker              │
│                                       ├ LivenessVerifier           │
│                                       └ AttendancePipeline         │
└───────┬──────────────────────────────────────┬─────────────────────┘
        │                                      │
        ▼                                      ▼
┌─────────────────────┐               ┌────────────────────────┐
│  PostgreSQL +       │               │   ONNX Runtime          │
│  pgvector           │               │   (CPU or CUDA)         │
│  - employees        │               │     SCRFD / ArcFace     │
│  - attendance_logs  │               │   or YOLO26-face        │
│  - attendance_log   │               └────────────────────────┘
│  - alembic_history  │
└─────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the deep dive.

---

## 🔀 The two core strategies

The pipeline (`app.core.pipeline.AttendancePipeline.process_frame`)
accepts a `strategy` argument with three modes:

| Strategy           | Detect | Align | Track | Liveness | Extract | DB match | Log | Use case                                  |
|--------------------|:-----:|:-----:|:-----:|:--------:|:-------:|:--------:|:---:|-------------------------------------------|
| `detection_only`   |   ✅  |   ✅  |   ❌  |    ❌    |   ✅    |    ✅    | ✅  | Highest accuracy, expensive (re-detect each frame) |
| `tracking_only`    |   ✅  |   ⚠️ |   ✅  |    ✅    |   ❌    |    ❌    | ❌  | Cheap motion tracking, no identity        |
| `both` (default)   |   ✅  |   ✅  |   ✅  |    ✅    |   ✅*   |    ✅*   | ✅  | Best of both — extract only on track birth/cooldown |

\* In `both` mode the extractor runs for a new track and re-extracts every
30 processed frames; it does not wait for the attendance cooldown to expire.
`detection_only` can log a matched identity without running liveness; its
attendance records leave the liveness result unknown (`null`). It does not
provide spoof protection, so use `both` for normal attendance operation.

```
Strategy "detection_only":
   frame ─▶ detect ─▶ align ─▶ extract ─▶ match DB ─▶ log

Strategy "tracking_only":
   frame ─▶ detect ─▶ track ─▶ liveness ─▶ (no extract, no match)

Strategy "both":
   frame ─▶ detect ─▶ track ─▶ liveness ─▶┬─▶ (track alive?)  ─▶ skip extract
                                          └─▶ (track birth?)  ─▶ extract ─▶ match ─▶ log
```

---

## 🎛 Two dashboard toggles

The dashboard (`frontend/index.html`) exposes **two independent toggles**
that are live-switchable without restarting the stream.

### 1. Input source toggle

```
 ┌───────────────────────────────┐
 │ Input source                  │
 │  ( ) Upload video             │
 │  (•) Live camera              │
 └───────────────────────────────┘
```

- **Upload video** — user picks a local file (`<input type=file
  accept="video/*">`); frames are decoded client-side with
  `<video>` + Canvas and pushed to the WebSocket as binary JPEGs.
- **Live camera** — `navigator.mediaDevices.getUserMedia()` opens the
  webcam; same Canvas → JPEG → WebSocket pipeline.

Both modes send a `{"type": "source", "mode": "upload" | "camera"}`
control message so the backend can apply source-specific policies
(e.g. lower FPS for uploads).

### 2. Strategy mode toggle

```
 ┌───────────────────────────────┐
 │ Strategy mode                 │
 │  (•) Both combined            │
 │  ( ) Detection only           │
 │  ( ) Tracking only            │
 └───────────────────────────────┘
```

Changing this radio sends a `{"type": "config", "strategy":
"detection_only" | "tracking_only" | "both"}` control message over the
same WebSocket. The backend's `StreamManager` forwards the new strategy
to `AttendancePipeline.process_frame(frame, strategy=...)`. The next
frame is processed with the new strategy — no reconnect required.

> See [`docs/WEBSOCKET_PROTOCOL.md`](docs/WEBSOCKET_PROTOCOL.md) for the
> full message spec, and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
> for sequence diagrams of all three strategies.

---

## 🚀 Quick start

```bash
# 1. Clone
git clone https://github.com/your-org/facial-attendance-system.git
cd facial-attendance-system

# 2. Configure
cp configs/.env.example configs/.env
# edit configs/.env — at minimum set POSTGRES_PASSWORD and JWT_SECRET

# 3. Launch (CPU variant by default)
docker compose -f deploy/docker/docker-compose.yml up --build

# 4. Open the dashboard
open http://localhost:8080

# 5. Enroll an employee via REST
curl -F employee_id=EMP001 -F name=Alice \
     -F image=@/path/to/face.jpg \
     http://localhost:8000/api/employees
```

For the **GPU variant** use
`docker compose -f deploy/docker/docker-compose.gpu.yml up`.

### Choosing a detector backend (SCRFD vs YOLO26)

The system ships with **two interchangeable face detectors**. Set
`DETECTOR_BACKEND` in your `.env` to choose:

| `DETECTOR_BACKEND` | Detector | Weights | Pros | Cons |
|---|---|---|---|---|
| `yolo26` (default) | Ultralytics YOLO26-face small | `yolo26s-face.pt` (~21 MB) | Selected based on the benchmarked detection results | Larger/slower than nano |
| `insightface` | InsightFace SCRFD-10g | `buffalo_l` pack (`det_10g.onnx`) | Alternative detector; already bundled with ArcFace model pack | Larger model pack |

The detector and recognizer are separate stages. `buffalo_l` supplies the ArcFace recognizer in either mode; selecting SCRFD uses its `det_10g.onnx` detector as well. Switching YOLO26 and SCRFD does not change the recognizer or its embeddings. Both detectors emit the same `DetectionResult` shape (bbox + 5 landmarks + score), so the rest of the pipeline is unchanged.

At application startup, the selected detector and ArcFace recognizer load and run a dummy inference. This moves model/runtime initialization off the first camera frame, but increases startup time; startup fails explicitly if either required model cannot warm up. The configured `MAX_STREAM_FPS=30` is an upper bound, not a throughput guarantee—actual FPS depends on detector inference speed, device, and frame size.

**Download YOLO26-s and the `buffalo_l` recognizer pack** before starting a fresh setup:

```bash
# Fetch YOLO26-s and the buffalo_l pack (which includes SCRFD-10g + ArcFace):
python scripts/download_models.py --yolo26 --pack buffalo_l

# Use the benchmarked YOLO26-s detector:
DETECTOR_BACKEND=yolo26
DETECTOR_MODEL=yolo26s-face.pt
MODEL_NAME=buffalo_l

# Or switch detection to SCRFD while retaining the same recognizer:
# DETECTOR_BACKEND=insightface
# MODEL_NAME=buffalo_l
docker compose up --build
```

The same config works in Kubernetes — set the env vars in
`deploy/k8s/configmap.yaml` (or `secret.yaml` if you prefer) and
re-apply: `kubectl apply -f deploy/k8s/configmap.yaml`.

---

## 📦 Prerequisites

| Component   | Minimum           | Notes                                           |
|-------------|-------------------|-------------------------------------------------|
| Docker      | 24.0              | Docker Compose v2 included                      |
| NVIDIA GPU  | optional          | Required for the GPU image; CUDA 12.x driver    |
| RAM         | 4 GB              | 8 GB+ recommended for production                |
| Disk        | 5 GB              | Models ~500 MB, Postgres ~1 GB, logs variable   |
| Browser     | Chrome 100+/Edge  | Webcam requires HTTPS or `localhost` origin     |

---

## ⚙ Configuration

All runtime config is driven by environment variables (see
`configs/.env.example`). Key variables:

| Variable                       | Default        | Description                                       |
|--------------------------------|----------------|---------------------------------------------------|
| `APP_HOST`                     | `0.0.0.0`      | Bind address                                      |
| `APP_PORT`                     | `8000`         | HTTP port                                         |
| `POSTGRES_DSN`                 | (required)     | `postgresql+asyncpg://user:pass@host:5432/db`     |
| `POSTGRES_PASSWORD`            | (required)     | Postgres password (used by compose healthchecks)  |
| `JWT_SECRET`                   | (required)     | HS256 secret for admin endpoints                  |
| `INSIGHTFACE_MODEL_PACK`       | `buffalo_s`    | `buffalo_l` / `buffalo_m` / `buffalo_s` / `antelopev2` |
| `INFERENCE_DEVICE`             | `cpu`          | `cpu` or `cuda`                                   |
| `ATTENDANCE_COOLDOWN_SECONDS`  | `30`           | Min seconds between logs of the same identity     |
| `MATCH_THRESHOLD`              | `0.45`         | Cosine distance cutoff (lower = stricter)         |
| `LIVENESS_ENABLED`             | `true`         | Toggle liveness gate in `both` strategy           |
| `LIVENESS_EAR_THRESHOLD`       | `0.20`         | Eye-Aspect-Ratio blink threshold                  |
| `LIVENESS_MIN_BLINKS`          | `1`            | Min blinks in the rolling window to pass          |
| `LIVENESS_WINDOW_SECONDS`      | `5`            | Rolling window for blink counting                 |
| `MAX_STREAM_FPS`               | `30`           | Server-side rate cap on the WebSocket             |
| `CPU_CORES`                    | `0`            | Logical CPU count override; `0` uses `os.cpu_count()` |
| `ARCFACE_THREADING`             | `auto`         | Per-face ArcFace threading (`auto`, `true`, `false`) |
| `MAX_ARCFACE_THREADS`           | `4`            | Maximum concurrent ArcFace calls                   |
| `MAX_FRAME_WIDTH`              | `1280`         | Downscale frames wider than this before inference |
| `LOG_LEVEL`                    | `INFO`         | `DEBUG` / `INFO` / `WARNING` / `ERROR`            |
| `CORS_ORIGINS`                 | `*`            | Comma-separated allowed origins                   |

ArcFace threading is used only when at least four cores are available and a
frame has more than two uncached face tracks. The track embedding/identity
cache remains enabled and re-verifies on its existing 30-frame interval.
The WebSocket pipeline uses two bounded frame queues and allows up to two
frames in flight so YOLO detection can overlap downstream processing.

---

## 📡 API reference

Full OpenAPI spec is served at `/docs` (Swagger UI) and `/redoc` once
the server is running. To export a static copy:

```bash
curl http://localhost:8000/openapi.json > docs/openapi.json
redoc-cli bundle docs/openapi.json --output docs/api.html
```

Key endpoints:

| Method | Path                              | Description                                            |
|--------|-----------------------------------|--------------------------------------------------------|
| GET    | `/health`                         | Liveness probe — always 200.                           |
| GET    | `/readyz`                         | Readiness probe — 200 only if DB + models loaded.      |
| POST   | `/api/employees`                  | Enroll a new employee (multipart image upload).       |
| GET    | `/api/employees`                  | List employees (paginated).                            |
| GET    | `/api/employees/{id}`             | Get a single employee.                                 |
| DELETE | `/api/employees/{id}`             | Delete an employee (and their embedding).              |
| GET    | `/api/attendance`                 | List attendance logs; supports `?date=` `?employee_id=` `?limit=` `?offset=` |
| GET    | `/api/attendance/export?format=csv` | Download CSV / JSON dump of attendance logs.        |
| WS     | `/ws/stream`                      | Bidirectional stream — see [WebSocket protocol](docs/WEBSOCKET_PROTOCOL.md). |

> Full reference: [`docs/API.md`](docs/API.md)

---

## 🛰 Deployment

Three supported deployment targets — pick what fits your scale:

| Target              | When to use                              | Docs                                              |
|---------------------|------------------------------------------|---------------------------------------------------|
| docker-compose      | Single-node dev / small office           | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#docker-compose) |
| docker-compose.prod | Single-node production (TLS, persistent) | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#production)     |
| Kubernetes (plain YAML) | Multi-node, autoscaling, GPU nodes       | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#kubernetes)     |

Quick GPU node snippet (Kubernetes):

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
nodeSelector:
  accelerator: nvidia
```

Monitoring (Prometheus + Grafana) and TLS termination are covered in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#monitoring).

---

## 🛠 Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/cpu.txt -r requirements/dev.txt
pre-commit install

# Run DB migrations
alembic upgrade head

# Download model pack (CPU)
python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])"

# Run the API
uvicorn app.main:app --reload --port 8000

# In another shell, run the worker (if separate)
python -m app.worker
```

For a CUDA 12.1 environment, install `requirements/gpu.txt` instead of
`requirements/cpu.txt`. Do not install both ONNX Runtime variants in one
environment. GPU inference also requires `GPU_ENABLED=true` (or
`DETECTOR_DEVICE=cuda`) and an NVIDIA GPU visible to the process; confirm
the selected device at `/readyz`.

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the full guide
(including debug tips, model downloads, and how to add a new strategy).

---

## 🧪 Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=app --cov-report=html

# Just unit tests (no DB, no GPU)
pytest tests/unit -v

# Just integration tests (needs SQLite; skips pgvector tests)
pytest tests/integration -v
```

- Unit tests use mocks for every heavy ML component — no GPU, no model
  files, no Postgres needed.
- Unit tests use in-memory SQLite and exercise the byte-backed cosine
  fallback. A PostgreSQL test database is only needed to verify the
  pgvector-accelerated path.
- See [`tests/fixtures/sample_faces/README.md`](tests/fixtures/sample_faces/README.md)
  for the privacy policy on test face images.

---

## 📊 Performance benchmarks

> Numbers below are placeholder targets — replace with measured values
> from `scripts/benchmark.py` (Task 1-d) before each release.

| Hardware                     | Strategy         | FPS (avg) | p95 latency | Memory   |
|------------------------------|------------------|----------:|------------:|---------:|
| Intel i7-12700, CPU          | detection_only   |        8 |     180 ms  |   1.2 GB |
| Intel i7-12700, CPU          | tracking_only    |       28 |      45 ms  |   1.0 GB |
| Intel i7-12700, CPU          | both             |       18 |      85 ms  |   1.4 GB |
| NVIDIA A10, CUDA             | detection_only   |       45 |      28 ms  |   2.1 GB |
| NVIDIA A10, CUDA             | both             |       60 |      22 ms  |   2.4 GB |

Re-run benchmarks with:

```bash
pytest tests/perf/ --benchmark-only   # if pytest-benchmark is installed
# or
python scripts/benchmark.py --strategy both --frames 1000
```

---

## 🩺 Troubleshooting

| Symptom                                              | Likely cause                              | Fix                                                                 |
|------------------------------------------------------|-------------------------------------------|---------------------------------------------------------------------|
| Webcam not showing in dashboard                      | Browser blocked camera, or non-HTTPS      | Use `localhost` or serve over HTTPS; check browser permissions.    |
| `/readyz` returns 503                                | DB unreachable or model not loaded        | Check `POSTGRES_DSN`, `INSIGHTFACE_MODEL_PACK`, `/health` logs.    |
| WebSocket disconnects every 30s                      | Reverse proxy idle timeout                | Set `proxy_read_timeout 600s;` in nginx.                            |
| Attendance not logged despite face in frame          | Cooldown active or match score too low    | Lower `MATCH_THRESHOLD`, or wait `ATTENDANCE_COOLDOWN_SECONDS`.    |
| Liveness always fails (`is_real=False`)              | EAR threshold too strict / face too small | Lower `LIVENESS_EAR_THRESHOLD`; ensure face ≥ 80×80 px.            |
| OOM on GPU                                           | Batch too large                           | Reduce `MAX_FRAME_WIDTH`; pin to `buffalo_s` model pack.           |
| `insightface` import error on Apple Silicon          | onnxruntime-silicon missing               | `pip install onnxruntime-silicon`                                  |

More in [`docs/DEVELOPMENT.md#debugging`](docs/DEVELOPMENT.md#debugging)
and [`docs/SECURITY.md`](docs/SECURITY.md).

---

## 🤝 Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md)
first — it covers branch naming, commit conventions, the PR process,
testing requirements, and our policy on biometric data.

Quick start:

```bash
pre-commit install
git checkout -b feat/my-feature
# ... hack ...
pytest tests/unit -v
git commit -m "feat(core): add new strategy variant"
git push -u origin feat/my-feature
# open PR — the template will guide you
```

---

## 📄 License

MIT — see [`LICENSE`](LICENSE).
