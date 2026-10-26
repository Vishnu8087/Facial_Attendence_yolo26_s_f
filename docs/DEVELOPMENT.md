# Development Guide

> **See also**: [README.md](../README.md) · [ARCHITECTURE.md](ARCHITECTURE.md) ·
> [CONTRIBUTING.md](../CONTRIBUTING.md) · [DEPLOYMENT.md](DEPLOYMENT.md) ·
> [SECURITY.md](SECURITY.md)

This guide walks you through setting up a local dev environment, running
the API and worker, applying DB migrations, downloading model packs,
running the test suite, and common debug tips.

## Table of contents

- [One-time setup](#one-time-setup)
- [Install Python dependencies](#install-python-dependencies)
- [Run DB migrations](#run-db-migrations)
- [Download model packs](#download-model-packs)
- [Run the API](#run-the-api)
- [Run the worker](#run-the-worker)
- [Run the dashboard](#run-the-dashboard)
- [Run tests](#run-tests)
- [Code quality](#code-quality)
- [Debugging](#debugging)
- [Adding a new strategy](#adding-a-new-strategy)
- [Working with the two toggles](#working-with-the-two-toggles)

## One-time setup

```bash
git clone https://github.com/your-org/facial-attendance-system.git
cd facial-attendance-system

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip wheel setuptools
```

Recommended Python versions: **3.10, 3.11, or 3.12** (matches the CI
matrix in `.github/workflows/ci.yml`).

## Install Python dependencies

```bash
pip install -r requirements/cpu.txt
pip install -r requirements/dev.txt
pre-commit install
```

`base.txt` contains shared runtime dependencies. `cpu.txt` adds the CPU
ONNX Runtime build; use `gpu.txt` instead of `cpu.txt` for a CUDA 12.1
machine with a compatible NVIDIA driver. Do not install both runtime
variants in the same environment. `dev.txt` adds test and lint dependencies.
Installing the GPU requirements does not turn GPU inference on by itself:
set `GPU_ENABLED=true` (or explicitly set `DETECTOR_DEVICE=cuda`) and make
the NVIDIA GPU available to the process. Check `/readyz` to confirm the
reported detector device after startup.

### Apple Silicon

`onnxruntime` from PyPI does not include ARM wheels; install the
`onnxruntime-silicon` package instead:

```bash
pip uninstall onnxruntime
pip install onnxruntime-silicon
```

## Run DB migrations

The project-level Docker Compose stack runs Postgres + pgvector. Start the
database and API from the repository root:

```bash
docker compose up -d db api
```

The API container waits for Postgres and applies Alembic migrations before
starting Uvicorn. The development override keeps this startup path enabled,
then runs Uvicorn with hot reload. To apply migrations manually, ensure
`DATABASE_URL` points to the database and run:

```bash
alembic upgrade head
```

The Compose database includes pgvector. For an existing local PostgreSQL
installation without the pgvector server extension, set `USE_PGVECTOR=false`
in `.env` before migrating. Embeddings are then stored as float32 bytes and
matched with a Python cosine scan; this is intended for small local datasets.
Keep `USE_PGVECTOR=true` (the default) for the indexed pgvector search path.

To roll back the latest migration:

```bash
alembic downgrade -1
```

To create a new migration after editing models in `app/db/models/`:

```bash
alembic revision --autogenerate -m "add tenant_id column"
# Edit the generated file under alembic/versions/ to verify.
alembic upgrade head
```

## Download model packs

InsightFace model packs are **not** bundled — they're downloaded on
first use from the InsightFace S3 bucket. To pre-warm the cache (avoids
a multi-second stall on first request):

```bash
python - <<'PY'
from insightface.app import FaceAnalysis
FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"]).prepare(ctx_id=0)
print("buffalo_s ready")
PY
```

Available packs: `buffalo_l`, `buffalo_m`, `buffalo_s`, `buffalo_sc`,
`antelopev2`. Set `INSIGHTFACE_MODEL_PACK` in `configs/.env` to switch.

For an **air-gapped** deployment, pre-download the pack on an
internet-connected machine and mount `~/.insightface/models/` into the
container.

## Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- `--reload` watches `app/` for changes.
- Swagger UI: http://localhost:8000/docs
- ReDoc:        http://localhost:8000/redoc
- Health:       http://localhost:8000/health
- Readiness:    http://localhost:8000/readyz

## Run the worker

The `AttendanceLogger` runs as a background task inside the API process
by default. For high-throughput deployments, run a separate worker that
consumes from a queue (Redis Streams / NATS JetStream):

```bash
python -m app.worker --queue attendance-logs --concurrency 4
```

## Run the dashboard

The dashboard is a static SPA — no build step. Serve it any way you
like:

```bash
# Option A — Python's built-in server
cd frontend && python -m http.server 8080

# Option B — via the FastAPI app (mounts /static → frontend/)
# Already wired in app/main.py; just open:
open http://localhost:8000/

# Option C — via docker-compose (recommended)
docker compose -f deploy/docker/docker-compose.yml up dashboard
```

Open http://localhost:8080 — the WebSocket will auto-connect to
`/ws/stream` on the same host (or to `localhost:8000` if served from the
FastAPI app).

## Run tests

```bash
# All tests with coverage
pytest tests/ -v --cov=app --cov-report=html
open htmlcov/index.html

# Just unit tests (no DB, no GPU, fastest)
pytest tests/unit -v

# Just integration tests (uses in-memory SQLite)
pytest tests/integration -v

# A single test
pytest tests/unit/test_liveness.py::test_blink_detected_in_synthetic_sequence -v

# With pytest-benchmark (perf tests)
pytest tests/perf/ --benchmark-only
```

Tests are **mock-first** — no GPU, no model files, no live Postgres
required. See `tests/conftest.py` for the shared fixtures.

To run integration tests against a real Postgres + pgvector:

```bash
export TEST_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/facial_test
pytest tests/integration -v
```

## Code quality

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Or invoke each tool directly
ruff check app tests
black --check app tests
mypy app --ignore-missing-imports
```

Pre-commit hooks (defined in `.pre-commit-config.yaml`):

- `ruff` lint + format
- `black` (secondary format check)
- `mypy` (best-effort)
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`,
  `check-added-large-files` (max 1MB), `detect-private-key`
- `bandit` (security lint)
- `yamllint` (relaxed)
- `markdownlint`

## Debugging

### Inspect a single frame offline

```python
import cv2
from app.core.pipeline import AttendancePipeline

pipe = AttendancePipeline()  # uses defaults
frame = cv2.imread("tests/fixtures/sample_faces/face_alpha.png")
result = pipe.process_frame(frame, strategy="both")
print(result)
```

### See what the WebSocket is sending

```python
# In a Python REPL with `websockets` installed
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:
        await ws.send(json.dumps({"type": "config", "strategy": "both"}))
        # Send a binary JPEG
        ...
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                print("binary frame:", len(msg), "bytes")
            else:
                print(json.loads(msg))

asyncio.run(main())
```

### Profile the pipeline

```bash
python -m cProfile -o profile.out -m app.cli profile-frame \
    --frame tests/fixtures/sample_faces/face_alpha.png --strategy both
snakeviz profile.out
```

### Postgres issues

```bash
# Connect to the dev DB
docker compose -f deploy/docker/docker-compose.yml exec postgres psql -U facial facial

# Useful queries
SELECT * FROM pg_extension WHERE extname = 'vector';
SELECT COUNT(*) FROM employees;
SELECT employee_id, timestamp, strategy, score FROM attendance_log ORDER BY timestamp DESC LIMIT 10;
```

### Common gotchas

| Symptom                                                  | Fix                                                              |
|----------------------------------------------------------|------------------------------------------------------------------|
| `CUDA error: no kernel image is available`               | Mismatched CUDA version. Use the GPU image matching your driver.|
| `insightface.utils.download FileNotFound`                | Network blocked from InsightFace S3. Pre-download the pack.     |
| `asyncpg.exceptions.UndefinedColumnError: embedding`     | Forgot to `CREATE EXTENSION vector;`. Re-run alembic.           |
| WebSocket disconnects after 60s on nginx                 | Set `proxy_read_timeout 600s;` in nginx.                        |
| `mypy` complains about `app.core.*` not being a package  | Add `__init__.py` to `app/core/` (Task 1-a owns this).          |

## Adding a new strategy

1. Add the strategy name to the `Literal` type accepted by
   `AttendancePipeline.process_frame` (in `app/core/pipeline.py`).
2. Implement the new branch in `process_frame`.
3. Add a unit test in `tests/unit/test_pipeline.py` mirroring the
   existing strategy tests — assert which sub-components are called.
4. Update `docs/ARCHITECTURE.md` with a sequence diagram.
5. Update `docs/WEBSOCKET_PROTOCOL.md` so the frontend knows the new
   value is valid for `{"type": "config", "strategy": "..."}`.
6. Update `frontend/index.html` to add a new radio button.
7. Open a PR — the **Strategy / toggle impact** section in the PR
   template is mandatory.

## Working with the two toggles

The two dashboard toggles are the **input source** (upload / live
camera) and the **strategy mode** (detection_only / tracking_only /
both). Both are documented at:

- [`README.md#-two-dashboard-toggles`](../README.md#-two-dashboard-toggles)
- [`docs/WEBSOCKET_PROTOCOL.md`](WEBSOCKET_PROTOCOL.md) — full message
  spec for the `config` and `source` control messages.
- [`docs/ARCHITECTURE.md#strategy-modes`](ARCHITECTURE.md#strategy-modes)
  — sequence diagrams for each strategy.

When testing locally, you can switch either toggle mid-stream — the
backend picks up the new value on the next frame. No reconnect needed.
