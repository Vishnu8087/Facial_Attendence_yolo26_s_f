# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — YOLO26 detector backend (Oct 2026)

- **New file** `app/core/detector_yolo26.py` — `YoloFaceDetector` class
  wrapping Ultralytics YOLO26 (released January 2026). Uses the
  WIDERFace-pretrained `yolo26n-face.pt` (5 MB) or `yolo26s-face.pt`
  (19 MB) weights from `akanametov/yolo-face`. Implements the same
  `detect(frame) -> List[DetectionResult]` contract as the default
  RetinaFace detector, so the rest of the pipeline is unchanged.
- **Config switch** `DETECTOR_BACKEND` (`insightface` | `yolo26`) in
  `app/config.py` + `configs/config.yaml` + `.env.example`. When set
  to `yolo26`, `EnrollmentService` and `StreamManager` construct a
  `YoloFaceDetector` instead of `FaceDetector`.
- **Config switches** `DETECTOR_MODEL` (`yolo26n-face.pt` |
  `yolo26s-face.pt`) and `DETECTOR_DEVICE` (`cpu` | `cuda` | `cuda:0`).
- **`scripts/download_models.py`** now accepts `--yolo26` and
  `--yolo26-variant` to pre-fetch YOLO26-face weights from
  `akanametov/yolo-face` GitHub releases.
- **`deploy/docker/entrypoint.sh`** pre-fetches the YOLO26-face model
  at container start when `DETECTOR_BACKEND=yolo26`.
- **`requirements/base.txt`** adds `ultralytics==8.3.45` (ships YOLO26).
- **14 new unit tests** in `tests/unit/test_detector_yolo26.py`
  (construction, lifecycle, detect contract, sort/cap/min-size filters,
  error paths, mock ultralytics).
- **Docs updated**: README, ARCHITECTURE, DEPLOYMENT, env example
  explain how to switch detector backends at runtime.

### Why YOLO26?
- NMS-free end-to-end head → predictable latency on dense scenes
- 43% faster CPU ONNX inference vs YOLO11 (Ultralytics benchmark, Jan 2026)
- 5-face-landmark output in ArcFace convention → drop-in replacement
- Native keypoint head (released Jan 2026 with `yolo26-pose` models)

## [0.1.0] - 2024-01-01

### Added
- **Core CV pipeline** (`app.core/`):
  - `FaceDetector` — RetinaFace-based detector with 5-point landmarks.
  - `FaceAligner` — affine alignment to canonical 112×112 crop.
  - `FeatureExtractor` — ArcFace 512-d L2-normalised embeddings.
  - `MotionTracker` — ByteTrack-style tracker with persistent IDs and
    cooldown-based ID reassignment.
  - `LivenessVerifier` — Eye-Aspect-Ratio blink detection + head-pose
    estimation; static-frame rejection.
  - `AttendancePipeline` — orchestrates the above with three
    `strategy` modes: `detection_only`, `tracking_only`, `both`.
- **API + DB** (`app/api/`, `app/db/`):
  - FastAPI app with `/health`, `/readyz`, `/api/employees`,
    `/api/attendance`, `/api/attendance/export`.
  - WebSocket endpoint at `/ws/stream` carrying both `config` and
    `source` control messages plus binary JPEG frames.
  - SQLAlchemy + Alembic models: `Employee`, `AttendanceLog`.
  - pgvector-backed `find_nearest_employee` cosine-similarity search.
- **Frontend dashboard** (`frontend/index.html`):
  - Single-page app, no build step.
  - Two live toggles: **input source** (upload video / live camera) and
    **strategy mode** (detection-only / tracking-only / both).
- **Deployment** (`deploy/`):
  - Multi-stage Dockerfiles (CPU + GPU).
  - docker-compose files for dev and prod.
  - Kubernetes release YAML manifests (no Helm).
- **Tests** (`tests/`):
  - Unit tests for every `app.core` component and `app.db.crud`.
  - Integration tests for every REST endpoint and the WebSocket.
  - End-to-end enroll→stream→attendance test.
  - 100% mock-based — no GPU, no model files, no live DB required.
- **CI/CD** (`.github/workflows/`):
  - `ci.yml` — lint (ruff + black + mypy) + pytest matrix on
    Python 3.10/3.11/3.12 with Codecov upload.
  - `docker.yml` — build & push CPU + GPU images to GHCR with SHA + semver tags.
  - `security.yml` — Trivy image + filesystem scan, pip-audit, dependency review.
  - `release.yml` — GitHub Release with body extracted from this file.
- **Documentation**:
  - Top-level `README.md` with badges, architecture diagram, toggle
    explanations, quick start, troubleshooting.
  - `docs/ARCHITECTURE.md` — system diagram, component breakdown,
    sequence diagrams for each strategy, anti-spoofing deep-dive.
  - `docs/API.md` — endpoint reference + OpenAPI export instructions.
  - `docs/DEPLOYMENT.md` — docker-compose / k8s (plain YAML) / GPU / TLS / monitoring.
  - `docs/DEVELOPMENT.md` — local dev setup, migrations, model download, debug tips.
  - `docs/SECURITY.md` — vulnerability reporting + deployment hardening.
  - `docs/WEBSOCKET_PROTOCOL.md` — full client↔server message spec with JSON examples.
  - `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE` (MIT), `.github/PULL_REQUEST_TEMPLATE.md`.
- **Tooling**:
  - `.pre-commit-config.yaml` — ruff, black, mypy, bandit, yamllint, markdownlint.
  - `.github/dependabot.yml` — pip + docker + github-actions updates.

### Security
- Liveness verifier rejects static photos and screen-replay attacks by
  requiring ≥1 blink within a 5-second rolling window.
- `ATTENDANCE_COOLDOWN_SECONDS` (default 30s) prevents duplicate logs
  of the same identity across consecutive frames.
- No real biometric data is ever committed to the repository — only
  synthetic test fixtures under `tests/fixtures/sample_faces/`.

### Known limitations
- Liveness verifier is heuristic-based (EAR + head pose); a learned
  anti-spoofing model is on the roadmap for v0.2.
- pgvector is required for nearest-neighbour search; SQLite fallback
  performs a linear scan (test-only).
- The dashboard does not yet support multi-camera layouts.

### Breaking changes
- None — this is the initial release.

[Unreleased]: https://github.com/your-org/facial-attendance-system/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/facial-attendance-system/releases/tag/v0.1.0
