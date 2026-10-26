#!/usr/bin/env bash
###############################################################################
# entrypoint.sh — Facial Attendance & Motion Tracking System
#
# Responsibilities (in order):
#   1. Wait for Postgres to accept connections.
#   2. Run Alembic migrations (`alembic upgrade head`).
#   3. Optionally pre-download InsightFace models if not present in $MODELS_DIR.
#   4. Exec into either the API (`uvicorn app.main:app`) or the worker
#      (`python -m app.worker`) based on $1.
#
# Image is non-root (USER app). The script is also safe to run as root in
# dev containers.
#
# Usage:
#   ./entrypoint.sh api     # default — FastAPI app
#   ./entrypoint.sh worker  # background processor
#   ./entrypoint.sh bash    # interactive shell (debug only)
###############################################################################
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()  { printf '[entrypoint] %s\n' "$*" >&2; }
warn() { printf '[entrypoint][WARN] %s\n' "$*" >&2; }
die()  { printf '[entrypoint][FATAL] %s\n' "$*" >&2; exit 1; }

trap 'die "Unhandled error on line $LINENO (exit=$?)"' ERR

# ---------------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------------
APP_HOME="${APP_HOME:-/app}"
MODELS_DIR="${MODELS_DIR:-/app/models}"
ALEMBIC_INI="${ALEMBIC_INI:-${APP_HOME}/alembic.ini}"
WORKERS="${WORKERS:-2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

DATABASE_URL="${DATABASE_URL:-}"
SKIP_MIGRATIONS="${SKIP_MIGRATIONS:-false}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-true}"
MODEL_NAME="${MODEL_NAME:-buffalo_l}"

WAIT_DB_TIMEOUT="${WAIT_DB_TIMEOUT:-60}"   # seconds
WAIT_DB_INTERVAL="${WAIT_DB_INTERVAL:-2}"

# ---------------------------------------------------------------------------
# 1. Wait for Postgres
# ---------------------------------------------------------------------------
wait_for_db() {
    [ -z "$DATABASE_URL" ] && { warn "DATABASE_URL empty — skipping DB wait"; return 0; }

    log "Waiting for Postgres (${DATABASE_URL%%@*}@***) up to ${WAIT_DB_TIMEOUT}s …"
    local elapsed=0
    # Use python so we don't depend on pg_isready in the API image.
    until python - <<EOF
import os, sys, socket, time, urllib.parse
url = os.environ["DATABASE_URL"]
try:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "db"
    port = parsed.port or 5432
    with socket.create_connection((host, port), timeout=2):
        pass
    sys.exit(0)
except Exception:
    sys.exit(1)
EOF
    do
        elapsed=$((elapsed + WAIT_DB_INTERVAL))
        if [ "$elapsed" -ge "$WAIT_DB_TIMEOUT" ]; then
            die "Postgres not reachable after ${WAIT_DB_TIMEOUT}s"
        fi
        sleep "$WAIT_DB_INTERVAL"
    done
    log "Postgres is accepting TCP connections."
}

# ---------------------------------------------------------------------------
# 2. Run Alembic migrations
# ---------------------------------------------------------------------------
run_migrations() {
    if [ "$SKIP_MIGRATIONS" = "true" ]; then
        warn "SKIP_MIGRATIONS=true — skipping alembic upgrade"
        return 0
    fi
    [ ! -f "$ALEMBIC_INI" ] && { warn "alembic.ini not found at $ALEMBIC_INI — skipping"; return 0; }

    log "Running alembic upgrade head …"
    cd "$APP_HOME"
    alembic upgrade head
    log "Migrations complete."
}

# ---------------------------------------------------------------------------
# 3. Pre-download models
# ---------------------------------------------------------------------------
ensure_models() {
    if [ "$DOWNLOAD_MODELS" != "true" ]; then
        log "DOWNLOAD_MODELS=false — skipping model download"
        return 0
    fi

    mkdir -p "$MODELS_DIR"

    # InsightFace ships models inside the package; the first instantiation
    # triggers a download into ~/.insightface unless HF_HOME / INSIGHTFACE_HOME
    # is set. We point it at MODELS_DIR so the volume persists.
    export INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-${MODELS_DIR}/insightface}"
    export HF_HOME="${HF_HOME:-${MODELS_DIR}/huggingface}"
    mkdir -p "$INSIGHTFACE_HOME" "$HF_HOME"

    if [ -f "${MODELS_DIR}/.models-ready" ]; then
        log "Models already present in ${MODELS_DIR} (marker .models-ready exists)."
        return 0
    fi

    log "Pre-warming InsightFace model '${MODEL_NAME}' into ${INSIGHTFACE_HOME} …"
    python - <<EOF
import os
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="${MODEL_NAME}", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("[entrypoint] model '${MODEL_NAME}' ready")
except Exception as exc:  # noqa: BLE001
    print(f"[entrypoint][WARN] model pre-warm failed: {exc}")
    raise SystemExit(0)  # do not block startup — worker will lazy-load
EOF

    # If DETECTOR_BACKEND=yolo26, also pre-fetch the YOLO26-face weights
    # so the first frame doesn't pay the download cost.
    if [ "${DETECTOR_BACKEND:-insightface}" = "yolo26" ]; then
        local variant="${DETECTOR_MODEL:-yolo26n-face.pt}"
        log "DETECTOR_BACKEND=yolo26 — pre-fetching ${variant} …"
        python - <<EOF
try:
    from ultralytics import YOLO
    model = YOLO("${variant}")  # auto-downloads on first call
    print(f"[entrypoint] yolo26 model '${variant}' ready")
except Exception as exc:
    print(f"[entrypoint][WARN] yolo26 pre-warm failed: {exc}")
    # Non-fatal — worker will retry on first frame
EOF
    fi

    touch "${MODELS_DIR}/.models-ready"
    log "Models cached."
}

# ---------------------------------------------------------------------------
# 4. Exec into requested role
# ---------------------------------------------------------------------------
run_api() {
    log "Starting API (uvicorn) — host=${HOST} port=${PORT} workers=${WORKERS}"
    cd "$APP_HOME"
    local args=(app.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS" \
        --proxy-headers --forwarded-allow-ips='*')
    if [ "${RELOAD:-false}" = "true" ]; then
        args+=(--reload)
    fi
    exec uvicorn "${args[@]}"
}

run_worker() {
    log "Starting worker (python -m app.worker)"
    cd "$APP_HOME"
    exec python -m app.worker
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ROLE="${1:-api}"
shift || true

case "$ROLE" in
    api|worker)
        wait_for_db
        run_migrations
        ensure_models
        case "$ROLE" in
            api)    run_api "$@" ;;
            worker) run_worker "$@" ;;
        esac
        ;;
    bash|sh)
        exec "$ROLE" "$@"
        ;;
    *)
        die "Unknown role: '${ROLE}'. Expected: api | worker | bash"
        ;;
esac
