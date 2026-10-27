"""Application configuration.

Settings are resolved in this order (later overrides earlier):

1. ``configs/config.yaml`` — defaults shipped with the repo.
2. Environment variables — set by Docker / Kubernetes / ``.env``.
3. Explicitly-passed overrides (tests, scripts).

We use :mod:`pydantic_settings` so that env vars map to fields
case-insensitively (``DATABASE_URL`` -> ``database_url``).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repo root on disk (``/app`` inside the container, the repo dir in dev)."""

CONFIG_YAML = PROJECT_ROOT / "configs" / "config.yaml"
"""Default YAML config file. May be absent (tests / minimal deploys)."""


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load the YAML config file if it exists, returning a flat-ish dict.

    The YAML file is structured in top-level sections (``app``, ``database``,
    ``cv``, ``streams``, ``api``, ``prometheus``, ``logging``). We flatten
    each section into ``section_key`` so that they line up with env-var names
    consumed by :class:`Settings`.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:  # pragma: no cover — defensive
        return {}

    flat: dict[str, Any] = {}
    for section_name, section_val in data.items():
        if isinstance(section_val, dict):
            for k, v in section_val.items():
                flat[f"{section_name}_{k}"] = v
        else:
            flat[section_name] = section_val
    return flat


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Typed application settings.

    Every field has a default so that the app can boot in test/CI
    environments without a real ``.env`` file. Production deployments
    override via env vars (see ``configs/.env.example``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # ----- App / server -----
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    workers: int = Field(2, description="uvicorn worker count")
    reload: bool = False
    app_home: str = "/app"
    log_level: str = "INFO"
    log_format: str = "json"  # json | console
    log_exceptions: bool = True

    # ----- Database -----
    database_url: str = "postgresql+asyncpg://attendance:change-me@localhost:5432/attendance"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "attendance"
    db_user: str = "attendance"
    db_password: str = "change-me"
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800
    db_echo: bool = False
    use_pgvector: bool = True

    # Postgres bootstrap (compose-only)
    postgres_user: str = "attendance"
    postgres_password: str = "change-me"
    postgres_db: str = "attendance"

    # ----- Redis -----
    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # ----- Auth / security -----
    jwt_secret: str = "change-me-jwt-secret-not-for-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_key: str = ""  # optional pre-shared key for non-JWT auth
    cors_origins: str = "*"

    # ----- Observability -----
    sentry_dsn: str = ""
    sentry_traces_sample_rate: float = 0.1
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    prometheus_path: str = "/metrics"

    # ----- CV pipeline -----
    gpu_enabled: bool = False
    # Default: YOLO26-s face detector, selected for its benchmarked accuracy.
    detector_backend: str = "yolo26"          # insightface | yolo26
    detector_model: str = "yolo26s-face.pt"
    detector_device: str = ""                 # e.g. "cuda:0"; empty = auto (cpu unless gpu_enabled)
    model_name: str = "buffalo_l"
    model_dir: str = "/app/models"
    det_size: int = 640
    similarity_threshold: float = 0.30  # lowered from 0.55 — LFW benchmark: 0% FAR + 99.33% accuracy
    cooldown_seconds: int = 30
    min_face_size: int = 60
    max_faces_per_frame: int = 50
    tracker_type: str = "bytetrack"
    liveness_enabled: bool = True
    liveness_min_score: float = 0.5
    default_strategy: str = "both"  # both | detection_only | tracking_only
    cpu_cores: int = Field(
        0,
        validation_alias=AliasChoices("CPU_CORES", "CV_CPU_CORES"),
        description="CPU cores available to inference; 0 detects via os.cpu_count().",
    )
    arcface_threading: str = Field(
        "auto",
        validation_alias=AliasChoices("ARCFACE_THREADING", "CV_ARCFACE_THREADING"),
        description="ArcFace per-face threading mode: auto, true, or false.",
    )
    max_arcface_threads: int = Field(
        4,
        validation_alias=AliasChoices("MAX_ARCFACE_THREADS", "CV_MAX_ARCFACE_THREADS"),
        description="Maximum concurrent ArcFace inference calls.",
    )

    # ----- Streams / WebSocket -----
    max_cameras: int = 32
    ws_max_queue: int = 128
    ws_heartbeat_interval: int = 5
    stream_frame_timeout: int = 30
    max_stream_fps: int = Field(
        15,
        validation_alias=AliasChoices(
            "MAX_STREAM_FPS",
            "STREAMS_MAX_STREAM_FPS",
        ),
    )
    max_frame_bytes: int = 2 * 1024 * 1024  # 2 MB

    # ----- API -----
    max_upload_mb: int = 32

    # ----- Filesystem paths -----
    models_dir: str = "/app/models"
    data_dir: str = "/app/data"
    log_dir: str = "/app/logs"
    tmp_dir: str = "/app/tmp"
    faces_dir: str = "/app/data/faces"

    # ----- Worker -----
    worker_concurrency: int = 4
    worker_queue_name: str = "facial-attendance:frames"
    worker_events_channel: str = "facial-attendance:events"

    # ----------------------------------------------------------------- #
    # Derived helpers
    # ----------------------------------------------------------------- #
    @field_validator("default_strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        allowed = {"both", "detection_only", "tracking_only"}
        if v not in allowed:
            raise ValueError(f"default_strategy must be one of {allowed}, got {v!r}")
        return v

    @field_validator("detector_backend")
    @classmethod
    def _validate_detector_backend(cls, v: str) -> str:
        allowed = {"insightface", "yolo26"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(
                f"detector_backend must be one of {allowed}, got {v!r}. "
                f"Use 'insightface' for SCRFD or 'yolo26' for "
                f"Ultralytics YOLO26-face (akanametov/yolo-face)."
            )
        return v_lower

    @field_validator("arcface_threading")
    @classmethod
    def _validate_arcface_threading(cls, v: str) -> str:
        value = v.lower()
        if value not in {"auto", "true", "false"}:
            raise ValueError("arcface_threading must be one of 'auto', 'true', or 'false'")
        return value

    @field_validator("cpu_cores")
    @classmethod
    def _validate_cpu_cores(cls, v: int) -> int:
        if v < 0:
            raise ValueError("cpu_cores must be 0 (auto) or a positive integer")
        return v

    @field_validator("max_arcface_threads")
    @classmethod
    def _validate_max_arcface_threads(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_arcface_threads must be at least 1")
        return v

    @field_validator("max_stream_fps")
    @classmethod
    def _validate_max_stream_fps(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_stream_fps must be at least 1")
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list. ``"*"`` is preserved verbatim."""
        if not self.cors_origins:
            return ["*"]
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_cpu_cores(self) -> int:
        """Return the configured core count, or the logical CPU count when auto."""
        return self.cpu_cores or os.cpu_count() or 1

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def frontend_dir(self) -> Path:
        return PROJECT_ROOT / "frontend"

    @property
    def alembic_ini(self) -> Path:
        return PROJECT_ROOT / "alembic.ini"

    def ensure_dirs(self) -> None:
        """Create the runtime directories on disk if missing."""
        for p in (self.models_dir, self.data_dir, self.log_dir, self.tmp_dir, self.faces_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Singleton
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton.

    The YAML config is loaded first and merged into the env (YAML values
    become the default — env vars still override).
    """
    yaml_defaults = _load_yaml_config(CONFIG_YAML)
    # Push YAML values into the env *only if the env var is not already set*
    # — env vars must always win.
    for key, val in yaml_defaults.items():
        env_key = key.upper()
        if env_key not in os.environ and val is not None:
            os.environ[env_key] = str(val)
    return Settings()


# Module-level singleton — most callers do ``from app.config import settings``.
settings = get_settings()
