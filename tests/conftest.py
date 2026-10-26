"""Shared pytest fixtures.

Every fixture here is designed to work *without* a GPU, *without* real
InsightFace / ONNX models, and *without* a running Postgres instance —
heavy components are mocked. If the real implementation under ``app/``
is importable, the fixtures will use it; otherwise they fall back to a
mock so the suite can still be collected and exercised.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Path setup — make ``app`` importable even when not installed as a package.
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_FACES_DIR = FIXTURES_DIR / "sample_faces"


# --------------------------------------------------------------------------- #
# Asyncio
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the whole session — avoids 'attached to a
    different loop' errors when the async DB session is session-scoped."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# --------------------------------------------------------------------------- #
# Image fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def sample_frame() -> np.ndarray:
    """A 480x640x3 uint8 BGR-ish synthetic frame (gradient + shapes).

    Replacing real camera frames in tests. NOT a face — used for the
    MotionTracker / pipeline tests where face detection is mocked.
    """
    rng = np.random.default_rng(seed=42)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Vertical gradient
    for y in range(480):
        frame[y, :, 0] = int(y / 480 * 255)
        frame[y, :, 1] = int((480 - y) / 480 * 255)
        frame[y, :, 2] = 128
    # Random noise overlay
    frame = np.clip(frame.astype(np.int16) + rng.integers(-10, 10, frame.shape), 0, 255).astype(np.uint8)
    return frame


@pytest.fixture(scope="session")
def sample_face_frame() -> np.ndarray:
    """A 250x250x3 uint8 frame containing a synthetic face-like pattern.

    Used by aligner / extractor tests where an actual face crop is needed.
    The real implementation in ``tests/fixtures/sample_faces/face_alpha.png``
    is generated from this function — kept here so tests don't depend on
    file I/O.
    """
    img = np.zeros((250, 250, 3), dtype=np.uint8)
    img[:] = (200, 200, 170)  # skin tone background
    # Eyes
    cv2 = _try_import_cv2()
    if cv2 is not None:
        cv2.circle(img, (85, 100), 12, (40, 40, 40), -1)
        cv2.circle(img, (165, 100), 12, (40, 40, 40), -1)
        # Nose
        cv2.line(img, (125, 110), (125, 145), (120, 90, 70), 3)
        # Mouth
        cv2.ellipse(img, (125, 170), (35, 12), 0, 0, 180, (40, 20, 20), 3)
        # Face outline
        cv2.ellipse(img, (125, 125), (110, 130), 0, 0, 360, (80, 60, 50), 2)
    else:
        # Fallback when cv2 is missing
        img[95:115, 75:100, :] = 40
        img[95:115, 155:180, :] = 40
        img[140:160, 120:135, :] = 120
    return img


@pytest.fixture(scope="session")
def sample_face_landmarks() -> np.ndarray:
    """5-point landmarks (x, y) matching :fixture:`sample_face_frame`.

    Order follows the InsightFace convention:
    ``[right_eye, left_eye, nose, right_mouth, left_mouth]``.
    """
    return np.array(
        [
            [85, 100],   # right eye (from viewer)
            [165, 100],  # left eye
            [125, 130],  # nose
            [95, 170],   # right mouth corner
            [155, 170],  # left mouth corner
        ],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# Mock CV components — return canned results so tests don't need real models.
# --------------------------------------------------------------------------- #
class _MockDetection:
    def __init__(self, bbox: np.ndarray, landmarks: np.ndarray, score: float):
        self.bbox = bbox            # [x1, y1, x2, y2]
        self.landmarks = landmarks  # (5, 2)
        self.score = score


class _MockTrack:
    def __init__(self, track_id: int, bbox: np.ndarray, landmarks: np.ndarray | None = None):
        self.track_id = track_id
        self.bbox = bbox
        self.landmarks = landmarks if landmarks is not None else np.zeros((5, 2), dtype=np.float32)


@pytest.fixture
def mock_detector(monkeypatch):
    """Return a MagicMock-style detector returning two fixed detections."""
    from unittest.mock import MagicMock

    detections = [
        _MockDetection(
            bbox=np.array([10, 10, 110, 130], dtype=np.float32),
            landmarks=np.array(
                [[35, 50], [80, 50], [60, 80], [40, 100], [80, 100]], dtype=np.float32
            ),
            score=0.97,
        ),
        _MockDetection(
            bbox=np.array([300, 50, 420, 200], dtype=np.float32),
            landmarks=np.array(
                [[330, 100], [390, 100], [360, 130], [335, 160], [385, 160]], dtype=np.float32
            ),
            score=0.91,
        ),
    ]
    detector = MagicMock(name="FaceDetector")
    detector.detect.return_value = detections
    return detector


@pytest.fixture
def mock_extractor(monkeypatch):
    """Return a mock FeatureExtractor producing deterministic 512-d L2-normalised embeddings."""
    from unittest.mock import MagicMock

    def _embed(face_img: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(seed=int(face_img.sum()) % (2**32 - 1))
        emb = rng.standard_normal(512).astype(np.float32)
        emb /= max(np.linalg.norm(emb), 1e-9)
        return emb

    extractor = MagicMock(name="FeatureExtractor")
    extractor.embedding_dim = 512
    extractor.extract.side_effect = _embed
    return extractor


@pytest.fixture
def mock_tracker():
    """Return a MagicMock tracker that assigns IDs 1, 2, 3 ... per new bbox."""
    from unittest.mock import MagicMock

    state = {"next_id": 1, "active": {}}

    def _update(detections: Iterable[Any]):
        out = []
        for det in detections:
            tid = state["active"].get(tuple(det.bbox.tolist()))
            if tid is None:
                tid = state["next_id"]
                state["next_id"] += 1
                state["active"][tuple(det.bbox.tolist())] = tid
            out.append(_MockTrack(track_id=tid, bbox=det.bbox, landmarks=det.landmarks))
        return out

    tracker = MagicMock(name="MotionTracker")
    tracker.update.side_effect = _update
    tracker.cooldown_seconds = 30.0
    tracker.reset = MagicMock(side_effect=lambda: state.update({"next_id": 1, "active": {}}))
    return tracker


@pytest.fixture
def mock_liveness():
    """Return a MagicMock LivenessVerifier returning 'real' for non-static frames."""
    from unittest.mock import MagicMock

    verifier = MagicMock(name="LivenessVerifier")
    verifier.verify.return_value = {
        "is_real": True,
        "ear": 0.28,
        "blink_count": 1,
        "head_pose": (0.0, 0.0, 0.0),
        "score": 0.93,
    }
    return verifier


# --------------------------------------------------------------------------- #
# Database fixtures
# --------------------------------------------------------------------------- #
def _can_import_sqlalchemy() -> bool:
    return importlib.util.find_spec("sqlalchemy") is not None


@pytest.fixture
async def db_session():
    """Async SQLAlchemy session against in-memory SQLite.

    Skips the test if SQLAlchemy / aiosqlite are not installed, or if the
    project's models module is not present yet.
    """
    if not _can_import_sqlalchemy():
        pytest.skip("SQLAlchemy is not installed")

    try:
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    except ImportError:
        pytest.skip("aiosqlite / async SQLAlchemy not installed")

    try:
        from app.db.base import Base  # type: ignore
        from app.db import models  # noqa: F401  # ensures tables registered
    except Exception as exc:  # pragma: no cover - depends on 1-b
        pytest.skip(f"app.db not yet implemented by Task 1-b: {exc!r}")

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with sessionmaker() as session:
        yield session
    await engine.dispose()


# --------------------------------------------------------------------------- #
# HTTP + WebSocket test clients
# --------------------------------------------------------------------------- #
@pytest.fixture
async def test_client():
    """httpx.AsyncClient wired to the FastAPI app via ASGITransport.

    Skips if the API app is not importable yet (Task 1-b).
    """
    try:
        from httpx import ASGITransport, AsyncClient
    except ImportError:
        pytest.skip("httpx not installed")
    try:
        from app.main import app  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on 1-b
        pytest.skip(f"app.main not yet implemented by Task 1-b: {exc!r}")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
async def ws_client(test_client):
    """Open a WebSocket connection to ``/ws/stream``.

    Requires :fixture:`test_client` to be functional (i.e. app importable).
    """
    try:
        async with test_client.websocket_connect("/ws/stream") as ws:
            yield ws
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"WebSocket endpoint not available: {exc!r}")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def sample_faces_dir() -> Path:
    return SAMPLE_FACES_DIR
