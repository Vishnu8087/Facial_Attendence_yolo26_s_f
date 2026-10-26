"""Unit tests for ``app.core.tracker.MotionTracker``."""
from __future__ import annotations

import time

import numpy as np
import pytest

MotionTracker = pytest.importorskip("app.core.tracker", reason="Task 1-a not yet merged").MotionTracker  # type: ignore[attr-defined]


def _bbox(cx, cy, w=80, h=100):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def _det(bbox):
    from types import SimpleNamespace
    return SimpleNamespace(bbox=bbox, landmarks=np.zeros((5, 2), dtype=np.float32), score=0.9)


def test_tracker_assigns_persistent_id_across_frames():
    tracker = MotionTracker(cooldown_seconds=30.0)
    bbox = _bbox(100, 100)

    out1 = tracker.update([_det(bbox)])
    out2 = tracker.update([_det(bbox + 5)])  # small motion
    out3 = tracker.update([_det(bbox + 10)])

    assert len(out1) == len(out2) == len(out3) == 1
    assert out1[0].track_id == out2[0].track_id == out3[0].track_id, "ID should persist"


def test_tracker_assigns_new_id_to_new_face():
    tracker = MotionTracker(cooldown_seconds=30.0)
    a = tracker.update([_det(_bbox(100, 100))])
    b = tracker.update([_det(_bbox(500, 400))])  # very different location
    assert a[0].track_id != b[0].track_id


def test_tracker_handles_empty_input():
    tracker = MotionTracker()
    out = tracker.update([])
    assert out == []


def test_tracker_reuses_id_after_short_gap_within_cooldown():
    tracker = MotionTracker(cooldown_seconds=30.0)
    bbox = _bbox(100, 100)
    first = tracker.update([_det(bbox)])
    # No detections for a brief gap (no sleep — within cooldown window)
    tracker.update([])
    second = tracker.update([_det(bbox)])
    assert first[0].track_id == second[0].track_id


def test_tracker_assigns_new_id_after_cooldown_expires():
    """After cooldown_seconds has elapsed since the last sighting, a
    reappearing face should get a fresh ID (it's treated as a new session)."""
    tracker = MotionTracker(cooldown_seconds=0.05)
    bbox = _bbox(100, 100)
    first = tracker.update([_det(bbox)])
    time.sleep(0.1)
    tracker.update([])  # tick tracker forward to expire
    second = tracker.update([_det(bbox)])
    assert first[0].track_id != second[0].track_id


def test_tracker_multiple_faces_keep_distinct_ids():
    tracker = MotionTracker(cooldown_seconds=30.0)
    a = tracker.update([_det(_bbox(100, 100)), _det(_bbox(400, 300))])
    b = tracker.update([_det(_bbox(105, 105)), _det(_bbox(405, 305))])
    ids_first = sorted(t.track_id for t in a)
    ids_second = sorted(t.track_id for t in b)
    assert ids_first == ids_second
    assert len(set(ids_first)) == 2


def test_tracker_reset_clears_state():
    tracker = MotionTracker(cooldown_seconds=30.0)
    tracker.update([_det(_bbox(100, 100))])
    tracker.reset()
    out = tracker.update([_det(_bbox(100, 100))])
    # After reset, IDs restart — assume starts at 1 again
    assert out[0].track_id in (1, 0)
