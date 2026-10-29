from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.core.liveness import LivenessVerifier
from app.core.pad import MODEL_SCALES, MiniFASNetPad


class _FakeSession:
    def __init__(self, logits: list[float]) -> None:
        self.logits = np.asarray([logits], dtype=np.float32)
        self.input_values: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, inputs):
        self.input_values.append(inputs["input"])
        return [self.logits]


def _pad(monkeypatch, tmp_path, model_logits):
    for filename, _ in MODEL_SCALES:
        (tmp_path / filename).write_bytes(b"test model")
    sessions = iter(_FakeSession(logits) for logits in model_logits)

    import onnxruntime

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(
        onnxruntime,
        "InferenceSession",
        lambda *_args, **_kwargs: next(sessions),
    )
    return MiniFASNetPad(tmp_path)


def test_minifasnet_fuses_scales_and_marks_live_class(monkeypatch, tmp_path):
    pad = _pad(monkeypatch, tmp_path, ([0.0, 5.0, 0.0], [0.0, 5.0, 0.0]))
    frame = np.full((240, 320, 3), 127, dtype=np.uint8)

    result = pad.predict(frame, (80, 40, 240, 220))

    assert result["is_real"] is True
    assert result["label"] == "real"
    assert result["scores"].shape == (3,)
    assert result["score"] > 0.5


def test_minifasnet_rejects_paper_prediction(monkeypatch, tmp_path):
    pad = _pad(monkeypatch, tmp_path, ([5.0, 0.0, 0.0], [5.0, 0.0, 0.0]))
    frame = np.full((240, 320, 3), 127, dtype=np.uint8)

    result = pad.predict(frame, (80, 40, 240, 220))

    assert result["is_real"] is False
    assert result["label"] == "paper"


def test_minifasnet_rejects_invalid_or_empty_boxes(monkeypatch, tmp_path):
    pad = _pad(monkeypatch, tmp_path, ([0.0, 5.0, 0.0], [0.0, 5.0, 0.0]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive width and height"):
        pad.predict(frame, (25, 25, 25, 50))


def test_pad_liveness_requires_bbox_and_uses_pad_result():
    class FakePad:
        def __init__(self):
            self.received_bbox = None

        def predict(self, _frame, bbox):
            self.received_bbox = bbox
            return {
                "is_real": False,
                "label": "paper",
                "score": 0.02,
                "scores": np.asarray([0.96, 0.02, 0.02]),
            }

    pad = FakePad()
    verifier = LivenessVerifier(pad_model=pad)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    landmarks = np.asarray(
        [[40, 40], [80, 40], [60, 60], [45, 80], [75, 80]],
        dtype=np.float32,
    )

    missing_box = verifier.verify(frame, landmarks, track_id=7)
    checked = verifier.verify(frame, landmarks, track_id=7, bbox=(20, 20, 100, 100))

    assert missing_box.is_live is False
    assert missing_box.reason == "pad_missing_frame_or_bbox"
    assert checked.is_live is False
    assert checked.pad_label == "paper"
    assert pad.received_bbox == (20, 20, 100, 100)
