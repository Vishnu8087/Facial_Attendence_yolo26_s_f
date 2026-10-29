"""ONNX inference for the MiniFASNet face presentation-attack models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_SCALES = (
    ("2.7_80x80_MiniFASNetV2.onnx", 2.7),
    ("4_0_0_80x80_MiniFASNetV1SE.onnx", 4.0),
)
CLASS_LABELS = ("paper", "real", "screen")


class MiniFASNetPad:
    """Fuse the released MiniFASNet scales and classify a detected face crop."""

    def __init__(
        self,
        model_dir: str | Path,
        threshold: float = 0.5,
        providers: Sequence[str] | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("PAD threshold must be between 0 and 1")
        self.model_dir = Path(model_dir)
        self.threshold = float(threshold)

        import onnxruntime as ort

        available = ort.get_available_providers()
        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
        unavailable = set(providers) - set(available)
        if unavailable:
            raise RuntimeError(
                f"Requested ONNX Runtime providers are unavailable: {sorted(unavailable)}"
            )
        self.providers = list(providers)
        self._models: list[tuple[str, float, Any, str, str]] = []

        for filename, scale in MODEL_SCALES:
            model_path = self.model_dir / filename
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"PAD model is missing: {model_path}. "
                    "Run scripts/download_pad_models.py to fetch the pinned models."
                )
            session = ort.InferenceSession(
                str(model_path),
                providers=self.providers,
            )
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError(f"Unexpected MiniFASNet ONNX signature: {model_path}")
            self._models.append(
                (filename, scale, session, inputs[0].name, outputs[0].name)
            )
            logger.info(
                "pad.model.loaded",
                extra={"model": filename, "providers": session.get_providers()},
            )

    def predict(self, frame: np.ndarray, bbox: Sequence[float]) -> dict[str, Any]:
        """Return the fused live probability and class for an ``xyxy`` face box."""
        face_box = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if face_box.size != 4 or not np.isfinite(face_box).all():
            raise ValueError("PAD bbox must contain four finite xyxy values")
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("PAD expects a BGR image with shape (height, width, 3)")

        probabilities = np.zeros(3, dtype=np.float64)
        for _, scale, session, input_name, output_name in self._models:
            patch = self._prepare_patch(frame, face_box, scale)
            logits = np.asarray(
                session.run([output_name], {input_name: patch})[0],
                dtype=np.float64,
            ).reshape(-1)
            if logits.size != len(CLASS_LABELS) or not np.isfinite(logits).all():
                raise RuntimeError("MiniFASNet returned invalid three-class logits")
            exp_logits = np.exp(logits - np.max(logits))
            probabilities += exp_logits / np.sum(exp_logits)

        probabilities /= len(self._models)
        label_index = int(np.argmax(probabilities))
        live_score = float(probabilities[1])
        is_real = label_index == 1 and live_score >= self.threshold
        return {
            "is_real": is_real,
            "label": CLASS_LABELS[label_index],
            "score": live_score,
            "scores": probabilities.astype(np.float32),
        }

    def warmup(self) -> None:
        """Run one zero-input inference per loaded model before camera startup."""
        sample = np.zeros((1, 3, 80, 80), dtype=np.float32)
        for _, _, session, input_name, output_name in self._models:
            session.run([output_name], {input_name: sample})

    @staticmethod
    def _prepare_patch(
        frame: np.ndarray,
        bbox: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Match the upstream 80x80 BGR, unnormalised, multi-scale preprocessing."""
        image_height, image_width = frame.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in bbox)
        x1 = min(max(x1, 0.0), image_width - 1.0)
        y1 = min(max(y1, 0.0), image_height - 1.0)
        x2 = min(max(x2, 0.0), image_width - 1.0)
        y2 = min(max(y2, 0.0), image_height - 1.0)
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError("PAD bbox must have positive width and height")

        expanded_scale = min(
            scale,
            (image_width - 1.0) / box_width,
            (image_height - 1.0) / box_height,
        )
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        half_width = box_width * expanded_scale / 2.0
        half_height = box_height * expanded_scale / 2.0
        crop_x1 = max(0, int(np.floor(center_x - half_width)))
        crop_y1 = max(0, int(np.floor(center_y - half_height)))
        crop_x2 = min(image_width, int(np.ceil(center_x + half_width)) + 1)
        crop_y2 = min(image_height, int(np.ceil(center_y + half_height)) + 1)
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            raise ValueError("PAD bbox produced an empty image crop")

        resized = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)
        chw = np.transpose(resized.astype(np.float32), (2, 0, 1))
        return np.expand_dims(chw, axis=0)


__all__ = ["MiniFASNetPad"]
