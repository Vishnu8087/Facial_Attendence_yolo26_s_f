# MiniFASNet PAD evaluation

The project now supports optional face presentation-attack detection (PAD)
using two ONNX MiniFASNet models. This is separate from ArcFace identity
recognition and ByteTrack tracking. PAD evaluates the detected face crop as
paper, bona fide, or screen; attendance still requires a matching identity.

## Enable the local application test

Use the project virtual environment and download the pinned, SHA-256-verified
models:

```powershell
python scripts\download_pad_models.py
```

Set these variables in the server's environment (or `configs/.env`):

```env
PAD_ENABLED=true
PAD_MODEL_DIR=models/pad
PAD_THRESHOLD=0.5
```

Restart the API. It warms the model sessions at startup. If either model
cannot be found or loaded, startup fails instead of silently reverting to
the Euler-angle/blink heuristic. If PAD inference fails for a face, that face
does not pass the attendance liveness gate.

When PAD rejects a face as `paper` or `screen`, the live overlay labels it
`PAPER - ATTENDANCE BLOCKED` or `SCREEN - ATTENDANCE BLOCKED`. Rejected faces
are not shown with an employee identity, and the stream does not emit an
attendance event for them. The face detector may still draw a box: PAD is a
separate presentation check, not a face-detection filter.

The default `PAD_THRESHOLD=0.5` is an initial test value, not a threshold
validated for this camera. The model accepts class 1 only when it is the
highest class and its fused score meets the threshold. Classes 0 and 2 are
rejected. The implementation uses the detector's `xyxy` box and two context
scales (2.7 and 4.0), preserving the MiniFASNet BGR 80×80 preprocessing.

## Full Colab / Kaggle attendance experiment

Open [`../notebooks/face_recognition_pad_attendance.ipynb`](../notebooks/face_recognition_pad_attendance.ipynb)
in Google Colab or Kaggle. It contains the complete face-detection + ArcFace
enrollment/recognition + PAD + attendance CSV workflow, Drive dataset layout,
and Colab camera capture helper. Place genuine enrollment samples in employee
folders and separately captured camera test images in `live`, `print`, and
`screen` folders. Use different capture sessions for threshold tuning and
held-out evaluation, including varied lighting, distance, angles, print
materials, and display devices.

The notebook reports PAD-only and combined-attendance false-accept and
false-reject rates, balanced accuracy, and inference time. The evaluation is
only as representative as the images provided; do not tune and report results
on the same small set as if it were an independent validation set.

For PAD scoring without employee enrollment or recognition, the smaller
[`../notebooks/pad_minifasnet_evaluation.ipynb`](../notebooks/pad_minifasnet_evaluation.ipynb)
is also available.

## Model provenance and limitations

The model architecture and pretrained weights originate from
[Minivision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing),
with ONNX files from the Apache-2.0
[ONNX conversion repository](https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/tree/584d4421d7ac42c59e640796f46e886b0095367a).
The downloader pins the conversion commit and verifies each file's SHA-256.
The model weights are downloaded locally and are not committed to this
repository.

This is a research baseline, not a certified anti-spoofing system. It may
misclassify photos, displays, genuine faces, or attacks unlike its training
data. Validate on your own camera and threat scenarios. For stronger
protection, evaluate a challenge-response flow and a PAD model trained and
validated for the deployment domain.
