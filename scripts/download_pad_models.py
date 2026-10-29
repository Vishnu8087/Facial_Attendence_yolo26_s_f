"""Download and verify the pinned MiniFASNet ONNX PAD models."""
from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_COMMIT = "584d4421d7ac42c59e640796f46e886b0095367a"
MODELS = {
    "2.7_80x80_MiniFASNetV2.onnx": "0CBE5CAEC95C31DE9D2EF845CB85407D76AECD1B6A2C0E343F7D35306BFBCCB8",
    "4_0_0_80x80_MiniFASNetV1SE.onnx": "A25886A85CDCFA2C4EA23EDB71DE35F250C17827B4CADD253A972B28C80FDF1E",
}


def _download_verified(filename: str, expected_sha256: str, output_dir: Path) -> None:
    url = (
        "https://raw.githubusercontent.com/"
        f"QingHeYang/Silent-Face-Anti-Spoofing-onnx/{MODEL_COMMIT}/onnx/{filename}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    digest = hashlib.sha256()
    temporary_path: Path | None = None
    try:
        with urlopen(url, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_dir,
                prefix=f".{filename}.",
                suffix=".download",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    temporary.write(chunk)
        actual_sha256 = digest.hexdigest().upper()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: expected {expected_sha256}, "
                f"received {actual_sha256}"
            )
        os.replace(temporary_path, target)
        temporary_path = None
        print(f"Verified {target} (SHA-256 {actual_sha256})")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "pad",
        help="Directory for verified ONNX files (default: models/pad).",
    )
    args = parser.parse_args()
    for filename, sha256 in MODELS.items():
        _download_verified(filename, sha256, args.output_dir)


if __name__ == "__main__":
    main()
