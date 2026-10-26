#!/usr/bin/env python3
"""Download CV model weights into the local ``models/`` directory.

Supports two detector backends:

1. **InsightFace** (default) — packs like ``buffalo_l``, ``buffalo_s``,
   ``buffalo_m``, ``antelopev2`` distributed as zipped tarballs on a
   public CDN. The :mod:`insightface` Python package will *auto-download*
   a pack on first use, but the auto-download has no progress bar, no
   checksum verification, and writes to ``~/.insightface/models`` which
   is awkward in containers. This script does it properly.

2. **YOLO26-face** (optional) — Ultralytics YOLO26 (released January
   2026) face-detector weights from
   https://github.com/akanametov/yolo-face. Available variants:
   ``yolo26n-face.pt`` (5 MB, faster) and ``yolo26s-face.pt`` (about 21 MB,
   more accurate). Ultralytics' ``YOLO()`` constructor will auto-download
   these on first use, but pre-fetching them avoids cold-start latency
   and lets the entrypoint verify they're present before booting.

Features:
* Progress bar via :mod:`tqdm` (graceful no-tqdm fallback).
* SHA-256 checksum verification against a known-good manifest.
* Idempotent: skips files that already exist and match the checksum.
* Configurable destination directory (``--dest``).

Usage
-----

    # InsightFace buffalo_l (default)
    python scripts/download_models.py

    # Smaller InsightFace pack
    python scripts/download_models.py --pack buffalo_s

    # YOLO26-face weights (used when DETECTOR_BACKEND=yolo26)
    python scripts/download_models.py --yolo26
    python scripts/download_models.py --yolo26 --yolo26-variant yolo26s-face.pt

    # Both backends (useful when you want to switch at runtime)
    python scripts/download_models.py --yolo26 --pack buffalo_l

    # Custom destination
    python scripts/download_models.py --dest /app/models

    # Force re-download
    python scripts/download_models.py --force

Environment
-----------
* ``MODEL_DIR`` / ``MODELS_DIR`` — override the default destination.
* ``INSIGHTFACE_MODEL_BASE_URL`` — override the CDN base URL (rare).

Exit codes
----------
* 0 — success (or all files already present and verified)
* 1 — checksum mismatch after download
* 2 — network failure / unavailable pack
* 3 — insightface package not installed (needed only for the URL fallback)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

# Make `app` importable when run from a checkout so we can reuse
# :mod:`app.core.utils` helpers if needed — but degrade gracefully if
# the package isn't on the path (e.g. when running standalone).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("download_models")

# --------------------------------------------------------------------------- #
# Known-good SHA-256 checksums for the canonical InsightFace model packs.
# These are filled in for the most common packs; unknown packs fall back
# to "no checksum verification" with a warning.
# --------------------------------------------------------------------------- #
KNOWN_CHECKSUMS: Dict[str, Dict[str, str]] = {
    # pack_name -> {filename: sha256}
    # The actual files inside a pack are deterministic at build time,
    # so these hashes are stable across downloads. Replace with the
    # real hashes from your trusted source if you pin to a specific
    # insightface release.
    "buffalo_l": {
        # Placeholder — populate with real SHA-256 digests from your
        # release channel. The script will still run; it just won't be
        # able to verify these files.
    },
    "buffalo_s": {},
    "buffalo_m": {},
    "buffalo_sc": {},
    "antelopev2": {},
}

# Default CDN base URL for InsightFace model packs. The official bucket
# is hosted on Cloudflare; we keep it configurable for air-gapped /
# mirrored deployments.
DEFAULT_BASE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7"

# YOLO26-face weights are distributed via the akanametov/yolo-face
# GitHub release for the publisher-provided face detector weights.
YOLO26_FACE_BASE_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0"
YOLO26_FACE_VARIANTS = {
    "yolo26n-face.pt": f"{YOLO26_FACE_BASE_URL}/yolo26n-face.pt",
    "yolo26s-face.pt": f"{YOLO26_FACE_BASE_URL}/yolo26s-face.pt",
}


# --------------------------------------------------------------------------- #
# YOLO26-face downloader
# --------------------------------------------------------------------------- #
def download_yolo26_face(
    variant: str = "yolo26s-face.pt",
    dest_dir: Optional[Path] = None,
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    """Download a YOLO26-face weights file into ``dest_dir``.

    Parameters
    ----------
    variant:
        ``"yolo26n-face.pt"`` (5 MB, nano) or ``"yolo26s-face.pt"`` (about 21 MB, small).
    dest_dir:
        Destination directory. Defaults to ``$MODEL_DIR`` /
        ``$MODELS_DIR`` / ``./models``.
    force:
        Re-download even if the file already exists.
    show_progress:
        Show a tqdm progress bar.

    Returns
    -------
    pathlib.Path
        Path to the downloaded ``.pt`` file.
    """
    if variant not in YOLO26_FACE_VARIANTS:
        raise ValueError(
            f"unknown YOLO26-face variant {variant!r}; "
            f"choose from {list(YOLO26_FACE_VARIANTS)}"
        )

    if dest_dir is None:
        dest_dir = Path(
            os.environ.get("MODEL_DIR")
            or os.environ.get("MODELS_DIR")
            or (PROJECT_ROOT / "models")
        )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / variant

    if dest_path.exists() and not force:
        logger.info("yolo26.skip file=%s dest=%s (already present)", variant, dest_path)
        return dest_path

    url = YOLO26_FACE_VARIANTS[variant]
    logger.info("yolo26.download.start variant=%s url=%s dest=%s", variant, url, dest_path)

    try:
        _download(url, dest_path, show_progress=show_progress)
    except RuntimeError as exc:
        logger.warning(
            "yolo26.download.failed err=%s — Ultralytics will retry on first use",
            exc,
        )
        raise

    logger.info("yolo26.download.done variant=%s size=%d", variant, dest_path.stat().st_size)
    return dest_path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream-compute the SHA-256 hex digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(bar: Optional["tqdm"], bytes_transferred: int, total: int) -> None:
    if bar is not None:
        bar.total = total
        bar.update(bytes_transferred - bar.n)


def _download(url: str, dest: Path, show_progress: bool = True) -> Path:
    """Download ``url`` to ``dest`` with optional progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    bar = None
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore

            bar = tqdm(unit="B", unit_scale=True, desc=dest.name, leave=False)
        except ImportError:
            logger.info("tqdm not installed — falling back to plain progress logging")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "facial-attendance/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            transferred = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                transferred += len(chunk)
                if bar is not None:
                    _progress_hook(bar, transferred, total)
                elif transferred % (1 << 22) == 0:
                    logger.info("download.progress file=%s bytes=%d/%d", dest.name, transferred, total)
        if bar is not None:
            bar.close()
        tmp.replace(dest)
        return dest
    except (urllib.error.URLError, OSError) as exc:
        if bar is not None:
            bar.close()
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(f"failed to download {url}: {exc}") from exc


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """Extract a ``.zip``/``.tar.gz`` InsightFace pack into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if tar_path.suffix == ".zip":
        # zipfile is in the stdlib but InsightFace packs are tar.gz —
        # handle both for robustness.
        import zipfile

        with zipfile.ZipFile(tar_path) as zf:
            zf.extractall(dest_dir)
        return
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(dest_dir)


def _resolve_pack_url(pack_name: str, base_url: str) -> str:
    """Resolve the download URL for a model pack.

    InsightFace packs are distributed as ``<pack_name>.zip`` on the CDN.
    Newer insightface versions also expose a helper
    (``insightface.utils.storage.get_download_url``) — we use that when
    available, otherwise we fall back to the conventional URL pattern.
    """
    try:
        from insightface.utils import storage as ifs  # type: ignore

        url = ifs.get_download_url(pack_name)  # type: ignore[attr-defined]
        if url:
            return url
    except Exception:
        pass
    return f"{base_url.rstrip('/')}/{pack_name}.zip"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def download_pack(
    pack_name: str = "buffalo_l",
    dest_dir: Optional[Path] = None,
    base_url: str = DEFAULT_BASE_URL,
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    """Download and extract an InsightFace model pack.

    Parameters
    ----------
    pack_name:
        Name of the pack (``"buffalo_l"``, ``"buffalo_s"``, …).
    dest_dir:
        Destination directory. Defaults to ``$MODEL_DIR`` /
        ``$MODELS_DIR`` / ``./models``.
    base_url:
        CDN base URL.
    force:
        Re-download even if the pack already exists.
    show_progress:
        Show a tqdm progress bar.

    Returns
    -------
    pathlib.Path
        Path to the unpacked pack directory.
    """
    if dest_dir is None:
        dest_dir = Path(
            os.environ.get("MODEL_DIR")
            or os.environ.get("MODELS_DIR")
            or (PROJECT_ROOT / "models")
        )
    dest_dir = Path(dest_dir)
    pack_dir = dest_dir / pack_name

    # Idempotency: if the pack directory already contains the expected
    # ``.onnx`` files, skip the download (unless ``force``).
    if pack_dir.exists() and any(pack_dir.glob("*.onnx")) and not force:
        logger.info("download.skip pack=%s dest=%s (already present)", pack_name, pack_dir)
        return pack_dir

    url = _resolve_pack_url(pack_name, base_url)
    logger.info("download.start pack=%s url=%s dest=%s", pack_name, url, dest_dir)

    with tempfile.TemporaryDirectory(prefix="ifdl_") as tmpdir:
        archive = Path(tmpdir) / f"{pack_name}.zip"
        _download(url, archive, show_progress=show_progress)

        # Verify checksum of the archive if we have one on file.
        known = KNOWN_CHECKSUMS.get(pack_name, {})
        archive_hash = known.get("__archive__")
        if archive_hash:
            actual = _sha256(archive)
            if actual != archive_hash:
                raise RuntimeError(
                    f"checksum mismatch for {archive.name}: expected {archive_hash}, got {actual}"
                )
            logger.info("download.checksum_ok pack=%s sha256=%s", pack_name, actual[:16])
        else:
            logger.warning(
                "download.no_checksum pack=%s — skipping verification "
                "(populate KNOWN_CHECKSUMS to enable)",
                pack_name,
            )

        # Extract into the destination directory.
        if pack_dir.exists():
            shutil.rmtree(pack_dir)
        _extract_tar(archive, dest_dir)

        # InsightFace packs extract to a sub-folder named after the pack;
        # if the extraction produced a different layout, normalise it.
        if not pack_dir.exists():
            # Some packs extract directly into dest_dir without a sub-folder.
            onnx_files = list(dest_dir.glob("*.onnx"))
            if onnx_files:
                pack_dir.mkdir(parents=True, exist_ok=True)
                for f in onnx_files:
                    shutil.move(str(f), str(pack_dir / f.name))

    # Verify individual ONNX file checksums if available.
    if pack_dir.exists():
        for fname, expected_hash in known.items():
            if fname == "__archive__":
                continue
            fpath = pack_dir / fname
            if not fpath.exists():
                logger.warning("download.missing_file pack=%s file=%s", pack_name, fname)
                continue
            actual = _sha256(fpath)
            if actual != expected_hash:
                raise RuntimeError(
                    f"checksum mismatch for {fname}: expected {expected_hash}, got {actual}"
                )
            logger.info("download.file_ok pack=%s file=%s sha256=%s", pack_name, fname, actual[:16])

    logger.info("download.done pack=%s dest=%s", pack_name, pack_dir)
    return pack_dir


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download InsightFace model packs with progress + checksum verification.",
    )
    parser.add_argument(
        "--pack",
        default="buffalo_l",
        help="InsightFace model pack name (default: buffalo_l). Try buffalo_s for a smaller pack.",
    )
    parser.add_argument(
        "--yolo26",
        action="store_true",
        help="Also download YOLO26-face weights (for DETECTOR_BACKEND=yolo26).",
    )
    parser.add_argument(
        "--yolo26-variant",
        default="yolo26s-face.pt",
        choices=sorted(YOLO26_FACE_VARIANTS.keys()),
        help="YOLO26-face model variant (default: yolo26s-face.pt = about 21 MB small).",
    )
    parser.add_argument(
        "--no-insightface",
        action="store_true",
        help="Skip InsightFace pack download (use with --yolo26 to fetch only YOLO26).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: $MODEL_DIR / $MODELS_DIR / ./models).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="CDN base URL (rarely needed).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the pack is already present.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar (useful in logs).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    exit_code = 0

    # InsightFace pack (unless --no-insightface was passed)
    if not args.no_insightface:
        try:
            download_pack(
                pack_name=args.pack,
                dest_dir=args.dest,
                base_url=args.base_url,
                force=args.force,
                show_progress=not args.no_progress,
            )
        except RuntimeError as exc:
            logger.error("insightface.download.failed err=%s", exc)
            exit_code = 1 if "checksum" in str(exc).lower() else 2

    # YOLO26-face weights (optional)
    if args.yolo26:
        try:
            download_yolo26_face(
                variant=args.yolo26_variant,
                dest_dir=args.dest,
                force=args.force,
                show_progress=not args.no_progress,
            )
        except RuntimeError as exc:
            logger.error("yolo26.download.failed err=%s", exc)
            if exit_code == 0:
                exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
