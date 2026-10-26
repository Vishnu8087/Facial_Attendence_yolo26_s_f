#!/usr/bin/env python3
"""CLI for enrolling an employee from a face image on disk.

Bypasses HTTP — calls :class:`EnrollmentService` directly. Useful for:

* Bulk-importing employees from a folder.
* Bootstrapping a fresh install before the API is up.
* CI / scripted setups.

Usage::

    python scripts/enroll_employee.py \\
        --name "John Doe" \\
        --employee-id EMP001 \\
        --department Engineering \\
        --email john@example.com \\
        --image /path/to/face.jpg

    # Or with a pre-computed embedding (512-d float32 saved as .npy):
    python scripts/enroll_employee.py \\
        --name "John Doe" \\
        --employee-id EMP001 \\
        --embedding /path/to/embedding.npy

Environment: reads ``DATABASE_URL`` (and other app config) via :mod:`app.config`.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make `app` importable when run from a checkout.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.db.session import dispose_engine, get_session  # noqa: E402
from app.services.enrollment import EnrollmentError, EnrollmentService  # noqa: E402
from app.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


async def enroll(
    *,
    name: str,
    employee_id: str,
    department: str | None,
    email: str | None,
    image_path: Path | None,
    embedding_path: Path | None,
) -> int:
    configure_logging()
    settings.ensure_dirs()
    service = EnrollmentService()

    image_bytes: bytes | None = None
    embedding: np.ndarray | None = None
    if image_path is not None:
        if not image_path.exists():
            logger.error("enroll_cli.image.missing", path=str(image_path))
            return 2
        image_bytes = image_path.read_bytes()
        logger.info("enroll_cli.image.loaded", path=str(image_path), bytes=len(image_bytes))
    elif embedding_path is not None:
        if not embedding_path.exists():
            logger.error("enroll_cli.embedding.missing", path=str(embedding_path))
            return 2
        embedding = np.load(embedding_path).astype(np.float32).reshape(-1)
        if embedding.shape[0] != 512:
            logger.error("enroll_cli.embedding.bad_shape", shape=embedding.shape)
            return 2
        logger.info("enroll_cli.embedding.loaded", path=str(embedding_path), dim=embedding.shape[0])
    else:
        logger.error("enroll_cli.no_image_or_embedding")
        return 2

    try:
        async with get_session() as session:
            result = await service.enroll_from_image(
                session,
                employee_id=employee_id,
                name=name,
                department=department,
                email=email,
                image_bytes=image_bytes,
                embedding=embedding,
                save_face=True,
            )
    except EnrollmentError as exc:
        logger.error("enroll_cli.enrollment_error", error=str(exc))
        return 1
    except Exception as exc:
        logger.exception("enroll_cli.unexpected_error", error=str(exc))
        return 1
    finally:
        await dispose_engine()

    emp = result.employee
    print(
        f"Enrolled: id={emp.id} employee_id={emp.employee_id} name={emp.name} "
        f"department={emp.department or '-'} embedding_norm={result.embedding_norm:.4f} "
        f"faces_detected={result.num_faces_detected} face_image={result.face_image_path or '-'}"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enroll an employee from a face image or pre-computed embedding."
    )
    parser.add_argument("--name", required=True, help="Display name.")
    parser.add_argument("--employee-id", required=True, help="Unique business ID (e.g. EMP001).")
    parser.add_argument("--department", default=None, help="Department (free-form).")
    parser.add_argument("--email", default=None, help="Email address.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--image", type=Path, help="Path to a face image (JPEG/PNG).")
    grp.add_argument(
        "--embedding",
        type=Path,
        help="Path to a .npy file containing a 512-d float32 embedding.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rc = asyncio.run(
        enroll(
            name=args.name,
            employee_id=args.employee_id,
            department=args.department,
            email=args.email,
            image_path=args.image,
            embedding_path=args.embedding,
        )
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
