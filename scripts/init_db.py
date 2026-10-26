#!/usr/bin/env python3
"""Initialise the database for the Facial Attendance System.

Idempotent. Safe to run multiple times.

Steps:

1. Connect to the Postgres ``postgres`` maintenance database.
2. ``CREATE DATABASE`` if it doesn't exist.
3. Connect to the new database; ``CREATE EXTENSION vector`` (and friends).
4. Run Alembic migrations to ``head`` (creates tables + indexes).
5. Optionally seed demo employees.

Usage::

    python scripts/init_db.py
    python scripts/init_db.py --seed
    python scripts/init_db.py --database-url postgresql://user:pass@host:5432/attendance
    python scripts/init_db.py --skip-migrations

Environment variables (read via :mod:`app.config`):

* ``DATABASE_URL`` — full async URL (``postgresql+asyncpg://...``)
* ``DB_HOST``, ``DB_PORT``, ``DB_NAME``, ``DB_USER``, ``DB_PASSWORD``
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# Make `app` importable when run from a checkout.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _split_db_url(url: str) -> tuple[str, str, str, int, str, str]:
    """Split an asyncpg URL into (sync_url, host, db_name, port, user, pass)."""
    from urllib.parse import urlparse

    parsed = urlparse(url.replace("postgresql+asyncpg://", "postgresql://"))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    user = parsed.username or "attendance"
    password = parsed.password or ""
    db_name = parsed.path.lstrip("/") or "attendance"
    sync_url = f"postgresql://{user}:{password}@{host}:{port}"
    return sync_url, host, db_name, port, user, password


async def _create_database_if_missing(database_url: str) -> None:
    """Connect to the maintenance DB and CREATE DATABASE if needed."""
    sync_url, host, db_name, port, user, password = _split_db_url(database_url)
    maintenance_url = f"postgresql://{user}:{password}@{host}:{port}/postgres"

    try:
        import asyncpg  # type: ignore
    except ImportError:
        logger.error("init_db.asyncpg_missing")
        return

    conn = await asyncpg.connect(maintenance_url)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        if exists:
            logger.info("init_db.database.exists", db_name=db_name)
            return
        await conn.execute(f'CREATE DATABASE "{db_name}"')
        logger.info("init_db.database.created", db_name=db_name)
    finally:
        await conn.close()


async def _create_extensions(database_url: str) -> None:
    """CREATE EXTENSION vector / uuid-ossp / pg_trgm."""
    try:
        import asyncpg  # type: ignore
    except ImportError:
        logger.warning("init_db.extensions.skipped_no_asyncpg")
        return

    conn = await asyncpg.connect(database_url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        for ext in ("vector", "uuid-ossp", "pg_trgm"):
            try:
                await conn.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}"')
                logger.info("init_db.extension.created", ext=ext)
            except Exception as exc:
                logger.warning("init_db.extension.failed", ext=ext, error=str(exc))
    finally:
        await conn.close()


async def _run_migrations() -> None:
    """Run ``alembic upgrade head``."""
    import subprocess

    ini = settings.alembic_ini
    if not ini.exists():
        logger.warning("init_db.alembic_ini.missing", path=str(ini))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        ["alembic", "-c", str(ini), "upgrade", "head"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("init_db.migrations.applied")
    else:
        logger.error(
            "init_db.migrations.failed",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


async def _seed_demo_employees() -> None:
    """Insert a handful of demo employees with zero embeddings."""
    from app.db import crud
    from app.db.session import get_session

    demos = [
        ("EMP001", "Alice Alpha", "Engineering", "alice@example.com"),
        ("EMP002", "Bob Beta", "Engineering", "bob@example.com"),
        ("EMP003", "Carol Gamma", "HR", "carol@example.com"),
        ("EMP004", "Dave Delta", "Sales", "dave@example.com"),
    ]
    async with get_session() as session:
        for emp_id, name, dept, email in demos:
            existing = await crud.get_employee_by_employee_id(session, emp_id)
            if existing is not None:
                logger.info("init_db.seed.exists", employee_id=emp_id)
                continue
            await crud.create_employee(
                session,
                employee_id=emp_id,
                name=name,
                department=dept,
                email=email,
                embedding=None,
                embedding_model="seed",
            )
            logger.info("init_db.seed.inserted", employee_id=emp_id, name=name)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main(
    *,
    database_url: str | None = None,
    seed: bool = False,
    skip_migrations: bool = False,
) -> int:
    configure_logging()
    url = database_url or settings.database_url
    logger.info("init_db.start", url=url.replace(settings.db_password, "***") if settings.db_password else url)

    # 1. Create database (idempotent)
    await _create_database_if_missing(url)

    # 2. Extensions
    await _create_extensions(url)

    # 3. Migrations
    if not skip_migrations:
        await _run_migrations()

    # 4. Seed (optional)
    if seed:
        await _seed_demo_employees()

    logger.info("init_db.complete")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialise the facial-attendance database.")
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override DATABASE_URL (postgresql+asyncpg://...).",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Insert demo employees (EMP001..EMP004) with null embeddings.",
    )
    parser.add_argument(
        "--skip-migrations",
        action="store_true",
        help="Skip `alembic upgrade head` (useful when migrations are run separately).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rc = asyncio.run(
        main(
            database_url=args.database_url,
            seed=args.seed,
            skip_migrations=args.skip_migrations,
        )
    )
    sys.exit(rc)
