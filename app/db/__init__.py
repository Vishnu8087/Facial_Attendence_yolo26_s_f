"""Database sub-package.

Exposes:

* :data:`Base` — declarative SQLAlchemy base.
* :func:`async_engine`, :func:`get_session`, :func:`get_db` — async engine
  and session dependency.
* :mod:`app.db.models` — SQLAlchemy ORM models.
* :mod:`app.db.crud` — async CRUD operations.
* :mod:`app.db.migrations` — Alembic environment.
"""
from __future__ import annotations

from app.db.base import Base
from app.db.session import (
    async_engine,
    async_session_factory,
    create_all,
    dispose_engine,
    get_db,
    get_session,
)

__all__ = [
    "Base",
    "async_engine",
    "async_session_factory",
    "create_all",
    "dispose_engine",
    "get_db",
    "get_session",
]
