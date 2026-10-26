"""SQLAlchemy declarative base.

A single ``Base`` is shared by every model in :mod:`app.db.models` so
that ``Base.metadata.create_all`` registers all tables at once. Alembic
also references this object (see ``app/db/migrations/env.py``).
"""
from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase, declared_attr


class Base(DeclarativeBase):
    """Project-wide declarative base."""

    # noqa: EAM - just metadata nicety
    @declared_attr.directive
    def __tablename__(cls) -> str:  # type: ignore[override]
        """Default table name = lowercased class name + 's'."""
        return cls.__name__.lower() + "s"


__all__ = ["Base"]
