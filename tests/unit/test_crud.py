"""Unit tests for ``app.db.crud``.

These tests run against an in-memory SQLite database (via the
``db_session`` fixture), exercising the byte-backed cosine fallback.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

crud = pytest.importorskip("app.db.crud", reason="Task 1-b not yet merged")  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Create employee
# --------------------------------------------------------------------------- #
async def test_create_employee_returns_object_with_id(db_session):
    emp = await crud.create_employee(
        db_session,
        employee_id="EMP001",
        name="Alice Alpha",
        department="Engineering",
        embedding=np.zeros(512, dtype=np.float32),
    )
    assert emp is not None
    assert emp.id is not None
    assert emp.employee_id == "EMP001"


async def test_create_employee_duplicate_employee_id_raises(db_session):
    await crud.create_employee(
        db_session,
        employee_id="EMP002",
        name="Bob Beta",
        department="Engineering",
        embedding=np.zeros(512, dtype=np.float32),
    )
    with pytest.raises(Exception):  # IntegrityError or custom
        await crud.create_employee(
            db_session,
            employee_id="EMP002",
            name="Bob Beta Clone",
            department="Engineering",
            embedding=np.zeros(512, dtype=np.float32),
        )


# --------------------------------------------------------------------------- #
# Read / list
# --------------------------------------------------------------------------- #
async def test_get_employee_by_employee_id(db_session):
    await crud.create_employee(
        db_session,
        employee_id="EMP003",
        name="Carol Gamma",
        department="HR",
        embedding=np.zeros(512, dtype=np.float32),
    )
    fetched = await crud.get_employee_by_employee_id(db_session, "EMP003")
    assert fetched is not None
    assert fetched.name == "Carol Gamma"


async def test_list_employees_paginated(db_session):
    for i in range(5):
        await crud.create_employee(
            db_session,
            employee_id=f"EMP{i:03d}",
            name=f"User {i}",
            department="Engineering",
            embedding=np.zeros(512, dtype=np.float32),
        )
    items = await crud.list_employees(db_session, limit=3, offset=0)
    assert len(items) == 3


# --------------------------------------------------------------------------- #
# Delete
# --------------------------------------------------------------------------- #
async def test_delete_employee(db_session):
    emp = await crud.create_employee(
        db_session,
        employee_id="EMP_DEL",
        name="Delta Delete",
        department="Engineering",
        embedding=np.zeros(512, dtype=np.float32),
    )
    await crud.delete_employee(db_session, emp.id)
    fetched = await crud.get_employee_by_employee_id(db_session, "EMP_DEL")
    assert fetched is None


# --------------------------------------------------------------------------- #
# find_nearest_employee — pgvector or byte-backed cosine fallback.
# --------------------------------------------------------------------------- #
async def test_find_nearest_employee_returns_closest_match(db_session):
    """Inserts two employees with known embeddings and verifies the nearest
    neighbour lookup returns the closest one by cosine similarity."""
    rng = np.random.default_rng(0)
    e1 = rng.standard_normal(512).astype(np.float32)
    e1 /= np.linalg.norm(e1)
    e2 = rng.standard_normal(512).astype(np.float32)
    e2 /= np.linalg.norm(e2)

    await crud.create_employee(
        db_session, employee_id="EMP_A", name="A", department="X", embedding=e1
    )
    await crud.create_employee(
        db_session, employee_id="EMP_B", name="B", department="X", embedding=e2
    )

    nearest, score = await crud.find_nearest_employee(db_session, e1, threshold=0.5)
    assert nearest is not None
    assert nearest.employee_id == "EMP_A"
    assert score >= 0.99


async def test_find_nearest_employee_respects_threshold(db_session):
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    other = np.zeros(512, dtype=np.float32)
    other[1] = 1.0
    await crud.create_employee(
        db_session, employee_id="EMP_C", name="C", embedding=other
    )

    nearest, score = await crud.find_nearest_employee(
        db_session, embedding, threshold=0.5
    )
    assert nearest is None
    assert score == 0.0


# --------------------------------------------------------------------------- #
# Cooldown query
# --------------------------------------------------------------------------- #
async def test_cooldown_query_blocks_recent_log(db_session):
    """If an attendance log exists for an employee within the cooldown
    window, the cooldown query must report it as 'recently logged'."""
    emp = await crud.create_employee(
        db_session,
        employee_id="EMP_CD",
        name="Cooldown Tester",
        department="Engineering",
        embedding=np.zeros(512, dtype=np.float32),
    )
    now = datetime.now(timezone.utc)
    if hasattr(crud, "log_attendance"):
        await crud.log_attendance(db_session, employee_id=emp.id, timestamp=now, strategy="both")
    if hasattr(crud, "recently_logged"):
        recent = await crud.recently_logged(db_session, employee_id=emp.id, cooldown_seconds=30)
        assert recent is True
    else:
        pytest.skip("crud.recently_logged not implemented yet")
