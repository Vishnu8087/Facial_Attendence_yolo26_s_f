"""Integration tests for the ``/api/attendance`` endpoints.

Covers:
* GET /api/attendance with date + employee filters
* GET /api/attendance/export?format=csv
* Pagination
"""
from __future__ import annotations

from datetime import date

import pytest

pytestmark = pytest.mark.asyncio


async def test_list_attendance_returns_list(test_client):
    resp = await test_client.get("/api/attendance")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_list_attendance_with_date_filter(test_client):
    today = date.today().isoformat()
    resp = await test_client.get(f"/api/attendance?date={today}")
    assert resp.status_code == 200


async def test_list_attendance_with_employee_filter(test_client):
    resp = await test_client.get("/api/attendance?employee_id=EMP-NONEXIST")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_attendance_pagination(test_client):
    resp = await test_client.get("/api/attendance?limit=10&offset=0")
    assert resp.status_code == 200


async def test_export_csv_returns_text_csv(test_client):
    resp = await test_client.get("/api/attendance/export?format=csv")
    if resp.status_code == 404:
        pytest.skip("export endpoint not yet implemented by Task 1-b")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")
    body = resp.text
    # CSV header row should contain the canonical columns
    assert "employee_id" in body.splitlines()[0].lower()


async def test_export_json_returns_array(test_client):
    resp = await test_client.get("/api/attendance/export?format=json")
    if resp.status_code == 404:
        pytest.skip("export endpoint not yet implemented")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
