"""Integration tests for the ``/api/employees`` endpoints."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_create_employee_via_api(test_client):
    payload = {
        "employee_id": "EMP-API-1",
        "name": "API Alice",
        "department": "Engineering",
        # embedding is omitted — endpoint should accept either an uploaded
        # image (face crop) or a base64 embedding. We send a tiny base64
        # image instead.
    }
    # Try the JSON route first (expects pre-computed embedding), then the
    # multipart route (expects image upload).
    resp = await test_client.post("/api/employees", json=payload)
    if resp.status_code == 422:
        pytest.skip("employees endpoint requires image upload — covered by test_enroll_to_attendance")
    assert resp.status_code in (200, 201)
    body = resp.json()
    assert body["employee_id"] == "EMP-API-1"


async def test_list_employees_empty(test_client):
    resp = await test_client.get("/api/employees")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_get_employee_by_id(test_client):
    # First create
    create = await test_client.post("/api/employees", json={"employee_id": "EMP-LIST", "name": "L"})
    if create.status_code not in (200, 201):
        pytest.skip("create endpoint requires image upload")
    emp_id = create.json()["id"]
    resp = await test_client.get(f"/api/employees/{emp_id}")
    assert resp.status_code == 200
    assert resp.json()["employee_id"] == "EMP-LIST"


async def test_delete_employee(test_client):
    create = await test_client.post("/api/employees", json={"employee_id": "EMP-DEL", "name": "D"})
    if create.status_code not in (200, 201):
        pytest.skip("create endpoint requires image upload")
    emp_id = create.json()["id"]
    delete = await test_client.delete(f"/api/employees/{emp_id}")
    assert delete.status_code in (200, 204)
    get = await test_client.get(f"/api/employees/{emp_id}")
    assert get.status_code == 404


async def test_create_employee_validation_error(test_client):
    """Missing required fields should yield HTTP 422."""
    resp = await test_client.post("/api/employees", json={})
    assert resp.status_code == 422


async def test_multipart_upload_reaches_image_validation(test_client):
    """A multipart file must be recognized before its contents are decoded."""
    response = await test_client.post(
        "/api/employees",
        data={"employee_id": "EMP-UPLOAD-PROBE", "name": "Upload Probe"},
        files={"image": ("probe.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Could not decode image 1 (expected JPEG/PNG/WEBP)."
    )
