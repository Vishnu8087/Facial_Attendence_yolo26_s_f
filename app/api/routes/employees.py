"""Employee CRUD endpoints.

``POST   /api/employees``            — enroll (multipart images or JSON embedding)
``GET    /api/employees``            — paginated listing
``GET    /api/employees/{id}``       — fetch by UUID
``DELETE /api/employees/{id}``       — soft-delete by UUID
``GET    /api/employees/{id}/face``  — stream the stored face image
"""
from __future__ import annotations

import json
import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    status,
)
from fastapi.responses import FileResponse, Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import UploadFile

from app.api.deps import get_db_session, get_enrollment_service
from app.config import settings
from app.schemas.common import ErrorResponse
from app.schemas.employee import (
    EmployeeCreateJSON,
    EmployeeListResponse,
    EmployeeResponse,
    decode_embedding_b64,
)
from app.services.enrollment import EnrollmentError, EnrollmentService
from app.utils.logging import get_logger
from app.utils.security import User, get_current_user, require_admin

logger = get_logger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _to_response(session: AsyncSession, emp) -> EmployeeResponse:
    """Build an EmployeeResponse, computing the face-image URL."""
    face_url = None
    if emp.face_image_path:
        face_url = f"/api/employees/{emp.id}/face"
    return EmployeeResponse(
        id=emp.id,
        employee_id=emp.employee_id,
        name=emp.name,
        department=emp.department,
        email=emp.email,
        active=emp.active,
        face_image_url=face_url,
        embedding_model=emp.embedding_model,
        created_at=emp.created_at,
        updated_at=emp.updated_at,
    )


# --------------------------------------------------------------------------- #
# Create
# --------------------------------------------------------------------------- #
@router.post(
    "",
    response_model=EmployeeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a new employee",
    description=(
        "Enroll a new employee. Accepts either:\n\n"
        "- **multipart/form-data** with one or more repeated `image` files + `employee_id`, `name`, "
        "`department?`, `email?` form fields — the backend extracts the embedding; OR\n"
        "- **application/json** with `employee_id`, `name`, and either "
        "`image_base64` or `embedding_base64`."
    ),
    responses={
        201: {"description": "Created"},
        409: {"model": ErrorResponse, "description": "employee_id already exists"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        400: {"model": ErrorResponse, "description": "No face detected in image"},
        503: {"model": ErrorResponse, "description": "CV models not loaded"},
    },
)
async def create_employee(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    enrollment_service: EnrollmentService = Depends(get_enrollment_service),
    _admin: User = Depends(require_admin),
) -> EmployeeResponse:
    """Enroll endpoint — supports multipart or JSON.

    FastAPI doesn't allow mixing ``Form`` parameters with a JSON ``Body``
    on the same signature, so we sniff ``Content-Type`` and parse the
    body ourselves. Both paths funnel into :meth:`EnrollmentService.enroll_from_image`.
    """
    content_type = (request.headers.get("content-type") or "").lower()
    employee_id: str | None = None
    name: str | None = None
    department: str | None = None
    email: str | None = None
    image_bytes: bytes | None = None
    image_bytes_list: list[bytes] | None = None
    embedding: "object | None" = None

    if "application/json" in content_type:
        # ---- JSON body path ---- #
        try:
            raw = await request.body()
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid JSON body: {exc}",
            ) from exc
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=422,
                detail="JSON body must be an object.",
            )
        # Validate via the Pydantic schema (raises 422 on missing fields).
        try:
            parsed = EmployeeCreateJSON(**body)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        employee_id = parsed.employee_id
        name = parsed.name
        department = parsed.department
        email = parsed.email
        if parsed.embedding_base64:
            try:
                embedding = decode_embedding_b64(parsed.embedding_base64)
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid embedding_base64: {exc}",
                ) from exc
        elif parsed.image_base64:
            import base64 as _b64

            try:
                image_bytes = _b64.b64decode(parsed.image_base64)
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid image_base64: {exc}",
                ) from exc
        else:
            raise HTTPException(
                status_code=422,
                detail="Either image (multipart), image_base64, or embedding_base64 must be provided.",
            )
    else:
        # ---- multipart/form-data path ---- #
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid form data: {exc}",
            ) from exc
        employee_id = (form.get("employee_id") or "").strip() or None
        name = (form.get("name") or "").strip() or None
        department = (form.get("department") or "").strip() or None
        email = (form.get("email") or "").strip() or None
        if not employee_id or not name:
            raise HTTPException(
                status_code=422,
                detail="employee_id and name are required",
            )
        uploads = form.getlist("image")
        if not uploads or any(not isinstance(upload, UploadFile) for upload in uploads):
            raise HTTPException(
                status_code=422,
                detail="At least one 'image' file is required for multipart enrollment.",
            )
        max_bytes = settings.max_upload_mb * 1024 * 1024
        image_bytes_list = [await upload.read() for upload in uploads]
        if sum(map(len, image_bytes_list)) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Combined images exceed the {settings.max_upload_mb}MB upload limit",
            )

    # Reject image enrollment when the CV models are unavailable. The
    # pre-computed embedding path does not require the CV pipeline.
    if embedding is None and not getattr(request.app.state, "models_loaded", True):
        logger.warning("employees.enroll.cv_models_not_loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face processing models are unavailable. Check the CV dependencies and model setup.",
        )

    try:
        result = await enrollment_service.enroll_from_image(
            session,
            employee_id=employee_id,
            name=name,
            department=department,
            email=email,
            image_bytes=image_bytes,
            image_bytes_list=image_bytes_list,
            embedding=embedding,
            save_face=True,
        )
    except EnrollmentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"employee_id {employee_id!r} already exists",
        ) from exc

    await session.commit()
    return await _to_response(session, result.employee)


# --------------------------------------------------------------------------- #
# List
# --------------------------------------------------------------------------- #
@router.get(
    "",
    response_model=list[EmployeeResponse],
    summary="List enrolled employees (paginated)",
    description="Returns up to `limit` employees starting at `offset`. "
    "Pass `department` or `q` (prefix match on name/employee_id) to filter. "
    "The response is a JSON array (the dashboard's api.js normalises either "
    "an array or `{items: [...]}` envelope — we return the array form for "
    "compatibility with the test suite).",
)
async def list_employees(
    session: AsyncSession = Depends(get_db_session),
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    department: Annotated[str | None, Query(max_length=128)] = None,
    q: Annotated[str | None, Query(max_length=128)] = None,
    include_inactive: bool = Query(False),
    _user: User = Depends(get_current_user),
) -> list[EmployeeResponse]:
    """List endpoint — returns a JSON array."""
    from app.db import crud

    items = await crud.list_employees(
        session,
        limit=limit,
        offset=offset,
        department=department,
        q=q,
        include_inactive=include_inactive,
    )
    return [await _to_response(session, e) for e in items]


# --------------------------------------------------------------------------- #
# Get by ID
# --------------------------------------------------------------------------- #
@router.get(
    "/{employee_pk}",
    response_model=EmployeeResponse,
    summary="Fetch an employee by UUID",
    responses={404: {"model": ErrorResponse, "description": "Not found"}},
)
async def get_employee(
    employee_pk: Annotated[uuid.UUID, Path()],
    session: AsyncSession = Depends(get_db_session),
    _user: User = Depends(get_current_user),  # noqa: ANN001
) -> EmployeeResponse:
    """Fetch by primary-key UUID."""
    from app.db import crud

    emp = await crud.get_employee(session, employee_pk)
    if emp is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Employee not found")
    return await _to_response(session, emp)


# --------------------------------------------------------------------------- #
# Delete
# --------------------------------------------------------------------------- #
@router.delete(
    "/{employee_pk}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
    summary="Delete an employee by UUID",
    description="Soft-deletes by default (sets active=False). Pass `?hard=true` for a real DELETE.",
    responses={
        204: {"description": "Deleted"},
        404: {"model": ErrorResponse, "description": "Not found"},
    },
)
async def delete_employee(
    employee_pk: Annotated[uuid.UUID, Path()],
    hard: bool = Query(False, description="Perform a hard DELETE (cascades to attendance logs)."),
    session: AsyncSession = Depends(get_db_session),
    _admin: User = Depends(require_admin),
) -> None:
    """Soft-delete (or hard-delete) an employee."""
    from app.db import crud

    ok = await crud.delete_employee(session, employee_pk, hard=hard)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Employee not found")
    await session.commit()


# --------------------------------------------------------------------------- #
# Face image
# --------------------------------------------------------------------------- #
@router.get(
    "/{employee_pk}/face",
    summary="Stream the stored face image",
    response_class=FileResponse,
    responses={
        200: {"content": {"image/jpeg": {}}},
        404: {"model": ErrorResponse, "description": "Not found"},
    },
)
async def get_employee_face(
    employee_pk: Annotated[uuid.UUID, Path()],
    session: AsyncSession = Depends(get_db_session),
):
    """Return the JPEG face crop stored at enroll time."""
    from app.db import crud

    emp = await crud.get_employee(session, employee_pk)
    if emp is None or not emp.face_image_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No face image stored")

    from pathlib import Path

    p = Path(emp.face_image_path)
    if not p.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Face image file missing")
    return FileResponse(str(p), media_type="image/jpeg")


__all__ = ["router"]
