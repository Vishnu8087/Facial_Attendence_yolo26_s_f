"""Employee-related schemas (Pydantic v2)."""
from __future__ import annotations

import base64
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------- #
# Create
# --------------------------------------------------------------------------- #
class EmployeeCreate(BaseModel):
    """Multipart enroll payload.

    Used by FastAPI's ``Form()`` parser — fields map 1:1 to form data.
    The ``image`` file is handled separately as an ``UploadFile``.
    """

    employee_id: str = Field(..., min_length=1, max_length=64, description="Unique business ID.")
    name: str = Field(..., min_length=1, max_length=255)
    department: str | None = Field(None, max_length=128)
    email: str | None = Field(None, max_length=255)

    model_config = ConfigDict(extra="forbid")


class EmployeeCreateJSON(BaseModel):
    """JSON enroll payload — accepts a pre-computed base64 embedding."""

    employee_id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=255)
    department: str | None = Field(None, max_length=128)
    email: str | None = Field(None, max_length=255)

    # Either an image (base64) or an embedding (base64 float32[512]).
    image_base64: str | None = Field(None, description="JPEG/PNG image, base64-encoded.")
    embedding_base64: str | None = Field(
        None,
        description="512-d L2-normalised embedding, float32 little-endian, base64-encoded.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("employee_id")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #
class EmployeeResponse(BaseModel):
    """Single-employee response shape."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    employee_id: str
    name: str
    department: str | None = None
    email: str | None = None
    active: bool = True
    face_image_url: str | None = Field(
        None, description="Relative URL to the stored face crop, if any."
    )
    embedding_model: str | None = None
    created_at: datetime
    updated_at: datetime


class EmployeeListResponse(BaseModel):
    """Paginated employee listing."""

    items: list[EmployeeResponse]
    total: int
    limit: int
    offset: int


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def decode_embedding_b64(b64: str) -> "Any":
    """Decode a base64-encoded 512-d float32 embedding.

    Returns a ``numpy.ndarray`` of shape ``(512,)`` dtype ``float32``.
    """
    import numpy as np

    raw = base64.b64decode(b64)
    if len(raw) != 512 * 4:
        raise ValueError(
            f"Embedding must be 512 float32 values ({512*4} bytes), got {len(raw)} bytes"
        )
    return np.frombuffer(raw, dtype=np.float32).copy()


__all__ = [
    "EmployeeCreate",
    "EmployeeCreateJSON",
    "EmployeeResponse",
    "EmployeeListResponse",
    "decode_embedding_b64",
]
