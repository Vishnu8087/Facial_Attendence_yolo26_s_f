"""initial schema — employees + attendance_logs with optional pgvector

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000

Creates:

* ``vector`` extension (pgvector), when ``USE_PGVECTOR=true``.
* ``employees`` table — enrolled employees with 512-d embedding column.
* ``attendance_logs`` table — deduplicated attendance events.
* Composite index on ``(employee_id, detected_at)`` for cooldown queries.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.config import settings

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # Extensions
    # ------------------------------------------------------------------ #
    if settings.use_pgvector:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    # ------------------------------------------------------------------ #
    # employees
    # ------------------------------------------------------------------ #
    op.create_table(
        "employees",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("employee_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("department", sa.String(128), nullable=True),
        sa.Column("email", sa.String(255), nullable=True),
        # LargeBinary is BYTEA in PostgreSQL; enabled pgvector installations
        # convert this column below.
        sa.Column("embedding", sa.LargeBinary, nullable=True),
        sa.Column("face_image_path", sa.Text, nullable=True),
        sa.Column("embedding_model", sa.String(64), nullable=True),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("employee_id", name="uq_employees_employee_id"),
    )
    if settings.use_pgvector:
        op.execute("ALTER TABLE employees ALTER COLUMN embedding TYPE vector(512) USING NULL;")

    op.create_index("ix_employees_employee_id", "employees", ["employee_id"])
    op.create_index("ix_employees_name", "employees", ["name"])
    op.create_index("ix_employees_department", "employees", ["department"])
    op.create_index("ix_employees_active", "employees", ["active"])
    if settings.use_pgvector:
        # IVFFlat index for fast cosine similarity search.
        op.execute(
            "CREATE INDEX ix_employees_embedding_cosine "
            "ON employees USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        )

    # ------------------------------------------------------------------ #
    # attendance_logs
    # ------------------------------------------------------------------ #
    op.create_table(
        "attendance_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "employee_pk",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("employees.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("employee_id", sa.String(64), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("similarity", sa.Float, nullable=True),
        sa.Column("track_id", sa.Integer, nullable=True),
        sa.Column("camera_id", sa.String(64), nullable=True),
        sa.Column("strategy", sa.String(32), nullable=False, server_default="both"),
        sa.Column("liveness_passed", sa.Boolean, nullable=True),
        sa.Column("liveness_confidence", sa.Float, nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
    )
    op.create_index("ix_attendance_logs_employee_pk", "attendance_logs", ["employee_pk"])
    op.create_index("ix_attendance_logs_employee_id", "attendance_logs", ["employee_id"])
    op.create_index("ix_attendance_logs_detected_at", "attendance_logs", ["detected_at"])
    op.create_index("ix_attendance_logs_camera_id", "attendance_logs", ["camera_id"])
    op.create_index(
        "ix_attendance_logs_employee_detected",
        "attendance_logs",
        ["employee_id", "detected_at"],
    )
    op.create_index(
        "ix_attendance_logs_detected_at_strategy",
        "attendance_logs",
        ["detected_at", "strategy"],
    )


def downgrade() -> None:
    op.drop_index("ix_attendance_logs_detected_at_strategy", table_name="attendance_logs")
    op.drop_index("ix_attendance_logs_employee_detected", table_name="attendance_logs")
    op.drop_index("ix_attendance_logs_camera_id", table_name="attendance_logs")
    op.drop_index("ix_attendance_logs_detected_at", table_name="attendance_logs")
    op.drop_index("ix_attendance_logs_employee_id", table_name="attendance_logs")
    op.drop_index("ix_attendance_logs_employee_pk", table_name="attendance_logs")
    op.drop_table("attendance_logs")

    op.execute("DROP INDEX IF EXISTS ix_employees_embedding_cosine;")
    op.drop_index("ix_employees_active", table_name="employees")
    op.drop_index("ix_employees_department", table_name="employees")
    op.drop_index("ix_employees_name", table_name="employees")
    op.drop_index("ix_employees_employee_id", table_name="employees")
    op.drop_table("employees")

    # Leave the extensions in place — they may be shared with other apps.
