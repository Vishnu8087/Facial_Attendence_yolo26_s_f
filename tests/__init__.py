"""Test suite for the Facial Attendance & Motion Tracking System.

Layout
------
``tests/unit/``
    Pure unit tests against ``app.core`` and ``app.db.crud`` — heavy ML
    models are mocked; no GPU or model files required.

``tests/integration/``
    End-to-end style tests against the FastAPI app via ``httpx.AsyncClient``
    and the WebSocket endpoint at ``/ws/stream``. Uses an in-memory SQLite
    database (or testcontainers Postgres+pgvector when available).

Run with::

    pytest tests/ -v --cov=app --cov-report=html
"""
