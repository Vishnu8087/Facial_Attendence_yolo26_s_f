"""Facial Attendance & Motion Tracking System — backend application package.

This package contains the FastAPI backend, database layer, schemas,
services, and utilities. The CV pipeline lives under :mod:`app.core`
(owned by Task 1-a) and is imported lazily so that the rest of the app
can boot even when the heavy ML dependencies / model weights are not
available (e.g. unit tests, the worker before model pre-warm).
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
