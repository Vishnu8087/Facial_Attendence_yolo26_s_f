"""Structured logging configuration.

We support two backends:

* ``json`` — :mod:`structlog` with a JSON renderer (production, k8s,
  ELK / Loki / Datadog friendly).
* ``console`` — pretty colored console output for local dev.

The chosen backend is taken from ``settings.log_format``; the level is
taken from ``settings.log_level``.

Calling :func:`configure_logging` once at process start wires up:

1. :mod:`structlog` processors (timestamper, log-level, exception
   formatter, renderer).
2. :mod:`logging` stdlib bridging so that libraries (uvicorn, sqlalchemy,
   asyncpg) flow through the same pipeline.
3. :mod:`uvicorn` / :mod:`uvicorn.access` loggers raised to the
   configured level.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config import settings

_CONFIGURED = False


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Initialise structured logging.

    Idempotent — safe to call multiple times (subsequent calls are no-ops
    unless the caller forces a different level/format).

    Parameters
    ----------
    level:
        Override ``settings.log_level`` (one of ``DEBUG/INFO/WARNING/ERROR``).
    fmt:
        Override ``settings.log_format`` (``json`` or ``console``).
    """
    global _CONFIGURED

    log_level = (level or settings.log_level or "INFO").upper()
    log_format = (fmt or settings.log_format or "json").lower()

    # ---- stdlib logging ----
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    # Tame noisy libs
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(log_level)

    # ---- structlog processors ----
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "console":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level, logging.INFO)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging into structlog so third-party logs render the
    # same way.
    structlog.stdlib.ProcessorFormatter.wrap_for_formatter

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )
    root = logging.getLogger()
    # Replace any pre-existing handlers so we own the output format.
    root.handlers = [handler]
    root.setLevel(log_level)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    The first call lazy-initialises the logging config so that importers
    that touch the logger before :func:`configure_logging` still get
    sensible output.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()
