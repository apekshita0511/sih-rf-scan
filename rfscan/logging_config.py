"""Centralised logging setup for the ``rfscan`` package.

All modules obtain loggers via :func:`get_logger`, which namespaces them under
``rfscan.*``. :func:`configure_logging` is called once at process start (CLI,
dashboard, or test fixtures) and is idempotent.
"""

from __future__ import annotations

import logging
import sys

_ROOT_NAME = "rfscan"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str | int = "INFO") -> None:
    """Attach a single stderr handler to the ``rfscan`` logger.

    Idempotent: repeated calls only adjust the level, never stack handlers.
    """
    logger = logging.getLogger(_ROOT_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``rfscan`` (e.g. ``rfscan.simulator``)."""
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
