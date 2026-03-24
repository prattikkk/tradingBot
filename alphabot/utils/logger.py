"""
AlphaBot Structured Logger — loguru + JSON.
Log rotation: 50 MB per file, 30-day retention, automatic compression.
Every event is logged with: timestamp, module, event_type, symbol, details.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from alphabot.config import settings


def setup_logger() -> None:
    """Configure loguru for structured JSON logging."""
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler — human-readable
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
        colorize=True,
    )

    # JSON file handler — structured, rotated
    logger.add(
        str(log_dir / "alphabot_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{message}",
        rotation="50 MB",
        retention="30 days",
        compression="gz",
        serialize=True,  # structured JSON
        enqueue=True,  # thread-safe
    )

    # Error-only file for critical issues
    logger.add(
        str(log_dir / "errors_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        serialize=True,
        enqueue=True,
    )

    logger.info("Logger initialized", log_level=settings.log_level, log_dir=str(log_dir))
