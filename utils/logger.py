"""
utils/logger.py — Rich coloured logging
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Reconfigure streams on Windows to avoid UnicodeEncodeErrors from emojis
if sys.platform.startswith("win"):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass

try:
    from rich.logging import RichHandler
    from rich.console import Console
    _RICH = True
except ImportError:
    _RICH = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    configured_level = os.getenv("LOG_LEVEL", level).upper()
    logger.setLevel(getattr(logging, configured_level, logging.INFO))
    log_format = os.getenv("LOG_FORMAT", "text").strip().lower()
    use_json = log_format == "json"

    # File handler
    fh = logging.FileHandler(
        log_dir / f"alphabot_{datetime.now().strftime('%Y%m%d')}.log",
        encoding="utf-8"
    )
    if use_json:
        fh.setFormatter(_JsonFormatter())
    else:
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
    logger.addHandler(fh)

    # Console handler
    if _RICH and not use_json:
        ch = RichHandler(rich_tracebacks=True, show_path=False)
    else:
        ch = logging.StreamHandler(sys.stdout)
        if use_json:
            ch.setFormatter(_JsonFormatter())
        else:
            ch.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S"
            ))
    logger.addHandler(ch)
    return logger
