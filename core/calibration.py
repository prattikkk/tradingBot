"""Symbol-level strategy calibration helpers.

Loads optional per-symbol overrides from a JSON file and caches by mtime.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import CONFIG


log = logging.getLogger("zlm")


class SymbolCalibration:
    def __init__(self) -> None:
        self._cached_path: Path | None = None
        self._cached_mtime: float | None = None
        self._cached_data: dict[str, dict[str, Any]] = {}

    def _resolve_path(self) -> Path:
        configured = str(getattr(CONFIG.strategy, "symbol_calibration_path", "data/symbol_calibration.json"))
        return Path(configured).expanduser()

    def _load_if_needed(self) -> None:
        path = self._resolve_path()
        try:
            stat = path.stat()
        except OSError:
            self._cached_path = path
            self._cached_mtime = None
            self._cached_data = {}
            return

        if self._cached_path == path and self._cached_mtime == stat.st_mtime:
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to load symbol calibration file %s: %s", path, exc)
            payload = {}

        normalized: dict[str, dict[str, Any]] = {}
        if isinstance(payload, dict):
            for symbol, values in payload.items():
                if isinstance(symbol, str) and isinstance(values, dict):
                    normalized[symbol.upper()] = values

        self._cached_path = path
        self._cached_mtime = stat.st_mtime
        self._cached_data = normalized

    def get(self, symbol: str, key: str, default: Any) -> Any:
        self._load_if_needed()
        values = self._cached_data.get((symbol or "").upper(), {})
        if key not in values:
            return default

        raw = values[key]
        try:
            if isinstance(default, bool):
                if isinstance(raw, bool):
                    return raw
                if isinstance(raw, str):
                    return raw.strip().lower() in {"1", "true", "yes", "on"}
                if isinstance(raw, (int, float)):
                    return raw != 0
                return default
            if isinstance(default, int) and not isinstance(default, bool):
                return int(raw)
            if isinstance(default, float):
                return float(raw)
        except (TypeError, ValueError):
            return default
        return raw


SYMBOL_CALIBRATION = SymbolCalibration()
