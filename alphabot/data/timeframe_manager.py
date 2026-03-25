"""
AlphaBot Timeframe Manager
===========================
Manages multiple timeframe data subscriptions and provides a unified
interface for multi-timeframe analysis.

Timeframe hierarchy (coarsest to finest):
  4H -> 1H -> 15M -> 5M

Rules:
  - Each lower timeframe only fires strategy evaluation when its candle closes
    AND when the next-higher-timeframe has sufficient data.
  - Higher timeframes provide trend bias only - they never trigger entries.
  - The 4H timeframe is purely for macro regime context (optional).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from loguru import logger

from alphabot.data.data_store import DataStore


TF_SECONDS: Dict[str, int] = {
    "1m":  60,
    "3m":  180,
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
    "2h":  7200,
    "4h":  14400,
    "1d":  86400,
}

BIAS_TIMEFRAME: Dict[str, str] = {
    "5m":  "1h",
    "15m": "1h",
    "30m": "4h",
    "1h":  "4h",
}

MIN_CANDLES_FOR_EVAL: Dict[str, int] = {
    "5m":  100,
    "15m": 100,
    "30m": 80,
    "1h":  80,
    "4h":  50,
}


@dataclass
class TimeframeConfig:
    timeframe: str
    is_entry_tf: bool = True
    is_bias_tf: bool = False
    min_candles: int = 100
    bias_from: Optional[str] = None
    enabled: bool = True


class TimeframeManager:
    """
    Central coordinator for multi-timeframe data and signal evaluation.

    Usage in main.py:
        tf_manager = TimeframeManager(data_store)
        tf_manager.configure_default_stack(["15m", "5m"], ["1h", "4h"])
        tf_manager.register_callback(on_entry_signal_ready)
    """

    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self._configs: Dict[str, TimeframeConfig] = {}
        self._callbacks: List[Callable] = []
        self._last_bar_time: Dict[str, Dict[str, datetime.datetime]] = {}

    def configure_default_stack(
        self,
        entry_timeframes: List[str],
        bias_timeframes: Optional[List[str]] = None,
    ) -> None:
        for tf in entry_timeframes:
            self._configs[tf] = TimeframeConfig(
                timeframe=tf,
                is_entry_tf=True,
                is_bias_tf=False,
                min_candles=MIN_CANDLES_FOR_EVAL.get(tf, 100),
                bias_from=BIAS_TIMEFRAME.get(tf),
            )
        for tf in (bias_timeframes or []):
            self._configs[tf] = TimeframeConfig(
                timeframe=tf,
                is_entry_tf=False,
                is_bias_tf=True,
                min_candles=MIN_CANDLES_FOR_EVAL.get(tf, 50),
            )
        logger.info(
            f"[TFManager] Stack: entry={entry_timeframes} "
            f"bias={bias_timeframes or []}"
        )
        logger.info(self.summary())

    def register_callback(self, callback: Callable) -> None:
        """Register async callback: async def cb(symbol, timeframe) -> None"""
        self._callbacks.append(callback)

    async def on_candle_close(self, candle) -> None:
        tf = candle.timeframe
        symbol = candle.symbol

        if tf not in self._configs:
            return

        if not self._is_new_candle(symbol, tf, candle.open_time):
            return

        cfg = self._configs[tf]
        if not cfg.is_entry_tf:
            logger.debug(f"[TFManager] {symbol} {tf}: bias bar updated")
            return

        if not self._has_sufficient_data(symbol, cfg):
            return

        for cb in self._callbacks:
            try:
                await cb(symbol, tf)
            except Exception as e:
                logger.error(f"[TFManager] Callback error {symbol} {tf}: {e}")

    def _is_new_candle(self, symbol: str, tf: str, open_time: datetime.datetime) -> bool:
        if symbol not in self._last_bar_time:
            self._last_bar_time[symbol] = {}
        if self._last_bar_time[symbol].get(tf) == open_time:
            return False
        self._last_bar_time[symbol][tf] = open_time
        return True

    def _has_sufficient_data(self, symbol: str, cfg: TimeframeConfig) -> bool:
        if not self.data_store.has_enough_data(symbol, cfg.timeframe, cfg.min_candles):
            return False
        if cfg.bias_from:
            min_b = MIN_CANDLES_FOR_EVAL.get(cfg.bias_from, 50)
            if not self.data_store.has_enough_data(symbol, cfg.bias_from, min_b):
                return False
        return True

    def get_all_timeframes(self) -> List[str]:
        return list(self._configs.keys())

    def get_bias_timeframe(self, entry_tf: str) -> Optional[str]:
        cfg = self._configs.get(entry_tf)
        return cfg.bias_from if cfg else None

    def get_entry_timeframes(self) -> List[str]:
        return [tf for tf, cfg in self._configs.items() if cfg.is_entry_tf]

    def summary(self) -> str:
        lines = ["TimeframeManager stack:"]
        for tf, cfg in sorted(self._configs.items(), key=lambda x: TF_SECONDS.get(x[0], 9999)):
            role = "ENTRY" if cfg.is_entry_tf else "BIAS"
            bias = f" <- bias from {cfg.bias_from}" if cfg.bias_from else ""
            lines.append(f"  {tf:6}  [{role}]{bias}")
        return "\n".join(lines)
