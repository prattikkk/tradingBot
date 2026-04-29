"""
AlphaBot Data Store — In-memory rolling OHLCV buffer.
Maintains a pandas DataFrame with the last N candles per symbol/timeframe.
All indicator calculations happen on closed candles only.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Dict, Optional, Tuple

import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.data.models import Candle


class DataStore:
    """
    Rolling OHLCV buffer per (symbol, timeframe).
    Thread-safe via GIL for reads; writes happen in the async event loop.
    """

    def __init__(self, lookback: int | None = None):
        self.lookback = lookback or settings.candle_lookback
        # Key: (symbol, timeframe) → pd.DataFrame
        self._buffers: Dict[Tuple[str, str], pd.DataFrame] = {}
        # Latest ticker price per symbol
        self._prices: Dict[str, Decimal] = {}

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["open_time", "open", "high", "low", "close", "volume", "close_time"]
        )

    def _key(self, symbol: str, timeframe: str) -> Tuple[str, str]:
        return (symbol.upper(), timeframe)

    def add_candle(self, candle: Candle) -> None:
        """Add a closed candle to the buffer. Trims to lookback size."""
        if not candle.is_closed:
            return  # Only store closed candles

        key = self._key(candle.symbol, candle.timeframe)
        if key not in self._buffers:
            self._buffers[key] = self._empty_df()

        new_row = {
            "open_time": candle.open_time,
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "volume": float(candle.volume),
            "close_time": candle.close_time,
        }

        df = self._buffers[key]

        # Avoid duplicate candles by open_time
        if not df.empty and candle.open_time in df["open_time"].values:
            return

        self._buffers[key] = pd.concat(
            [df, pd.DataFrame([new_row])], ignore_index=True
        ).tail(self.lookback)

        logger.debug(
            f"Candle stored: {candle.symbol} {candle.timeframe} "
            f"close={candle.close} vol={candle.volume} "
            f"buffer_size={len(self._buffers[key])}"
        )

    def load_historical(self, symbol: str, timeframe: str,
                        candles: list[Candle]) -> None:
        """Bulk-load historical candles (e.g., from REST API on startup)."""
        key = self._key(symbol, timeframe)
        rows = []
        for c in candles:
            rows.append({
                "open_time": c.open_time,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
                "close_time": c.close_time,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.drop_duplicates(subset=["open_time"]).tail(self.lookback)
        self._buffers[key] = df.reset_index(drop=True)
        logger.info(
            f"Loaded {len(df)} historical candles for {symbol} {timeframe}"
        )

    def get_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get OHLCV DataFrame for indicator calculations."""
        key = self._key(symbol, timeframe)
        if key not in self._buffers:
            return self._empty_df()
        return self._buffers[key].copy()

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update latest ticker price."""
        self._prices[symbol.upper()] = price

    def get_price(self, symbol: str) -> Optional[Decimal]:
        """Get latest ticker price."""
        return self._prices.get(symbol.upper())

    def has_enough_data(self, symbol: str, timeframe: str,
                        min_candles: int = 50) -> bool:
        """Check if we have enough candles for indicator calculation."""
        key = self._key(symbol, timeframe)
        if key not in self._buffers:
            return False
        return len(self._buffers[key]) >= min_candles

    def candle_count(self, symbol: str, timeframe: str) -> int:
        key = self._key(symbol, timeframe)
        if key not in self._buffers:
            return 0
        return len(self._buffers[key])

    def latest_open_time(self, symbol: str, timeframe: str) -> Optional[datetime.datetime]:
        key = self._key(symbol, timeframe)
        df = self._buffers.get(key)
        if df is None or df.empty:
            return None
        value = df.iloc[-1].get("open_time")
        if pd.isna(value):
            return None
        return value
