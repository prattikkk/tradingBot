"""
AlphaBot Market Regime Detector.
Classifies market as: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, UNCLEAR.

Uses ADX, ATR, Bollinger Band Width, and EMA slope.
Computed on each closed candle.
"""

from __future__ import annotations

import enum
from typing import Optional

import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.data.data_store import DataStore
from alphabot.utils.indicators import compute_all_indicators


class MarketRegime(str, enum.Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNCLEAR = "UNCLEAR"


class RegimeDetector:
    """
    Detects market regime per symbol using:
    - ADX > 25 → trending; ADX < 20 → ranging
    - ATR spike > 2× 20-period avg → high volatility
    - BB Width expansion → volatile
    - EMA slope direction → up-trend vs down-trend
    """

    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self._last_regime: dict[str, MarketRegime] = {}

    def detect(self, symbol: str, timeframe: str | None = None) -> MarketRegime:
        """
        Classify the current market regime for a symbol.
        Returns MarketRegime enum.
        """
        tf = timeframe or settings.primary_timeframe

        if not self.data_store.has_enough_data(symbol, tf, min_candles=50):
            logger.warning(
                f"Not enough data for regime detection: {symbol} {tf} "
                f"({self.data_store.candle_count(symbol, tf)} candles)"
            )
            return MarketRegime.UNCLEAR

        df = self.data_store.get_dataframe(symbol, tf)
        df = compute_all_indicators(df, {
            "ema_fast": settings.ema_fast,
            "ema_slow": settings.ema_slow,
            "atr_period": settings.atr_period,
            "adx_period": settings.adx_period,
            "rsi_period": 14,
            "bb_period": settings.bb_period,
            "bb_std": settings.bb_std,
        })

        if df.empty or len(df) < 30:
            return MarketRegime.UNCLEAR

        regime = self._classify(df, symbol)
        prev = self._last_regime.get(symbol)
        if regime != prev:
            logger.info(
                f"Regime change: {symbol} {prev} → {regime}"
            )
        self._last_regime[symbol] = regime
        return regime

    def _classify(self, df: pd.DataFrame, symbol: str) -> MarketRegime:
        """Internal classification logic using latest indicator values."""
        latest = df.iloc[-1]

        # Get ADX value
        adx_col = [c for c in df.columns if c.startswith("ADX_")]
        adx_val = float(latest[adx_col[0]]) if adx_col else 0.0

        # Get ATR value and check for spike
        atr_val = float(latest.get("atr", 0))
        atr_sma_20 = df["atr"].rolling(20).mean().iloc[-1] if "atr" in df.columns else atr_val
        atr_spike = atr_val > (float(settings.atr_volatility_multiplier) * atr_sma_20) if atr_sma_20 > 0 else False

        # BB Width
        bb_width = float(latest.get("bb_width", 0))
        bb_width_sma = df["bb_width"].rolling(20).mean().iloc[-1] if "bb_width" in df.columns else bb_width
        bb_expanded = bb_width > (float(settings.atr_volatility_multiplier) * bb_width_sma) if bb_width_sma > 0 else False

        # EMA slopes
        ema_fast_slope = float(latest.get("ema_fast_slope", 0))
        ema_slow_slope = float(latest.get("ema_slow_slope", 0))

        # EMA positions
        ema_fast_val = float(latest.get("ema_fast", 0))
        ema_slow_val = float(latest.get("ema_slow", 0))

        # ---- Classification Logic ----
        # HIGH VOLATILITY: ATR spike or extreme BB expansion
        if atr_spike or bb_expanded:
            logger.debug(
                f"{symbol} HIGH_VOLATILITY: atr_spike={atr_spike} bb_expanded={bb_expanded}"
            )
            return MarketRegime.HIGH_VOLATILITY

        # TRENDING: ADX > threshold
        if adx_val > settings.adx_trending_threshold:
            if ema_fast_val > ema_slow_val and ema_fast_slope > 0:
                logger.debug(
                    f"{symbol} TRENDING_UP: ADX={adx_val:.1f} ema_fast>ema_slow slope>0"
                )
                return MarketRegime.TRENDING_UP
            elif ema_fast_val < ema_slow_val and ema_fast_slope < 0:
                logger.debug(
                    f"{symbol} TRENDING_DOWN: ADX={adx_val:.1f} ema_fast<ema_slow slope<0"
                )
                return MarketRegime.TRENDING_DOWN
            else:
                # ADX high but direction unclear
                logger.debug(
                    f"{symbol} TRENDING (direction ambiguous): ADX={adx_val:.1f}"
                )
                if ema_fast_slope > 0:
                    return MarketRegime.TRENDING_UP
                elif ema_fast_slope < 0:
                    return MarketRegime.TRENDING_DOWN
                return MarketRegime.UNCLEAR

        # RANGING: ADX < threshold
        if adx_val < settings.adx_ranging_threshold:
            logger.debug(
                f"{symbol} RANGING: ADX={adx_val:.1f}"
            )
            return MarketRegime.RANGING

        # In-between (ADX 20-25): UNCLEAR
        logger.debug(f"{symbol} UNCLEAR: ADX={adx_val:.1f} (between thresholds)")
        return MarketRegime.UNCLEAR

    def get_regime(self, symbol: str) -> MarketRegime:
        """Get last detected regime for a symbol (cached)."""
        return self._last_regime.get(symbol, MarketRegime.UNCLEAR)

    def get_current_regimes(self) -> dict[str, MarketRegime]:
        """Return all cached regimes."""
        return dict(self._last_regime)
