"""
AlphaBot Strategy - EMA 9/21 with ADX and volume confirmation.

Entry Long:  EMA fast crosses above EMA slow, ADX above threshold, volume surge
Entry Short: EMA fast crosses below EMA slow, ADX above threshold, volume surge
SL/TP:       ATR-based with configurable multiples
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.strategies.base import BaseStrategy
from alphabot.strategies.signal import Signal, SignalDirection, compute_confidence


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


class EmaAdxVolumeStrategy(BaseStrategy):
    name = "ema_adx_volume"

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str,
        timeframe: str,
        higher_tf_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        if len(df) < 3:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        ema_fast_now = _to_float_or_none(latest.get("ema_fast"))
        ema_slow_now = _to_float_or_none(latest.get("ema_slow"))
        ema_fast_prev = _to_float_or_none(prev.get("ema_fast"))
        ema_slow_prev = _to_float_or_none(prev.get("ema_slow"))

        if (
            ema_fast_now is None
            or ema_slow_now is None
            or ema_fast_prev is None
            or ema_slow_prev is None
        ):
            return None

        bullish_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast_now > ema_slow_now)
        bearish_cross = (ema_fast_prev >= ema_slow_prev) and (ema_fast_now < ema_slow_now)

        ema_fast_slope = float(latest.get("ema_fast_slope", 0.0))
        trend_long = ema_fast_now > ema_slow_now and ema_fast_slope > 0
        trend_short = ema_fast_now < ema_slow_now and ema_fast_slope < 0

        # When crossover_only mode is enabled, only fresh EMA crosses are accepted.
        # Continuation entries (trend_long / trend_short) are skipped to avoid
        # entering mid-trend where noise is highest and post-entry edge is weakest.
        if settings.ema_adx_crossover_only and not (bullish_cross or bearish_cross):
            return None

        if not (bullish_cross or bearish_cross or trend_long or trend_short):
            return None

        direction = SignalDirection.LONG if (bullish_cross or trend_long) else SignalDirection.SHORT

        adx_col = next((c for c in df.columns if c.startswith("ADX_")), None)
        dmp_col = next((c for c in df.columns if c.startswith("DMP_")), None)
        dmn_col = next((c for c in df.columns if c.startswith("DMN_")), None)

        if not adx_col or not dmp_col or not dmn_col:
            return None

        adx_val = float(latest.get(adx_col, 0))
        dmp_val = float(latest.get(dmp_col, 0))
        dmn_val = float(latest.get(dmn_col, 0))
        if any(pd.isna(v) for v in [adx_val, dmp_val, dmn_val]):
            return None

        if adx_val < float(settings.ema_adx_min):
            return None

        if direction == SignalDirection.LONG and dmp_val <= dmn_val:
            return None
        if direction == SignalDirection.SHORT and dmn_val <= dmp_val:
            return None

        volume_now = float(latest.get("volume", 0))
        vol_sma = float(latest.get("volume_sma", 1))
        if pd.isna(volume_now) or pd.isna(vol_sma):
            return None
        vol_ratio = volume_now / vol_sma if vol_sma > 0 else 0.0

        vol_min = float(settings.ema_adx_volume_multiplier)
        if vol_ratio < vol_min:
            return None

        close = float(latest.get("close", 0))
        atr_val = float(latest.get("atr", 0))
        if pd.isna(close) or pd.isna(atr_val) or close <= 0 or atr_val <= 0:
            return None

        open_price = float(latest.get("open", close))
        high_price = float(latest.get("high", close))
        low_price = float(latest.get("low", close))
        if any(pd.isna(v) for v in [open_price, high_price, low_price]):
            return None
        if high_price <= low_price:
            return None

        # Price-action guard: avoid signaling against the active candle direction.
        # This prevents lagging EMA state from emitting continuation entries into
        # obvious reversal candles.
        if direction == SignalDirection.LONG:
            if close < open_price or close < ema_fast_now:
                return None
        else:
            if close > open_price or close > ema_fast_now:
                return None

        # Anti-exhaustion guard: skip entries on oversized expansion candles
        # which are statistically prone to mean-reversion right after entry.
        entry_range_atr = (high_price - low_price) / atr_val
        if entry_range_atr > float(settings.ema_adx_max_entry_range_atr):
            return None

        regime_align = self._regime_alignment(regime, direction)
        primary_score = 1.0 if (bullish_cross or bearish_cross) else 0.65
        confirmation_score = min(adx_val / max(float(settings.ema_adx_min) * 1.5, 1.0), 1.0)
        volume_score = min(vol_ratio / max(vol_min * 1.5, 1e-6), 1.0)
        htf_score = self._higher_tf_alignment(higher_tf_df, direction)

        confidence = compute_confidence(
            regime_alignment=regime_align,
            primary_indicator=primary_score,
            confirmation=confirmation_score,
            volume=volume_score,
            higher_tf=htf_score,
        )

        if confidence < settings.min_signal_confidence:
            logger.debug(
                f"[{self.name}] {symbol} rejected: confidence={confidence:.1f}"
            )
            return None

        sl_mult = float(settings.ema_adx_atr_sl_multiplier)
        tp1_mult = float(settings.ema_adx_atr_tp1_multiplier)
        tp2_mult = float(settings.ema_adx_atr_tp2_multiplier)

        if direction == SignalDirection.LONG:
            sl = close - (sl_mult * atr_val)
            tp1 = close + (tp1_mult * atr_val)
            tp2 = close + (tp2_mult * atr_val)
        else:
            sl = close + (sl_mult * atr_val)
            tp1 = close - (tp1_mult * atr_val)
            tp2 = close - (tp2_mult * atr_val)

        sl_dist = abs(close - sl)
        tp_dist = abs(tp1 - close)
        if sl_dist == 0 or (tp_dist / sl_dist) < float(settings.min_risk_reward):
            return None

        signal = Signal(
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 1),
            entry_price=Decimal(str(round(close, 8))),
            stop_loss=Decimal(str(round(sl, 8))),
            take_profit_1=Decimal(str(round(tp1, 8))),
            take_profit_2=Decimal(str(round(tp2, 8))),
            strategy_name=self.name,
            regime=regime,
            timeframe=timeframe,
            regime_alignment_score=regime_align,
            primary_indicator_score=primary_score,
            confirmation_score=confirmation_score,
            volume_score=volume_score,
            higher_tf_score=htf_score,
        )

        logger.info(
            f"[{self.name}] {symbol} {direction.value} conf={confidence:.1f} "
            f"entry={close:.4f} SL={sl:.4f} TP1={tp1:.4f}"
        )
        return signal

    @staticmethod
    def _regime_alignment(regime: str, direction: SignalDirection) -> float:
        if direction == SignalDirection.LONG and regime == "TRENDING_UP":
            return 1.0
        if direction == SignalDirection.SHORT and regime == "TRENDING_DOWN":
            return 1.0
        if "TRENDING" in regime:
            return 0.7
        return 0.3

    @staticmethod
    def _higher_tf_alignment(
        htf_df: Optional[pd.DataFrame],
        direction: SignalDirection,
    ) -> float:
        if htf_df is None or htf_df.empty:
            return 0.5

        latest = htf_df.iloc[-1]
        ema_fast = latest.get("ema_fast")
        ema_slow = latest.get("ema_slow")
        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return 0.5

        if direction == SignalDirection.LONG and ema_fast > ema_slow:
            return 1.0
        if direction == SignalDirection.SHORT and ema_fast < ema_slow:
            return 1.0
        return 0.2
