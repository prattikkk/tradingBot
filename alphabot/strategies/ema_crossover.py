"""
AlphaBot Strategy A — EMA Crossover Trend Following.
Active during TRENDING_UP and TRENDING_DOWN regimes.

Entry Long:  EMA fast crosses above EMA slow + ADX > 25 + Volume > 1.2× avg
Entry Short: EMA fast crosses below EMA slow + ADX > 25 + Volume > 1.2× avg
Exit:        Opposite EMA cross OR ATR-based trailing stop
SL:          1.5 × ATR below/above entry
TP:          3 × ATR from entry (minimum 2:1 R:R)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional, cast

import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.strategies.base import BaseStrategy
from alphabot.strategies.signal import Signal, SignalDirection, compute_confidence


class EMACrossoverStrategy(BaseStrategy):
    name = "ema_crossover"

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

        # --- Indicator values ---
        ema_fast_now = latest.get("ema_fast")
        ema_slow_now = latest.get("ema_slow")
        ema_fast_prev = prev.get("ema_fast")
        ema_slow_prev = prev.get("ema_slow")

        if any(pd.isna(v) for v in [ema_fast_now, ema_slow_now, ema_fast_prev, ema_slow_prev]):
            return None

        if any(v is None for v in [ema_fast_now, ema_slow_now, ema_fast_prev, ema_slow_prev]):
            return None

        ema_fast_now = float(cast(float, ema_fast_now))
        ema_slow_now = float(cast(float, ema_slow_now))
        ema_fast_prev = float(cast(float, ema_fast_prev))
        ema_slow_prev = float(cast(float, ema_slow_prev))

        # ADX
        adx_col = [c for c in df.columns if c.startswith("ADX_")]
        adx_val = float(latest[adx_col[0]]) if adx_col else 0.0

        # MACD histogram for confirmation
        macdh_col = [c for c in df.columns if c.startswith("MACDh_")]
        macd_hist = float(latest[macdh_col[0]]) if macdh_col else 0.0

        # Volume confirmation
        volume_now = float(latest.get("volume", 0))
        vol_sma = float(latest.get("volume_sma", 1))
        vol_ratio = volume_now / vol_sma if vol_sma > 0 else 0

        atr_val = float(latest.get("atr", 0))
        close = float(latest["close"])

        # --- Crossover detection ---
        bullish_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast_now > ema_slow_now)
        bearish_cross = (ema_fast_prev >= ema_slow_prev) and (ema_fast_now < ema_slow_now)

        # Also allow continuation entries while trend is still strong.
        ema_fast_slope = float(latest.get("ema_fast_slope", 0.0))
        bullish_continuation = (
            ema_fast_now > ema_slow_now
            and ema_fast_slope > 0
            and close > ema_fast_now
        )
        bearish_continuation = (
            ema_fast_now < ema_slow_now
            and ema_fast_slope < 0
            and close < ema_fast_now
        )

        if not bullish_cross and not bearish_cross and not bullish_continuation and not bearish_continuation:
            return None

        # Direction
        if bullish_cross or bullish_continuation:
            direction = SignalDirection.LONG
        else:
            direction = SignalDirection.SHORT

        # --- Filters ---
        adx_ok = adx_val > settings.adx_trending_threshold
        vol_ok = vol_ratio > 1.0
        if not adx_ok:
            return None

        # Fresh cross is strongest; continuation has slightly lower base weight.
        is_fresh_cross = (direction == SignalDirection.LONG and bullish_cross) or (
            direction == SignalDirection.SHORT and bearish_cross
        )

        # Continuation entries are stricter to avoid chasing extended moves.
        if not is_fresh_cross:
            if direction == SignalDirection.LONG and macd_hist < 0.2:
                return None
            if direction == SignalDirection.SHORT and macd_hist > -0.2:
                return None

            max_extension = (atr_val * 1.2) if atr_val > 0 else (close * 0.003)
            if abs(close - ema_fast_now) > max_extension:
                return None

            if vol_ratio < 1.1:
                return None

        # --- Scoring ---
        regime_align = 1.0 if "TRENDING" in regime else 0.3
        primary_score = 1.0 if is_fresh_cross else 0.65
        if direction == SignalDirection.LONG:
            if macd_hist > 0.5:
                confirm_score = 1.0
            elif macd_hist > 0:
                confirm_score = 0.7
            elif macd_hist > -0.5:
                confirm_score = 0.5
            else:
                confirm_score = 0.2
        else:
            if macd_hist < -0.5:
                confirm_score = 1.0
            elif macd_hist < 0:
                confirm_score = 0.7
            elif macd_hist < 0.5:
                confirm_score = 0.5
            else:
                confirm_score = 0.2
        volume_sc = min(vol_ratio / 1.5, 1.0) if vol_ok else 0.2
        htf_score = self._higher_tf_alignment(higher_tf_df, direction)

        confidence = compute_confidence(
            regime_alignment=regime_align,
            primary_indicator=primary_score,
            confirmation=confirm_score,
            volume=volume_sc,
            higher_tf=htf_score,
        )

        if confidence < settings.min_signal_confidence:
            logger.debug(
                f"[{self.name}] {symbol} signal rejected: confidence={confidence:.1f} < {settings.min_signal_confidence}"
            )
            return None

        # --- SL / TP ---
        sl_mult = float(settings.atr_sl_multiplier)
        tp_mult = float(settings.atr_tp_multiplier)

        if direction == SignalDirection.LONG:
            sl = close - (sl_mult * atr_val)
            tp1 = close + (tp_mult * atr_val)
            tp2 = close + (tp_mult * 1.5 * atr_val)
        else:
            sl = close + (sl_mult * atr_val)
            tp1 = close - (tp_mult * atr_val)
            tp2 = close - (tp_mult * 1.5 * atr_val)

        signal = Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=Decimal(str(round(close, 8))),
            stop_loss=Decimal(str(round(sl, 8))),
            take_profit_1=Decimal(str(round(tp1, 8))),
            take_profit_2=Decimal(str(round(tp2, 8))),
            strategy_name=self.name,
            regime=regime,
            timeframe=timeframe,
            regime_alignment_score=regime_align,
            primary_indicator_score=primary_score,
            confirmation_score=confirm_score,
            volume_score=volume_sc,
            higher_tf_score=htf_score,
        )

        logger.info(
            f"[{self.name}] {symbol} {direction.value} signal: "
            f"conf={confidence:.1f} entry={close} SL={sl:.2f} TP1={tp1:.2f} R:R={signal.risk_reward_ratio:.2f}"
        )
        return signal

    @staticmethod
    def _higher_tf_alignment(htf_df: Optional[pd.DataFrame],
                              direction: SignalDirection) -> float:
        """Check if higher-timeframe EMA trend agrees with signal direction."""
        if htf_df is None or htf_df.empty or len(htf_df) < 2:
            return 0.5  # neutral if no data

        latest = htf_df.iloc[-1]
        ema_fast = latest.get("ema_fast")
        ema_slow = latest.get("ema_slow")

        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return 0.5

        if direction == SignalDirection.LONG and ema_fast > ema_slow:
            return 1.0
        elif direction == SignalDirection.SHORT and ema_fast < ema_slow:
            return 1.0
        return 0.2
