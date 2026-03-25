"""
AlphaBot Strategy C — ATR Breakout (High Volatility / Transition Regime).
Active during HIGH_VOLATILITY regime or regime transitions.

Entry Long:  Price breaks above previous high + volume > 2× avg + ATR expanding
Entry Short: Price breaks below previous low + volume > 2× avg + ATR expanding
Exit:        Trailing stop at 2× ATR from high/low since entry
SL:          2× ATR from entry (wider for volatility)
TP:          None — trailing stop manages exit
Position:    Size reduced to 50% to account for regime uncertainty.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.strategies.base import BaseStrategy
from alphabot.strategies.signal import Signal, SignalDirection, compute_confidence


class ATRBreakoutStrategy(BaseStrategy):
    name = "atr_breakout"

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str,
        timeframe: str,
        higher_tf_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        if len(df) < 5:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(latest["close"])
        high_now = float(latest["high"])
        low_now = float(latest["low"])
        atr_val = float(latest.get("atr", 0))

        if atr_val == 0:
            return None

        # Previous candle highs/lows for breakout reference
        lookback = min(20, len(df) - 1)
        recent = df.iloc[-lookback - 1:-1]
        prev_high = float(recent["high"].max())
        prev_low = float(recent["low"].min())

        # ATR expansion check
        atr_sma = float(df["atr"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else atr_val
        atr_expanding = atr_val > atr_sma * 1.2

        # Volume surge check
        volume_now = float(latest.get("volume", 0))
        vol_sma = float(latest.get("volume_sma", 1))
        vol_ratio = volume_now / vol_sma if vol_sma > 0 else 0
        vol_surge = vol_ratio > 2.0

        # ADX rising check
        adx_col = [c for c in df.columns if c.startswith("ADX_")]
        adx_val = float(latest[adx_col[0]]) if adx_col else 0.0
        adx_prev = float(prev[adx_col[0]]) if adx_col else 0.0
        adx_rising = adx_val > adx_prev

        direction = None

        # Breakout above previous high
        if high_now > prev_high and vol_surge and atr_expanding:
            direction = SignalDirection.LONG

        # Breakout below previous low
        elif low_now < prev_low and vol_surge and atr_expanding:
            direction = SignalDirection.SHORT

        if direction is None:
            return None

        # --- Scoring ---
        regime_align = 0.8 if regime == "HIGH_VOLATILITY" else 0.5
        primary_score = 1.0  # Breakout confirmed
        confirm_score = 0.8 if adx_rising else 0.4
        volume_sc = min(vol_ratio / 3.0, 1.0)
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
                f"[{self.name}] {symbol} signal rejected: confidence={confidence:.1f}"
            )
            return None

        # --- SL / TP ---
        sl_mult = 2.5  # Wider for volatility
        trailing_mult = 2.0

        if direction == SignalDirection.LONG:
            sl = close - (sl_mult * atr_val)
            tp1 = close + (trailing_mult * 2 * atr_val)
            tp2 = close + (trailing_mult * 3 * atr_val)
        else:
            sl = close + (sl_mult * atr_val)
            tp1 = close - (trailing_mult * 2 * atr_val)
            tp2 = close - (trailing_mult * 3 * atr_val)

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
            f"conf={confidence:.1f} entry={close} SL={sl:.2f} "
            f"ATR={atr_val:.4f} vol_ratio={vol_ratio:.1f}x"
        )
        return signal

    @staticmethod
    def _higher_tf_alignment(htf_df: Optional[pd.DataFrame],
                              direction: SignalDirection) -> float:
        if htf_df is None or htf_df.empty or len(htf_df) < 2:
            return 0.5
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
