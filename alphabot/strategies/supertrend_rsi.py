"""
AlphaBot Strategy - Supertrend + RSI + EMA trend filter.

Entry Long:  Supertrend flips bullish, RSI above threshold, close above EMA long
Entry Short: Supertrend flips bearish, RSI below threshold, close below EMA long
SL/TP:       ATR-based with configurable multiples
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.strategies.base import BaseStrategy
from alphabot.strategies.signal import Signal, SignalDirection, compute_confidence


class SupertrendRsiStrategy(BaseStrategy):
    name = "supertrend_rsi"

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

        close = float(latest.get("close", 0))
        atr_val = float(latest.get("atr", 0))
        if pd.isna(close) or pd.isna(atr_val) or atr_val <= 0 or close <= 0:
            return None

        st_dir_col, st_line_col = self._get_supertrend_columns(df)
        if not st_dir_col or not st_line_col:
            return None

        st_dir_now = float(latest.get(st_dir_col, 0))
        st_dir_prev = float(prev.get(st_dir_col, 0))
        st_line = float(latest.get(st_line_col, 0))
        if any(pd.isna(v) for v in [st_dir_now, st_dir_prev, st_line]):
            return None

        ema_long = float(latest.get("ema_long", 0))
        if ema_long <= 0 or pd.isna(ema_long):
            return None

        rsi_val = float(latest.get("rsi", 50))
        if pd.isna(rsi_val):
            return None

        volume_now = float(latest.get("volume", 0))
        vol_sma = float(latest.get("volume_sma", 1))
        if pd.isna(volume_now) or pd.isna(vol_sma):
            return None
        vol_ratio = volume_now / vol_sma if vol_sma > 0 else 0.0

        flip_long = st_dir_prev < 0 and st_dir_now > 0
        flip_short = st_dir_prev > 0 and st_dir_now < 0
        trend_long = st_dir_now > 0 and close > st_line
        trend_short = st_dir_now < 0 and close < st_line

        if not (flip_long or flip_short or trend_long or trend_short):
            return None

        direction = SignalDirection.LONG if (flip_long or trend_long) else SignalDirection.SHORT

        if direction == SignalDirection.LONG:
            if close <= ema_long:
                return None
            if rsi_val < float(settings.supertrend_rsi_long_min):
                return None
            if close < st_line:
                return None
        else:
            if close >= ema_long:
                return None
            if rsi_val > float(settings.supertrend_rsi_short_max):
                return None
            if close > st_line:
                return None

        vol_min = float(settings.supertrend_volume_multiplier)
        if vol_ratio < vol_min:
            return None

        if not (flip_long or flip_short):
            max_ext = float(settings.supertrend_max_extension_atr)
            if abs(close - st_line) > max_ext * atr_val:
                return None

        regime_align = self._regime_alignment(regime, direction)
        primary_score = 1.0 if (flip_long or flip_short) else 0.7
        confirmation_score = self._rsi_score(direction, rsi_val)
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

        sl_mult = float(settings.supertrend_atr_sl_multiplier)
        tp1_mult = float(settings.supertrend_atr_tp1_multiplier)
        tp2_mult = float(settings.supertrend_atr_tp2_multiplier)

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
    def _get_supertrend_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        dir_col = next((c for c in df.columns if c.startswith("SUPERTd_")), None)
        line_col = next(
            (
                c
                for c in df.columns
                if c.startswith("SUPERT_")
                and not c.startswith("SUPERTd_")
                and not c.startswith("SUPERTl_")
                and not c.startswith("SUPERTs_")
            ),
            None,
        )
        return dir_col, line_col

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
    def _rsi_score(direction: SignalDirection, rsi_val: float) -> float:
        if direction == SignalDirection.LONG:
            base = float(settings.supertrend_rsi_long_min)
            score = (rsi_val - base) / 20.0
        else:
            base = float(settings.supertrend_rsi_short_max)
            score = (base - rsi_val) / 20.0
        return max(0.4, min(score, 1.0))

    @staticmethod
    def _higher_tf_alignment(
        htf_df: Optional[pd.DataFrame],
        direction: SignalDirection,
    ) -> float:
        if htf_df is None or htf_df.empty:
            return 0.5

        latest = htf_df.iloc[-1]
        ema_long = latest.get("ema_long")
        close = latest.get("close")
        if pd.isna(ema_long) or pd.isna(close):
            return 0.5

        if direction == SignalDirection.LONG and close > ema_long:
            return 1.0
        if direction == SignalDirection.SHORT and close < ema_long:
            return 1.0
        return 0.2
