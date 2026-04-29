"""
AlphaBot Strategy - Supertrend Pullback Continuation.

A stricter sibling of supertrend_rsi that waits for:
  1) established Supertrend direction,
  2) a pullback toward the Supertrend line, and
  3) a recovery candle in trend direction.

This generally trades less frequently but aims for cleaner continuation entries.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.strategies.base import BaseStrategy
from alphabot.strategies.signal import Signal, SignalDirection, compute_confidence


class SupertrendPullbackStrategy(BaseStrategy):
    name = "supertrend_pullback"

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str,
        timeframe: str,
        higher_tf_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        if len(df) < 4:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(latest.get("close", 0))
        open_price = float(latest.get("open", close))
        high = float(latest.get("high", close))
        low = float(latest.get("low", close))
        atr_val = float(latest.get("atr", 0))

        if any(pd.isna(v) for v in [close, open_price, high, low, atr_val]):
            return None
        if atr_val <= 0 or close <= 0 or high <= low:
            return None

        st_dir_col, st_line_col = self._get_supertrend_columns(df)
        if not st_dir_col or not st_line_col:
            return None

        st_dir_now = float(latest.get(st_dir_col, 0))
        st_dir_prev = float(prev.get(st_dir_col, 0))
        st_line = float(latest.get(st_line_col, 0))
        st_line_prev = float(prev.get(st_line_col, st_line))

        if any(pd.isna(v) for v in [st_dir_now, st_dir_prev, st_line, st_line_prev]):
            return None

        # Wait for continuation setups after trend is already established.
        just_flipped = (st_dir_prev < 0 and st_dir_now > 0) or (st_dir_prev > 0 and st_dir_now < 0)
        if just_flipped:
            return None

        ema_long = float(latest.get("ema_long", 0))
        if pd.isna(ema_long) or ema_long <= 0:
            return None

        rsi_val = float(latest.get("rsi", 50))
        if pd.isna(rsi_val):
            return None

        volume_now = float(latest.get("volume", 0))
        vol_sma = float(latest.get("volume_sma", 1))
        if pd.isna(volume_now) or pd.isna(vol_sma):
            return None
        vol_ratio = volume_now / vol_sma if vol_sma > 0 else 0.0

        adx_col = next((c for c in df.columns if c.startswith("ADX_")), None)
        if not adx_col:
            return None
        adx_val = float(latest.get(adx_col, 0))
        if pd.isna(adx_val) or adx_val < float(settings.supertrend_pullback_adx_min):
            return None

        trend_long = st_dir_now > 0 and close > st_line and close > ema_long
        trend_short = st_dir_now < 0 and close < st_line and close < ema_long
        if not (trend_long or trend_short):
            return None

        direction = SignalDirection.LONG if trend_long else SignalDirection.SHORT

        pullback_band = float(settings.supertrend_pullback_band_atr) * atr_val

        prev_high = float(prev.get("high", close))
        prev_low = float(prev.get("low", close))
        if any(pd.isna(v) for v in [prev_high, prev_low]):
            return None

        if direction == SignalDirection.LONG:
            if rsi_val < float(settings.supertrend_pullback_rsi_long_min):
                return None
            pullback_ok = prev_low <= (st_line_prev + pullback_band)
            recovery_ok = close > prev_high and close >= open_price
        else:
            if rsi_val > float(settings.supertrend_pullback_rsi_short_max):
                return None
            pullback_ok = prev_high >= (st_line_prev - pullback_band)
            recovery_ok = close < prev_low and close <= open_price

        if not (pullback_ok and recovery_ok):
            return None

        vol_min = float(settings.supertrend_pullback_volume_multiplier)
        if vol_ratio < vol_min:
            return None

        max_ext = float(settings.supertrend_pullback_max_extension_atr)
        if abs(close - st_line) > max_ext * atr_val:
            return None

        regime_align = self._regime_alignment(regime, direction)
        primary_score = 1.0
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
            return None

        sl_mult = float(settings.supertrend_pullback_atr_sl_multiplier)
        tp1_mult = float(settings.supertrend_pullback_atr_tp1_multiplier)
        tp2_mult = float(settings.supertrend_pullback_atr_tp2_multiplier)

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
            return 0.6
        return 0.2

    @staticmethod
    def _rsi_score(direction: SignalDirection, rsi_val: float) -> float:
        if direction == SignalDirection.LONG:
            base = float(settings.supertrend_pullback_rsi_long_min)
            score = (rsi_val - base) / 18.0
        else:
            base = float(settings.supertrend_pullback_rsi_short_max)
            score = (base - rsi_val) / 18.0
        return max(0.4, min(score, 1.0))

    @staticmethod
    def _higher_tf_alignment(
        htf_df: Optional[pd.DataFrame],
        direction: SignalDirection,
    ) -> float:
        if htf_df is None or htf_df.empty:
            return 0.5

        latest = htf_df.iloc[-1]
        close = latest.get("close")
        ema_long = latest.get("ema_long")
        st_dir_col = next((c for c in htf_df.columns if c.startswith("SUPERTd_")), None)
        st_dir = latest.get(st_dir_col) if st_dir_col else None

        if pd.isna(close) or pd.isna(ema_long):
            return 0.5

        trend_ok = False
        if direction == SignalDirection.LONG:
            trend_ok = bool(close > ema_long)
            if st_dir is not None and not pd.isna(st_dir):
                trend_ok = trend_ok and bool(float(st_dir) > 0)
        else:
            trend_ok = bool(close < ema_long)
            if st_dir is not None and not pd.isna(st_dir):
                trend_ok = trend_ok and bool(float(st_dir) < 0)

        return 1.0 if trend_ok else 0.2
