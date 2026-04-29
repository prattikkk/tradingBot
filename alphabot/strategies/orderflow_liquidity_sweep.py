"""
AlphaBot Strategy - Order Flow + Liquidity Sweep.

This strategy combines two ideas:
1) liquidity sweep detection around recent swing highs/lows, and
2) a candle-volume-based order-flow proxy (signed volume imbalance).

The data path currently has closed candles and volume but no full L2 book state,
so "order flow" here is implemented as directional volume imbalance.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.strategies.base import BaseStrategy
from alphabot.strategies.signal import Signal, SignalDirection, compute_confidence


class OrderFlowLiquiditySweepStrategy(BaseStrategy):
    name = "orderflow_liquidity_sweep"

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str,
        timeframe: str,
        higher_tf_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        lookback = int(settings.orderflow_sweep_lookback)
        if len(df) < lookback + 1:
            return None

        latest = df.iloc[-1]

        close = float(latest.get("close", 0))
        open_price = float(latest.get("open", close))
        high = float(latest.get("high", close))
        low = float(latest.get("low", close))
        atr_val = float(latest.get("atr", 0))
        if any(pd.isna(v) for v in [close, open_price, high, low, atr_val]):
            return None
        if close <= 0 or atr_val <= 0 or high <= low:
            return None

        rsi_val = float(latest.get("rsi", 50.0))
        if pd.isna(rsi_val):
            return None

        volume_now = float(latest.get("volume", 0))
        vol_sma = float(latest.get("volume_sma", 0))
        if pd.isna(volume_now) or pd.isna(vol_sma):
            return None
        if vol_sma <= 0:
            vol_sma = float(df["volume"].tail(lookback).mean())
        if vol_sma <= 0:
            return None

        vol_ratio = volume_now / vol_sma
        vol_min = float(settings.orderflow_sweep_volume_multiplier)
        if vol_ratio < vol_min:
            return None

        sweep_window = df.iloc[-(lookback + 1):-1]
        swing_high = float(sweep_window["high"].max())
        swing_low = float(sweep_window["low"].min())
        if any(pd.isna(v) for v in [swing_high, swing_low]):
            return None

        candle_range = max(high - low, 1e-9)
        upper_wick = max(high - max(open_price, close), 0.0)
        lower_wick = max(min(open_price, close) - low, 0.0)
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        min_wick_ratio = float(settings.orderflow_sweep_min_wick_ratio)
        bullish_sweep = (
            low < swing_low
            and close > swing_low
            and close > open_price
            and lower_wick_ratio >= min_wick_ratio
        )
        bearish_sweep = (
            high > swing_high
            and close < swing_high
            and close < open_price
            and upper_wick_ratio >= min_wick_ratio
        )

        if not (bullish_sweep or bearish_sweep):
            return None

        direction = SignalDirection.LONG if bullish_sweep else SignalDirection.SHORT

        if direction == SignalDirection.LONG and rsi_val > float(settings.orderflow_sweep_rsi_long_max):
            return None
        if direction == SignalDirection.SHORT and rsi_val < float(settings.orderflow_sweep_rsi_short_min):
            return None

        max_reclaim = float(settings.orderflow_sweep_max_reclaim_atr) * atr_val
        if direction == SignalDirection.LONG and (close - swing_low) > max_reclaim:
            return None
        if direction == SignalDirection.SHORT and (swing_high - close) > max_reclaim:
            return None

        imbalance = self._orderflow_imbalance(df, lookback)
        if imbalance is None:
            return None

        min_imbalance = float(settings.orderflow_sweep_min_imbalance)
        if direction == SignalDirection.LONG and imbalance < min_imbalance:
            return None
        if direction == SignalDirection.SHORT and imbalance > -min_imbalance:
            return None

        regime_align = self._regime_alignment(regime)
        primary_score = lower_wick_ratio if direction == SignalDirection.LONG else upper_wick_ratio
        primary_score = max(0.5, min(primary_score, 1.0))

        imbalance_score = min(abs(imbalance) / max(min_imbalance * 2.0, 1e-6), 1.0)
        rsi_score = self._rsi_reversal_score(direction, rsi_val)
        confirmation_score = (imbalance_score * 0.7) + (rsi_score * 0.3)
        volume_score = min(vol_ratio / max(vol_min * 1.8, 1e-6), 1.0)
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

        stop_buffer = float(settings.orderflow_sweep_stop_buffer_atr) * atr_val
        tp1_mult = float(settings.orderflow_sweep_atr_tp1_multiplier)
        tp2_mult = float(settings.orderflow_sweep_atr_tp2_multiplier)

        if direction == SignalDirection.LONG:
            sl = min(low, swing_low) - stop_buffer
        else:
            sl = max(high, swing_high) + stop_buffer

        sl_dist = abs(close - sl)
        min_rr = float(settings.min_risk_reward)
        tp1_dist = max(tp1_mult * atr_val, sl_dist * min_rr)
        tp2_dist = max(tp2_mult * atr_val, tp1_dist * 1.35)

        if direction == SignalDirection.LONG:
            tp1 = close + tp1_dist
            tp2 = close + tp2_dist
        else:
            tp1 = close - tp1_dist
            tp2 = close - tp2_dist

        tp_dist = abs(tp1 - close)
        min_rr_required = float(settings.min_risk_reward)
        if sl_dist <= 0 or (tp_dist + 1e-9) < (sl_dist * min_rr_required):
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
            f"imbalance={imbalance:.3f} entry={close:.4f} SL={sl:.4f} TP1={tp1:.4f}"
        )
        return signal

    @staticmethod
    def _signed_volume(df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        open_price = df["open"].astype(float)
        volume = df["volume"].astype(float)

        direction = np.sign(close - open_price)
        fallback = np.sign(close - close.shift(1))
        direction = np.where(direction == 0, fallback, direction)
        direction = np.nan_to_num(direction, nan=0.0)
        return pd.Series(direction, index=df.index) * volume

    def _orderflow_imbalance(self, df: pd.DataFrame, lookback: int) -> Optional[float]:
        signed_volume = self._signed_volume(df)
        flow = signed_volume.tail(lookback)
        if flow.empty:
            return None
        flow_sum = float(flow.sum())
        flow_abs = float(flow.abs().sum())
        if flow_abs <= 0:
            return None
        return flow_sum / flow_abs

    @staticmethod
    def _regime_alignment(regime: str) -> float:
        if regime in {"RANGING", "HIGH_VOLATILITY"}:
            return 1.0
        if "TRENDING" in regime:
            return 0.55
        return 0.35

    @staticmethod
    def _rsi_reversal_score(direction: SignalDirection, rsi_val: float) -> float:
        if direction == SignalDirection.LONG:
            cap = float(settings.orderflow_sweep_rsi_long_max)
            score = (cap - rsi_val) / max(cap - 30.0, 1e-6)
        else:
            floor = float(settings.orderflow_sweep_rsi_short_min)
            score = (rsi_val - floor) / max(70.0 - floor, 1e-6)
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
        if pd.isna(close) or pd.isna(ema_long):
            return 0.5

        if direction == SignalDirection.LONG:
            return 0.8 if close >= ema_long else 0.5
        return 0.8 if close <= ema_long else 0.5
