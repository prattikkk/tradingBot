"""
strategies/mean_reversion.py
Range-focused mean-reversion strategy.

Entry logic:
  LONG  - Price tests/losses lower Bollinger Band
           + Stochastic oversold and curling up
           + Quiet relative volume (non-breakout tape)

  SHORT - Inverse of above
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import CONFIG
from core.calibration import SYMBOL_CALIBRATION
from core.indicators import atr, bollinger_bands, ema, pivot_points, stochastic, volume_profile
from core.signal import Direction, Signal
from utils.logger import get_logger

log = get_logger("MeanReversion")
cfg = CONFIG.strategy
risk = CONFIG.risk


class MeanReversionStrategy:
    name = "mean_reversion"

    def generate(
        self,
        symbol: str,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        htf_df2: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        if df is None or len(df) < 80:
            return None
        try:
            return self._compute(symbol, df, htf_df, htf_df2)
        except Exception as exc:
            log.error(f"[{symbol}] Mean-reversion error: {exc}", exc_info=True)
            return None

    def _compute(self, symbol, df, htf_df, htf_df2):
        signal_idx = -2
        prev_idx = -3

        close = df["close"]
        curr_price = close.iloc[signal_idx]

        bb_upper, bb_mid, bb_lower = bollinger_bands(close, period=20, std_dev=2.0)
        stoch_k, stoch_d = stochastic(df, k_period=14, d_period=3)
        rel_vol = volume_profile(df, cfg.volume_ma_period)
        atr_vals = atr(df, 14)

        curr_upper = bb_upper.iloc[signal_idx]
        curr_mid = bb_mid.iloc[signal_idx]
        curr_lower = bb_lower.iloc[signal_idx]
        curr_k = stoch_k.iloc[signal_idx]
        curr_d = stoch_d.iloc[signal_idx]
        prev_k = stoch_k.iloc[prev_idx]
        prev_d = stoch_d.iloc[prev_idx]
        curr_vol = rel_vol.iloc[signal_idx]
        curr_atr = atr_vals.iloc[signal_idx]

        vol_ratio_max = float(SYMBOL_CALIBRATION.get(symbol, "mean_reversion_vol_ratio_max", 1.05))
        width_pct_max = float(SYMBOL_CALIBRATION.get(symbol, "mean_reversion_width_pct_max", 0.60))
        stoch_oversold_k = float(SYMBOL_CALIBRATION.get(symbol, "mean_reversion_stoch_oversold_k", 22.0))
        stoch_oversold_d = float(SYMBOL_CALIBRATION.get(symbol, "mean_reversion_stoch_oversold_d", 28.0))
        stoch_overbought_k = float(SYMBOL_CALIBRATION.get(symbol, "mean_reversion_stoch_overbought_k", 78.0))
        stoch_overbought_d = float(SYMBOL_CALIBRATION.get(symbol, "mean_reversion_stoch_overbought_d", 72.0))

        if (
            np.isnan(curr_upper)
            or np.isnan(curr_mid)
            or np.isnan(curr_lower)
            or np.isnan(curr_k)
            or np.isnan(curr_d)
            or np.isnan(curr_vol)
            or np.isnan(curr_atr)
            or curr_atr <= 0
        ):
            return None

        # Mean reversion works best in quieter tape; avoid high-volume expansion bars.
        if curr_vol > vol_ratio_max:
            return None

        bb_width = ((bb_upper - bb_lower) / bb_mid.replace(0, np.nan)).rolling(80).rank(pct=True)
        width_pct = bb_width.iloc[signal_idx]
        if np.isnan(width_pct) or width_pct > width_pct_max:
            return None

        touch_lower = curr_price <= curr_lower * 1.002
        touch_upper = curr_price >= curr_upper * 0.998

        stoch_oversold = curr_k <= stoch_oversold_k and curr_d <= stoch_oversold_d
        stoch_overbought = curr_k >= stoch_overbought_k and curr_d >= stoch_overbought_d
        stoch_bull_turn = prev_k <= prev_d and curr_k > curr_d
        stoch_bear_turn = prev_k >= prev_d and curr_k < curr_d

        htf_bias = self._htf_bias(htf_df, htf_df2)
        pivots = pivot_points(df.iloc[:-1]) if bool(cfg.use_support_resistance) else None

        direction = Direction.FLAT
        confidence = 0.0

        if touch_lower and stoch_oversold and (stoch_bull_turn or curr_k > prev_k):
            if htf_bias == Direction.SHORT:
                return None

            direction = Direction.LONG
            confidence += 0.46

            if curr_price < curr_lower:
                confidence += min(0.10, ((curr_lower - curr_price) / curr_lower) * 25)
            if curr_k <= 15 and curr_d <= 20:
                confidence += 0.10
            else:
                confidence += 0.06
            if stoch_bull_turn:
                confidence += 0.08
            if curr_vol <= 0.90:
                confidence += 0.08
            elif curr_vol <= 1.00:
                confidence += 0.04
            if width_pct <= 0.35:
                confidence += 0.08
            elif width_pct <= 0.50:
                confidence += 0.04
            if htf_bias == Direction.FLAT:
                confidence += 0.08
            elif htf_bias == Direction.LONG:
                confidence += 0.03
            if pivots and curr_price <= pivots["S1"]:
                confidence += 0.04

        elif touch_upper and stoch_overbought and (stoch_bear_turn or curr_k < prev_k):
            if htf_bias == Direction.LONG:
                return None

            direction = Direction.SHORT
            confidence += 0.46

            if curr_price > curr_upper:
                confidence += min(0.10, ((curr_price - curr_upper) / curr_upper) * 25)
            if curr_k >= 85 and curr_d >= 80:
                confidence += 0.10
            else:
                confidence += 0.06
            if stoch_bear_turn:
                confidence += 0.08
            if curr_vol <= 0.90:
                confidence += 0.08
            elif curr_vol <= 1.00:
                confidence += 0.04
            if width_pct <= 0.35:
                confidence += 0.08
            elif width_pct <= 0.50:
                confidence += 0.04
            if htf_bias == Direction.FLAT:
                confidence += 0.08
            elif htf_bias == Direction.SHORT:
                confidence += 0.03
            if pivots and curr_price >= pivots["R1"]:
                confidence += 0.04

        if direction == Direction.FLAT:
            return None

        confidence = min(confidence, 1.0)

        sl_dist = max(risk.atr_sl_multiplier, 1.2) * curr_atr
        tp2_dist = max(risk.atr_tp2_multiplier, 2.2) * curr_atr

        if direction == Direction.LONG:
            sl = min(curr_price - sl_dist, curr_lower - 0.6 * curr_atr)
            tp1 = max(curr_mid, curr_price + 0.8 * curr_atr)
            tp2 = min(curr_upper, curr_price + tp2_dist)
            tp2 = max(tp2, tp1 + 0.4 * curr_atr)
        else:
            sl = max(curr_price + sl_dist, curr_upper + 0.6 * curr_atr)
            tp1 = min(curr_mid, curr_price - 0.8 * curr_atr)
            tp2 = max(curr_lower, curr_price - tp2_dist)
            tp2 = min(tp2, tp1 - 0.4 * curr_atr)

        reason = (
            f"MR {'↑' if direction == Direction.LONG else '↓'} | "
            f"BB[pos={((curr_price - curr_lower) / max(curr_upper - curr_lower, 1e-9)):.2f}] | "
            f"StochK={curr_k:.1f} D={curr_d:.1f} | Vol={curr_vol:.2f}x | "
            f"HTF={htf_bias.value} | width_pct={width_pct:.2f} | closed_bar=true"
        )

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 3),
            strategy=self.name,
            entry_price=curr_price,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            atr=curr_atr,
            reason=reason,
            htf_bias=htf_bias,
        )

    @staticmethod
    def _htf_bias(htf_df, htf_df2) -> Direction:
        score = 0
        for frame in (htf_df, htf_df2):
            if frame is None or len(frame) < 55:
                continue
            e9 = ema(frame["close"], 9).iloc[-2]
            e21 = ema(frame["close"], 21).iloc[-2]
            e50 = ema(frame["close"], 50).iloc[-2]
            if e9 > e21 > e50:
                score += 1
            elif e9 < e21 < e50:
                score -= 1

        if score >= 1:
            return Direction.LONG
        if score <= -1:
            return Direction.SHORT
        return Direction.FLAT
