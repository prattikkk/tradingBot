"""
strategies/breakout_momentum.py
Volatility breakout strategy using Bollinger Bands + Donchian Channel.

Entry logic:
  LONG  — Price closes above upper Bollinger Band
           + Closes above N-period high (Donchian breakout)
           + RSI > 55 (momentum confirmation)
           + Volume > 1.8× average (conviction)
           + ATR expanding (volatility increasing)

  SHORT — Inverse of above

Best on 15m–1h during high-volatility sessions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from core.signal import Signal, Direction
from core.indicators import bollinger_bands, atr, rsi, ema, pivot_points, volume_profile
from core.calibration import SYMBOL_CALIBRATION
from config import CONFIG
from utils.logger import get_logger

log = get_logger("Breakout")
cfg = CONFIG.strategy
risk = CONFIG.risk

DEFAULT_DONCHIAN_PERIOD = cfg.breakout_period
DEFAULT_VOL_RATIO_MIN = 1.8
DEFAULT_RSI_BULL_MIN = 52.0
DEFAULT_RSI_BEAR_MAX = 48.0


class BreakoutMomentumStrategy:
    name = "breakout_momentum"

    def generate(
        self,
        symbol: str,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        htf_df2: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        donchian_period = int(SYMBOL_CALIBRATION.get(symbol, "breakout_period", DEFAULT_DONCHIAN_PERIOD))
        if df is None or len(df) < max(12, donchian_period + 10):
            return None
        try:
            return self._compute(symbol, df, htf_df, htf_df2, donchian_period)
        except Exception as e:
            log.error(f"[{symbol}] Breakout error: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------ #

    def _compute(self, symbol, df, htf_df, htf_df2, donchian_period: int):
        # Evaluate on closed bars only (-2 signal bar, -3 previous bar).
        signal_idx = -2
        prev_idx = -3

        vol_ratio_min = float(SYMBOL_CALIBRATION.get(symbol, "breakout_volume_ratio_min", DEFAULT_VOL_RATIO_MIN))
        rsi_bull_min = float(SYMBOL_CALIBRATION.get(symbol, "breakout_rsi_bull_min", DEFAULT_RSI_BULL_MIN))
        rsi_bear_max = float(SYMBOL_CALIBRATION.get(symbol, "breakout_rsi_bear_max", DEFAULT_RSI_BEAR_MAX))

        close = df["close"]
        curr_price = close.iloc[signal_idx]
        prev_price = close.iloc[prev_idx]

        # Bollinger Bands (20, 2.0)
        bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)

        # Donchian Channel
        don_high = df["high"].rolling(donchian_period).max().shift(1)
        don_low  = df["low"].rolling(donchian_period).min().shift(1)

        # ATR & expansion
        _atr = atr(df, 14)
        atr_ma = _atr.rolling(20).mean()
        atr_expanding = _atr.iloc[signal_idx] > atr_ma.iloc[signal_idx]

        # Volume
        vol_ratio = volume_profile(df, 20)

        # RSI
        rsi_vals = rsi(close, cfg.rsi_period)

        curr_bb_upper = bb_upper.iloc[signal_idx]
        curr_bb_lower = bb_lower.iloc[signal_idx]
        prev_bb_upper = bb_upper.iloc[prev_idx]
        prev_bb_lower = bb_lower.iloc[prev_idx]
        curr_don_high = don_high.iloc[signal_idx]
        curr_don_low  = don_low.iloc[signal_idx]
        curr_vol      = vol_ratio.iloc[signal_idx]
        curr_rsi      = rsi_vals.iloc[signal_idx]
        curr_atr      = _atr.iloc[signal_idx]
        pivots = pivot_points(df.iloc[:-1]) if bool(cfg.use_support_resistance) else None

        if np.isnan(curr_atr) or curr_atr == 0:
            return None

        # BB width percentile (avoid squeezes)
        bb_width = ((bb_upper - bb_lower) / bb_mid).rolling(50).rank(pct=True)
        bb_width_pct = bb_width.iloc[signal_idx]
        if np.isnan(bb_width_pct) or bb_width_pct < 0.30:
            return None   # In a squeeze — skip

        direction = Direction.FLAT
        confidence = 0.0

        # --- LONG BREAKOUT ---
        bull_bb     = prev_price <= prev_bb_upper and curr_price > curr_bb_upper
        bull_don    = curr_price > curr_don_high
        bull_rsi    = curr_rsi > rsi_bull_min
        bull_vol    = curr_vol >= vol_ratio_min

        # --- SHORT BREAKOUT ---
        bear_bb     = prev_price >= prev_bb_lower and curr_price < curr_bb_lower
        bear_don    = curr_price < curr_don_low
        bear_rsi    = curr_rsi < rsi_bear_max
        bear_vol    = curr_vol >= vol_ratio_min

        if bull_bb and bull_don:
            direction = Direction.LONG
            confidence += 0.35   # dual breakout
            if bull_rsi:   confidence += 0.20
            if bull_vol:   confidence += 0.20
            if atr_expanding: confidence += 0.15
            if self._htf_aligned(htf_df, htf_df2, bullish=True):
                confidence += 0.10
            if pivots:
                if curr_price >= pivots["R1"]:
                    confidence += 0.08
                elif curr_price <= pivots["PP"]:
                    confidence -= 0.05

        elif bear_bb and bear_don:
            direction = Direction.SHORT
            confidence += 0.35
            if bear_rsi:   confidence += 0.20
            if bear_vol:   confidence += 0.20
            if atr_expanding: confidence += 0.15
            if self._htf_aligned(htf_df, htf_df2, bullish=False):
                confidence += 0.10
            if pivots:
                if curr_price <= pivots["S1"]:
                    confidence += 0.08
                elif curr_price >= pivots["PP"]:
                    confidence -= 0.05

        if direction == Direction.FLAT:
            return None

        confidence = max(0.0, min(confidence, 1.0))

        # Wider SL for breakouts (volatility event)
        sl_mult  = risk.atr_sl_multiplier * 1.2
        tp1_mult = risk.atr_tp1_multiplier
        tp2_mult = risk.atr_tp2_multiplier * 1.1   # let breakouts run

        if direction == Direction.LONG:
            sl  = curr_price - sl_mult  * curr_atr
            tp1 = curr_price + tp1_mult * curr_atr
            tp2 = curr_price + tp2_mult * curr_atr
        else:
            sl  = curr_price + sl_mult  * curr_atr
            tp1 = curr_price - tp1_mult * curr_atr
            tp2 = curr_price - tp2_mult * curr_atr

        reason = (
            f"BB breakout {'↑' if direction==Direction.LONG else '↓'} | "
            f"Donchian={'yes' if (bull_don or bear_don) else 'no'} | "
            f"Vol={curr_vol:.1f}x | RSI={curr_rsi:.1f} | "
            f"ATR_exp={'yes' if atr_expanding else 'no'} | "
            f"SR={'on' if pivots else 'off'} | closed_bar=true"
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
        )

    @staticmethod
    def _htf_aligned(htf_df, htf_df2, bullish: bool) -> bool:
        for df in [htf_df, htf_df2]:
            if df is None or len(df) < 30:
                continue
            e9  = ema(df["close"], 9).iloc[-2]
            e21 = ema(df["close"], 21).iloc[-2]
            if bullish and e9 > e21:
                return True
            if not bullish and e9 < e21:
                return True
        return False
