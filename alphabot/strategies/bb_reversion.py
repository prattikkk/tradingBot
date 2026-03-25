"""
AlphaBot Strategy B — Bollinger Band Mean Reversion.
Active during RANGING regime.

Entry Long:  Price touches/breaks lower BB + RSI < 35 + Stoch RSI oversold (< 20)
Entry Short: Price touches/breaks upper BB + RSI > 65 + Stoch RSI overbought (> 80)
Exit:        Price returns to BB midline (EMA 20) OR RSI hits 50
SL:          Beyond the BB extreme + 1.2× ATR buffer
TP:          Middle BB line (conservative, high win-rate)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.strategies.base import BaseStrategy
from alphabot.strategies.signal import Signal, SignalDirection, compute_confidence


class BBReversionStrategy(BaseStrategy):
    name = "bb_reversion"

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
        close = float(latest["close"])
        atr_val = float(latest.get("atr", 0))
        rsi_val = float(latest.get("rsi", 50))

        # Bollinger Band columns
        bbl_col = [c for c in df.columns if c.startswith("BBL_")]
        bbu_col = [c for c in df.columns if c.startswith("BBU_")]
        bbm_col = [c for c in df.columns if c.startswith("BBM_")]

        if not bbl_col or not bbu_col or not bbm_col:
            return None

        bb_lower = float(latest[bbl_col[0]])
        bb_upper = float(latest[bbu_col[0]])
        bb_mid = float(latest[bbm_col[0]])

        if any(pd.isna(v) for v in [bb_lower, bb_upper, bb_mid]):
            return None

        # Stochastic RSI
        stoch_k_col = [c for c in df.columns if c.startswith("STOCHRSIk_")]
        stoch_k_val = float(latest[stoch_k_col[0]]) if stoch_k_col else 50.0

        # Volume
        volume_now = float(latest.get("volume", 0))
        vol_sma = float(latest.get("volume_sma", 1))
        vol_ratio = volume_now / vol_sma if vol_sma > 0 else 0

        direction = None
        stoch_bonus = 0.0

        # --- Long Entry: price at/below lower BB, RSI oversold ---
        if close <= bb_lower and rsi_val < 38:
            direction = SignalDirection.LONG
            stoch_bonus = 0.3 if stoch_k_val < 25 else 0.0

        # --- Short Entry: price at/above upper BB, RSI overbought ---
        elif close >= bb_upper and rsi_val > 62:
            direction = SignalDirection.SHORT
            stoch_bonus = 0.3 if stoch_k_val > 75 else 0.0

        if direction is None:
            return None

        # --- Scoring ---
        regime_align = 1.0 if regime == "RANGING" else 0.3
        primary_score = 1.0  # BB touch confirmed

        # RSI confirmation strength
        if direction == SignalDirection.LONG:
            rsi_strength = max(0, (35 - rsi_val) / 35)
        else:
            rsi_strength = max(0, (rsi_val - 65) / 35)
        confirm_score = min(rsi_strength + 0.3, 1.0)

        volume_sc = min(vol_ratio / 1.5, 1.0) if vol_ratio > 0.8 else 0.3
        volume_sc = min(volume_sc + stoch_bonus, 1.0)
        htf_score = 0.5  # Neutral for mean-reversion

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
        sl_buffer = 1.2 * atr_val

        if direction == SignalDirection.LONG:
            sl = bb_lower - sl_buffer
            tp1 = bb_mid  # Conservative: return to midline
            tp2 = bb_mid + (bb_mid - bb_lower) * 0.3
        else:
            sl = bb_upper + sl_buffer
            tp1 = bb_mid
            tp2 = bb_mid - (bb_upper - bb_mid) * 0.3

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
            f"conf={confidence:.1f} entry={close} SL={sl:.2f} TP1={tp1:.2f} "
            f"RSI={rsi_val:.1f} StochRSI={stoch_k_val:.1f}"
        )
        return signal
