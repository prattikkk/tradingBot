"""
AlphaBot Strategy D - Pullback Momentum Confluence (PMC)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.strategies.base import BaseStrategy
from alphabot.strategies.signal import Signal, SignalDirection, compute_confidence


class PullbackMomentumStrategy(BaseStrategy):
    """Pullback Momentum Confluence strategy."""

    name = "pullback_momentum"

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str,
        timeframe: str,
        higher_tf_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        # Rate-limit per (symbol,timeframe) to avoid log spam: log at most once per candle timestamp.
        key = f"{symbol}:{timeframe}"
        candle_ts = None
        try:
            candle_ts = str(df.iloc[-1].get("timestamp"))
        except Exception:
            candle_ts = None

        def dbg(reason: str) -> None:
            if not candle_ts:
                return

            # Track block reasons for auditability (in-memory, per-process)
            counts = getattr(self, "_pmc_block_counts", {})
            counts[reason] = int(counts.get(reason, 0)) + 1
            setattr(self, "_pmc_block_counts", counts)

            # Emit a periodic summary (once per hour per process)
            try:
                import time
                now = int(time.time())
                last_sum = int(getattr(self, "_pmc_last_summary", 0) or 0)
                if now - last_sum >= 3600 and counts:
                    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
                    top_str = ", ".join([f"{k}={v}" for k, v in top])
                    logger.info(f"[{self.name}] hourly block summary: {top_str}")
                    setattr(self, "_pmc_last_summary", now)
            except Exception:
                pass

            last = getattr(self, "_pmc_last_dbg", {})
            if last.get(key) == candle_ts:
                return
            last[key] = candle_ts
            setattr(self, "_pmc_last_dbg", last)
            logger.info(f"[{self.name}] {symbol} {timeframe} blocked: {reason}")

        if higher_tf_df is None or higher_tf_df.empty or len(higher_tf_df) < 80:
            dbg("htf_df_missing")
            return None
        if len(df) < 100:
            dbg("ltf_df_insufficient")
            return None

        cur = df.iloc[-1]
        prev = df.iloc[-2]

        htf_bias, htf_score = self._htf_trend_bias(higher_tf_df)
        if htf_bias is None:
            try:
                htf_cur = higher_tf_df.iloc[-1]
                htf_ema_fast = float(htf_cur.get("ema_fast", 0))
                htf_ema_slow = float(htf_cur.get("ema_slow", 0))
                htf_slope = float(htf_cur.get("ema_fast_slope", 0))
                htf_adx_col = [c for c in higher_tf_df.columns if c.startswith("ADX_")]
                htf_adx = float(htf_cur[htf_adx_col[0]]) if htf_adx_col else 0.0
                dbg(
                    "htf_bias_none "
                    f"ema_fast={htf_ema_fast:.4f} ema_slow={htf_ema_slow:.4f} "
                    f"slope={htf_slope:.6f} adx={htf_adx:.2f} "
                    f"adx_min={float(settings.pmc_adx_min_htf):.2f}"
                )
            except Exception:
                dbg("htf_bias_none")
            return None

        close = float(cur["close"])
        high = float(cur["high"])
        low = float(cur["low"])
        open_ = float(cur["open"])
        ema_fast = float(cur.get("ema_fast", 0))
        atr_val = float(cur.get("atr", 0))
        if ema_fast <= 0 or atr_val <= 0:
            dbg("missing_ema_or_atr")
            return None

        pullback_ok, pullback_score = self._check_pullback(htf_bias, close, high, low, ema_fast, atr_val, df)
        if not pullback_ok:
            dbg("pullback_fail")
            return None

        rsi_now = float(cur.get("rsi", 50))
        rsi_prev = float(prev.get("rsi", 50))
        rsi_ok, rsi_score = self._check_rsi(htf_bias, rsi_now, rsi_prev)
        if not rsi_ok:
            dbg("rsi_fail")
            return None

        candle_ok, candle_score = self._check_candle(htf_bias, open_, close, high, low)
        if not candle_ok:
            dbg("candle_fail")
            return None

        volume_now = float(cur.get("volume", 0))
        vol_sma = float(cur.get("volume_sma", 1))
        vol_ratio = volume_now / vol_sma if vol_sma > 0 else 0
        vol_mult = float(settings.pmc_volume_multiplier)
        if vol_ratio < vol_mult:
            dbg(f"volume_fail ratio={vol_ratio:.2f}x < mult={vol_mult:.2f}x")
            return None
        vol_score = min(vol_ratio / (vol_mult * 1.5), 1.0)

        macdh_col = [c for c in df.columns if c.startswith("MACDh_")]
        macd_hist = float(cur[macdh_col[0]]) if macdh_col else 0.0
        macd_prev = float(prev[macdh_col[0]]) if macdh_col else 0.0
        _, macd_score = self._check_macd(htf_bias, macd_hist, macd_prev)

        direction = SignalDirection.LONG if htf_bias == "up" else SignalDirection.SHORT
        regime_align = self._regime_align(regime, direction)

        confidence = compute_confidence(
            regime_alignment=regime_align,
            primary_indicator=htf_score,
            confirmation=(rsi_score * 0.5 + candle_score * 0.5),
            volume=vol_score,
            higher_tf=macd_score,
            weights={
                "regime": 30,
                "primary": 25,
                "confirmation": 20,
                "volume": 15,
                "higher_tf": 10,
            },
        )

        if confidence < settings.min_signal_confidence:
            dbg(f"confidence_fail conf={confidence:.1f} < min={float(settings.min_signal_confidence):.1f}")
            return None

        sl_mult = float(settings.pmc_atr_sl_multiplier)
        tp1_mult = float(settings.pmc_atr_tp1_multiplier)
        tp2_mult = float(settings.pmc_atr_tp2_multiplier)

        if direction == SignalDirection.LONG:
            sl = close - (sl_mult * atr_val)
            tp1 = close + (tp1_mult * atr_val)
            tp2 = close + (tp2_mult * atr_val)
        else:
            sl = close + (sl_mult * atr_val)
            tp1 = close - (tp1_mult * atr_val)
            tp2 = close - (tp2_mult * atr_val)

        sl_dist = abs(close - sl)
        tp_dist = abs(close - tp1)
        if sl_dist == 0 or (tp_dist / sl_dist) < float(settings.min_risk_reward):
            rr = (tp_dist / sl_dist) if sl_dist else 0.0
            dbg(f"rr_fail rr={rr:.2f} < min={float(settings.min_risk_reward):.2f}")
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
            primary_indicator_score=htf_score,
            confirmation_score=(rsi_score * 0.5 + candle_score * 0.5),
            volume_score=vol_score,
            higher_tf_score=macd_score,
        )
        logger.info(f"[{self.name}] {symbol} {direction.value} conf={confidence:.1f} pullback={pullback_score:.2f} vol={vol_ratio:.2f}x")
        return signal

    def _htf_trend_bias(self, htf_df: pd.DataFrame) -> tuple[Optional[str], float]:
        cur = htf_df.iloc[-1]
        ema_fast = float(cur.get("ema_fast", 0))
        ema_slow = float(cur.get("ema_slow", 0))
        ema_slope = float(cur.get("ema_fast_slope", 0))
        adx_col = [c for c in htf_df.columns if c.startswith("ADX_")]
        adx_val = float(cur[adx_col[0]]) if adx_col else 0.0
        if ema_fast == 0 or ema_slow == 0 or adx_val < float(settings.pmc_adx_min_htf):
            return None, 0.0
        if ema_fast > ema_slow and ema_slope > 0:
            return "up", min(adx_val / 40.0, 1.0)
        if ema_fast < ema_slow and ema_slope < 0:
            return "down", min(adx_val / 40.0, 1.0)
        return None, 0.0

    def _check_pullback(self, bias: str, close: float, high: float, low: float, ema_fast: float, atr_val: float, df: pd.DataFrame) -> tuple[bool, float]:
        tol = atr_val * 0.75
        if bias == "up":
            ema_touched = low <= ema_fast + tol
            resuming = close > ema_fast
            prev_lows = [float(df.iloc[i]["low"]) for i in range(-4, -1)]
            deep_enough = any(l <= ema_fast + tol * 1.5 for l in prev_lows)
        else:
            ema_touched = high >= ema_fast - tol
            resuming = close < ema_fast
            prev_highs = [float(df.iloc[i]["high"]) for i in range(-4, -1)]
            deep_enough = any(h >= ema_fast - tol * 1.5 for h in prev_highs)
        if not (ema_touched and resuming and deep_enough):
            return False, 0.0
        distance_score = max(0.5, 1.0 - abs(close - ema_fast) / atr_val)
        return True, distance_score

    def _check_rsi(self, bias: str, rsi_now: float, rsi_prev: float) -> tuple[bool, float]:
        if bias == "up":
            was_oversold = rsi_prev <= float(settings.pmc_rsi_oversold) + 5
            recovering = rsi_now >= float(settings.pmc_rsi_confirmation_long) and rsi_now > rsi_prev
            if not (was_oversold and recovering):
                return False, 0.0
            score = min((rsi_now - rsi_prev) / 10.0, 1.0)
        else:
            was_overbought = rsi_prev >= float(settings.pmc_rsi_overbought) - 5
            recovering = rsi_now <= float(settings.pmc_rsi_confirmation_short) and rsi_now < rsi_prev
            if not (was_overbought and recovering):
                return False, 0.0
            score = min((rsi_prev - rsi_now) / 10.0, 1.0)
        return True, max(score, 0.3)

    def _check_candle(self, bias: str, open_: float, close: float, high: float, low: float) -> tuple[bool, float]:
        candle_range = high - low
        if candle_range <= 0:
            return False, 0.0
        body = abs(close - open_)
        body_pct = body / candle_range
        upper_wick = high - max(close, open_)
        lower_wick = min(close, open_) - low
        min_body = float(settings.pmc_min_body_pct)

        if bias == "up":
            if close <= open_ or body_pct < min_body:
                return False, 0.0
            if (upper_wick / candle_range) > float(settings.pmc_max_upper_wick_long):
                return False, 0.0
        else:
            if close >= open_ or body_pct < min_body:
                return False, 0.0
            if (lower_wick / candle_range) > float(settings.pmc_max_lower_wick_short):
                return False, 0.0
        return True, min(body_pct / 0.6, 1.0)

    def _check_macd(self, bias: str, macd_hist: float, macd_prev: float) -> tuple[bool, float]:
        if bias == "up":
            return True, 1.0 if macd_hist > 0 else (0.6 if macd_hist > macd_prev else 0.2)
        return True, 1.0 if macd_hist < 0 else (0.6 if macd_hist < macd_prev else 0.2)

    @staticmethod
    def _regime_align(regime: str, direction: SignalDirection) -> float:
        if "HIGH_VOLATILITY" in regime:
            return 0.4
        if direction == SignalDirection.LONG:
            return 1.0 if "TRENDING_UP" in regime else 0.5 if "RANGING" in regime else 0.3
        return 1.0 if "TRENDING_DOWN" in regime else 0.5 if "RANGING" in regime else 0.3
