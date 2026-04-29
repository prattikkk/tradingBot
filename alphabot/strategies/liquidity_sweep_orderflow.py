"""
AlphaBot Strategy - Liquidity Sweep + Order Flow.

Detects institutional stop hunts (liquidity sweeps) confirmed by an
order-flow proxy (delta divergence from OHLCV candles).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.strategies.base import BaseStrategy
from alphabot.strategies.signal import Signal, SignalDirection


DEFAULT_CONFIG = {
    "enabled": True,
    "swing_lookback": 10,
    "sweep_min_wick_pct": 0.05,
    "delta_window": 5,
    "cvd_slope_window": 20,
    "min_delta_ratio": 0.1,
    "htf_ema_fast": 20,
    "htf_ema_slow": 50,
    "min_confidence": 0.40,
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp1_atr_mult": 2.5,
    "tp2_atr_mult": 4.0,
}


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_delta(df: pd.DataFrame) -> pd.Series:
    """
    Approximate per-bar delta (buy vol - sell vol) from OHLCV candles.
    """
    hl_range = df["high"] - df["low"]
    safe_range = hl_range.replace(0, np.nan)

    buy_vol = df["volume"] * (df["close"] - df["low"]) / safe_range
    sell_vol = df["volume"] * (df["high"] - df["close"]) / safe_range
    return (buy_vol - sell_vol).fillna(0)


def compute_cvd(delta: pd.Series) -> pd.Series:
    return delta.cumsum()


def find_swing_highs(df: pd.DataFrame, lookback: int) -> pd.Series:
    highs = df["high"]
    is_swing = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window = highs.iloc[i - lookback : i + lookback + 1]
        if highs.iloc[i] == window.max():
            is_swing.iloc[i] = True
    return is_swing


def find_swing_lows(df: pd.DataFrame, lookback: int) -> pd.Series:
    lows = df["low"]
    is_swing = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window = lows.iloc[i - lookback : i + lookback + 1]
        if lows.iloc[i] == window.min():
            is_swing.iloc[i] = True
    return is_swing


def get_recent_swing_levels(
    df: pd.DataFrame,
    lookback: int,
    n_levels: int = 3,
    current_idx: int = -1,
) -> Tuple[list[float], list[float]]:
    swing_highs = find_swing_highs(df, lookback)
    swing_lows = find_swing_lows(df, lookback)

    past_highs = df.loc[swing_highs, "high"].iloc[:current_idx]
    past_lows = df.loc[swing_lows, "low"].iloc[:current_idx]

    return (
        past_highs.iloc[-n_levels:].tolist() if len(past_highs) >= 1 else [],
        past_lows.iloc[-n_levels:].tolist() if len(past_lows) >= 1 else [],
    )


def htf_bias(htf_df: Optional[pd.DataFrame], fast: int = 20, slow: int = 50) -> str:
    if htf_df is None or len(htf_df) < slow:
        return "neutral"

    ema_fast = htf_df["close"].ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_slow = htf_df["close"].ewm(span=slow, adjust=False).mean().iloc[-1]

    if ema_fast > ema_slow * 1.001:
        return "bull"
    if ema_fast < ema_slow * 0.999:
        return "bear"
    return "neutral"


def detect_bullish_sweep(
    bar: pd.Series,
    swing_lows: list[float],
    min_wick_pct: float,
) -> Tuple[bool, float, float]:
    if not swing_lows:
        return False, 0.0, 0.0

    low = float(bar["low"])
    close = float(bar["close"])

    for level in sorted(swing_lows, reverse=True):
        wick_size = level - low
        wick_pct = (wick_size / level) * 100 if level else 0.0

        if low < level and close > level and wick_pct >= min_wick_pct:
            return True, level, wick_pct

    return False, 0.0, 0.0


def detect_bearish_sweep(
    bar: pd.Series,
    swing_highs: list[float],
    min_wick_pct: float,
) -> Tuple[bool, float, float]:
    if not swing_highs:
        return False, 0.0, 0.0

    high = float(bar["high"])
    close = float(bar["close"])

    for level in sorted(swing_highs):
        wick_size = high - level
        wick_pct = (wick_size / level) * 100 if level else 0.0

        if high > level and close < level and wick_pct >= min_wick_pct:
            return True, level, wick_pct

    return False, 0.0, 0.0


def score_signal(
    direction: str,
    wick_pct: float,
    delta_ratio: float,
    cvd_slope: float,
    bias: str,
    min_delta_ratio: float,
) -> float:
    """
    Returns a confidence score in [0.0, 1.0].
    """
    score = 0.0

    sweep_score = min(wick_pct / 0.30, 1.0)
    score += 0.35 * sweep_score

    if delta_ratio >= min_delta_ratio:
        delta_score = min(delta_ratio / 0.50, 1.0)
        score += 0.35 * delta_score

    if direction == "long" and bias == "bull":
        score += 0.20
    elif direction == "short" and bias == "bear":
        score += 0.20
    elif bias == "neutral":
        score += 0.10

    if direction == "long" and cvd_slope > 0:
        score += 0.10
    elif direction == "short" and cvd_slope < 0:
        score += 0.10

    return round(min(score, 1.0), 4)


class LiquiditySweepOrderFlowStrategy(BaseStrategy):
    name = "liquidity_sweep_orderflow"

    @staticmethod
    def _runtime_config() -> Dict[str, float | int]:
        return {
            "enabled": bool(settings.liquidity_sweep_orderflow_enabled),
            "swing_lookback": int(settings.liquidity_sweep_orderflow_swing_lookback),
            "sweep_min_wick_pct": float(settings.liquidity_sweep_orderflow_sweep_min_wick_pct),
            "delta_window": int(settings.liquidity_sweep_orderflow_delta_window),
            "cvd_slope_window": int(settings.liquidity_sweep_orderflow_cvd_slope_window),
            "min_delta_ratio": float(settings.liquidity_sweep_orderflow_min_delta_ratio),
            "htf_ema_fast": int(settings.liquidity_sweep_orderflow_htf_ema_fast),
            "htf_ema_slow": int(settings.liquidity_sweep_orderflow_htf_ema_slow),
            "min_confidence": float(settings.liquidity_sweep_orderflow_min_confidence),
            "atr_period": int(settings.liquidity_sweep_orderflow_atr_period),
            "sl_atr_mult": float(settings.liquidity_sweep_orderflow_sl_atr_mult),
            "tp1_atr_mult": float(settings.liquidity_sweep_orderflow_tp1_atr_mult),
            "tp2_atr_mult": float(settings.liquidity_sweep_orderflow_tp2_atr_mult),
        }

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str,
        timeframe: str,
        higher_tf_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        cfg = {**DEFAULT_CONFIG, **self._runtime_config()}

        if not cfg.get("enabled", True):
            return None

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            return None

        min_bars = int(cfg["swing_lookback"]) * 2 + int(cfg["cvd_slope_window"]) + 5
        if len(df) < min_bars:
            logger.debug(
                f"[{self.name}] {symbol}: not enough bars ({len(df)} < {min_bars})"
            )
            return None

        work_df = df.copy()
        delta = compute_delta(work_df)
        cvd = compute_cvd(delta)
        atr = compute_atr(work_df, int(cfg["atr_period"]))

        avg_vol = work_df["volume"].rolling(int(cfg["cvd_slope_window"])).mean()
        avg_vol_val = float(avg_vol.iloc[-1]) if len(avg_vol) else 0.0
        if not np.isfinite(avg_vol_val) or avg_vol_val <= 0:
            return None

        recent_delta = float(delta.iloc[-int(cfg["delta_window"]) :].sum())
        delta_ratio = abs(recent_delta) / avg_vol_val if avg_vol_val > 0 else 0.0

        current_atr = float(atr.iloc[-1]) if len(atr) else 0.0
        if not np.isfinite(current_atr) or current_atr <= 0:
            return None

        cvd_now = float(cvd.iloc[-1])
        cvd_past = float(cvd.iloc[-int(cfg["cvd_slope_window"])])
        cvd_slope = cvd_now - cvd_past
        if not np.isfinite(cvd_slope):
            return None

        swing_highs_list, swing_lows_list = get_recent_swing_levels(
            work_df,
            lookback=int(cfg["swing_lookback"]),
            n_levels=3,
            current_idx=-1,
        )

        bar = work_df.iloc[-1]
        bias = htf_bias(
            higher_tf_df,
            fast=int(cfg["htf_ema_fast"]),
            slow=int(cfg["htf_ema_slow"]),
        )

        bull_sweep, bull_level, bull_wick_pct = detect_bullish_sweep(
            bar, swing_lows_list, float(cfg["sweep_min_wick_pct"])
        )
        bear_sweep, bear_level, bear_wick_pct = detect_bearish_sweep(
            bar, swing_highs_list, float(cfg["sweep_min_wick_pct"])
        )

        if bull_sweep and recent_delta > 0 and bias != "bear":
            confidence = score_signal(
                direction="long",
                wick_pct=bull_wick_pct,
                delta_ratio=delta_ratio,
                cvd_slope=cvd_slope,
                bias=bias,
                min_delta_ratio=float(cfg["min_delta_ratio"]),
            )
            if confidence >= float(cfg["min_confidence"]):
                entry = float(bar["close"])
                sl = entry - float(cfg["sl_atr_mult"]) * current_atr
                tp1 = entry + float(cfg["tp1_atr_mult"]) * current_atr
                tp2 = entry + float(cfg["tp2_atr_mult"]) * current_atr
                return self._build_signal(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    confidence_ratio=confidence,
                    entry=entry,
                    sl=sl,
                    tp1=tp1,
                    tp2=tp2,
                    regime=regime,
                    timeframe=timeframe,
                    reason=(
                        f"Bullish liquidity sweep below {bull_level:.4f} "
                        f"(wick {bull_wick_pct:.3f}%), "
                        f"positive delta ratio {delta_ratio:.3f}, "
                        f"HTF bias={bias}, CVD slope={cvd_slope:.2f}"
                    ),
                )

        if bear_sweep and recent_delta < 0 and bias != "bull":
            confidence = score_signal(
                direction="short",
                wick_pct=bear_wick_pct,
                delta_ratio=delta_ratio,
                cvd_slope=cvd_slope,
                bias=bias,
                min_delta_ratio=float(cfg["min_delta_ratio"]),
            )
            if confidence >= float(cfg["min_confidence"]):
                entry = float(bar["close"])
                sl = entry + float(cfg["sl_atr_mult"]) * current_atr
                tp1 = entry - float(cfg["tp1_atr_mult"]) * current_atr
                tp2 = entry - float(cfg["tp2_atr_mult"]) * current_atr
                return self._build_signal(
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    confidence_ratio=confidence,
                    entry=entry,
                    sl=sl,
                    tp1=tp1,
                    tp2=tp2,
                    regime=regime,
                    timeframe=timeframe,
                    reason=(
                        f"Bearish liquidity sweep above {bear_level:.4f} "
                        f"(wick {bear_wick_pct:.3f}%), "
                        f"negative delta ratio {delta_ratio:.3f}, "
                        f"HTF bias={bias}, CVD slope={cvd_slope:.2f}"
                    ),
                )

        return None

    def _build_signal(
        self,
        symbol: str,
        direction: SignalDirection,
        confidence_ratio: float,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        regime: str,
        timeframe: str,
        reason: str,
    ) -> Optional[Signal]:
        if not all(np.isfinite(v) for v in [entry, sl, tp1, tp2]):
            return None
        if entry <= 0 or sl <= 0 or tp1 <= 0 or tp2 <= 0:
            return None

        confidence_pct = round(min(max(confidence_ratio * 100.0, 0.0), 100.0), 1)

        signal = Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence_pct,
            entry_price=Decimal(str(round(entry, 8))),
            stop_loss=Decimal(str(round(sl, 8))),
            take_profit_1=Decimal(str(round(tp1, 8))),
            take_profit_2=Decimal(str(round(tp2, 8))),
            strategy_name=self.name,
            regime=regime,
            timeframe=timeframe,
        )

        logger.info(
            f"[{self.name}] {symbol} {direction.value} conf={confidence_pct:.1f} "
            f"entry={entry:.4f} SL={sl:.4f} TP1={tp1:.4f} | {reason}"
        )
        return signal
