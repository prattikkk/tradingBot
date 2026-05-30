"""
strategies/ensemble.py
Ensemble strategy — aggregates signals from all strategies.

A trade is only taken when ≥ MIN_AGREEMENT strategies agree on direction.
The final confidence is the weighted average of agreeing signals.
"""
from __future__ import annotations
from typing import Optional

import pandas as pd

from core.regime import MarketRegime, detect_market_regime
from core.signal import Signal, Direction
from core.calibration import SYMBOL_CALIBRATION
from strategies.supertrend_rsi import SuperTrendRSIStrategy
from strategies.ema_adx_volume import EMAAdxVolumeStrategy
from strategies.breakout_momentum import BreakoutMomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from config import CONFIG
from utils.logger import get_logger

log = get_logger("Ensemble")

# Weights per strategy (must sum to 1.0)
WEIGHTS = {
    SuperTrendRSIStrategy.name:     0.30,
    EMAAdxVolumeStrategy.name:      0.30,
    BreakoutMomentumStrategy.name:  0.20,
    MeanReversionStrategy.name:     0.20,
}

REGIME_ALLOWLIST = {
    MarketRegime.TRENDING: {
        SuperTrendRSIStrategy.name,
        EMAAdxVolumeStrategy.name,
    },
    MarketRegime.RANGING: {
        MeanReversionStrategy.name,
        BreakoutMomentumStrategy.name,
    },
    MarketRegime.HIGH_VOL: {
        EMAAdxVolumeStrategy.name,
        BreakoutMomentumStrategy.name,
    },
    MarketRegime.UNKNOWN: set(WEIGHTS.keys()),
}


class EnsembleStrategy:
    name = "ensemble"

    def __init__(self):
        self._strategies = [
            SuperTrendRSIStrategy(),
            EMAAdxVolumeStrategy(),
            BreakoutMomentumStrategy(),
            MeanReversionStrategy(),
        ]
        self.last_skip_reason: str = ""

    @staticmethod
    def _min_agree(runnable_count: int) -> int:
        return max(1, int(runnable_count) - 2)

    @staticmethod
    def _regime_settings(symbol: str) -> tuple[float, int, float]:
        return (
            float(
                SYMBOL_CALIBRATION.get(
                    symbol,
                    "regime_adx_trending",
                    CONFIG.strategy.regime_adx_trending,
                )
            ),
            int(
                SYMBOL_CALIBRATION.get(
                    symbol,
                    "regime_vol_window",
                    CONFIG.strategy.regime_vol_window,
                )
            ),
            float(
                SYMBOL_CALIBRATION.get(
                    symbol,
                    "regime_high_vol_quantile",
                    CONFIG.strategy.regime_high_vol_quantile,
                )
            ),
        )

    @staticmethod
    def _fallback_streak_threshold(symbol: str) -> int:
        default_cycles = int(getattr(CONFIG.strategy, "no_signal_fallback_cycles", 3))
        return max(1, int(SYMBOL_CALIBRATION.get(symbol, "fallback_no_signal_cycles", default_cycles)))

    def _fallback_signal(
        self,
        symbol: str,
        regime: MarketRegime,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame],
        htf_df2: Optional[pd.DataFrame],
        no_signal_streak: int,
    ) -> Optional[Signal]:
        expand_allowlist = bool(SYMBOL_CALIBRATION.get(symbol, "fallback_expand_allowlist", True))
        if expand_allowlist:
            candidates_pool = list(self._strategies)
        else:
            allowed = REGIME_ALLOWLIST.get(regime, set(WEIGHTS.keys()))
            candidates_pool = [s for s in self._strategies if s.name in allowed]

        fallback_candidates: list[Signal] = []
        for strat in candidates_pool:
            sig = strat.generate(symbol, df, htf_df, htf_df2)
            if sig is not None and sig.direction != Direction.FLAT:
                fallback_candidates.append(sig)

        if not fallback_candidates:
            return None

        selected = max(fallback_candidates, key=lambda s: s.confidence)
        confidence_scale = float(SYMBOL_CALIBRATION.get(symbol, "fallback_confidence_scale", 0.95))
        scaled_confidence = max(0.0, min(1.0, selected.confidence * confidence_scale))

        return Signal(
            symbol=symbol,
            direction=selected.direction,
            confidence=round(scaled_confidence, 3),
            strategy=self.name,
            entry_price=selected.entry_price,
            stop_loss=selected.stop_loss,
            take_profit_1=selected.take_profit_1,
            take_profit_2=selected.take_profit_2,
            atr=selected.atr,
            reason=(
                f"Regime={regime.value} | Fallback[{selected.strategy}] "
                f"streak={no_signal_streak}"
            ),
            htf_bias=selected.htf_bias,
            extra={
                **(selected.extra or {}),
                "regime": regime.value,
                "fallback_activated": True,
                "fallback_source": selected.strategy,
                "fallback_streak": int(no_signal_streak),
            },
        )

    def generate(
        self,
        symbol: str,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        htf_df2: Optional[pd.DataFrame] = None,
        no_signal_streak: int = 0,
    ) -> Optional[Signal]:
        self.last_skip_reason = ""

        trend_adx_threshold, vol_window, high_vol_quantile = self._regime_settings(symbol)

        regime = detect_market_regime(
            df,
            adx_period=CONFIG.strategy.adx_period,
            trend_adx_threshold=trend_adx_threshold,
            vol_window=vol_window,
            high_vol_quantile=high_vol_quantile,
        )
        allowed = REGIME_ALLOWLIST.get(regime, set(WEIGHTS.keys()))
        runnable = [s for s in self._strategies if s.name in allowed]
        if not runnable:
            self.last_skip_reason = f"regime_blocked({regime.value})"
            return None

        signals: list[Signal] = []

        for strat in runnable:
            sig = strat.generate(symbol, df, htf_df, htf_df2)
            if sig is not None and sig.direction != Direction.FLAT:
                signals.append(sig)
                log.debug(f"  [{symbol}] {strat.name}: {sig.direction.value} conf={sig.confidence:.0%}")

        if not signals:
            fallback_enabled = bool(getattr(CONFIG.strategy, "no_signal_fallback_enabled", True))
            threshold = self._fallback_streak_threshold(symbol)
            if fallback_enabled and no_signal_streak >= threshold:
                fallback = self._fallback_signal(
                    symbol,
                    regime,
                    df,
                    htf_df,
                    htf_df2,
                    no_signal_streak,
                )
                if fallback is not None:
                    return fallback
            self.last_skip_reason = f"no_substrategy_signal(regime={regime.value})"
            return None

        # Count votes by direction
        long_sigs  = [s for s in signals if s.direction == Direction.LONG]
        short_sigs = [s for s in signals if s.direction == Direction.SHORT]

        min_agree = self._min_agree(len(runnable))
        winning_sigs = None
        direction = Direction.FLAT

        if len(long_sigs) >= min_agree:
            winning_sigs = long_sigs
            direction = Direction.LONG
        elif len(short_sigs) >= min_agree:
            winning_sigs = short_sigs
            direction = Direction.SHORT

        if not winning_sigs:
            log.debug(f"[{symbol}] No ensemble agreement (L={len(long_sigs)} S={len(short_sigs)})")
            self.last_skip_reason = (
                f"no_ensemble_agreement(regime={regime.value},"
                f"L={len(long_sigs)},S={len(short_sigs)},min={min_agree})"
            )
            return None

        # Weighted confidence
        total_weight = sum(WEIGHTS.get(s.strategy, 0.33) for s in winning_sigs)
        weighted_conf = sum(
            s.confidence * WEIGHTS.get(s.strategy, 0.33)
            for s in winning_sigs
        ) / max(total_weight, 1e-9)

        disagreement_count = len(signals) - len(winning_sigs)
        disagreement_penalty = 1.0
        if disagreement_count > 0:
            disagreement_penalty = max(0.65, 1.0 - 0.12 * disagreement_count)
            weighted_conf *= disagreement_penalty

        # Use the highest-weighted signal's price levels as the reference
        ref = max(winning_sigs, key=lambda s: WEIGHTS.get(s.strategy, 0.33))

        reasons = " | ".join(
            f"{s.strategy}({s.confidence:.0%})" for s in winning_sigs
        )

        if disagreement_count > 0:
            reasons = (
                f"{reasons} | dissent={disagreement_count} "
                f"(penalty x{disagreement_penalty:.2f})"
            )

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=round(weighted_conf, 3),
            strategy=self.name,
            entry_price=ref.entry_price,
            stop_loss=ref.stop_loss,
            take_profit_1=ref.take_profit_1,
            take_profit_2=ref.take_profit_2,
            atr=ref.atr,
            reason=(
                f"Regime={regime.value} | Ensemble "
                f"[{len(winning_sigs)}/{len(runnable)}]: {reasons}"
            ),
            htf_bias=ref.htf_bias,
            extra={
                "regime": regime.value,
                "disagreement_count": disagreement_count,
                "disagreement_penalty": round(disagreement_penalty, 3),
            },
        )
