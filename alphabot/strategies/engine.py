"""
AlphaBot Strategy Engine — Regime-to-Strategy Router.
Reads the current regime and activates the appropriate sub-strategy.
Produces BUY/SELL/HOLD signals with confidence scoring.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from alphabot.config import settings
from alphabot.data.data_store import DataStore
from alphabot.regime.detector import MarketRegime, RegimeDetector
from alphabot.strategies.base import BaseStrategy
from alphabot.strategies.ema_adx_volume import EmaAdxVolumeStrategy
from alphabot.strategies.liquidity_sweep_orderflow import LiquiditySweepOrderFlowStrategy
from alphabot.strategies.orderflow_liquidity_sweep import OrderFlowLiquiditySweepStrategy
from alphabot.strategies.signal import Signal
from alphabot.strategies.supertrend_pullback import SupertrendPullbackStrategy
from alphabot.strategies.supertrend_rsi import SupertrendRsiStrategy
from alphabot.strategies.supertrend_trail import SupertrendTrailStrategy
from alphabot.utils.indicators import compute_all_indicators


# Regime → Strategy mapping
REGIME_STRATEGY_MAP: Dict[MarketRegime, List[type]] = {
    MarketRegime.TRENDING_UP: [
        SupertrendTrailStrategy,
        SupertrendPullbackStrategy,
        SupertrendRsiStrategy,
        EmaAdxVolumeStrategy,
    ],
    MarketRegime.TRENDING_DOWN: [
        SupertrendTrailStrategy,
        SupertrendPullbackStrategy,
        SupertrendRsiStrategy,
        EmaAdxVolumeStrategy,
    ],
    MarketRegime.RANGING: [
        LiquiditySweepOrderFlowStrategy,
        OrderFlowLiquiditySweepStrategy,
    ],
    MarketRegime.HIGH_VOLATILITY: [
        LiquiditySweepOrderFlowStrategy,
        OrderFlowLiquiditySweepStrategy,
    ],
    MarketRegime.UNCLEAR: [],  # No trades in unclear regime
}

SPECIALIZED_TREND_STRATEGIES = {"supertrend_trail", "supertrend_pullback"}
SPECIALIZED_SELECTION_MARGIN = 3.0


class StrategyEngine:
    """
    Strategy selector — reads regime, routes to correct strategy,
    emits signals with confidence scoring.
    """

    def __init__(self, data_store: DataStore, regime_detector: RegimeDetector):
        self.data_store = data_store
        self.regime_detector = regime_detector
        self._strategies: Dict[str, BaseStrategy] = {
            "supertrend_trail": SupertrendTrailStrategy(),
            "supertrend_pullback": SupertrendPullbackStrategy(),
            "supertrend_rsi": SupertrendRsiStrategy(),
            "ema_adx_volume": EmaAdxVolumeStrategy(),
            "liquidity_sweep_orderflow": LiquiditySweepOrderFlowStrategy(),
            "orderflow_liquidity_sweep": OrderFlowLiquiditySweepStrategy(),
        }

    def evaluate(
        self,
        symbol: str,
        timeframe: str | None = None,
        bias_timeframe: str | None = None,
    ) -> Optional[Signal]:
        """
        Evaluate current conditions for a symbol and generate a signal.

        1. Detect regime
        2. Select strategy based on regime
        3. Compute indicators
        4. Run strategy logic
        5. Return signal or None
        """
        tf = timeframe or settings.primary_timeframe
        regime = self.regime_detector.detect(symbol, tf)

        if regime == MarketRegime.UNCLEAR:
            logger.debug(f"[Engine] {symbol}: regime UNCLEAR — no trade")
            return None

        if regime == MarketRegime.HIGH_VOLATILITY:
            logger.info(f"[Engine] {symbol}: HIGH_VOLATILITY regime — no trades")

        # Get strategy classes for this regime
        strategy_classes = REGIME_STRATEGY_MAP.get(regime, [])
        if not strategy_classes:
            logger.debug(f"[Engine] {symbol}: no strategy for regime {regime}")
            return None

        # Get OHLCV data with indicators
        df = self.data_store.get_dataframe(symbol, tf)
        if df.empty or len(df) < 50:
            logger.warning(f"[Engine] {symbol}: insufficient data ({len(df)} candles)")
            return None

        indicator_config = {
            "ema_fast": settings.ema_fast,
            "ema_slow": settings.ema_slow,
            "ema_long": settings.ema_long_period,
            "atr_period": settings.atr_period,
            "adx_period": settings.adx_period,
            "rsi_period": settings.rsi_period,
            "volume_sma_period": settings.volume_sma_period,
            "ema_slope_period": settings.ema_slope_period,
            "supertrend_period": settings.supertrend_period,
            "supertrend_multiplier": settings.supertrend_multiplier,
        }
        df = compute_all_indicators(df, indicator_config)

        # Higher timeframe for confirmation
        htf_df = None
        htf = bias_timeframe or ("1h" if tf != "1h" else "4h")
        if htf and self.data_store.has_enough_data(symbol, htf, min_candles=30):
            htf_df = self.data_store.get_dataframe(symbol, htf)
            htf_df = compute_all_indicators(htf_df, indicator_config)

        # Run each eligible strategy and pick the strongest signal
        candidate_signals: List[Signal] = []
        for strategy_cls in strategy_classes:
            strategy_name = strategy_cls.name if hasattr(strategy_cls, 'name') else strategy_cls.__name__
            strategy = self._strategies.get(strategy_name)
            if strategy is None:
                strategy = strategy_cls()
                self._strategies[strategy_name] = strategy
            signal = strategy.generate_signal(
                symbol=symbol,
                df=df,
                regime=regime.value,
                timeframe=tf,
                higher_tf_df=htf_df,
            )
            if signal:
                candidate_signals.append(signal)

        best_signal = self._select_best_signal(candidate_signals)

        if best_signal:
            candidate_summary = ", ".join(
                f"{sig.strategy_name}:{sig.direction.value}:{sig.confidence:.1f}"
                for sig in sorted(candidate_signals, key=lambda s: s.confidence, reverse=True)
            )
            logger.info(
                f"[Engine] {symbol}: Signal generated — "
                f"{best_signal.direction.value} via {best_signal.strategy_name} "
                f"confidence={best_signal.confidence:.1f} regime={regime.value} "
                f"candidates=[{candidate_summary}]"
            )
        else:
            logger.debug(f"[Engine] {symbol}: no signal generated for regime {regime.value}")

        return best_signal

    @staticmethod
    def _select_best_signal(candidates: List[Signal]) -> Optional[Signal]:
        if not candidates:
            return None

        best = max(candidates, key=lambda s: s.confidence)
        if best.strategy_name != "supertrend_rsi":
            return best

        specialized = [
            sig
            for sig in candidates
            if sig.strategy_name in SPECIALIZED_TREND_STRATEGIES
        ]
        if not specialized:
            return best

        specialized_best = max(specialized, key=lambda s: s.confidence)
        if specialized_best.confidence >= (best.confidence - SPECIALIZED_SELECTION_MARGIN):
            return specialized_best
        return best

    def evaluate_all(self) -> List[Signal]:
        """Evaluate all configured trading pairs and return any signals."""
        signals = []
        for symbol in settings.trading_pairs:
            try:
                signal = self.evaluate(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"[Engine] Error evaluating {symbol}: {e}")
        return signals
