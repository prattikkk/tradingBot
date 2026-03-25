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
from alphabot.strategies.signal import Signal
from alphabot.strategies.ema_crossover import EMACrossoverStrategy
from alphabot.strategies.bb_reversion import BBReversionStrategy
from alphabot.strategies.atr_breakout import ATRBreakoutStrategy
from alphabot.strategies.pullback_momentum import PullbackMomentumStrategy
from alphabot.utils.indicators import compute_all_indicators


# Regime → Strategy mapping
REGIME_STRATEGY_MAP: Dict[MarketRegime, List[type]] = {
    MarketRegime.TRENDING_UP: [PullbackMomentumStrategy, EMACrossoverStrategy],
    MarketRegime.TRENDING_DOWN: [PullbackMomentumStrategy, EMACrossoverStrategy],
    # Balanced option: allow PMC to run in RANGING, but it will self-filter via HTF bias gates.
    MarketRegime.RANGING: [PullbackMomentumStrategy, BBReversionStrategy],
    MarketRegime.HIGH_VOLATILITY: [ATRBreakoutStrategy],
    MarketRegime.UNCLEAR: [],  # No trades in unclear regime
}


class StrategyEngine:
    """
    Strategy selector — reads regime, routes to correct strategy,
    emits signals with confidence scoring.
    """

    def __init__(self, data_store: DataStore, regime_detector: RegimeDetector):
        self.data_store = data_store
        self.regime_detector = regime_detector
        self._strategies: Dict[str, BaseStrategy] = {
            "ema_crossover": EMACrossoverStrategy(),
            "bb_reversion": BBReversionStrategy(),
            "atr_breakout": ATRBreakoutStrategy(),
            "pullback_momentum": PullbackMomentumStrategy(),
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

        # HIGH_VOLATILITY — ATR breakout only, with caution
        if regime == MarketRegime.HIGH_VOLATILITY:
            logger.info(f"[Engine] {symbol}: HIGH_VOLATILITY regime — cautious mode")

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
            "atr_period": settings.atr_period,
            "adx_period": settings.adx_period,
            "rsi_period": 14,
            "bb_period": settings.bb_period,
            "bb_std": settings.bb_std,
        }
        df = compute_all_indicators(df, indicator_config)

        # Higher timeframe for confirmation
        htf_df = None
        htf = bias_timeframe or ("1h" if tf != "1h" else "4h")
        if htf and self.data_store.has_enough_data(symbol, htf, min_candles=30):
            htf_df = self.data_store.get_dataframe(symbol, htf)
            htf_df = compute_all_indicators(htf_df, indicator_config)

        # Run each eligible strategy and pick the strongest signal
        best_signal: Optional[Signal] = None
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
            if signal and (best_signal is None or signal.confidence > best_signal.confidence):
                best_signal = signal

        if best_signal:
            logger.info(
                f"[Engine] {symbol}: Signal generated — "
                f"{best_signal.direction.value} via {best_signal.strategy_name} "
                f"confidence={best_signal.confidence:.1f} regime={regime.value}"
            )
        else:
            logger.debug(f"[Engine] {symbol}: no signal generated for regime {regime.value}")

        return best_signal

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
