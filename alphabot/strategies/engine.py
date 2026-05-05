"""
AlphaBot Strategy Engine — Regime-to-Strategy Router.
Reads the current regime and activates the appropriate sub-strategy.
Produces BUY/SELL/HOLD signals with confidence scoring.
"""

from __future__ import annotations

import datetime
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
        LiquiditySweepOrderFlowStrategy,
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
SPECIALIZED_SELECTION_MARGIN = 1.0


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
            logger.info(f"[Engine] {symbol}: HIGH_VOLATILITY regime — range strategies only")

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

        # Higher timeframe confirmation frames (strict AND gate across configured bias tfs).
        bias_timeframes: List[str] = []
        if bias_timeframe:
            bias_timeframes.append(bias_timeframe)
        for tf_name in list(getattr(settings, "bias_timeframes", []) or []):
            if tf_name and tf_name not in bias_timeframes and tf_name != tf:
                bias_timeframes.append(tf_name)
        if not bias_timeframes:
            bias_timeframes = ["1h" if tf != "1h" else "4h"]

        htf_frames: Dict[str, pd.DataFrame] = {}
        for htf in bias_timeframes:
            if not self.data_store.has_enough_data(symbol, htf, min_candles=30):
                continue
            htf_df = self.data_store.get_dataframe(symbol, htf)
            htf_df = compute_all_indicators(htf_df, indicator_config)
            if self._is_htf_stale(htf_df, htf):
                logger.warning(f"[Engine] {symbol}: stale HTF data on {htf} — skipping signal")
                return None
            htf_frames[htf] = htf_df

        missing_bias = [name for name in bias_timeframes if name not in htf_frames]
        if missing_bias:
            # FIX[5]: Require all configured bias timeframes (strict AND gate).
            logger.debug(
                f"[Engine] {symbol}: missing HTF bias frames {missing_bias} — no trade"
            )
            return None

        primary_htf_df = htf_frames[bias_timeframes[0]]

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
                higher_tf_df=primary_htf_df,
            )
            if signal:
                if not self._passes_multi_htf_gate(signal, htf_frames):
                    logger.info(
                        f"[Engine] {symbol}: {signal.strategy_name} rejected by multi-HTF bias gate"
                    )
                    continue
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

    @staticmethod
    def _timeframe_seconds(timeframe: str) -> int:
        units = {
            "m": 60,
            "h": 3600,
            "d": 86400,
        }
        if not timeframe:
            return 0
        unit = timeframe[-1].lower()
        try:
            value = int(timeframe[:-1])
        except Exception:
            return 0
        return value * units.get(unit, 0)

    @classmethod
    def _is_htf_stale(cls, htf_df: pd.DataFrame, timeframe: str) -> bool:
        if htf_df.empty:
            return True
        if "open_time" not in htf_df.columns:
            return False

        latest_open = htf_df.iloc[-1].get("open_time")
        if latest_open is None or pd.isna(latest_open):
            return True

        try:
            latest_dt = pd.Timestamp(latest_open).to_pydatetime()
        except Exception:
            return True
        if latest_dt.tzinfo is not None:
            latest_dt = latest_dt.astimezone(datetime.UTC).replace(tzinfo=None)

        now_utc = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        age_seconds = max(0.0, (now_utc - latest_dt).total_seconds())
        tf_seconds = cls._timeframe_seconds(timeframe)
        if tf_seconds <= 0:
            return False
        # FIX[5]: Staleness threshold is configurable in bars (default 2.5 bars).
        max_staleness_bars = float(getattr(settings, "max_htf_staleness_bars", 2.5) or 2.5)
        return age_seconds > (tf_seconds * max_staleness_bars)

    @staticmethod
    def _passes_single_htf_gate(signal: Signal, htf_df: pd.DataFrame) -> bool:
        if htf_df.empty:
            return False

        latest = htf_df.iloc[-1]
        close = latest.get("close")
        ema_long = latest.get("ema_long")
        rsi_val = latest.get("rsi")

        if pd.isna(close) or pd.isna(ema_long):
            return False

        is_long = signal.direction.value == "LONG"
        if is_long and float(close) < float(ema_long):
            return False
        if not is_long and float(close) > float(ema_long):
            return False

        if rsi_val is not None and not pd.isna(rsi_val):
            if is_long and float(rsi_val) < 50.0:
                return False
            if not is_long and float(rsi_val) > 50.0:
                return False

        st_dir_col = next((c for c in htf_df.columns if c.startswith("SUPERTd_")), None)
        if st_dir_col:
            st_dir = latest.get(st_dir_col)
            if st_dir is not None and not pd.isna(st_dir):
                if is_long and float(st_dir) <= 0:
                    return False
                if not is_long and float(st_dir) >= 0:
                    return False

        return True

    def _passes_multi_htf_gate(self, signal: Signal, htf_frames: Dict[str, pd.DataFrame]) -> bool:
        frame_results = {
            tf_name: self._passes_single_htf_gate(signal, htf_df)
            for tf_name, htf_df in htf_frames.items()
        }

        # supertrend_rsi can proceed when at least one bias timeframe confirms.
        if signal.strategy_name == "supertrend_rsi":
            passed = [tf_name for tf_name, ok in frame_results.items() if ok]
            if passed:
                return True

        for tf_name, ok in frame_results.items():
            if not ok:
                logger.info(
                    f"[Engine] {signal.symbol}: HTF gate failed on {tf_name} "
                    f"for {signal.direction.value}"
                )
                return False
        return True

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
