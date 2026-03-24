"""Tests for trading strategies and signal generation."""
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal

from alphabot.strategies.signal import Signal, SignalDirection, compute_confidence
from alphabot.strategies.ema_crossover import EMACrossoverStrategy
from alphabot.strategies.bb_reversion import BBReversionStrategy
from alphabot.strategies.atr_breakout import ATRBreakoutStrategy
from alphabot.regime.detector import MarketRegime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, base: float = 100.0, trend: float = 0.0,
             vol_factor: float = 1.0) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame."""
    np.random.seed(0)
    closes = []
    c = base
    for _ in range(n):
        c += trend + np.random.normal(0, 0.5 * vol_factor)
        closes.append(c)
    highs = [c + abs(np.random.normal(0, 0.3 * vol_factor)) for c in closes]
    lows = [c - abs(np.random.normal(0, 0.3 * vol_factor)) for c in closes]
    opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    volumes = [1000 + np.random.uniform(-200, 200) for _ in range(n)]
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ---------------------------------------------------------------------------
# Signal model tests
# ---------------------------------------------------------------------------

class TestSignalModel:
    def test_signal_creation(self):
        sig = Signal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            confidence=75.0,
            entry_price=Decimal("50000"),
            stop_loss=Decimal("49000"),
            take_profit_1=Decimal("51500"),
            take_profit_2=Decimal("52500"),
            strategy_name="ema_crossover",
            regime=MarketRegime.TRENDING_UP.value,
            timeframe="5m",
        )
        assert sig.direction == SignalDirection.LONG
        assert sig.confidence == 75.0
        assert sig.entry_price == Decimal("50000")

    def test_signal_risk_reward(self):
        sig = Signal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            confidence=70.0,
            entry_price=Decimal("100"),
            stop_loss=Decimal("95"),
            take_profit_1=Decimal("110"),
            take_profit_2=Decimal("115"),
            strategy_name="test",
            regime=MarketRegime.TRENDING_UP.value,
            timeframe="5m",
        )
        # SL distance = 5, TP1 distance = 10 → R:R = 2.0
        assert sig.risk_reward_ratio == pytest.approx(2.0)


class TestConfidenceScoring:
    def test_perfect_scores(self):
        # Each input is a 0-1 fraction. Perfect = 1.0 × weights = 100
        confidence = compute_confidence(1.0, 1.0, 1.0, 1.0, 1.0)
        assert confidence == pytest.approx(100.0, abs=0.1)

    def test_zero_scores(self):
        confidence = compute_confidence(0.0, 0.0, 0.0, 0.0, 0.0)
        assert confidence == pytest.approx(0.0, abs=0.1)

    def test_partial_scores(self):
        confidence = compute_confidence(0.8, 0.6, 0.5, 0.4, 0.3)
        assert 0 < confidence < 100


# ---------------------------------------------------------------------------
# EMA Crossover Strategy tests
# ---------------------------------------------------------------------------

class TestEmaCrossoverStrategy:
    @pytest.fixture
    def strategy(self):
        return EMACrossoverStrategy()

    def test_returns_signal_or_none(self, strategy):
        df = _make_df(n=100, trend=0.3)
        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.TRENDING_UP.value, "5m"
        )
        assert result is None or isinstance(result, Signal)

    def test_wrong_regime_returns_none(self, strategy):
        df = _make_df(n=100, trend=0.3)
        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.RANGING.value, "5m"
        )
        # EMA strategy should not fire in RANGING regime
        assert result is None

    def test_strategy_name(self, strategy):
        assert strategy.name == "ema_crossover"


# ---------------------------------------------------------------------------
# BB Mean Reversion Strategy tests
# ---------------------------------------------------------------------------

class TestBBReversionStrategy:
    @pytest.fixture
    def strategy(self):
        return BBReversionStrategy()

    def test_returns_signal_or_none(self, strategy):
        df = _make_df(n=100, trend=0.0, vol_factor=0.5)
        result = strategy.generate_signal(
            "ETHUSDT", df, MarketRegime.RANGING.value, "5m"
        )
        assert result is None or isinstance(result, Signal)

    def test_wrong_regime_returns_none(self, strategy):
        df = _make_df(n=100, trend=0.0)
        result = strategy.generate_signal(
            "ETHUSDT", df, MarketRegime.TRENDING_UP.value, "5m"
        )
        assert result is None

    def test_strategy_name(self, strategy):
        assert strategy.name == "bb_reversion"


# ---------------------------------------------------------------------------
# ATR Breakout Strategy tests
# ---------------------------------------------------------------------------

class TestATRBreakoutStrategy:
    @pytest.fixture
    def strategy(self):
        return ATRBreakoutStrategy()

    def test_returns_signal_or_none(self, strategy):
        df = _make_df(n=100, trend=0.1, vol_factor=3.0)
        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.HIGH_VOLATILITY.value, "5m"
        )
        assert result is None or isinstance(result, Signal)

    def test_wrong_regime_returns_none(self, strategy):
        df = _make_df(n=100, trend=0.1)
        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.RANGING.value, "5m"
        )
        assert result is None

    def test_strategy_name(self, strategy):
        assert strategy.name == "atr_breakout"
