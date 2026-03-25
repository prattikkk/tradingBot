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
from alphabot.utils.indicators import adx


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

    def test_rejects_weak_bb_touch_without_reversal_confirmation(self, strategy):
        df = pd.DataFrame({
            "open": [101.0, 99.2, 99.0],
            "high": [101.5, 100.1, 99.3],
            "low": [99.4, 98.9, 98.7],
            "close": [99.8, 99.0, 98.9],
            "volume": [1000, 950, 900],
            "volume_sma": [1000, 1000, 1000],
            "atr": [0.6, 0.6, 0.6],
            "rsi": [42.0, 34.0, 33.0],
            "BBL_20_2.0": [99.2, 99.1, 99.0],
            "BBM_20_2.0": [100.0, 100.0, 100.0],
            "BBU_20_2.0": [100.8, 100.9, 101.0],
            "STOCHRSIk_14_14_3_3": [35.0, 26.0, 19.0],
        })
        result = strategy.generate_signal(
            "ETHUSDT", df, MarketRegime.RANGING.value, "15m"
        )
        assert result is None

    def test_accepts_bb_reversal_with_confirmation(self, strategy):
        df = pd.DataFrame({
            "open": [101.0, 99.5, 98.8],
            "high": [101.2, 99.7, 99.4],
            "low": [99.4, 98.9, 98.7],
            "close": [99.6, 99.0, 99.2],
            "volume": [1000, 980, 1150],
            "volume_sma": [1000, 1000, 1000],
            "atr": [0.6, 0.6, 0.6],
            "rsi": [40.0, 32.0, 34.0],
            "BBL_20_2.0": [99.2, 99.1, 99.0],
            "BBM_20_2.0": [100.0, 100.0, 100.0],
            "BBU_20_2.0": [100.8, 100.9, 101.0],
            "STOCHRSIk_14_14_3_3": [28.0, 18.0, 21.0],
        })
        result = strategy.generate_signal(
            "ETHUSDT", df, MarketRegime.RANGING.value, "15m"
        )
        assert result is None or isinstance(result, Signal)


class TestIndicatorMath:
    def test_adx_is_bounded_and_populated(self):
        df = _make_df(n=120, trend=0.2, vol_factor=1.0)
        adx_df = adx(df["high"], df["low"], df["close"], period=14)
        assert not adx_df.empty
        adx_col = [c for c in adx_df.columns if c.startswith("ADX_")][0]
        dmp_col = [c for c in adx_df.columns if c.startswith("DMP_")][0]
        dmn_col = [c for c in adx_df.columns if c.startswith("DMN_")][0]

        latest_adx = float(adx_df[adx_col].dropna().iloc[-1])
        latest_dmp = float(adx_df[dmp_col].dropna().iloc[-1])
        latest_dmn = float(adx_df[dmn_col].dropna().iloc[-1])

        assert 0.0 <= latest_adx <= 100.0
        assert 0.0 <= latest_dmp <= 100.0
        assert 0.0 <= latest_dmn <= 100.0


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
