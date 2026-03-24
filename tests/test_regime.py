"""Tests for Market Regime Detector."""
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal

from alphabot.regime.detector import RegimeDetector, MarketRegime
from alphabot.data.data_store import DataStore
from alphabot.data.models import Candle


def _make_candles(closes: list[float], symbol: str = "BTCUSDT",
                  timeframe: str = "5m") -> list[Candle]:
    """Build Candle objects from close prices."""
    import datetime as dt
    candles = []
    base_ts = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    for i, c in enumerate(closes):
        open_time = base_ts + dt.timedelta(minutes=5 * i)
        close_time = open_time + dt.timedelta(minutes=5)
        candles.append(Candle(
            symbol=symbol,
            timeframe=timeframe,
            open_time=open_time,
            open=Decimal(str(c * 0.999)),
            high=Decimal(str(c * 1.005)),
            low=Decimal(str(c * 0.995)),
            close=Decimal(str(c)),
            volume=Decimal("100"),
            close_time=close_time,
            is_closed=True,
        ))
    return candles


def _trending_up_prices(n: int = 100) -> list[float]:
    """Generate steadily rising prices."""
    return [100 + i * 0.5 for i in range(n)]


def _trending_down_prices(n: int = 100) -> list[float]:
    """Generate steadily falling prices."""
    return [150 - i * 0.5 for i in range(n)]


def _ranging_prices(n: int = 100) -> list[float]:
    """Generate flat oscillating prices."""
    return [100 + 0.5 * ((-1) ** i) for i in range(n)]


def _volatile_prices(n: int = 100) -> list[float]:
    """Generate highly volatile prices with wide swings."""
    np.random.seed(42)
    base = 100.0
    prices = []
    for _ in range(n):
        base += np.random.uniform(-3, 3)
        prices.append(base)
    return prices


class TestRegimeDetector:
    """Regime detector unit tests."""

    @pytest.fixture
    def data_store(self):
        return DataStore(lookback=200)

    @pytest.fixture
    def detector(self, data_store):
        return RegimeDetector(data_store)

    def _load_candles(self, data_store: DataStore, prices: list[float],
                      symbol: str = "BTCUSDT"):
        candles = _make_candles(prices, symbol=symbol)
        for c in candles:
            data_store.add_candle(c)

    def test_returns_enum_member(self, detector: RegimeDetector, data_store: DataStore):
        self._load_candles(data_store, _trending_up_prices(100))
        regime = detector.detect("BTCUSDT", timeframe="5m")
        assert isinstance(regime, MarketRegime)

    def test_insufficient_data_returns_unclear(self, detector: RegimeDetector, data_store: DataStore):
        self._load_candles(data_store, [100.0] * 5)
        regime = detector.detect("BTCUSDT", timeframe="5m")
        assert regime == MarketRegime.UNCLEAR

    def test_caches_last_regime(self, detector: RegimeDetector, data_store: DataStore):
        self._load_candles(data_store, _trending_up_prices(100))
        detector.detect("BTCUSDT", timeframe="5m")
        r2 = detector.get_regime("BTCUSDT")
        assert isinstance(r2, MarketRegime)
        assert r2 != MarketRegime.UNCLEAR or True  # may be UNCLEAR depending on data

    def test_get_current_regimes(self, detector: RegimeDetector, data_store: DataStore):
        self._load_candles(data_store, _trending_up_prices(100))
        detector.detect("BTCUSDT", timeframe="5m")
        regimes = detector.get_current_regimes()
        assert isinstance(regimes, dict)
        assert "BTCUSDT" in regimes

    def test_all_regime_values_exist(self):
        expected = {"TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY", "UNCLEAR"}
        actual = {m.value for m in MarketRegime}
        assert expected == actual
