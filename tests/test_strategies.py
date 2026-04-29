"""Tests for trading strategies and signal generation."""
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal

from alphabot.config import settings
from alphabot.strategies.signal import Signal, SignalDirection, compute_confidence
from alphabot.strategies.engine import StrategyEngine
from alphabot.strategies.liquidity_sweep_orderflow import LiquiditySweepOrderFlowStrategy
from alphabot.strategies.orderflow_liquidity_sweep import OrderFlowLiquiditySweepStrategy
from alphabot.strategies.supertrend_pullback import SupertrendPullbackStrategy
from alphabot.strategies.supertrend_rsi import SupertrendRsiStrategy
from alphabot.strategies.supertrend_trail import SupertrendTrailStrategy
from alphabot.strategies.ema_adx_volume import EmaAdxVolumeStrategy
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


def _make_supertrend_df(direction: SignalDirection) -> pd.DataFrame:
    long_side = direction == SignalDirection.LONG
    rsi_val = (
        float(settings.supertrend_rsi_long_min) + 10.0
        if long_side
        else float(settings.supertrend_rsi_short_max) - 10.0
    )
    close = 110.0 if long_side else 90.0
    ema_long = 100.0
    st_line = 105.0 if long_side else 95.0
    st_dir_prev = -1.0 if long_side else 1.0
    st_dir_now = 1.0 if long_side else -1.0

    return pd.DataFrame({
        "open": [100.0, 101.0, close],
        "high": [101.5, 102.0, close + 1.0],
        "low": [99.0, 100.0, close - 1.0],
        "close": [100.0, close - 1.0, close],
        "volume": [1000.0, 1200.0, 2000.0],
        "atr": [1.5, 1.6, 2.0],
        "rsi": [50.0, rsi_val - 5.0, rsi_val],
        "ema_long": [ema_long, ema_long, ema_long],
        "volume_sma": [1000.0, 1000.0, 1000.0],
        "SUPERTd_10_3.0": [st_dir_prev, st_dir_prev, st_dir_now],
        "SUPERT_10_3.0": [st_line, st_line, st_line],
    })


def _make_ema_adx_df(direction: SignalDirection) -> pd.DataFrame:
    long_side = direction == SignalDirection.LONG
    ema_fast_prev = 9.0 if long_side else 21.0
    ema_slow_prev = 21.0 if long_side else 9.0
    ema_fast_now = 22.0 if long_side else 8.0
    ema_slow_now = 20.0 if long_side else 10.0
    ema_fast_slope = 0.2 if long_side else -0.2

    adx_val = float(settings.ema_adx_min) + 10.0
    dmp_val = 30.0 if long_side else 10.0
    dmn_val = 10.0 if long_side else 30.0

    vol_min = float(settings.ema_adx_volume_multiplier)
    volume_sma = 1000.0
    volume = volume_sma * (vol_min + 0.6)
    close = 110.0 if long_side else 90.0

    return pd.DataFrame({
        "open": [100.0, 101.0, close],
        "high": [101.5, 102.0, close + 1.0],
        "low": [99.0, 100.0, close - 1.0],
        "close": [100.0, 101.0, close],
        "volume": [volume_sma, volume_sma, volume],
        "volume_sma": [volume_sma, volume_sma, volume_sma],
        "ema_fast": [ema_fast_prev, ema_fast_prev, ema_fast_now],
        "ema_slow": [ema_slow_prev, ema_slow_prev, ema_slow_now],
        "ema_fast_slope": [ema_fast_slope, ema_fast_slope, ema_fast_slope],
        "atr": [1.5, 1.6, 2.0],
        "ADX_14": [adx_val, adx_val, adx_val],
        "DMP_14": [dmp_val, dmp_val, dmp_val],
        "DMN_14": [dmn_val, dmn_val, dmn_val],
    })


def _make_supertrend_pullback_df(direction: SignalDirection) -> pd.DataFrame:
    long_side = direction == SignalDirection.LONG
    if long_side:
        # Established uptrend, previous bar pulls toward ST line, latest recovers above prev high.
        return pd.DataFrame({
            "open": [100.0, 102.8, 104.1, 105.0],
            "high": [101.0, 103.3, 104.6, 106.2],
            "low": [99.3, 101.7, 102.9, 104.8],
            "close": [100.7, 103.0, 104.2, 106.0],
            "volume": [1000.0, 1100.0, 1200.0, 1500.0],
            "volume_sma": [1000.0, 1000.0, 1000.0, 1000.0],
            "atr": [1.8, 1.8, 1.8, 1.8],
            "rsi": [56.0, 57.0, 58.0, 60.0],
            "ema_long": [98.0, 99.0, 100.0, 101.0],
            "SUPERTd_10_3.0": [1.0, 1.0, 1.0, 1.0],
            "SUPERT_10_3.0": [100.5, 102.0, 103.2, 104.0],
            "ADX_14": [25.0, 26.0, 27.0, 28.0],
        })

    return pd.DataFrame({
        "open": [110.0, 108.8, 106.9, 106.1],
        "high": [110.7, 109.3, 108.2, 106.2],
        "low": [108.9, 107.7, 106.7, 105.0],
        "close": [109.2, 108.0, 107.1, 105.3],
        "volume": [1000.0, 1100.0, 1200.0, 1500.0],
        "volume_sma": [1000.0, 1000.0, 1000.0, 1000.0],
        "atr": [1.7, 1.7, 1.7, 1.7],
        "rsi": [44.0, 42.0, 41.0, 39.0],
        "ema_long": [111.0, 110.0, 109.0, 108.0],
        "SUPERTd_10_3.0": [-1.0, -1.0, -1.0, -1.0],
        "SUPERT_10_3.0": [109.5, 108.6, 107.6, 106.3],
        "ADX_14": [24.0, 25.0, 26.0, 27.0],
    })


def _make_supertrend_trail_df(direction: SignalDirection) -> pd.DataFrame:
    long_side = direction == SignalDirection.LONG
    if long_side:
        return pd.DataFrame({
            "open": [100.0, 101.8, 103.1, 104.6],
            "high": [101.0, 102.7, 103.6, 105.8],
            "low": [99.4, 101.2, 102.5, 104.2],
            "close": [100.8, 102.3, 103.2, 105.7],
            "volume": [1000.0, 1050.0, 1100.0, 1400.0],
            "volume_sma": [1000.0, 1000.0, 1000.0, 1000.0],
            "atr": [1.5, 1.5, 1.5, 1.5],
            "rsi": [54.0, 55.0, 56.0, 60.0],
            "ema_long": [98.0, 99.0, 100.0, 101.0],
            "SUPERTd_10_3.0": [1.0, 1.0, 1.0, 1.0],
            "SUPERT_10_3.0": [99.8, 101.0, 102.0, 103.4],
            "ADX_14": [22.0, 23.0, 24.0, 26.0],
        })

    return pd.DataFrame({
        "open": [111.0, 109.2, 108.0, 106.7],
        "high": [111.4, 109.6, 108.4, 107.1],
        "low": [109.8, 108.3, 106.9, 105.1],
        "close": [110.2, 108.7, 107.4, 105.2],
        "volume": [1000.0, 1050.0, 1100.0, 1400.0],
        "volume_sma": [1000.0, 1000.0, 1000.0, 1000.0],
        "atr": [1.6, 1.6, 1.6, 1.6],
        "rsi": [46.0, 45.0, 44.0, 40.0],
        "ema_long": [112.0, 111.0, 110.0, 109.0],
        "SUPERTd_10_3.0": [-1.0, -1.0, -1.0, -1.0],
        "SUPERT_10_3.0": [110.8, 109.5, 108.6, 106.9],
        "ADX_14": [21.0, 22.0, 23.0, 25.0],
    })


def _make_orderflow_sweep_df(direction: SignalDirection) -> pd.DataFrame:
    lookback = max(int(getattr(settings, "orderflow_sweep_lookback", 20)), 20)
    long_side = direction == SignalDirection.LONG

    rows = []
    for i in range(lookback):
        if long_side:
            open_price = 100.0 + (0.05 if i % 2 == 0 else -0.03)
            close = open_price + 0.2
        else:
            open_price = 100.0 + (0.03 if i % 2 == 0 else -0.05)
            close = open_price - 0.2

        rows.append(
            {
                "open": open_price,
                "high": max(open_price, close) + 0.3,
                "low": min(open_price, close) - 0.3,
                "close": close,
                "volume": 950.0 + i * 8.0,
                "atr": 1.0,
                "rsi": 45.0 if long_side else 55.0,
                "ema_long": 100.0,
                "volume_sma": 1000.0,
            }
        )

    if long_side:
        rows.append(
            {
                "open": 99.6,
                "high": 100.4,
                "low": 98.2,
                "close": 100.2,
                "volume": 1900.0,
                "atr": 1.0,
                "rsi": 44.0,
                "ema_long": 100.0,
                "volume_sma": 1000.0,
            }
        )
    else:
        rows.append(
            {
                "open": 100.4,
                "high": 101.8,
                "low": 99.4,
                "close": 99.6,
                "volume": 1900.0,
                "atr": 1.0,
                "rsi": 56.0,
                "ema_long": 100.0,
                "volume_sma": 1000.0,
            }
        )

    return pd.DataFrame(rows)


def _make_liquidity_sweep_orderflow_df(direction: SignalDirection) -> pd.DataFrame:
    lookback = int(getattr(settings, "liquidity_sweep_orderflow_swing_lookback", 10))
    n = max(60, lookback * 2 + 25)
    long_side = direction == SignalDirection.LONG

    rows = []
    for i in range(n - 1):
        wave = np.sin(i / 2.2) * 0.9
        close = 100.0 + wave
        open_price = close - (0.15 if i % 2 == 0 else -0.15)
        high = max(open_price, close) + 0.35
        low = min(open_price, close) - 0.35
        rows.append(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000.0 + (i % 5) * 30.0,
            }
        )

    temp = pd.DataFrame(rows)
    if long_side:
        ref_low = float(temp["low"].iloc[-(lookback + 8):-1].min())
        rows.append(
            {
                "open": ref_low - 0.05,
                "high": ref_low + 0.55,
                "low": ref_low - 0.20,
                "close": ref_low + 0.25,
                "volume": 2600.0,
            }
        )
    else:
        ref_high = float(temp["high"].iloc[-(lookback + 8):-1].max())
        rows.append(
            {
                "open": ref_high + 0.05,
                "high": ref_high + 0.20,
                "low": ref_high - 0.55,
                "close": ref_high - 0.25,
                "volume": 2600.0,
            }
        )

    return pd.DataFrame(rows)


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
            strategy_name="supertrend_rsi",
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
        # SL distance = 5, TP1 distance = 10 -> R:R = 2.0
        assert sig.risk_reward_ratio == pytest.approx(2.0)


class TestConfidenceScoring:
    def test_perfect_scores(self):
        confidence = compute_confidence(1.0, 1.0, 1.0, 1.0, 1.0)
        assert confidence == pytest.approx(100.0, abs=0.1)

    def test_zero_scores(self):
        confidence = compute_confidence(0.0, 0.0, 0.0, 0.0, 0.0)
        assert confidence == pytest.approx(0.0, abs=0.1)

    def test_partial_scores(self):
        confidence = compute_confidence(0.8, 0.6, 0.5, 0.4, 0.3)
        assert 0 < confidence < 100


# ---------------------------------------------------------------------------
# Supertrend + RSI strategy tests
# ---------------------------------------------------------------------------

class TestSupertrendRsiStrategy:
    @pytest.fixture
    def strategy(self):
        return SupertrendRsiStrategy()

    def test_generates_signal_when_conditions_met(self, strategy):
        df = _make_supertrend_df(SignalDirection.LONG)
        htf_df = df.copy()
        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.TRENDING_UP.value, "15m", higher_tf_df=htf_df
        )
        assert isinstance(result, Signal)
        assert result.direction == SignalDirection.LONG

    def test_strategy_name(self, strategy):
        assert strategy.name == "supertrend_rsi"


class TestSupertrendPullbackStrategy:
    @pytest.fixture
    def strategy(self):
        return SupertrendPullbackStrategy()

    def test_generates_signal_when_pullback_and_recovery_align(self, strategy):
        df = _make_supertrend_pullback_df(SignalDirection.LONG)
        htf_df = df.copy()
        result = strategy.generate_signal(
            "ETHUSDT", df, MarketRegime.TRENDING_UP.value, "15m", higher_tf_df=htf_df
        )
        assert isinstance(result, Signal)
        assert result.direction == SignalDirection.LONG
        assert result.strategy_name == "supertrend_pullback"

    def test_rejects_without_pullback_recovery_pattern(self, strategy):
        df = _make_supertrend_pullback_df(SignalDirection.LONG)
        # Break recovery condition for LONG by pushing close below previous high.
        df.loc[df.index[-1], "close"] = float(df.iloc[-2]["high"]) - 0.2
        htf_df = df.copy()

        result = strategy.generate_signal(
            "ETHUSDT", df, MarketRegime.TRENDING_UP.value, "15m", higher_tf_df=htf_df
        )
        assert result is None

    def test_strategy_name(self, strategy):
        assert strategy.name == "supertrend_pullback"


class TestSupertrendTrailStrategy:
    @pytest.fixture
    def strategy(self):
        return SupertrendTrailStrategy()

    def test_generates_signal_on_trend_breakout(self, strategy):
        df = _make_supertrend_trail_df(SignalDirection.LONG)
        htf_df = df.copy()
        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.TRENDING_UP.value, "15m", higher_tf_df=htf_df
        )
        assert isinstance(result, Signal)
        assert result.direction == SignalDirection.LONG
        assert result.strategy_name == "supertrend_trail"

    def test_rejects_without_breakout(self, strategy):
        df = _make_supertrend_trail_df(SignalDirection.LONG)
        # Break breakout condition: close not above previous high + buffer.
        df.loc[df.index[-1], "close"] = float(df.iloc[-2]["high"])
        htf_df = df.copy()

        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.TRENDING_UP.value, "15m", higher_tf_df=htf_df
        )
        assert result is None

    def test_strategy_name(self, strategy):
        assert strategy.name == "supertrend_trail"


class TestOrderFlowLiquiditySweepStrategy:
    @pytest.fixture
    def strategy(self):
        return OrderFlowLiquiditySweepStrategy()

    def test_generates_long_signal_on_bullish_sweep_with_positive_flow(self, strategy):
        df = _make_orderflow_sweep_df(SignalDirection.LONG)
        htf_df = df.copy()

        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.RANGING.value, "15m", higher_tf_df=htf_df
        )
        assert isinstance(result, Signal)
        assert result.direction == SignalDirection.LONG
        assert result.strategy_name == "orderflow_liquidity_sweep"

    def test_generates_short_signal_on_bearish_sweep_with_negative_flow(self, strategy):
        df = _make_orderflow_sweep_df(SignalDirection.SHORT)
        htf_df = df.copy()

        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.HIGH_VOLATILITY.value, "15m", higher_tf_df=htf_df
        )
        assert isinstance(result, Signal)
        assert result.direction == SignalDirection.SHORT
        assert result.strategy_name == "orderflow_liquidity_sweep"

    def test_rejects_when_no_sweep_occurs(self, strategy):
        df = _make_orderflow_sweep_df(SignalDirection.LONG)
        df.loc[df.index[-1], "low"] = float(df.iloc[-2]["low"]) + 0.1
        df.loc[df.index[-1], "open"] = 100.0
        df.loc[df.index[-1], "close"] = 100.1
        htf_df = df.copy()

        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.RANGING.value, "15m", higher_tf_df=htf_df
        )
        assert result is None

    def test_strategy_name(self, strategy):
        assert strategy.name == "orderflow_liquidity_sweep"


class TestLiquiditySweepOrderFlowStrategy:
    @pytest.fixture
    def strategy(self):
        return LiquiditySweepOrderFlowStrategy()

    def test_generates_long_signal(self, strategy):
        df = _make_liquidity_sweep_orderflow_df(SignalDirection.LONG)
        htf_df = df.copy()
        htf_df["close"] = np.linspace(98.0, 112.0, len(htf_df))

        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.RANGING.value, "15m", higher_tf_df=htf_df
        )
        assert isinstance(result, Signal)
        assert result.direction == SignalDirection.LONG
        assert result.strategy_name == "liquidity_sweep_orderflow"

    def test_generates_short_signal(self, strategy):
        df = _make_liquidity_sweep_orderflow_df(SignalDirection.SHORT)
        htf_df = df.copy()
        htf_df["close"] = np.linspace(112.0, 98.0, len(htf_df))

        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.HIGH_VOLATILITY.value, "15m", higher_tf_df=htf_df
        )
        assert isinstance(result, Signal)
        assert result.direction == SignalDirection.SHORT
        assert result.strategy_name == "liquidity_sweep_orderflow"

    def test_rejects_when_no_sweep(self, strategy):
        df = _make_liquidity_sweep_orderflow_df(SignalDirection.LONG)
        idx = df.index[-1]
        df.loc[idx, "low"] = float(df.iloc[-2]["low"]) + 0.05
        df.loc[idx, "high"] = float(df.iloc[-2]["high"]) - 0.05
        df.loc[idx, "open"] = float(df.iloc[-2]["close"])
        df.loc[idx, "close"] = float(df.iloc[-2]["close"])

        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.RANGING.value, "15m", higher_tf_df=df.copy()
        )
        assert result is None

    def test_strategy_name(self, strategy):
        assert strategy.name == "liquidity_sweep_orderflow"


# ---------------------------------------------------------------------------
# EMA + ADX + Volume strategy tests
# ---------------------------------------------------------------------------

class TestEmaAdxVolumeStrategy:
    @pytest.fixture
    def strategy(self):
        return EmaAdxVolumeStrategy()

    def test_generates_signal_when_conditions_met(self, strategy):
        df = _make_ema_adx_df(SignalDirection.LONG)
        htf_df = df.copy()
        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.TRENDING_UP.value, "15m", higher_tf_df=htf_df
        )
        assert isinstance(result, Signal)
        assert result.direction == SignalDirection.LONG

    def test_rejects_long_on_bearish_reversal_candle(self, strategy):
        df = _make_ema_adx_df(SignalDirection.LONG)
        close = float(df.iloc[-1]["close"])
        df.loc[df.index[-1], "open"] = close + 1.0
        htf_df = df.copy()

        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.TRENDING_UP.value, "15m", higher_tf_df=htf_df
        )
        assert result is None

    def test_rejects_oversized_entry_candle(self, strategy):
        df = _make_ema_adx_df(SignalDirection.LONG)
        idx = df.index[-1]
        close = float(df.iloc[-1]["close"])
        atr_val = float(df.iloc[-1]["atr"])

        df.loc[idx, "open"] = close - 0.5
        df.loc[idx, "high"] = close + (atr_val * 2.5)
        df.loc[idx, "low"] = close - (atr_val * 2.5)
        htf_df = df.copy()

        result = strategy.generate_signal(
            "BTCUSDT", df, MarketRegime.TRENDING_UP.value, "15m", higher_tf_df=htf_df
        )
        assert result is None

    def test_strategy_name(self, strategy):
        assert strategy.name == "ema_adx_volume"


class TestStrategyEngineSelection:
    @staticmethod
    def _mk_signal(strategy_name: str, confidence: float) -> Signal:
        return Signal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            confidence=confidence,
            entry_price=Decimal("100"),
            stop_loss=Decimal("95"),
            take_profit_1=Decimal("108"),
            take_profit_2=Decimal("112"),
            strategy_name=strategy_name,
            regime=MarketRegime.TRENDING_UP.value,
            timeframe="15m",
        )

    def test_prefers_specialized_signal_when_close_confidence(self):
        candidates = [
            self._mk_signal("supertrend_rsi", 80.0),
            self._mk_signal("supertrend_trail", 79.5),  # within 1.0 margin
        ]

        selected = StrategyEngine._select_best_signal(candidates)
        assert selected is not None
        assert selected.strategy_name == "supertrend_trail"

    def test_keeps_supertrend_rsi_when_specialized_gap_is_large(self):
        candidates = [
            self._mk_signal("supertrend_rsi", 80.0),
            self._mk_signal("supertrend_pullback", 74.0),
        ]

        selected = StrategyEngine._select_best_signal(candidates)
        assert selected is not None
        assert selected.strategy_name == "supertrend_rsi"


# ---------------------------------------------------------------------------
# Indicator math tests
# ---------------------------------------------------------------------------

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
