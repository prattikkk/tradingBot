import unittest
from unittest import mock

import pandas as pd

from config import CONFIG
from core.regime import MarketRegime
from core.signal import Direction, Signal
from strategies.ensemble import EnsembleStrategy, REGIME_ALLOWLIST


class _NoneStrategy:
    def __init__(self, name: str):
        self.name = name

    def generate(self, symbol, df, htf_df=None, htf_df2=None):
        return None


class _SignalStrategy:
    def __init__(self, name: str, signal: Signal):
        self.name = name
        self._signal = signal

    def generate(self, symbol, df, htf_df=None, htf_df2=None):
        return self._signal


class EnsembleTests(unittest.TestCase):
    def test_dynamic_min_agreement(self):
        self.assertEqual(EnsembleStrategy._min_agree(1), 1)
        self.assertEqual(EnsembleStrategy._min_agree(2), 1)
        self.assertEqual(EnsembleStrategy._min_agree(3), 1)
        self.assertEqual(EnsembleStrategy._min_agree(4), 2)

    def test_ranging_allowlist_includes_mean_reversion(self):
        allowed = REGIME_ALLOWLIST[MarketRegime.RANGING]
        self.assertIn("mean_reversion", allowed)

    def test_no_signal_fallback_activates_after_streak(self):
        strategy = EnsembleStrategy()
        fallback_sig = Signal(
            symbol="ETHUSDT",
            direction=Direction.LONG,
            confidence=0.72,
            strategy="supertrend_rsi",
            entry_price=100.0,
            stop_loss=99.0,
            take_profit_1=101.0,
            take_profit_2=102.0,
            atr=1.0,
            reason="fallback candidate",
        )
        strategy._strategies = [
            _NoneStrategy("mean_reversion"),
            _NoneStrategy("breakout_momentum"),
            _SignalStrategy("supertrend_rsi", fallback_sig),
        ]
        df = pd.DataFrame({"close": [1.0, 1.0, 1.0, 1.0]})

        with mock.patch("strategies.ensemble.detect_market_regime", return_value=MarketRegime.RANGING), mock.patch.object(
            CONFIG.strategy,
            "no_signal_fallback_enabled",
            True,
        ), mock.patch.object(CONFIG.strategy, "no_signal_fallback_cycles", 3):
            sig = strategy.generate("ETHUSDT", df, no_signal_streak=3)

        self.assertIsNotNone(sig)
        self.assertTrue(bool(sig.extra.get("fallback_activated")))
        self.assertEqual(sig.extra.get("fallback_source"), "supertrend_rsi")
        self.assertEqual(sig.strategy, "ensemble")

    def test_no_signal_fallback_respects_threshold(self):
        strategy = EnsembleStrategy()
        strategy._strategies = [
            _NoneStrategy("mean_reversion"),
            _NoneStrategy("breakout_momentum"),
            _SignalStrategy(
                "supertrend_rsi",
                Signal(
                    symbol="ETHUSDT",
                    direction=Direction.LONG,
                    confidence=0.70,
                    strategy="supertrend_rsi",
                    entry_price=100.0,
                    stop_loss=99.0,
                    take_profit_1=101.0,
                    take_profit_2=102.0,
                    atr=1.0,
                    reason="fallback candidate",
                ),
            ),
        ]
        df = pd.DataFrame({"close": [1.0, 1.0, 1.0, 1.0]})

        with mock.patch("strategies.ensemble.detect_market_regime", return_value=MarketRegime.RANGING), mock.patch.object(
            CONFIG.strategy,
            "no_signal_fallback_enabled",
            True,
        ), mock.patch.object(CONFIG.strategy, "no_signal_fallback_cycles", 5):
            sig = strategy.generate("ETHUSDT", df, no_signal_streak=2)

        self.assertIsNone(sig)
        self.assertTrue(strategy.last_skip_reason.startswith("no_substrategy_signal"))


if __name__ == "__main__":
    unittest.main()
