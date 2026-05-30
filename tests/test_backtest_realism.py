import unittest

import pandas as pd

from backtest import Backtest
from core.signal import Direction, Signal


class BacktestRealismTests(unittest.TestCase):
    def _make_backtest(self, **kwargs) -> Backtest:
        params = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "days": 1,
            "strategy_name": "ensemble",
            "slippage_bps": 0,
            "spread_bps": 0,
            "fee_bps": 4,
            "funding_rate_8h": 0,
            "maintenance_margin_rate": 0.005,
            "liquidation_fee_bps": 30,
            "conservative_ohlc_path": True,
            "max_hold_hours": 8,
        }
        params.update(kwargs)
        return Backtest(**params)

    def test_fee_applied_to_entry_and_exit(self):
        bt = self._make_backtest(fee_bps=4)
        long_entry = bt._apply_entry_cost(100.0, Direction.LONG)
        short_entry = bt._apply_entry_cost(100.0, Direction.SHORT)
        long_exit = bt._apply_exit_cost(100.0, Direction.LONG, is_stop=False)
        short_exit = bt._apply_exit_cost(100.0, Direction.SHORT, is_stop=False)

        self.assertAlmostEqual(long_entry, 100.04, places=5)
        self.assertAlmostEqual(short_entry, 99.96, places=5)
        self.assertAlmostEqual(long_exit, 99.96, places=5)
        self.assertAlmostEqual(short_exit, 100.04, places=5)

    def test_funding_cost_for_full_interval(self):
        bt = self._make_backtest(funding_rate_8h=0.0005)
        # 15m bars, 32 bars = 8 hours
        funding = bt._funding_cost(entry_price=100.0, quantity=2.0, bars_held=32)
        self.assertAlmostEqual(funding, 0.1, places=8)

    def test_conservative_same_candle_conflict_prefers_stop(self):
        bt = self._make_backtest(funding_rate_8h=0)
        idx = pd.date_range("2025-01-01", periods=4, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0],
                "high": [100.0, 112.0, 100.0, 100.0],
                "low": [100.0, 94.0, 100.0, 100.0],
                "close": [100.0, 101.0, 100.0, 100.0],
                "volume": [1.0, 1.0, 1.0, 1.0],
            },
            index=idx,
        )
        signal = Signal(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            confidence=1.0,
            strategy="test",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=105.0,
            take_profit_2=110.0,
            atr=1.0,
        )

        pnl, exit_fill, bars, reason, funding = bt._simulate_exit(
            df=df,
            entry_bar=0,
            signal=signal,
            qty=1.0,
            entry_price=100.0,
        )

        self.assertEqual(reason, "SL")
        self.assertEqual(bars, 1)
        self.assertEqual(funding, 0.0)
        self.assertLess(exit_fill, 95.0)
        self.assertLess(pnl, 0.0)
        self.assertEqual(bt._tie_events, 1)

    def test_htf_resample_builds_expected_bars(self):
        bt = self._make_backtest()
        idx = pd.date_range("2025-01-01", periods=8, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104, 105, 106, 107],
                "high": [101, 102, 103, 104, 105, 106, 107, 108],
                "low": [99, 100, 101, 102, 103, 104, 105, 106],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
                "volume": [10, 20, 30, 40, 50, 60, 70, 80],
            },
            index=idx,
        )

        htf = bt._build_htf_frame(df, "1h")

        self.assertIsNotNone(htf)
        self.assertEqual(len(htf), 2)
        self.assertEqual(float(htf.iloc[0]["open"]), 100)
        self.assertEqual(float(htf.iloc[0]["high"]), 104)
        self.assertEqual(float(htf.iloc[0]["low"]), 99)
        self.assertEqual(float(htf.iloc[0]["close"]), 103.5)
        self.assertEqual(float(htf.iloc[0]["volume"]), 100)


if __name__ == "__main__":
    unittest.main()
