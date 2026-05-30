import unittest
from unittest import mock

import pandas as pd

from config import CONFIG
from core.signal import Direction, Signal
from main import TradingBot


class _DummyExecutor:
    def __init__(self, exchange_positions=None, open_orders=None):
        self._exchange_positions = dict(exchange_positions or {})
        self._open_orders = dict(open_orders or {})
        self.closed_calls = []
        self.cancel_all_calls = []

    def get_open_positions(self, symbols=None):
        if not symbols:
            return dict(self._exchange_positions)
        return {
            symbol: self._exchange_positions[symbol]
            for symbol in symbols
            if symbol in self._exchange_positions
        }

    def close_position_market(self, symbol, direction, quantity):
        self.closed_calls.append((symbol, direction, float(quantity)))
        self._exchange_positions.pop(symbol, None)
        return True

    def cancel_all_open_orders(self, symbol):
        self.cancel_all_calls.append(symbol)
        return True

    def get_open_orders(self, symbol):
        return list(self._open_orders.get(symbol, []))


class _DummyPortfolio:
    def __init__(self, open_positions=None):
        self.open_positions = dict(open_positions or {})
        self.closed = []
        self.saved = 0
        self.metrics = {}

    def close_position(self, symbol, exit_price, reason="TP2"):
        self.open_positions.pop(symbol, None)
        self.closed.append((symbol, float(exit_price), reason))
        return 1.0

    def _save(self):
        self.saved += 1

    def increment_metric(self, name, amount=1, persist=False):
        self.metrics[name] = self.metrics.get(name, 0) + int(amount)
        return self.metrics[name]


class _DummyFetcher:
    def __init__(self, price=100.0):
        self.price = float(price)

    def get_current_price(self, symbol):
        return self.price


class ReconciliationTests(unittest.TestCase):
    @staticmethod
    def _build_primary_df(rows=130, freq="15min"):
        close_times = pd.date_range("2024-01-01", periods=rows, freq=freq)
        return pd.DataFrame(
            {
                "open": [100.0] * rows,
                "high": [101.0] * rows,
                "low": [99.0] * rows,
                "close": [100.0] * rows,
                "close_time": close_times,
            }
        )

    @staticmethod
    def _build_bot(portfolio, executor, fetcher=None):
        bot = TradingBot.__new__(TradingBot)
        bot.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        bot.portfolio = portfolio
        bot.executor = executor
        bot.fetcher = fetcher or _DummyFetcher()
        bot.dry_run = False
        bot._no_signal_streaks = {}
        bot._last_evaluated_signal_bar_close = {}
        return bot

    def test_periodic_reconciliation_closes_stale_local_position(self):
        portfolio = _DummyPortfolio(
            open_positions={
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "entry_price": 100.0,
                    "quantity": 1.0,
                }
            }
        )
        executor = _DummyExecutor(exchange_positions={})
        bot = self._build_bot(portfolio, executor)

        with mock.patch.object(CONFIG.trading, "force_close_untracked_exchange_positions", True):
            bot._reconcile_positions_from_exchange(startup=False)

        self.assertEqual(len(portfolio.closed), 1)
        self.assertEqual(portfolio.closed[0][0], "BTCUSDT")
        self.assertEqual(portfolio.closed[0][2], "PERIODIC_SYNC_CLOSED_ON_EXCHANGE")
        self.assertNotIn("BTCUSDT", portfolio.open_positions)

    def test_periodic_reconciliation_force_closes_untracked_exchange_position(self):
        portfolio = _DummyPortfolio(open_positions={})
        executor = _DummyExecutor(
            exchange_positions={
                "ETHUSDT": {
                    "symbol": "ETHUSDT",
                    "quantity": -2.5,
                    "entry_price": 2000.0,
                    "mark_price": 1990.0,
                    "notional": -5000.0,
                    "leverage": 10,
                }
            }
        )
        bot = self._build_bot(portfolio, executor)

        with mock.patch.object(CONFIG.trading, "force_close_untracked_exchange_positions", True):
            bot._reconcile_positions_from_exchange(startup=False)

        self.assertEqual(len(executor.closed_calls), 1)
        symbol, direction, qty = executor.closed_calls[0]
        self.assertEqual(symbol, "ETHUSDT")
        self.assertEqual(direction, "SHORT")
        self.assertAlmostEqual(qty, 2.5, places=6)
        self.assertGreaterEqual(portfolio.metrics.get("exchange_local_drift_detected", 0), 1)
        self.assertGreaterEqual(portfolio.metrics.get("exchange_local_drift_corrected", 0), 1)

    def test_periodic_reconciliation_aligns_quantity_and_risk(self):
        portfolio = _DummyPortfolio(
            open_positions={
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "stop_loss": 95.0,
                    "quantity": 1.0,
                    "notional": 100.0,
                    "risk_usdt": 5.0,
                }
            }
        )
        executor = _DummyExecutor(
            exchange_positions={
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "quantity": 2.0,
                    "entry_price": 100.0,
                    "mark_price": 100.0,
                    "notional": 200.0,
                    "leverage": 10,
                }
            }
        )
        bot = self._build_bot(portfolio, executor)

        with mock.patch.object(CONFIG.trading, "force_close_untracked_exchange_positions", True):
            bot._reconcile_positions_from_exchange(startup=False)

        pos = portfolio.open_positions["BTCUSDT"]
        self.assertAlmostEqual(float(pos["quantity"]), 2.0, places=6)
        self.assertAlmostEqual(float(pos["notional"]), 200.0, places=6)
        self.assertAlmostEqual(float(pos["risk_usdt"]), 10.0, places=6)

    def test_import_exchange_position_uses_real_protective_orders(self):
        portfolio = _DummyPortfolio(open_positions={})
        executor = _DummyExecutor(
            exchange_positions={},
            open_orders={
                "ETHUSDT": [
                    {
                        "symbol": "ETHUSDT",
                        "side": "BUY",
                        "type": "STOP_MARKET",
                        "stopPrice": "2050",
                        "reduceOnly": True,
                        "orderId": 101,
                    },
                    {
                        "symbol": "ETHUSDT",
                        "side": "BUY",
                        "type": "TAKE_PROFIT_MARKET",
                        "stopPrice": "1950",
                        "reduceOnly": True,
                        "orderId": 102,
                    },
                    {
                        "symbol": "ETHUSDT",
                        "side": "BUY",
                        "type": "TAKE_PROFIT_MARKET",
                        "stopPrice": "1900",
                        "reduceOnly": True,
                        "orderId": 103,
                    },
                ]
            },
        )
        bot = self._build_bot(portfolio, executor)

        imported = bot._import_exchange_position(
            {
                "symbol": "ETHUSDT",
                "quantity": -2.0,
                "entry_price": 2000.0,
                "mark_price": 1990.0,
                "notional": -4000.0,
                "leverage": 10,
            }
        )

        self.assertTrue(imported)
        pos = portfolio.open_positions["ETHUSDT"]
        self.assertEqual(pos["direction"], "SHORT")
        self.assertAlmostEqual(float(pos["stop_loss"]), 2050.0, places=6)
        self.assertAlmostEqual(float(pos["take_profit_1"]), 1950.0, places=6)
        self.assertAlmostEqual(float(pos["take_profit_2"]), 1900.0, places=6)
        self.assertEqual(pos["protection_source"], "exchange")
        self.assertFalse(bool(pos["unprotected_sync"]))
        self.assertEqual(pos["order_ids"]["sl"], 101)
        self.assertEqual(pos["order_ids"]["tp1"], 102)
        self.assertEqual(pos["order_ids"]["tp2"], 103)

    def test_import_exchange_position_marks_synthetic_fallback(self):
        portfolio = _DummyPortfolio(open_positions={})
        executor = _DummyExecutor(exchange_positions={}, open_orders={"BTCUSDT": []})
        bot = self._build_bot(portfolio, executor)

        imported = bot._import_exchange_position(
            {
                "symbol": "BTCUSDT",
                "quantity": 1.5,
                "entry_price": 100.0,
                "mark_price": 100.0,
                "notional": 150.0,
                "leverage": 5,
            }
        )

        self.assertTrue(imported)
        pos = portfolio.open_positions["BTCUSDT"]
        self.assertEqual(pos["protection_source"], "synthetic")
        self.assertTrue(bool(pos["unprotected_sync"]))
        self.assertEqual(int(portfolio.metrics.get("import_unprotected_sync", 0)), 1)

    def test_generate_signal_candidate_skips_when_same_bar_already_evaluated(self):
        class _NoSignalStrategy:
            def __init__(self):
                self.last_skip_reason = "no_substrategy_signal(regime=RANGING)"

            def generate(self, symbol, df, htf_df=None, htf_df2=None):
                return None

        portfolio = _DummyPortfolio(open_positions={})
        bot = self._build_bot(portfolio, _DummyExecutor())
        bot.strategy_cls = _NoSignalStrategy

        df = self._build_primary_df()
        bar_close = TradingBot._signal_bar_close_epoch(df)
        bot._last_evaluated_signal_bar_close["BTCUSDT"] = float(bar_close)

        with mock.patch.object(CONFIG.strategy, "align_signal_to_new_candle", True):
            result = bot._generate_signal_candidate(
                symbol="BTCUSDT",
                multi_tf={"15m": df, "1h": df, "4h": df},
                primary_tf="15m",
                htf_1="1h",
                htf_2="4h",
                min_confidence=0.60,
            )

        self.assertEqual(result["rejection_reason"], "same_bar_already_evaluated")

    def test_generate_signal_candidate_increments_no_signal_streak(self):
        class _NoSignalStrategy:
            def __init__(self):
                self.last_skip_reason = "no_substrategy_signal(regime=RANGING)"

            def generate(self, symbol, df, htf_df=None, htf_df2=None):
                return None

        portfolio = _DummyPortfolio(open_positions={})
        bot = self._build_bot(portfolio, _DummyExecutor())
        bot.strategy_cls = _NoSignalStrategy

        df = self._build_primary_df()
        with mock.patch.object(CONFIG.strategy, "align_signal_to_new_candle", False):
            result = bot._generate_signal_candidate(
                symbol="BTCUSDT",
                multi_tf={"15m": df, "1h": df, "4h": df},
                primary_tf="15m",
                htf_1="1h",
                htf_2="4h",
                min_confidence=0.60,
            )

        self.assertEqual(result["rejection_reason"], "no_substrategy_signal(regime=RANGING)")
        self.assertEqual(int(bot._no_signal_streaks.get("BTCUSDT", 0)), 1)

    def test_generate_signal_candidate_relaxes_confidence_for_fallback(self):
        class _FallbackSignalStrategy:
            def __init__(self):
                self.last_skip_reason = ""

            def generate(self, symbol, df, htf_df=None, htf_df2=None):
                return Signal(
                    symbol=symbol,
                    direction=Direction.LONG,
                    confidence=0.58,
                    strategy="ensemble",
                    entry_price=100.0,
                    stop_loss=99.0,
                    take_profit_1=101.0,
                    take_profit_2=102.0,
                    atr=1.0,
                    reason="fallback",
                    extra={"fallback_activated": True},
                )

        portfolio = _DummyPortfolio(open_positions={})
        bot = self._build_bot(portfolio, _DummyExecutor())
        bot.strategy_cls = _FallbackSignalStrategy

        df = self._build_primary_df()
        with mock.patch.object(CONFIG.strategy, "align_signal_to_new_candle", False), mock.patch.object(
            CONFIG.strategy,
            "no_signal_fallback_confidence_relaxation",
            0.05,
        ):
            result = bot._generate_signal_candidate(
                symbol="BTCUSDT",
                multi_tf={"15m": df, "1h": df, "4h": df},
                primary_tf="15m",
                htf_1="1h",
                htf_2="4h",
                min_confidence=0.60,
            )

        self.assertIn("signal", result)
        self.assertNotIn("rejection_reason", result)


if __name__ == "__main__":
    unittest.main()
