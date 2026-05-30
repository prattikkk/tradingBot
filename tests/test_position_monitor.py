import unittest

from config import CONFIG
from core.position_monitor import PositionMonitor


class _DummyExecutor:
    def __init__(self, open_positions=None, close_success=True, has_credentials=False):
        self.closed_calls = []
        self.cancel_calls = []
        self._open_positions = dict(open_positions or {})
        self._close_success = bool(close_success)
        self.has_credentials = bool(has_credentials)

    def cancel_order(self, symbol, order_id):
        self.cancel_calls.append((symbol, order_id))
        return True

    def close_position_market(self, symbol, direction, quantity):
        self.closed_calls.append((symbol, direction, float(quantity)))
        if not self._close_success:
            return False
        self._open_positions.pop(symbol, None)
        return True

    def get_open_positions(self, symbols=None):
        if not symbols:
            return dict(self._open_positions)
        return {symbol: self._open_positions[symbol] for symbol in symbols if symbol in self._open_positions}

    def place_stop_loss(self, symbol, direction, quantity, stop_price):
        return "DRY_RUN"

    def place_trailing_stop(self, symbol, direction, quantity, callback_rate_pct, activation_price=None):
        return None


class _DummyPortfolio:
    def __init__(self, symbol, pos):
        self.symbol = symbol
        self.open_positions = {symbol: pos}
        self.closed = []
        self.metrics = {}

    def update_tp1(self, symbol, exit_price):
        pos = self.open_positions[symbol]
        qty_exit = float(pos["quantity"]) * float(CONFIG.risk.partial_exit_pct)
        pos["quantity"] = float(pos["quantity"]) - qty_exit
        pos["tp1_hit"] = True
        pos["status"] = "TP1_HIT"
        pos["pnl"] = float(pos.get("pnl", 0.0)) + 1.0
        return 1.0

    def close_position(self, symbol, exit_price, reason="TP2"):
        pos = self.open_positions.pop(symbol, None)
        if pos is not None:
            self.closed.append((symbol, float(exit_price), reason))
            return 1.0
        return None

    def _save(self):
        return None

    def increment_metric(self, name, amount=1, persist=False):
        self.metrics[name] = self.metrics.get(name, 0) + int(amount)
        return self.metrics[name]


class _DummyFetcher:
    def __init__(self, price=100.0):
        self.price = float(price)

    def get_current_price(self, symbol):
        return self.price


class _RestFallbackFetcher:
    def __init__(self, rest_price=98.0, confirm_price=100.0):
        self.rest_price = float(rest_price)
        self.confirm_price = float(confirm_price)

    def get_current_price_with_meta(self, symbol):
        return self.rest_price, "rest"

    def get_current_price(self, symbol):
        return self.confirm_price


def _base_pos():
    return {
        "symbol": "TESTUSDT",
        "direction": "LONG",
        "entry_price": 100.0,
        "stop_loss": 99.0,
        "take_profit_1": 101.0,
        "take_profit_2": 102.0,
        "quantity": 10.0,
        "pnl": 0.0,
        "tp1_hit": False,
        "order_ids": {
            "entry": 1,
            "sl": None,
            "tp1": None,
            "tp2": None,
        },
    }


class PositionMonitorTests(unittest.TestCase):
    def test_pending_tp_orders_are_treated_as_client_side_exit(self):
        pos = _base_pos()
        pos["order_ids"]["tp1"] = "pending:tp1_abc"
        pos["order_ids"]["tp2"] = "pending:tp2_abc"

        self.assertTrue(PositionMonitor._is_client_side_exit(pos))

    def test_confirmed_tp_orders_are_exchange_managed(self):
        pos = _base_pos()
        pos["order_ids"]["tp1"] = 123456

        self.assertFalse(PositionMonitor._is_client_side_exit(pos))

    def test_client_side_tp1_executes_partial_market_close(self):
        symbol = "TESTUSDT"
        pos = _base_pos()
        portfolio = _DummyPortfolio(symbol, pos)
        executor = _DummyExecutor()
        monitor = PositionMonitor(portfolio, executor, _DummyFetcher())

        monitor._close_tp1(symbol, 101.0, pos)

        self.assertEqual(len(executor.closed_calls), 1)
        _, _, qty = executor.closed_calls[0]
        self.assertAlmostEqual(qty, 5.0, places=6)
        self.assertTrue(portfolio.open_positions[symbol]["tp1_hit"])
        self.assertAlmostEqual(portfolio.open_positions[symbol]["quantity"], 5.0, places=6)

    def test_client_side_tp2_executes_market_close(self):
        symbol = "TESTUSDT"
        pos = _base_pos()
        portfolio = _DummyPortfolio(symbol, pos)
        executor = _DummyExecutor()
        monitor = PositionMonitor(portfolio, executor, _DummyFetcher())

        monitor._close_tp2(symbol, pos, 102.0)

        self.assertEqual(len(executor.closed_calls), 1)
        _, _, qty = executor.closed_calls[0]
        self.assertAlmostEqual(qty, 10.0, places=6)
        self.assertEqual(len(portfolio.closed), 1)
        self.assertEqual(portfolio.closed[0][2], "TP2_HIT")

    def test_client_side_sl_executes_market_close(self):
        symbol = "TESTUSDT"
        pos = _base_pos()
        portfolio = _DummyPortfolio(symbol, pos)
        executor = _DummyExecutor()
        monitor = PositionMonitor(portfolio, executor, _DummyFetcher())

        monitor._close_sl(symbol, pos, 99.0)

        self.assertEqual(len(executor.closed_calls), 1)
        _, _, qty = executor.closed_calls[0]
        self.assertAlmostEqual(qty, 10.0, places=6)
        self.assertEqual(len(portfolio.closed), 1)
        self.assertEqual(portfolio.closed[0][2], "SL_HIT")

    def test_same_cycle_tp1_then_tp2_when_price_is_above_tp2(self):
        symbol = "TESTUSDT"
        pos = _base_pos()
        portfolio = _DummyPortfolio(symbol, pos)
        executor = _DummyExecutor()
        monitor = PositionMonitor(portfolio, executor, _DummyFetcher(price=103.0))

        monitor._check_position(symbol, pos)

        self.assertEqual(len(executor.closed_calls), 2)
        self.assertEqual(len(portfolio.closed), 1)
        self.assertEqual(portfolio.closed[0][2], "TP2_HIT")

    def test_rest_fallback_price_is_confirmed_before_exit_decision(self):
        symbol = "TESTUSDT"
        pos = _base_pos()
        pos["stop_loss"] = 99.0

        portfolio = _DummyPortfolio(symbol, pos)
        executor = _DummyExecutor()
        fetcher = _RestFallbackFetcher(rest_price=98.0, confirm_price=100.0)
        monitor = PositionMonitor(portfolio, executor, fetcher)

        monitor._check_position(symbol, pos)

        self.assertEqual(len(executor.closed_calls), 0)
        self.assertEqual(len(portfolio.closed), 0)

    def test_exchange_residual_triggers_corrective_close(self):
        symbol = "TESTUSDT"
        pos = _base_pos()

        executor = _DummyExecutor(
            open_positions={
                symbol: {
                    "symbol": symbol,
                    "quantity": -3.0,
                }
            },
            has_credentials=True,
        )
        portfolio = _DummyPortfolio(symbol, pos)
        monitor = PositionMonitor(portfolio, executor, _DummyFetcher())

        ok = monitor._ensure_exchange_flattened(symbol, reason="SL_HIT")

        self.assertTrue(ok)
        self.assertEqual(len(executor.closed_calls), 1)
        _, direction, qty = executor.closed_calls[0]
        self.assertEqual(direction, "SHORT")
        self.assertAlmostEqual(qty, 3.0, places=6)
        self.assertGreaterEqual(portfolio.metrics.get("exchange_local_drift_detected", 0), 1)
        self.assertGreaterEqual(portfolio.metrics.get("exchange_local_drift_corrected", 0), 1)


if __name__ == "__main__":
    unittest.main()
