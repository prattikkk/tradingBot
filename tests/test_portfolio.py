import unittest
from pathlib import Path
import json
import os
from core.portfolio import Portfolio

class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.test_file = Path("data/portfolio_test.json")
        if self.test_file.exists():
            self.test_file.unlink()
        tmp_file = self.test_file.with_suffix(".tmp")
        if tmp_file.exists():
            tmp_file.unlink()

        # Patch DATA_FILE path for testing
        Portfolio.DATA_FILE = self.test_file

    def tearDown(self):
        if self.test_file.exists():
            self.test_file.unlink()
        tmp_file = self.test_file.with_suffix(".tmp")
        if tmp_file.exists():
            tmp_file.unlink()

    def test_portfolio_save_is_atomic(self):
        portfolio = Portfolio()
        portfolio._balance = 1234.56
        portfolio.total_capital = 2000.0
        portfolio.telemetry = {"exchange_local_drift_detected": 2}

        portfolio._save()

        # Verify main file is created and has correct contents
        self.assertTrue(self.test_file.exists())
        with open(self.test_file, "r") as f:
            data = json.load(f)

        self.assertEqual(data["balance"], 1234.56)
        self.assertEqual(data["total_capital"], 2000.0)
        self.assertEqual(data["telemetry"], {"exchange_local_drift_detected": 2})

        # Temp file should not exist after successful save
        tmp_file = self.test_file.with_suffix(".tmp")
        self.assertFalse(tmp_file.exists())

    def test_portfolio_load(self):
        # Create a pre-existing portfolio file
        data = {
            "balance": 9876.54,
            "total_capital": 5000.0,
            "open_positions": {"BTCUSDT": {"symbol": "BTCUSDT", "direction": "LONG"}},
            "closed_trades": []
        }
        with open(self.test_file, "w") as f:
            json.dump(data, f)

        portfolio = Portfolio()
        self.assertEqual(portfolio._balance, 9876.54)
        self.assertEqual(portfolio.total_capital, 5000.0)
        self.assertIn("BTCUSDT", portfolio.open_positions)

    def test_update_tp1_recomputes_remaining_risk(self):
        portfolio = Portfolio()
        portfolio.open_positions = {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "stop_loss": 95.0,
                "take_profit_1": 101.0,
                "take_profit_2": 102.0,
                "quantity": 10.0,
                "risk_usdt": 50.0,
                "tp1_hit": False,
                "pnl": 0.0,
            }
        }

        pnl = portfolio.update_tp1("BTCUSDT", 101.0)

        self.assertIsNotNone(pnl)
        pos = portfolio.open_positions["BTCUSDT"]
        self.assertAlmostEqual(pos["quantity"], 5.0, places=6)
        self.assertAlmostEqual(pos["risk_usdt"], 25.0, places=6)

    def test_increment_metric_persists_when_requested(self):
        portfolio = Portfolio()

        updated = portfolio.increment_metric("protection_unconfirmed_mode", persist=True)

        self.assertEqual(updated, 1)
        reloaded = Portfolio()
        self.assertEqual(reloaded.telemetry.get("protection_unconfirmed_mode"), 1)

if __name__ == "__main__":
    unittest.main()
