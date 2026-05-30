import unittest
from unittest import mock

from core.risk_manager import RiskManager
from core.signal import Direction, Signal
from core import risk_manager as risk_manager_module


class _DummyPortfolio:
    def __init__(self, balance: float = 1000.0, total_capital: float = 1000.0):
        self._balance = float(balance)
        self.total_capital = float(total_capital)
        self.open_positions: dict[str, dict] = {}
        self.closed_trades: list[dict] = []

    def available_capital(self) -> float:
        return self._balance

    def equity(self) -> float:
        return self._balance


class RiskManagerTests(unittest.TestCase):
    @staticmethod
    def _make_signal(entry: float, stop: float) -> Signal:
        return Signal(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            confidence=0.8,
            strategy="test",
            entry_price=float(entry),
            stop_loss=float(stop),
            take_profit_1=float(entry) + 10.0,
            take_profit_2=float(entry) + 20.0,
            atr=1.0,
        )

    def test_size_position_risk_is_leverage_independent(self):
        portfolio = _DummyPortfolio(balance=1000.0, total_capital=1000.0)
        manager = RiskManager(portfolio)
        signal = self._make_signal(entry=100.0, stop=95.0)

        with mock.patch.object(risk_manager_module.risk_cfg, "max_risk_per_trade", 0.02), mock.patch.object(
            risk_manager_module.risk_cfg,
            "leverage",
            10,
        ), mock.patch.object(risk_manager_module.risk_cfg, "max_open_positions", 6), mock.patch.object(
            risk_manager_module.risk_cfg,
            "max_portfolio_risk",
            0.5,
        ), mock.patch.object(manager, "_adaptive_sizing_multiplier", return_value=1.0):
            pos = manager.size_position(
                signal,
                {
                    "step_size": 0.001,
                    "min_qty": 0.001,
                    "min_notional": 5.0,
                },
            )

        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos.quantity, 4.0, places=3)
        self.assertAlmostEqual(pos.notional_usdt, 400.0, places=3)
        self.assertAlmostEqual(pos.risk_usdt, 20.0, places=3)
        self.assertAlmostEqual(pos.risk_usdt, pos.quantity * abs(signal.entry_price - signal.stop_loss), places=6)

    def test_size_position_margin_cap_still_applies(self):
        portfolio = _DummyPortfolio(balance=1000.0, total_capital=1000.0)
        manager = RiskManager(portfolio)
        signal = self._make_signal(entry=50000.0, stop=49900.0)

        with mock.patch.object(risk_manager_module.risk_cfg, "max_risk_per_trade", 0.02), mock.patch.object(
            risk_manager_module.risk_cfg,
            "leverage",
            10,
        ), mock.patch.object(risk_manager_module.risk_cfg, "max_open_positions", 6), mock.patch.object(
            risk_manager_module.risk_cfg,
            "max_portfolio_risk",
            0.5,
        ), mock.patch.object(manager, "_adaptive_sizing_multiplier", return_value=1.0):
            pos = manager.size_position(
                signal,
                {
                    "step_size": 0.001,
                    "min_qty": 0.001,
                    "min_notional": 5.0,
                },
            )

        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos.quantity, 0.19, places=3)
        self.assertLessEqual(pos.notional_usdt, 9500.0 + 1e-6)
        self.assertLessEqual(pos.risk_usdt, 20.0 + 1e-6)

    def test_size_position_blocks_when_portfolio_risk_limit_reached(self):
        portfolio = _DummyPortfolio(balance=1000.0, total_capital=1000.0)
        portfolio.open_positions = {
            "ETHUSDT": {
                "risk_usdt": 100.0,
            }
        }
        manager = RiskManager(portfolio)
        signal = self._make_signal(entry=100.0, stop=99.0)

        with mock.patch.object(risk_manager_module.risk_cfg, "max_risk_per_trade", 0.02), mock.patch.object(
            risk_manager_module.risk_cfg,
            "leverage",
            10,
        ), mock.patch.object(risk_manager_module.risk_cfg, "max_open_positions", 6), mock.patch.object(
            risk_manager_module.risk_cfg,
            "max_portfolio_risk",
            0.1,
        ), mock.patch.object(manager, "_adaptive_sizing_multiplier", return_value=1.0):
            pos = manager.size_position(
                signal,
                {
                    "step_size": 0.001,
                    "min_qty": 0.001,
                    "min_notional": 5.0,
                },
            )

        self.assertIsNone(pos)

    def test_portfolio_risk_basis_equity_blocks_earlier_than_total_capital(self):
        portfolio = _DummyPortfolio(balance=500.0, total_capital=1000.0)
        portfolio.open_positions = {
            "ETHUSDT": {
                "risk_usdt": 80.0,
            }
        }
        manager = RiskManager(portfolio)

        with mock.patch.object(risk_manager_module.risk_cfg, "max_portfolio_risk", 0.1), mock.patch.object(
            risk_manager_module.risk_cfg,
            "portfolio_risk_basis",
            "equity",
        ):
            self.assertFalse(manager._portfolio_risk_ok())

    def test_portfolio_risk_basis_total_capital_allows_same_risk(self):
        portfolio = _DummyPortfolio(balance=500.0, total_capital=1000.0)
        portfolio.open_positions = {
            "ETHUSDT": {
                "risk_usdt": 80.0,
            }
        }
        manager = RiskManager(portfolio)

        with mock.patch.object(risk_manager_module.risk_cfg, "max_portfolio_risk", 0.1), mock.patch.object(
            risk_manager_module.risk_cfg,
            "portfolio_risk_basis",
            "total_capital",
        ):
            self.assertTrue(manager._portfolio_risk_ok())


if __name__ == "__main__":
    unittest.main()
