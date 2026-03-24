"""Tests for risk management rules."""
import pytest
from decimal import Decimal

from alphabot.risk.position_sizer import PositionSizer
from alphabot.risk.risk_manager import RiskManager
from alphabot.strategies.signal import Signal, SignalDirection
from alphabot.regime.detector import MarketRegime
from alphabot.database.db import Database


# ---------------------------------------------------------------------------
# Position Sizer tests
# ---------------------------------------------------------------------------

class TestPositionSizer:
    @pytest.fixture
    def sizer(self):
        return PositionSizer()

    def test_basic_size(self, sizer: PositionSizer):
        result = sizer.calculate_position_size(
            account_balance=Decimal("10000"),
            entry_price=Decimal("100"),
            stop_loss=Decimal("95"),
            leverage=1,
            regime="TRENDING_UP",
            existing_exposure=Decimal("0"),
        )
        assert result["quantity"] > 0

    def test_halves_in_high_volatility(self, sizer: PositionSizer):
        normal = sizer.calculate_position_size(
            account_balance=Decimal("10000"),
            entry_price=Decimal("100"),
            stop_loss=Decimal("95"),
            leverage=1,
            regime="TRENDING_UP",
            existing_exposure=Decimal("0"),
        )
        volatile = sizer.calculate_position_size(
            account_balance=Decimal("10000"),
            entry_price=Decimal("100"),
            stop_loss=Decimal("95"),
            leverage=1,
            regime="HIGH_VOLATILITY",
            existing_exposure=Decimal("0"),
        )
        # High vol should risk less / smaller size
        assert volatile["risk_amount"] < normal["risk_amount"]

    def test_zero_sl_distance_returns_zero(self, sizer: PositionSizer):
        result = sizer.calculate_position_size(
            account_balance=Decimal("10000"),
            entry_price=Decimal("100"),
            stop_loss=Decimal("100"),
            leverage=1,
            regime="TRENDING_UP",
            existing_exposure=Decimal("0"),
        )
        assert result["quantity"] == Decimal("0")


# ---------------------------------------------------------------------------
# Risk Manager tests
# ---------------------------------------------------------------------------

class TestRiskManager:
    @pytest.fixture
    def manager(self, tmp_path):
        db = Database(db_path=str(tmp_path / "test.db"))
        rm = RiskManager(db)
        rm.initialize(Decimal("10000"))
        return rm

    def _make_signal(self, confidence=75.0, direction=SignalDirection.LONG,
                     entry=Decimal("100"), sl=Decimal("95"),
                     tp1=Decimal("110"), tp2=Decimal("115")):
        return Signal(
            symbol="BTCUSDT",
            direction=direction,
            confidence=confidence,
            entry_price=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            strategy_name="test",
            regime=MarketRegime.TRENDING_UP.value,
            timeframe="5m",
        )

    def test_approve_good_signal(self, manager: RiskManager):
        signal = self._make_signal()
        approved, reason, size_info = manager.validate_signal(
            signal=signal,
            account_balance=Decimal("10000"),
            open_positions=[],
            existing_exposure=Decimal("0"),
        )
        assert approved is True

    def test_reject_low_confidence(self, manager: RiskManager):
        signal = self._make_signal(confidence=40.0)
        approved, reason, _ = manager.validate_signal(
            signal=signal,
            account_balance=Decimal("10000"),
            open_positions=[],
            existing_exposure=Decimal("0"),
        )
        assert approved is False
        assert "confidence" in reason.lower()

    def test_reject_low_risk_reward(self, manager: RiskManager):
        # TP1 too close → R:R < 1.5
        signal = self._make_signal(
            entry=Decimal("100"),
            sl=Decimal("95"),
            tp1=Decimal("102"),
            tp2=Decimal("104"),
        )
        approved, reason, _ = manager.validate_signal(
            signal=signal,
            account_balance=Decimal("10000"),
            open_positions=[],
            existing_exposure=Decimal("0"),
        )
        assert approved is False
        assert "r:r" in reason.lower() or "reward" in reason.lower() or "risk" in reason.lower()

    def test_reject_max_positions(self, manager: RiskManager):
        signal = self._make_signal()
        # Create enough fake open positions to hit the cap
        fake_positions = [
            {"symbol": f"PAIR{i}USDT", "direction": "LONG", "status": "OPEN"}
            for i in range(10)  # More than any reasonable max_positions
        ]
        approved, reason, _ = manager.validate_signal(
            signal=signal,
            account_balance=Decimal("10000"),
            open_positions=fake_positions,
            existing_exposure=Decimal("0"),
        )
        assert approved is False
        assert "position" in reason.lower()

    def test_reject_duplicate_symbol(self, manager: RiskManager):
        signal = self._make_signal()
        open_positions = [
            {"symbol": "BTCUSDT", "direction": "LONG", "status": "OPEN"}
        ]
        approved, reason, _ = manager.validate_signal(
            signal=signal,
            account_balance=Decimal("10000"),
            open_positions=open_positions,
            existing_exposure=Decimal("0"),
        )
        assert approved is False
        assert "correlation" in reason.lower() or "already" in reason.lower() or "block" in reason.lower()
