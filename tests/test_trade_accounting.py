"""Regression tests for trade accounting and stop-loss PnL handling."""

import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from alphabot.config import settings
from alphabot.database.db import Database
from alphabot.positions.pnl_tracker import PnLTracker
from alphabot.positions.position_manager import Position, PositionManager


def test_pnl_tracker_does_not_multiply_by_leverage(tmp_path):
    db = Database(db_path=str(tmp_path / "accounting.db"))
    tracker = PnLTracker(db)

    tracker.record_trade(
        trade_id="t-no-lev",
        symbol="ETHUSDT",
        direction="LONG",
        entry_price=100.0,
        exit_price=99.0,
        quantity=10.0,
        leverage=5,
        fees=0.0,
        strategy_name="unit",
        regime="TRENDING_UP",
        signal_confidence=80.0,
        exit_reason="SL_HIT",
        open_time=datetime.datetime.now(datetime.timezone.utc),
        close_time=datetime.datetime.now(datetime.timezone.utc),
    )

    # Correct contract PnL is (99 - 100) * 10 = -10, not -50.
    persisted = db.get_trade("t-no-lev")
    assert persisted is not None
    assert float(getattr(persisted, "gross_pnl")) == -10.0
    assert float(getattr(persisted, "net_pnl")) == -10.0


def test_pnl_tracker_accepts_precomputed_partial_close_totals(tmp_path):
    db = Database(db_path=str(tmp_path / "partials.db"))
    tracker = PnLTracker(db)

    tracker.record_trade(
        trade_id="t-partial",
        symbol="BTCUSDT",
        direction="LONG",
        entry_price=100.0,
        exit_price=100.0,
        quantity=1.0,
        leverage=5,
        fees=0.8,
        strategy_name="unit",
        regime="TRENDING_UP",
        signal_confidence=90.0,
        exit_reason="TRAILING_STOP",
        open_time=datetime.datetime.now(datetime.timezone.utc),
        close_time=datetime.datetime.now(datetime.timezone.utc),
        gross_pnl_override=12.5,
        net_pnl_override=11.7,
    )

    persisted = db.get_trade("t-partial")
    assert persisted is not None
    assert float(getattr(persisted, "gross_pnl")) == 12.5
    assert float(getattr(persisted, "net_pnl")) == 11.7


def test_position_manager_partial_realization_accumulates_correctly():
    manager = PositionManager(
        data_store=Mock(),
        database=Mock(),
        risk_manager=Mock(),
        pnl_tracker=Mock(),
    )

    pos = Position(
        position_id="p1",
        symbol="ETHUSDT",
        direction="LONG",
        quantity=Decimal("10"),
        entry_price=Decimal("100"),
        leverage=5,
        sl_price=Decimal("95"),
        tp1_price=Decimal("110"),
        tp2_price=Decimal("120"),
        strategy_name="unit",
        regime="TRENDING_UP",
        signal_confidence=80.0,
        size_usdt=Decimal("1000"),
    )

    # First partial: close 5 at 110 (+50 gross)
    manager._record_realized_component(pos, Decimal("5"), Decimal("110"))
    # Final partial: close 5 at 95 (-25 gross)
    manager._record_realized_component(pos, Decimal("5"), Decimal("95"))

    # Gross total = +25; total roundtrip fees = 1000 * 0.001 = 1.0
    assert pos.realized_pnl == Decimal("25")
    assert pos.fees_paid == Decimal("1.0")

    effective_exit = manager._effective_exit_price(
        direction=pos.direction,
        entry_price=pos.entry_price,
        quantity=pos.quantity,
        gross_pnl=pos.realized_pnl,
    )
    assert effective_exit == Decimal("102.5")


def test_monitor_prices_prefer_live_ticker_over_candle_range():
    data_store = Mock()
    data_store.get_dataframe.return_value = pd.DataFrame(
        [
            {
                "open": 100.0,
                "high": 106.5,
                "low": 99.5,
                "close": 102.0,
                "volume": 1.0,
            }
        ]
    )
    data_store.get_price.return_value = Decimal("103.25")

    manager = PositionManager(
        data_store=data_store,
        database=Mock(),
        risk_manager=Mock(),
        pnl_tracker=Mock(),
    )

    pos = Position(
        position_id="p-high-low",
        symbol="ETHUSDT",
        direction="LONG",
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
        leverage=3,
        sl_price=Decimal("98"),
        tp1_price=Decimal("105"),
        tp2_price=Decimal("108"),
        strategy_name="unit",
        regime="TRENDING_UP",
        signal_confidence=80.0,
        size_usdt=Decimal("100"),
    )

    prices = manager._get_monitor_prices(pos)
    assert prices is not None
    monitor_price, high_price, low_price = prices
    assert monitor_price == Decimal("103.25")
    assert high_price == Decimal("103.25")
    assert low_price == Decimal("103.25")


def test_monitor_prices_fall_back_to_candle_close_when_no_live_price():
    data_store = Mock()
    data_store.get_dataframe.return_value = pd.DataFrame(
        [
            {
                "open": 100.0,
                "high": 106.5,
                "low": 99.5,
                "close": 102.0,
                "volume": 1.0,
            }
        ]
    )
    data_store.get_price.return_value = None

    manager = PositionManager(
        data_store=data_store,
        database=Mock(),
        risk_manager=Mock(),
        pnl_tracker=Mock(),
    )

    pos = Position(
        position_id="p-close-fallback",
        symbol="ETHUSDT",
        direction="LONG",
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
        leverage=3,
        sl_price=Decimal("98"),
        tp1_price=Decimal("105"),
        tp2_price=Decimal("108"),
        strategy_name="unit",
        regime="TRENDING_UP",
        signal_confidence=80.0,
        size_usdt=Decimal("100"),
    )

    prices = manager._get_monitor_prices(pos)
    assert prices is not None
    monitor_price, high_price, low_price = prices
    assert monitor_price == Decimal("102.0")
    assert high_price == Decimal("102.0")
    assert low_price == Decimal("102.0")


def test_breakeven_activation_uses_configurable_r_threshold(monkeypatch):
    manager = PositionManager(
        data_store=Mock(),
        database=Mock(),
        risk_manager=Mock(),
        pnl_tracker=Mock(),
    )

    pos = Position(
        position_id="p-breakeven-r",
        symbol="BTCUSDT",
        direction="LONG",
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
        leverage=3,
        sl_price=Decimal("95"),
        tp1_price=Decimal("108"),
        tp2_price=Decimal("112"),
        strategy_name="unit",
        regime="TRENDING_UP",
        signal_confidence=80.0,
        size_usdt=Decimal("100"),
    )

    monkeypatch.setattr(settings, "breakeven_activation_r", Decimal("0.8"), raising=False)

    pos.update_price(Decimal("102.5"))  # 0.5R above entry
    assert manager._should_move_to_breakeven(pos) is False

    pos.update_price(Decimal("104.0"))  # 0.8R above entry
    assert manager._should_move_to_breakeven(pos) is True


def test_fee_aware_breakeven_price_covers_roundtrip_fee():
    manager = PositionManager(
        data_store=Mock(),
        database=Mock(),
        risk_manager=Mock(),
        pnl_tracker=Mock(),
    )

    long_pos = Position(
        position_id="p-be-long",
        symbol="BTCUSDT",
        direction="LONG",
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
        leverage=3,
        sl_price=Decimal("95"),
        tp1_price=Decimal("105"),
        tp2_price=Decimal("110"),
        strategy_name="unit",
        regime="TRENDING_UP",
        signal_confidence=80.0,
        size_usdt=Decimal("100"),
    )
    short_pos = Position(
        position_id="p-be-short",
        symbol="BTCUSDT",
        direction="SHORT",
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
        leverage=3,
        sl_price=Decimal("105"),
        tp1_price=Decimal("95"),
        tp2_price=Decimal("90"),
        strategy_name="unit",
        regime="TRENDING_DOWN",
        signal_confidence=80.0,
        size_usdt=Decimal("100"),
    )

    assert manager._fee_aware_breakeven_price(long_pos) == Decimal("100.100")
    assert manager._fee_aware_breakeven_price(short_pos) == Decimal("99.900")


@pytest.mark.asyncio
async def test_tp1_resyncs_protective_stop_order():
    order_executor = Mock()
    order_executor.place_market_order = AsyncMock(return_value={"id": "tp1-close"})
    order_executor.cancel_order = AsyncMock(return_value=None)
    order_executor.place_stop_market = AsyncMock(return_value={"id": "new-sl"})

    db = Mock()
    risk_manager = Mock()
    pnl_tracker = Mock()
    manager = PositionManager(
        data_store=Mock(),
        database=db,
        risk_manager=risk_manager,
        pnl_tracker=pnl_tracker,
        order_executor=order_executor,
    )

    pos = Position(
        position_id="p-sync",
        symbol="ETHUSDT",
        direction="LONG",
        quantity=Decimal("10"),
        entry_price=Decimal("100"),
        leverage=5,
        sl_price=Decimal("95"),
        tp1_price=Decimal("110"),
        tp2_price=Decimal("120"),
        strategy_name="unit",
        regime="TRENDING_UP",
        signal_confidence=80.0,
        size_usdt=Decimal("1000"),
    )
    pos.sl_order_ids = ["old-sl"]

    await manager._handle_tp1(pos, Decimal("110"))

    assert pos.remaining_qty == Decimal("5")
    assert pos.sl_price == Decimal("100.100")
    assert pos.sl_order_ids == ["new-sl"]

    order_executor.cancel_order.assert_awaited_with("ETHUSDT", "old-sl")
    order_executor.place_stop_market.assert_awaited_with(
        symbol="ETHUSDT",
        side="SELL",
        quantity=5.0,
        stop_price=100.1,
        reduce_only=True,
    )
