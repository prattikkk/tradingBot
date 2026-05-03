"""Regression tests for trade accounting and stop-loss PnL handling."""

import asyncio
import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from alphabot.config import settings
from alphabot.database.db import Database
from alphabot.execution.order_executor import OrderExecutor
from alphabot.positions.pnl_tracker import PnLTracker
from alphabot.positions.position_manager import Position, PositionManager
from alphabot.strategies.signal import Signal, SignalDirection


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ({"id": "111", "orderId": "222"}, "111"),
        ({"orderId": "222"}, "222"),
        ({"info": {"orderId": "333"}}, "333"),
        ({"info": {"id": "444"}}, "444"),
        ({}, ""),
    ],
)
def test_extract_order_id_normalizes_ccxt_schemas(order, expected):
    # FIX[2]: Ensure order-id extraction is stable across ccxt/Binance field variants.
    assert OrderExecutor._extract_order_id(order) == expected


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


@pytest.mark.asyncio
async def test_close_position_reconciles_reduceonly_rejection():
    order_executor = Mock()
    order_executor.place_market_order = AsyncMock(
        side_effect=Exception('binance {"code":-2022,"msg":"ReduceOnly Order is rejected."}')
    )
    order_executor.cancel_all_orders = AsyncMock(return_value=None)
    order_executor.get_order = AsyncMock(return_value={"avgPrice": "94.5"})
    order_executor.get_my_trades = AsyncMock(return_value=[])

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
        position_id="p-reduceonly",
        symbol="SOLUSDT",
        direction="LONG",
        quantity=Decimal("2"),
        entry_price=Decimal("100"),
        leverage=5,
        sl_price=Decimal("95"),
        tp1_price=Decimal("110"),
        tp2_price=Decimal("120"),
        strategy_name="unit",
        regime="TRENDING_UP",
        signal_confidence=80.0,
        size_usdt=Decimal("200"),
    )
    pos.sl_order_ids = ["sl-order-1"]
    pos.order_ids = ["entry-order-1", "sl-order-1"]
    manager._positions[pos.id] = pos

    await manager.close_position(pos.id, "SL_HIT", Decimal("95"))

    assert pos.status == "CLOSED"
    assert pos.remaining_qty == Decimal("0")
    assert pos.current_price == Decimal("94.5")
    assert pos.realized_pnl == Decimal("-11.0")
    order_executor.cancel_all_orders.assert_awaited_with("SOLUSDT")
    order_executor.get_order.assert_awaited_with("SOLUSDT", "sl-order-1")
    risk_manager.record_trade_result.assert_called_once()
    pnl_tracker.record_trade.assert_called_once()


@pytest.mark.asyncio
async def test_open_position_tracks_ccxt_id_fields_for_entry_and_sl():
    order_executor = Mock()
    order_executor.set_margin_mode = AsyncMock(return_value=None)
    order_executor.set_leverage = AsyncMock(return_value=None)
    order_executor.place_market_order = AsyncMock(return_value={"id": "entry-123", "avgPrice": "100.5"})
    order_executor.wait_for_order_fill = AsyncMock(return_value={"id": "entry-123", "avgPrice": "100.5"})
    order_executor.place_stop_market = AsyncMock(return_value={"id": "sl-789"})

    manager = PositionManager(
        data_store=Mock(),
        database=Mock(),
        risk_manager=Mock(),
        pnl_tracker=Mock(),
        order_executor=order_executor,
    )

    signal = Signal(
        symbol="BTCUSDT",
        direction=SignalDirection.LONG,
        confidence=80.0,
        entry_price=Decimal("100"),
        stop_loss=Decimal("98"),
        take_profit_1=Decimal("104"),
        take_profit_2=Decimal("108"),
        strategy_name="unit",
        regime="TRENDING_UP",
        timeframe="15m",
    )
    size_info = {
        "quantity": Decimal("0.1"),
        "leverage": 5,
        "size_usdt": Decimal("10"),
    }

    pos = await manager.open_position(signal, size_info)

    assert pos is not None
    assert "entry-123" in pos.order_ids
    assert "sl-789" in pos.order_ids
    assert pos.sl_order_ids == ["sl-789"]
    assert pos.entry_price == Decimal("100.5")
    order_executor.set_margin_mode.assert_awaited_with("BTCUSDT", settings.margin_mode.upper())
    order_executor.wait_for_order_fill.assert_awaited_once_with(
        "BTCUSDT",
        "entry-123",
        initial_order={"id": "entry-123", "avgPrice": "100.5"},
        timeout_seconds=float(getattr(settings, "order_fill_timeout_seconds", 6.0)),
    )


@pytest.mark.asyncio
async def test_open_position_concurrent_calls_allow_single_create():
    # FIX[1]: Concurrent opens on the same symbol must resolve to exactly one position.
    order_executor = Mock()
    order_executor.set_margin_mode = AsyncMock(return_value=None)
    order_executor.set_leverage = AsyncMock(return_value=None)
    order_executor.place_market_order = AsyncMock(
        return_value={"id": "entry-conc", "avgPrice": "101.0"}
    )

    async def _wait_fill(symbol, order_id, initial_order=None, timeout_seconds=None):
        await asyncio.sleep(0.01)
        return {"id": order_id, "avgPrice": "101.0", "status": "closed"}

    order_executor.wait_for_order_fill = AsyncMock(side_effect=_wait_fill)
    order_executor.place_stop_market = AsyncMock(return_value={"id": "sl-conc"})

    manager = PositionManager(
        data_store=Mock(),
        database=Mock(),
        risk_manager=Mock(),
        pnl_tracker=Mock(),
        order_executor=order_executor,
    )

    signal = Signal(
        symbol="BTCUSDT",
        direction=SignalDirection.LONG,
        confidence=80.0,
        entry_price=Decimal("100"),
        stop_loss=Decimal("98"),
        take_profit_1=Decimal("104"),
        take_profit_2=Decimal("108"),
        strategy_name="unit",
        regime="TRENDING_UP",
        timeframe="15m",
    )
    size_info = {
        "quantity": Decimal("0.1"),
        "leverage": 5,
        "size_usdt": Decimal("10"),
    }

    first, second = await asyncio.gather(
        manager.open_position(signal, size_info),
        manager.open_position(signal, size_info),
    )

    opened = [pos for pos in (first, second) if pos is not None]
    assert len(opened) == 1
    assert len(manager.open_positions) == 1
    assert order_executor.place_market_order.await_count == 1


@pytest.mark.asyncio
async def test_normalize_quantity_uses_cached_market_metadata():
    # FIX[3]: load_markets should be cached and not called for every order normalization.
    class _StubExchange:
        def __init__(self):
            self.load_calls = 0

        async def load_markets(self):
            self.load_calls += 1
            return {
                "BTCUSDT": {
                    "limits": {"amount": {"min": 0.01}},
                    "precision": {"amount": 3},
                }
            }

        def market(self, symbol):
            return {
                "limits": {"amount": {"min": 0.01}},
                "precision": {"amount": 3},
            }

        def amount_to_precision(self, symbol, quantity):
            return f"{quantity:.3f}"

    client = Mock()
    client.exchange = _StubExchange()
    executor = OrderExecutor(client)

    first = await executor._normalize_quantity("BTCUSDT", 0.12345)
    second = await executor._normalize_quantity("BTCUSDT", 0.45678)

    assert first == pytest.approx(0.123)
    assert second == pytest.approx(0.457)
    assert client.exchange.load_calls == 1


@pytest.mark.asyncio
async def test_open_position_skips_duplicate_symbol_open():
    manager = PositionManager(
        data_store=Mock(),
        database=Mock(),
        risk_manager=Mock(),
        pnl_tracker=Mock(),
        order_executor=None,
    )

    existing = Position(
        position_id="existing",
        symbol="ETHUSDT",
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
    manager._positions[existing.id] = existing

    signal = Signal(
        symbol="ETHUSDT",
        direction=SignalDirection.LONG,
        confidence=80.0,
        entry_price=Decimal("101"),
        stop_loss=Decimal("99"),
        take_profit_1=Decimal("105"),
        take_profit_2=Decimal("108"),
        strategy_name="unit",
        regime="TRENDING_UP",
        timeframe="15m",
    )
    size_info = {
        "quantity": Decimal("0.5"),
        "leverage": 3,
        "size_usdt": Decimal("50"),
    }

    pos = await manager.open_position(signal, size_info)
    assert pos is None
