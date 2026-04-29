"""
AlphaBot Position Manager — Full lifecycle orchestrator.
Tracks open positions, manages trailing stops, partial exits,
time-based stops, and breakeven moves.

Position Lifecycle:
  1. Signal approved → Position created + market order placed
    2. Protective SL order placed; TP logic handled by monitor loop
        3. Monitor loop (every 1s): update PnL, check SL/TP/trailing
             and keep protective SL aligned with remaining quantity
  4. TP1 hit → partial close 50%, move SL to breakeven
  5. TP2 hit → partial close 30%, trailing stop final 20%
  6. Position fully closed → trade logged, PnL updated, alert sent
"""

from __future__ import annotations

import asyncio
import datetime
import uuid
from decimal import Decimal
from typing import Any, Callable, Coroutine, Dict, List, Optional

from loguru import logger

from alphabot.config import settings
from alphabot.data.data_store import DataStore
from alphabot.database.db import Database
from alphabot.database.models import PositionRecord
from alphabot.positions.pnl_tracker import PnLTracker
from alphabot.risk.risk_manager import RiskManager
from alphabot.strategies.signal import Signal, SignalDirection


class Position:
    """In-memory position object for the monitor loop."""

    def __init__(
        self,
        position_id: str,
        symbol: str,
        direction: str,
        quantity: Decimal,
        entry_price: Decimal,
        leverage: int,
        sl_price: Decimal,
        tp1_price: Decimal,
        tp2_price: Optional[Decimal],
        strategy_name: str,
        regime: str,
        signal_confidence: float,
        size_usdt: Decimal = Decimal("0"),
    ):
        self.id = position_id
        self.symbol = symbol
        self.direction = direction  # "LONG" or "SHORT"
        self.status = "OPEN"       # OPEN, PARTIAL, CLOSED
        self.quantity = quantity
        self.remaining_qty = quantity
        self.entry_price = entry_price
        self.current_price = entry_price
        self.leverage = leverage
        self.sl_price = sl_price
        self.original_sl = sl_price
        self.tp1_price = tp1_price
        self.tp2_price = tp2_price
        self.trailing_stop_active = False
        self.trailing_stop_price: Optional[Decimal] = None
        self.breakeven_moved = False
        self.tp1_hit = False
        self.tp2_hit = False
        self.strategy_name = strategy_name
        self.regime = regime
        self.signal_confidence = signal_confidence
        self.size_usdt = size_usdt
        self.unrealized_pnl = Decimal("0")
        self.realized_pnl = Decimal("0")
        self.fees_paid = Decimal("0")
        self.opened_at = datetime.datetime.now(datetime.UTC)
        self.closed_at: Optional[datetime.datetime] = None
        self.close_reason: Optional[str] = None
        self.order_ids: List[str] = []
        self.tp_order_ids: List[str] = []
        self.sl_order_ids: List[str] = []
        self._last_sl_sync_price: Optional[Decimal] = None
        self._last_sl_sync_qty: Optional[Decimal] = None
        self._last_sl_sync_at: Optional[datetime.datetime] = None
        # Track highest/lowest price since entry for trailing stop
        self._peak_price = entry_price
        self._trough_price = entry_price

    @property
    def r_multiple(self) -> float:
        """Current profit in R multiples."""
        sl_dist = abs(float(self.entry_price) - float(self.original_sl))
        if sl_dist == 0:
            return 0.0
        if self.direction == "LONG":
            profit = float(self.current_price) - float(self.entry_price)
        else:
            profit = float(self.entry_price) - float(self.current_price)
        return profit / sl_dist

    def update_price(self, price: Decimal) -> None:
        """Update current price and peak/trough tracking."""
        self.current_price = price
        if self.direction == "LONG":
            if price > self._peak_price:
                self._peak_price = price
            self.unrealized_pnl = (price - self.entry_price) * self.remaining_qty
        else:
            if price < self._trough_price:
                self._trough_price = price
            self.unrealized_pnl = (self.entry_price - price) * self.remaining_qty

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "status": self.status,
            "quantity": float(self.remaining_qty),
            "entry_price": float(self.entry_price),
            "current_price": float(self.current_price),
            "sl_price": float(self.sl_price),
            "tp1_price": float(self.tp1_price),
            "tp2_price": float(self.tp2_price) if self.tp2_price else None,
            "unrealized_pnl": float(self.unrealized_pnl),
            "leverage": self.leverage,
            "strategy": self.strategy_name,
            "regime": self.regime,
            "opened_at": self.opened_at.isoformat(),
            "r_multiple": round(self.r_multiple, 2),
        }


class PositionManager:
    """
    Manages position lifecycle — open, monitor, partial close, full close.
    Runs a continuous async monitor loop.
    """

    def __init__(
        self,
        data_store: DataStore,
        database: Database,
        risk_manager: RiskManager,
        pnl_tracker: PnLTracker,
        order_executor=None,  # Injected later to avoid circular imports
        notifier=None,        # Telegram notifier
        on_position_closed: Optional[Callable[[], Coroutine[Any, Any, None]]] = None,
    ):
        self.data_store = data_store
        self.db = database
        self.risk_manager = risk_manager
        self.pnl_tracker = pnl_tracker
        self.order_executor = order_executor
        self.notifier = notifier
        self.on_position_closed = on_position_closed
        self._positions: Dict[str, Position] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        # Cache for per-symbol ATR values used in trailing stop calculations.
        # Populated from strategy/execution layer via update_atr_cache().
        self._atr_cache: Dict[str, float] = {}

    def set_order_executor(self, executor) -> None:
        self.order_executor = executor

    def set_notifier(self, notifier) -> None:
        self.notifier = notifier

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self._positions.values() if p.status in ("OPEN", "PARTIAL")]

    @property
    def open_positions_dicts(self) -> List[dict]:
        return [p.to_dict() for p in self.open_positions]

    @property
    def total_exposure(self) -> Decimal:
        return sum((p.size_usdt for p in self.open_positions), Decimal("0"))

    async def start_monitor(self) -> None:
        """Start the position monitor loop."""
        self._running = True
        # Recover open positions from DB
        await self._recover_positions()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("[PosManager] Monitor loop started")

    async def stop_monitor(self) -> None:
        """Stop the monitor loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("[PosManager] Monitor loop stopped")

    async def open_position(self, signal: Signal, size_info: dict) -> Optional[Position]:
        """
        Create a new position from an approved signal.
        Places entry order + protective SL order.
        """
        position_id = str(uuid.uuid4())
        qty = size_info["quantity"]
        leverage = size_info["leverage"]

        pos = Position(
            position_id=position_id,
            symbol=signal.symbol,
            direction=signal.direction.value,
            quantity=qty,
            entry_price=signal.entry_price,
            leverage=leverage,
            sl_price=signal.stop_loss,
            tp1_price=signal.take_profit_1,
            tp2_price=signal.take_profit_2,
            strategy_name=signal.strategy_name,
            regime=signal.regime,
            signal_confidence=signal.confidence,
            size_usdt=size_info["size_usdt"],
        )

        # Place orders via executor
        if self.order_executor:
            try:
                # Set leverage
                await self.order_executor.set_leverage(signal.symbol, leverage)

                # Market entry order
                entry_order = await self.order_executor.place_market_order(
                    symbol=signal.symbol,
                    side="BUY" if signal.direction == SignalDirection.LONG else "SELL",
                    quantity=float(qty),
                )
                if entry_order:
                    pos.order_ids.append(str(entry_order.get("orderId", "")))
                    # Update entry price to actual exchange fill price to avoid SL
                    # distance mismatch caused by REST polling lag (signal price can
                    # be 0-30s stale when using REST fallback).
                    actual_fill = (
                        entry_order.get("avgPrice")
                        or entry_order.get("average")
                        or entry_order.get("price")
                    )
                    if actual_fill:
                        try:
                            fill_price = Decimal(str(actual_fill))
                            if fill_price > 0:
                                pos.entry_price = fill_price
                                logger.info(
                                    f"[PosManager] {signal.symbol}: entry fill "
                                    f"{fill_price} (signal was {signal.entry_price})"
                                )
                        except Exception:
                            pass

                # Stop-loss order (placed immediately after entry)
                sl_side = "SELL" if signal.direction == SignalDirection.LONG else "BUY"
                sl_order = await self.order_executor.place_stop_market(
                    symbol=signal.symbol,
                    side=sl_side,
                    quantity=float(qty),
                    stop_price=float(signal.stop_loss),
                    reduce_only=True,
                )
                if sl_order:
                    sl_id = str(sl_order.get("orderId", "") or sl_order.get("id", ""))
                    if sl_id:
                        pos.sl_order_ids = [sl_id]
                        pos.order_ids.append(sl_id)

            except Exception as e:
                logger.error(f"[PosManager] Failed to place orders for {signal.symbol}: {e}")
                return None

        # Register position
        self._positions[position_id] = pos

        # Persist to DB
        self._persist_position(pos)

        # Notify
        if self.notifier:
            asyncio.create_task(self.notifier.notify_trade_opened(pos))

        logger.info(
            f"[PosManager] Position opened: {position_id} {signal.symbol} "
            f"{signal.direction.value} qty={qty} entry={signal.entry_price} "
            f"SL={signal.stop_loss} TP1={signal.take_profit_1}"
        )
        return pos

    async def close_position(self, position_id: str, reason: str,
                              exit_price: Optional[Decimal] = None) -> None:
        """Close a position fully."""
        pos = self._positions.get(position_id)
        if not pos or pos.status == "CLOSED":
            return

        price = exit_price or pos.current_price
        pos.current_price = price
        pos.status = "CLOSED"
        pos.closed_at = datetime.datetime.now(datetime.UTC)
        pos.close_reason = reason

        # Realize remaining quantity using unlevered contract PnL.
        # Quantity already represents notional exposure; multiplying by leverage double-counts risk.
        close_qty = pos.remaining_qty if pos.remaining_qty > 0 else Decimal("0")
        if close_qty > 0:
            self._record_realized_component(pos, close_qty, price)
            pos.remaining_qty = Decimal("0")

        net_pnl = pos.realized_pnl - pos.fees_paid

        # Close via executor
        if self.order_executor and close_qty > 0:
            try:
                close_side = "SELL" if pos.direction == "LONG" else "BUY"
                await self.order_executor.place_market_order(
                    symbol=pos.symbol,
                    side=close_side,
                    quantity=float(close_qty),
                    reduce_only=True,
                )
                # Cancel any outstanding orders for this position
                await self.order_executor.cancel_all_orders(pos.symbol)
            except Exception as e:
                logger.error(f"[PosManager] Error closing position {position_id}: {e}")

        # Persist CLOSED position early so crash/restart recovery can't re-record trades.
        try:
            self._persist_position(pos)
        except Exception as e:
            logger.error(f"[PosManager] Failed to persist closed position {position_id}: {e}")

        # Record in PnL tracker (use remaining_qty to account for partial closes)
        try:
            effective_exit = self._effective_exit_price(
                direction=pos.direction,
                entry_price=pos.entry_price,
                quantity=pos.quantity,
                gross_pnl=pos.realized_pnl,
            )
            self.pnl_tracker.record_trade(
                trade_id=pos.id,
                symbol=pos.symbol,
                direction=pos.direction,
                entry_price=float(pos.entry_price),
                exit_price=float(effective_exit),
                quantity=float(pos.quantity),
                leverage=pos.leverage,
                fees=float(pos.fees_paid),
                strategy_name=pos.strategy_name,
                regime=pos.regime,
                signal_confidence=pos.signal_confidence,
                exit_reason=reason,
                open_time=pos.opened_at,
                close_time=pos.closed_at,
                gross_pnl_override=float(pos.realized_pnl),
                net_pnl_override=float(net_pnl),
            )
        except Exception as e:
            logger.error(f"[PosManager] Failed to record trade for {position_id}: {e}")

        # Update risk manager
        try:
            self.risk_manager.record_trade_result(
                pos.symbol,
                net_pnl,
                is_win=(net_pnl > 0),
                strategy_name=pos.strategy_name,
                db=self.db,
            )
        except Exception as e:
            logger.error(f"[PosManager] Failed to record trade result for {position_id}: {e}")

        # Notify
        if self.notifier:
            asyncio.create_task(self.notifier.notify_trade_closed(pos, float(net_pnl)))

        if self.on_position_closed:
            asyncio.create_task(self.on_position_closed())

        logger.info(
            f"[PosManager] Position closed: {position_id} {pos.symbol} "
            f"reason={reason} PnL={float(net_pnl):.2f}"
        )

    async def close_all_positions(self, reason: str = "EMERGENCY_CLOSE") -> None:
        """Emergency: close all open positions."""
        for pos in list(self.open_positions):
            await self.close_position(pos.id, reason)
        logger.warning(f"[PosManager] All positions closed: {reason}")

    async def _monitor_loop(self) -> None:
        """Continuous loop: update prices, check SL/TP/trailing for all positions."""
        while self._running:
            try:
                for pos in list(self.open_positions):
                    price_snapshot = self._get_monitor_prices(pos)
                    if price_snapshot is None:
                        continue
                    monitor_price, _, _ = price_snapshot

                    pos.update_price(monitor_price)

                    # Use live mark/ticker price for trigger checks. This avoids falsely
                    # triggering SL/TP from the already-closed candle range that happened
                    # before the position was opened.
                    stop_probe = monitor_price
                    tp_probe = monitor_price

                    # Check stop-loss
                    if self._check_stop_loss(pos, stop_probe):
                        stop_reason = "BREAKEVEN_STOP" if pos.breakeven_moved else "SL_HIT"
                        await self.close_position(pos.id, stop_reason, pos.sl_price)
                        continue

                    # Check TP1
                    if not pos.tp1_hit and self._check_tp1(pos, tp_probe):
                        await self._handle_tp1(pos, pos.tp1_price)

                    # Check TP2
                    if pos.tp1_hit and not pos.tp2_hit and self._check_tp2(pos, tp_probe):
                        await self._handle_tp2(pos, pos.tp2_price or monitor_price)

                    # Trailing stop check
                    if pos.trailing_stop_active and self._check_trailing_stop(pos, stop_probe):
                        trailing_exit = pos.trailing_stop_price or monitor_price
                        await self.close_position(pos.id, "TRAILING_STOP", trailing_exit)
                        continue

                    # Breakeven move: only after configurable R progress.
                    if self._should_move_to_breakeven(pos):
                        pos.sl_price = self._fee_aware_breakeven_price(pos)
                        pos.breakeven_moved = True
                        await self._sync_protective_stop(pos, reason="BREAKEVEN_MOVE")
                        logger.info(f"[PosManager] {pos.id}: SL moved to breakeven")

                    # Update trailing stop price
                    if pos.trailing_stop_active:
                        trail_moved = self._update_trailing_stop(pos, monitor_price)
                        if trail_moved:
                            await self._sync_protective_stop(pos, reason="TRAIL_UPDATE")

                    # Time-based stop: if > N hours with < 20% progress toward TP
                    elapsed_hours = (datetime.datetime.now(datetime.UTC) - pos.opened_at).total_seconds() / 3600
                    if elapsed_hours > settings.time_stop_hours:
                        tp_dist = abs(float(pos.tp1_price) - float(pos.entry_price))
                        if tp_dist > 0:
                            if pos.direction == "LONG":
                                progress = (float(monitor_price) - float(pos.entry_price)) / tp_dist * 100
                            else:
                                progress = (float(pos.entry_price) - float(monitor_price)) / tp_dist * 100
                            if progress < float(settings.time_stop_progress_pct):
                                # Only close if still losing; if slightly profitable, lock breakeven.
                                if pos.r_multiple <= 0:
                                    await self.close_position(pos.id, "TIME_STOP", monitor_price)
                                    continue
                                if not pos.breakeven_moved:
                                    pos.sl_price = self._fee_aware_breakeven_price(pos)
                                    pos.breakeven_moved = True
                                    await self._sync_protective_stop(pos, reason="TIME_STOP_BREAKEVEN")
                                    logger.info(
                                        f"[PosManager] {pos.id}: TIME_STOP triggered but trade is green — SL moved to breakeven"
                                    )

                    # Persist updated state
                    self.db.update_position(
                        pos.id,
                        current_price=float(pos.current_price),
                        unrealized_pnl=float(pos.unrealized_pnl),
                        sl_price=float(pos.sl_price),
                        trailing_stop_price=float(pos.trailing_stop_price) if pos.trailing_stop_price else None,
                        trailing_stop_active=1 if pos.trailing_stop_active else 0,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[PosManager] Monitor loop error: {e}")

            await asyncio.sleep(1)  # Poll every 1 second

    def _get_monitor_prices(self, pos: Position) -> Optional[tuple[Decimal, Decimal, Decimal]]:
        """
        Return (price, price, price) where price is:
        1) latest live ticker/mark price when available, else
        2) latest closed-candle close as a safe fallback.

        We intentionally avoid candle high/low probes here because those highs/lows can
        include movement that happened before a newly-opened position existed.
        """
        tick_price = self.data_store.get_price(pos.symbol)
        if tick_price is not None:
            return tick_price, tick_price, tick_price

        try:
            tf = settings.primary_timeframe
            df = self.data_store.get_dataframe(pos.symbol, tf)
            if not df.empty:
                row = df.iloc[-1]
                close_val = row.get("close")
                if close_val is not None:
                    close_price = Decimal(str(close_val))
                    return close_price, close_price, close_price
        except Exception:
            pass

        return None

    def _check_stop_loss(self, pos: Position, price: Decimal) -> bool:
        if pos.direction == "LONG":
            return price <= pos.sl_price
        return price >= pos.sl_price

    def _check_tp1(self, pos: Position, price: Decimal) -> bool:
        if pos.direction == "LONG":
            return price >= pos.tp1_price
        return price <= pos.tp1_price

    def _check_tp2(self, pos: Position, price: Decimal) -> bool:
        if pos.tp2_price is None:
            return False
        if pos.direction == "LONG":
            return price >= pos.tp2_price
        return price <= pos.tp2_price

    def _check_trailing_stop(self, pos: Position, price: Decimal) -> bool:
        if pos.trailing_stop_price is None:
            return False
        if pos.direction == "LONG":
            return price <= pos.trailing_stop_price
        return price >= pos.trailing_stop_price

    @staticmethod
    def _should_move_to_breakeven(pos: Position) -> bool:
        return (not pos.breakeven_moved) and (
            pos.r_multiple >= float(settings.breakeven_activation_r)
        )

    @staticmethod
    def _fee_aware_breakeven_price(pos: Position) -> Decimal:
        fee_rate = Decimal(str(settings.estimated_roundtrip_fee_rate))
        if pos.direction == "LONG":
            return pos.entry_price * (Decimal("1") + fee_rate)
        return pos.entry_price * (Decimal("1") - fee_rate)

    def _update_trailing_stop(self, pos: Position, price: Decimal) -> bool:
        """Move trailing stop in the direction of profit."""
        atr_val = self._get_atr(pos.symbol)
        trail_dist = Decimal(str(atr_val * 1.5))
        moved = False

        if pos.direction == "LONG":
            new_stop = price - trail_dist
            if pos.trailing_stop_price is None or new_stop > pos.trailing_stop_price:
                pos.trailing_stop_price = new_stop
                pos.sl_price = new_stop
                moved = True
        else:
            new_stop = price + trail_dist
            if pos.trailing_stop_price is None or new_stop < pos.trailing_stop_price:
                pos.trailing_stop_price = new_stop
                pos.sl_price = new_stop
                moved = True

        return moved

    async def _handle_tp1(self, pos: Position, price: Decimal) -> None:
        """Partial close at TP1: close 50%, move SL to breakeven."""
        close_qty = pos.remaining_qty * Decimal("0.5")
        if close_qty <= 0:
            return

        # Close partial via executor
        executed = True
        if self.order_executor:
            try:
                close_side = "SELL" if pos.direction == "LONG" else "BUY"
                await self.order_executor.place_market_order(
                    symbol=pos.symbol,
                    side=close_side,
                    quantity=float(close_qty),
                    reduce_only=True,
                )
            except Exception as e:
                logger.error(f"[PosManager] TP1 partial close error: {e}")
                executed = False

        if not executed:
            return

        self._record_realized_component(pos, close_qty, price)
        pos.remaining_qty -= close_qty
        pos.tp1_hit = True
        pos.sl_price = self._fee_aware_breakeven_price(pos)
        pos.breakeven_moved = True
        pos.status = "PARTIAL"
        await self._sync_protective_stop(pos, reason="TP1")

        if self.notifier:
            asyncio.create_task(self.notifier.notify_tp1_hit(pos))

        logger.info(
            f"[PosManager] {pos.id}: TP1 hit — closed {close_qty}, "
            f"remaining={pos.remaining_qty}, SL→breakeven"
        )

    async def _handle_tp2(self, pos: Position, price: Decimal) -> None:
        """Partial close at TP2: close 30% of original, trailing stop on rest."""
        close_qty = pos.quantity * Decimal("0.3")
        close_qty = min(close_qty, pos.remaining_qty)
        if close_qty <= 0:
            return

        executed = True
        if self.order_executor:
            try:
                close_side = "SELL" if pos.direction == "LONG" else "BUY"
                await self.order_executor.place_market_order(
                    symbol=pos.symbol,
                    side=close_side,
                    quantity=float(close_qty),
                    reduce_only=True,
                )
            except Exception as e:
                logger.error(f"[PosManager] TP2 partial close error: {e}")
                executed = False

        if not executed:
            return

        self._record_realized_component(pos, close_qty, price)
        pos.remaining_qty -= close_qty
        pos.tp2_hit = True

        if pos.remaining_qty <= 0:
            await self.close_position(pos.id, "TP2_HIT", price)
            return

        # Activate trailing stop on remainder
        pos.trailing_stop_active = True
        await self._sync_protective_stop(pos, reason="TP2")

        logger.info(
            f"[PosManager] {pos.id}: TP2 hit — closed {close_qty}, "
            f"remaining={pos.remaining_qty} with trailing stop"
        )

    async def _sync_protective_stop(self, pos: Position, reason: str) -> None:
        """Recreate reduce-only stop order to match latest SL price and remaining size."""
        if not self.order_executor:
            return
        if pos.status == "CLOSED" or pos.remaining_qty <= 0:
            return

        now = datetime.datetime.now(datetime.UTC)
        if reason == "TRAIL_UPDATE":
            # Throttle frequent trailing updates to avoid excessive cancel/recreate churn.
            if (
                pos._last_sl_sync_price is not None
                and pos._last_sl_sync_qty is not None
                and pos._last_sl_sync_qty == pos.remaining_qty
                and pos._last_sl_sync_at is not None
            ):
                age_seconds = (now - pos._last_sl_sync_at).total_seconds()
                atr_step = Decimal(str(self._get_atr(pos.symbol) * 0.2))
                price_delta = abs(pos.sl_price - pos._last_sl_sync_price)
                if age_seconds < 20 and price_delta < atr_step:
                    return

        try:
            # Cancel previously tracked protective stops for this position.
            for order_id in list(pos.sl_order_ids):
                if not order_id:
                    continue
                try:
                    await self.order_executor.cancel_order(pos.symbol, order_id)
                except Exception as cancel_err:
                    logger.debug(
                        f"[PosManager] SL sync cancel skipped for {pos.id} order={order_id}: {cancel_err}"
                    )

            sl_side = "SELL" if pos.direction == "LONG" else "BUY"
            sl_order = await self.order_executor.place_stop_market(
                symbol=pos.symbol,
                side=sl_side,
                quantity=float(pos.remaining_qty),
                stop_price=float(pos.sl_price),
                reduce_only=True,
            )

            new_sl_id = ""
            if sl_order:
                new_sl_id = str(sl_order.get("orderId", "") or sl_order.get("id", ""))

            pos.sl_order_ids = [new_sl_id] if new_sl_id else []
            if new_sl_id and new_sl_id not in pos.order_ids:
                pos.order_ids.append(new_sl_id)
            pos._last_sl_sync_price = pos.sl_price
            pos._last_sl_sync_qty = pos.remaining_qty
            pos._last_sl_sync_at = now

            # Persist refreshed stop IDs/prices for restart consistency.
            self._persist_position(pos)
            logger.info(
                f"[PosManager] {pos.id}: Protective SL synced ({reason}) "
                f"qty={pos.remaining_qty} stop={pos.sl_price}"
            )
        except Exception as e:
            logger.error(f"[PosManager] Protective SL sync failed for {pos.id}: {e}")

    @staticmethod
    def _calculate_gross_pnl(
        direction: str,
        entry_price: Decimal,
        exit_price: Decimal,
        quantity: Decimal,
    ) -> Decimal:
        if direction == "LONG":
            return (exit_price - entry_price) * quantity
        return (entry_price - exit_price) * quantity

    def _estimate_fee_component(self, pos: Position, close_qty: Decimal) -> Decimal:
        if pos.quantity <= 0 or close_qty <= 0:
            return Decimal("0")
        ratio = close_qty / pos.quantity
        total_roundtrip_fee = pos.size_usdt * Decimal(str(settings.estimated_roundtrip_fee_rate))
        return total_roundtrip_fee * ratio

    def _record_realized_component(self, pos: Position, close_qty: Decimal, exit_price: Decimal) -> None:
        gross_component = self._calculate_gross_pnl(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=close_qty,
        )
        fee_component = self._estimate_fee_component(pos, close_qty)
        pos.realized_pnl += gross_component
        pos.fees_paid += fee_component

    @staticmethod
    def _effective_exit_price(
        direction: str,
        entry_price: Decimal,
        quantity: Decimal,
        gross_pnl: Decimal,
    ) -> Decimal:
        if quantity <= 0:
            return entry_price
        if direction == "LONG":
            return entry_price + (gross_pnl / quantity)
        return entry_price - (gross_pnl / quantity)

    def _get_atr(self, symbol: str) -> float:
        """Get latest ATR value for a symbol from cache.
        Cache is updated only on candle close, not on every monitor loop tick.
        """
        # Return cached value if available
        if symbol in self._atr_cache:
            return self._atr_cache[symbol]
        
        # Fallback: compute once and cache
        df = self.data_store.get_dataframe(symbol, settings.primary_timeframe)
        if df.empty or 'atr' not in df.columns:
            return 0.0
        
        atr_val = float(df['atr'].iloc[-1]) if not df['atr'].isna().all() else 0.0
        self._atr_cache[symbol] = atr_val
        return atr_val
    
    def update_atr_cache(self, symbol: str, atr_val: float) -> None:
        """Update ATR cache when new candle closes. Call this from strategy engine."""
        self._atr_cache[symbol] = atr_val

    def _persist_position(self, pos: Position) -> None:
        """Save position to database."""
        import json
        rec = PositionRecord(
            id=pos.id,
            symbol=pos.symbol,
            direction=pos.direction,
            status=pos.status,
            size_usdt=float(pos.size_usdt),
            quantity=float(pos.quantity),
            leverage=pos.leverage,
            entry_price=float(pos.entry_price),
            current_price=float(pos.current_price),
            sl_price=float(pos.sl_price),
            tp1_price=float(pos.tp1_price),
            tp2_price=float(pos.tp2_price) if pos.tp2_price else None,
            trailing_stop_price=float(pos.trailing_stop_price) if pos.trailing_stop_price else None,
            trailing_stop_active=1 if pos.trailing_stop_active else 0,
            unrealized_pnl=float(pos.unrealized_pnl),
            realized_pnl=float(pos.realized_pnl),
            fees_paid=float(pos.fees_paid),
            strategy_name=pos.strategy_name,
            regime_at_entry=pos.regime,
            signal_confidence=pos.signal_confidence,
            open_timestamp=pos.opened_at,
            close_timestamp=pos.closed_at,
            exit_reason=pos.close_reason,
            order_ids=json.dumps(pos.order_ids),
            tp_order_ids=json.dumps(pos.tp_order_ids),
            sl_order_ids=json.dumps(pos.sl_order_ids),
        )
        self.db.save_position(rec)

    async def _recover_positions(self) -> None:
        """Recover open positions from DB on restart."""
        import json
        open_recs = self.db.get_open_positions()
        for rec in open_recs:
            # If a trade already exists for this position ID, do not recover it as open.
            # This protects against crashes that happened after recording the trade but
            # before persisting the position as CLOSED.
            try:
                existing_trade = self.db.get_trade(str(rec.id))
            except Exception:
                existing_trade = None
            if existing_trade is not None:
                logger.warning(
                    f"[PosManager] Recovery reconcile: {rec.id} is OPEN in DB but trade exists; marking CLOSED"
                )
                try:
                    self.db.update_position(
                        str(rec.id),
                        status="CLOSED",
                        close_timestamp=getattr(existing_trade, "close_timestamp", None),
                        exit_reason=getattr(existing_trade, "exit_reason", None),
                    )
                except Exception as e:
                    logger.error(f"[PosManager] Recovery reconcile failed for {rec.id}: {e}")
                continue

            tp1 = getattr(rec, "tp1_price", None)
            tp2 = getattr(rec, "tp2_price", None)
            leverage_raw = getattr(rec, "leverage", settings.max_leverage)
            signal_conf_raw = getattr(rec, "signal_confidence", 0.0)
            pos = Position(
                position_id=str(rec.id),
                symbol=str(rec.symbol),
                direction=str(rec.direction),
                quantity=Decimal(str(rec.quantity)),
                entry_price=Decimal(str(rec.entry_price)),
                leverage=int(leverage_raw),
                sl_price=Decimal(str(rec.sl_price)),
                tp1_price=Decimal(str(tp1)) if tp1 is not None else Decimal("0"),
                tp2_price=Decimal(str(tp2)) if tp2 is not None else None,
                strategy_name=str(rec.strategy_name),
                regime=str(rec.regime_at_entry),
                signal_confidence=float(signal_conf_raw or 0.0),
                size_usdt=Decimal(str(rec.size_usdt)),
            )
            pos.status = str(rec.status)
            pos.trailing_stop_active = bool(rec.trailing_stop_active)
            if rec.trailing_stop_price is not None:
                pos.trailing_stop_price = Decimal(str(rec.trailing_stop_price))
            order_ids_raw = getattr(rec, "order_ids", None)
            if order_ids_raw is not None and str(order_ids_raw).strip():
                pos.order_ids = json.loads(str(order_ids_raw))
            if getattr(rec, "tp_order_ids", None):
                pos.tp_order_ids = json.loads(str(rec.tp_order_ids))
            if getattr(rec, "sl_order_ids", None):
                pos.sl_order_ids = json.loads(str(rec.sl_order_ids))
            self._positions[str(rec.id)] = pos
            logger.info(f"[PosManager] Recovered position: {rec.id} {rec.symbol} {rec.direction}")

        if open_recs:
            logger.info(f"[PosManager] Recovered {len(open_recs)} open positions from DB")
