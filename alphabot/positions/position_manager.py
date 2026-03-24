"""
AlphaBot Position Manager — Full lifecycle orchestrator.
Tracks open positions, manages trailing stops, partial exits,
time-based stops, and breakeven moves.

Position Lifecycle:
  1. Signal approved → Position created + market order placed
  2. SL/TP1 limit orders placed simultaneously
  3. Monitor loop (every 1s): update PnL, check trailing stop
  4. TP1 hit → partial close 50%, move SL to breakeven
  5. TP2 hit → partial close 30%, trailing stop final 20%
  6. Position fully closed → trade logged, PnL updated, alert sent
"""

from __future__ import annotations

import asyncio
import datetime
import uuid
from decimal import Decimal
from typing import Callable, Dict, List, Optional

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
            self.unrealized_pnl = (price - self.entry_price) * self.remaining_qty * self.leverage
        else:
            if price < self._trough_price:
                self._trough_price = price
            self.unrealized_pnl = (self.entry_price - price) * self.remaining_qty * self.leverage

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
    ):
        self.data_store = data_store
        self.db = database
        self.risk_manager = risk_manager
        self.pnl_tracker = pnl_tracker
        self.order_executor = order_executor
        self.notifier = notifier
        self._positions: Dict[str, Position] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

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
        Places entry order + SL/TP orders.
        """
        position_id = str(uuid.uuid4())[:12]
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

                # Stop-loss order (placed immediately after entry)
                sl_side = "SELL" if signal.direction == SignalDirection.LONG else "BUY"
                sl_order = await self.order_executor.place_stop_market(
                    symbol=signal.symbol,
                    side=sl_side,
                    quantity=float(qty),
                    stop_price=float(signal.stop_loss),
                )
                if sl_order:
                    pos.order_ids.append(str(sl_order.get("orderId", "")))

                # TP1 limit order
                tp_side = "SELL" if signal.direction == SignalDirection.LONG else "BUY"
                tp1_qty = float(qty * Decimal("0.5"))
                tp1_order = await self.order_executor.place_limit_order(
                    symbol=signal.symbol,
                    side=tp_side,
                    quantity=tp1_qty,
                    price=float(signal.take_profit_1),
                )
                if tp1_order:
                    pos.order_ids.append(str(tp1_order.get("orderId", "")))

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

        # Calculate final PnL
        if pos.direction == "LONG":
            pos.realized_pnl = (price - pos.entry_price) * pos.remaining_qty * pos.leverage
        else:
            pos.realized_pnl = (pos.entry_price - price) * pos.remaining_qty * pos.leverage

        # Estimate fees (0.04% taker fee × 2 for open+close)
        fees = float(pos.size_usdt) * 0.0004 * 2
        pos.fees_paid = Decimal(str(fees))
        net_pnl = pos.realized_pnl - pos.fees_paid

        # Close via executor
        if self.order_executor and pos.remaining_qty > 0:
            try:
                close_side = "SELL" if pos.direction == "LONG" else "BUY"
                await self.order_executor.place_market_order(
                    symbol=pos.symbol,
                    side=close_side,
                    quantity=float(pos.remaining_qty),
                )
                # Cancel any outstanding orders for this position
                await self.order_executor.cancel_all_orders(pos.symbol)
            except Exception as e:
                logger.error(f"[PosManager] Error closing position {position_id}: {e}")

        # Record in PnL tracker
        self.pnl_tracker.record_trade(
            trade_id=pos.id,
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=float(pos.entry_price),
            exit_price=float(price),
            quantity=float(pos.quantity),
            leverage=pos.leverage,
            fees=fees,
            strategy_name=pos.strategy_name,
            regime=pos.regime,
            signal_confidence=pos.signal_confidence,
            exit_reason=reason,
            open_time=pos.opened_at,
            close_time=pos.closed_at,
        )

        # Update risk manager
        self.risk_manager.record_trade_result(
            pos.symbol, net_pnl, is_win=(net_pnl > 0)
        )

        # Persist
        self._persist_position(pos)

        # Notify
        if self.notifier:
            asyncio.create_task(self.notifier.notify_trade_closed(pos, float(net_pnl)))

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
                    price = self.data_store.get_price(pos.symbol)
                    if price is None:
                        continue

                    pos.update_price(price)

                    # Check stop-loss
                    if self._check_stop_loss(pos, price):
                        await self.close_position(pos.id, "SL_HIT", price)
                        continue

                    # Check TP1
                    if not pos.tp1_hit and self._check_tp1(pos, price):
                        await self._handle_tp1(pos, price)

                    # Check TP2
                    if pos.tp1_hit and not pos.tp2_hit and self._check_tp2(pos, price):
                        await self._handle_tp2(pos, price)

                    # Trailing stop check
                    if pos.trailing_stop_active and self._check_trailing_stop(pos, price):
                        await self.close_position(pos.id, "TRAILING_STOP", price)
                        continue

                    # Breakeven move: at 0.5R profit, move SL to entry
                    if not pos.breakeven_moved and pos.r_multiple >= 0.5:
                        pos.sl_price = pos.entry_price
                        pos.breakeven_moved = True
                        logger.info(f"[PosManager] {pos.id}: SL moved to breakeven")

                    # Trailing stop activation: at 1R profit
                    if not pos.trailing_stop_active and pos.r_multiple >= float(settings.trailing_stop_activation_r):
                        pos.trailing_stop_active = True
                        atr_val = self._get_atr(pos.symbol)
                        if pos.direction == "LONG":
                            pos.trailing_stop_price = price - Decimal(str(atr_val * 1.5))
                        else:
                            pos.trailing_stop_price = price + Decimal(str(atr_val * 1.5))
                        logger.info(
                            f"[PosManager] {pos.id}: Trailing stop activated at {pos.trailing_stop_price}"
                        )

                    # Update trailing stop price
                    if pos.trailing_stop_active:
                        self._update_trailing_stop(pos, price)

                    # Time-based stop: if > N hours with < 20% progress toward TP
                    elapsed_hours = (datetime.datetime.now(datetime.UTC) - pos.opened_at).total_seconds() / 3600
                    if elapsed_hours > settings.time_stop_hours:
                        tp_dist = abs(float(pos.tp1_price) - float(pos.entry_price))
                        if tp_dist > 0:
                            if pos.direction == "LONG":
                                progress = (float(price) - float(pos.entry_price)) / tp_dist * 100
                            else:
                                progress = (float(pos.entry_price) - float(price)) / tp_dist * 100
                            if progress < float(settings.time_stop_progress_pct):
                                await self.close_position(pos.id, "TIME_STOP", price)
                                continue

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

    def _update_trailing_stop(self, pos: Position, price: Decimal) -> None:
        """Move trailing stop in the direction of profit."""
        atr_val = self._get_atr(pos.symbol)
        trail_dist = Decimal(str(atr_val * 1.5))

        if pos.direction == "LONG":
            new_stop = price - trail_dist
            if pos.trailing_stop_price is None or new_stop > pos.trailing_stop_price:
                pos.trailing_stop_price = new_stop
                pos.sl_price = new_stop
        else:
            new_stop = price + trail_dist
            if pos.trailing_stop_price is None or new_stop < pos.trailing_stop_price:
                pos.trailing_stop_price = new_stop
                pos.sl_price = new_stop

    async def _handle_tp1(self, pos: Position, price: Decimal) -> None:
        """Partial close at TP1: close 50%, move SL to breakeven."""
        close_qty = pos.remaining_qty * Decimal("0.5")
        pos.remaining_qty -= close_qty
        pos.tp1_hit = True
        pos.sl_price = pos.entry_price  # Breakeven
        pos.breakeven_moved = True
        pos.status = "PARTIAL"

        # Close partial via executor
        if self.order_executor:
            try:
                close_side = "SELL" if pos.direction == "LONG" else "BUY"
                await self.order_executor.place_market_order(
                    symbol=pos.symbol, side=close_side, quantity=float(close_qty)
                )
            except Exception as e:
                logger.error(f"[PosManager] TP1 partial close error: {e}")

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
        pos.remaining_qty -= close_qty
        pos.tp2_hit = True

        if pos.remaining_qty <= 0:
            await self.close_position(pos.id, "TP2_HIT", price)
            return

        # Activate trailing stop on remainder
        pos.trailing_stop_active = True

        if self.order_executor:
            try:
                close_side = "SELL" if pos.direction == "LONG" else "BUY"
                await self.order_executor.place_market_order(
                    symbol=pos.symbol, side=close_side, quantity=float(close_qty)
                )
            except Exception as e:
                logger.error(f"[PosManager] TP2 partial close error: {e}")

        logger.info(
            f"[PosManager] {pos.id}: TP2 hit — closed {close_qty}, "
            f"remaining={pos.remaining_qty} with trailing stop"
        )

    def _get_atr(self, symbol: str) -> float:
        """Get latest ATR value for a symbol."""
        df = self.data_store.get_dataframe(symbol, settings.primary_timeframe)
        if df.empty:
            return 0.0
        from alphabot.utils.indicators import atr
        atr_series = atr(df["high"], df["low"], df["close"], settings.atr_period)
        if atr_series is None or atr_series.empty:
            return 0.0
        return float(atr_series.iloc[-1])

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
        )
        self.db.save_position(rec)

    async def _recover_positions(self) -> None:
        """Recover open positions from DB on restart."""
        open_recs = self.db.get_open_positions()
        for rec in open_recs:
            pos = Position(
                position_id=rec.id,
                symbol=rec.symbol,
                direction=rec.direction,
                quantity=Decimal(str(rec.quantity)),
                entry_price=Decimal(str(rec.entry_price)),
                leverage=rec.leverage,
                sl_price=Decimal(str(rec.sl_price)),
                tp1_price=Decimal(str(rec.tp1_price)) if rec.tp1_price else Decimal("0"),
                tp2_price=Decimal(str(rec.tp2_price)) if rec.tp2_price else None,
                strategy_name=rec.strategy_name,
                regime=rec.regime_at_entry,
                signal_confidence=rec.signal_confidence or 0.0,
                size_usdt=Decimal(str(rec.size_usdt)),
            )
            pos.status = rec.status
            pos.trailing_stop_active = bool(rec.trailing_stop_active)
            if rec.trailing_stop_price:
                pos.trailing_stop_price = Decimal(str(rec.trailing_stop_price))
            self._positions[rec.id] = pos
            logger.info(f"[PosManager] Recovered position: {rec.id} {rec.symbol} {rec.direction}")

        if open_recs:
            logger.info(f"[PosManager] Recovered {len(open_recs)} open positions from DB")
