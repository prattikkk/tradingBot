"""
core/position_monitor.py
Monitors open positions and manages exits (TP1/TP2/SL/trailing stop).
Runs continuously; cadence is controlled by the main loop.
"""
from __future__ import annotations

from datetime import datetime, timezone

from core.portfolio import Portfolio
from core.executor import TestnetExecutor
from core.data_fetcher import DataFetcher
from core.signal import Direction
from config import CONFIG
from utils.logger import get_logger

log = get_logger("PosMon")
risk_cfg = CONFIG.risk


class PositionMonitor:
    def __init__(
        self,
        portfolio: Portfolio,
        executor: TestnetExecutor,
        fetcher: DataFetcher,
    ):
        self.portfolio = portfolio
        self.executor  = executor
        self.fetcher   = fetcher
        self._trailing_pct = max(0.0, float(risk_cfg.trailing_stop_pct))
        self._max_hold_hours = max(0.0, float(CONFIG.trading.max_hold_hours))

    def check_all(self):
        """Called periodically to check all open positions."""
        positions = dict(self.portfolio.open_positions)   # copy to avoid mutation
        for symbol, pos in positions.items():
            try:
                self._check_position(symbol, pos)
            except Exception as e:
                log.error(f"Error monitoring {symbol}: {e}", exc_info=True)

    # ------------------------------------------------------------------ #

    def _check_position(self, symbol: str, pos: dict):
        price = self._get_decision_price(symbol)
        if price is None:
            return

        order_ids = pos.setdefault("order_ids", {})

        if self._should_force_time_exit(pos):
            self._close_time_exit(symbol, pos, price)
            return

        direction = pos["direction"]
        sl        = pos["stop_loss"]
        tp1       = pos["take_profit_1"]
        tp2       = pos["take_profit_2"]
        tp1_hit   = pos.get("tp1_hit", False)

        # --- SL check ---
        if direction == Direction.LONG.value:
            if price <= sl:
                self._close_sl(symbol, pos, price)
                return

            if not tp1_hit and price >= tp1:
                self._close_tp1(symbol, price, pos)
                if symbol not in self.portfolio.open_positions:
                    return
                try:
                    pos = self.portfolio.open_positions[symbol]
                except KeyError:
                    return
                tp1_hit = True

            if tp1_hit and price >= tp2:
                self._close_tp2(symbol, pos, price)
                return

            # Trailing stop: once TP1 hit, move SL to breakeven
            if tp1_hit:
                if pos.get("exchange_trailing_active"):
                    return
                self._update_client_trailing_stop(symbol, pos, price)

        else:  # SHORT
            if price >= sl:
                self._close_sl(symbol, pos, price)
                return

            if not tp1_hit and price <= tp1:
                self._close_tp1(symbol, price, pos)
                if symbol not in self.portfolio.open_positions:
                    return
                try:
                    pos = self.portfolio.open_positions[symbol]
                except KeyError:
                    return
                tp1_hit = True

            if tp1_hit and price <= tp2:
                self._close_tp2(symbol, pos, price)
                return

            if tp1_hit:
                if pos.get("exchange_trailing_active"):
                    return
                self._update_client_trailing_stop(symbol, pos, price)

    def _close_sl(self, symbol: str, pos: dict, price: float):
        log.warning(f"⛔ SL triggered [{symbol}] @ {price:.4f}")
        order_ids = pos.setdefault("order_ids", {})
        self._track_protection_unconfirmed(symbol, order_ids)
        # Cancel remaining TP orders on testnet
        for key in ["sl", "tp1", "tp2", "trail"]:
            oid = order_ids.get(key)
            if oid:
                self.executor.cancel_order(symbol, oid)

        if self._is_client_side_exit(pos):
            qty = float(pos.get("quantity", 0.0))
            if qty > 0:
                ok = self.executor.close_position_market(
                    symbol=symbol,
                    direction=pos.get("direction", Direction.LONG.value),
                    quantity=qty,
                )
                if not ok:
                    log.error("[%s] failed to close SL on exchange; keeping local position open", symbol)
                    return

        if not self._ensure_exchange_flattened(symbol, reason="SL_HIT"):
            log.error("[%s] exchange still open after SL handling; keeping local position open", symbol)
            return

        self.portfolio.close_position(symbol, price, reason="SL_HIT")

    def _close_tp1(self, symbol: str, price: float, pos: dict):
        if self._is_client_side_exit(pos):
            qty_exit = float(pos.get("quantity", 0.0)) * float(CONFIG.risk.partial_exit_pct)
            if qty_exit > 0:
                ok = self.executor.close_position_market(
                    symbol=symbol,
                    direction=pos.get("direction", Direction.LONG.value),
                    quantity=qty_exit,
                )
                if not ok:
                    log.error("[%s] failed to execute TP1 partial close on exchange", symbol)
                    return

        pnl = self.portfolio.update_tp1(symbol, price)
        if pnl is None:
            return

        live_pos = self.portfolio.open_positions.get(symbol)
        if not live_pos:
            return

        order_ids = live_pos.setdefault("order_ids", {})
        new_sl = self._breakeven_stop(live_pos)

        sl_id = order_ids.get("sl")
        if sl_id:
            self.executor.cancel_order(symbol, sl_id)

        new_sl_id = self.executor.place_stop_loss(
            symbol=symbol,
            direction=live_pos["direction"],
            quantity=float(live_pos.get("quantity", 0.0)),
            stop_price=float(new_sl),
        )
        if not new_sl_id:
            log.error("[%s] failed to re-arm stop after TP1; flattening position", symbol)
            self._emergency_flatten(symbol, live_pos, price, reason="PROTECTION_REARM_FAIL")
            return

        live_pos["stop_loss"] = float(new_sl)
        order_ids["sl"] = new_sl_id
        order_ids["tp1"] = None
        if live_pos["direction"] == Direction.LONG.value:
            live_pos["highest_price"] = float(price)
        else:
            live_pos["lowest_price"] = float(price)

        self._activate_exchange_trailing_stop(symbol, live_pos, float(price))

        self.portfolio._save()
        log.info(f"🎯 TP1 [{symbol}] @ {price:.4f} | pnl=${pnl:+.2f} | SL re-armed @ {new_sl:.4f}")

    def _close_tp2(self, symbol: str, pos: dict, price: float):
        log.info(f"🏁 TP2 hit [{symbol}] @ {price:.4f}")
        order_ids = pos.setdefault("order_ids", {})
        self._track_protection_unconfirmed(symbol, order_ids)
        # Cancel protection orders on testnet
        for key in ["sl", "trail"]:
            oid = order_ids.get(key)
            if oid:
                self.executor.cancel_order(symbol, oid)

        if self._is_client_side_exit(pos):
            qty = float(pos.get("quantity", 0.0))
            if qty > 0:
                ok = self.executor.close_position_market(
                    symbol=symbol,
                    direction=pos.get("direction", Direction.LONG.value),
                    quantity=qty,
                )
                if not ok:
                    log.error("[%s] failed to execute TP2 close on exchange", symbol)
                    return

        if not self._ensure_exchange_flattened(symbol, reason="TP2_HIT"):
            log.error("[%s] exchange still open after TP2 handling; keeping local position open", symbol)
            return

        self.portfolio.close_position(symbol, price, reason="TP2_HIT")

    @staticmethod
    def _is_client_side_exit(pos: dict) -> bool:
        order_ids = pos.get("order_ids")
        if not isinstance(order_ids, dict):
            return True
        # Only confirmed TP orders can be trusted for exchange-managed exits.
        tp1 = order_ids.get("tp1")
        tp2 = order_ids.get("tp2")
        return not (
            PositionMonitor._is_confirmed_order_ref(tp1)
            or PositionMonitor._is_confirmed_order_ref(tp2)
        )

    @staticmethod
    def _is_confirmed_order_ref(order_ref: object) -> bool:
        if order_ref in (None, "", "DRY_RUN"):
            return False
        if isinstance(order_ref, str):
            return not order_ref.startswith("pending:")
        return True

    def _track_protection_unconfirmed(self, symbol: str, order_ids: dict) -> None:
        refs = [order_ids.get("sl"), order_ids.get("tp1"), order_ids.get("tp2"), order_ids.get("trail")]
        has_pending = any(isinstance(ref, str) and ref.startswith("pending:") for ref in refs)
        if not has_pending:
            return

        self._increment_metric("protection_unconfirmed_mode")
        log.warning(
            "[%s] protection_unconfirmed_mode active; pending protective IDs detected",
            symbol,
        )

    def _activate_exchange_trailing_stop(self, symbol: str, pos: dict, price: float) -> None:
        if self._trailing_pct <= 0:
            return
        if pos.get("exchange_trailing_active"):
            return
        if not self.executor.has_credentials:
            return

        qty = float(pos.get("quantity", 0.0))
        if qty <= 0:
            return

        order_ids = pos.setdefault("order_ids", {})
        current_sl_id = order_ids.get("sl")
        callback_rate_pct = self._trailing_pct * 100.0

        trail_id = self.executor.place_trailing_stop(
            symbol=symbol,
            direction=pos.get("direction", Direction.LONG.value),
            quantity=qty,
            callback_rate_pct=callback_rate_pct,
            activation_price=price,
        )
        if not trail_id:
            log.warning("[%s] exchange trailing stop unavailable; using client-side trailing", symbol)
            return

        if current_sl_id and current_sl_id != "DRY_RUN":
            self.executor.cancel_order(symbol, current_sl_id)

        order_ids["trail"] = trail_id
        order_ids["sl"] = None
        pos["exchange_trailing_active"] = True
        self.portfolio._save()
        log.info(
            "[%s] exchange trailing stop activated | callback=%.2f%%",
            symbol,
            callback_rate_pct,
        )

    def _update_client_trailing_stop(self, symbol: str, pos: dict, price: float):
        if self._trailing_pct <= 0:
            return

        qty = float(pos.get("quantity", 0.0))
        if qty <= 0:
            return

        direction = pos["direction"]
        current_sl = float(pos["stop_loss"])
        entry = float(pos["entry_price"])

        if direction == Direction.LONG.value:
            highest = max(float(pos.get("highest_price", price)), float(price))
            pos["highest_price"] = highest
            candidate = max(current_sl, highest * (1 - self._trailing_pct), entry)
            improved = candidate > current_sl * (1 + max(self._trailing_pct * 0.1, 0.0001))
        else:
            lowest = min(float(pos.get("lowest_price", price)), float(price))
            pos["lowest_price"] = lowest
            candidate = min(current_sl, lowest * (1 + self._trailing_pct), entry)
            improved = candidate < current_sl * (1 - max(self._trailing_pct * 0.1, 0.0001))

        if not improved:
            return

        order_ids = pos.setdefault("order_ids", {})
        old_sl = order_ids.get("sl")
        if old_sl:
            self.executor.cancel_order(symbol, old_sl)

        new_sl_id = self.executor.place_stop_loss(
            symbol=symbol,
            direction=direction,
            quantity=qty,
            stop_price=float(candidate),
        )
        if not new_sl_id:
            log.error("[%s] trailing update failed; flattening position for safety", symbol)
            self._emergency_flatten(symbol, pos, price, reason="TRAIL_GUARD_FAIL")
            return

        pos["stop_loss"] = float(candidate)
        order_ids["sl"] = new_sl_id
        self.portfolio._save()
        log.info("[%s] trailing SL updated to %.4f", symbol, candidate)

    def _emergency_flatten(self, symbol: str, pos: dict, price: float, reason: str):
        order_ids = pos.get("order_ids", {})
        self._track_protection_unconfirmed(symbol, order_ids)
        for key in ["sl", "tp1", "tp2", "trail"]:
            oid = order_ids.get(key)
            if oid:
                self.executor.cancel_order(symbol, oid)

        ok = self.executor.close_position_market(
            symbol=symbol,
            direction=pos.get("direction", Direction.LONG.value),
            quantity=float(pos.get("quantity", 0.0)),
        )
        if not ok:
            log.error("[%s] %s: emergency market close failed", symbol, reason)
            return

        if not self._ensure_exchange_flattened(symbol, reason=reason):
            log.error("[%s] %s: exchange still open after emergency flatten", symbol, reason)
            return

        self.portfolio.close_position(symbol, price, reason=reason)

    def _ensure_exchange_flattened(self, symbol: str, reason: str) -> bool:
        if not getattr(self.executor, "has_credentials", False):
            return True

        signed_qty = self._exchange_signed_quantity(symbol)
        if signed_qty == 0.0:
            return True

        qty = abs(signed_qty)
        direction = Direction.LONG.value if signed_qty > 0 else Direction.SHORT.value

        self._increment_metric("exchange_local_drift_detected")
        log.error(
            "[%s] %s: exchange quantity %.6f remained after local close intent; forcing corrective close",
            symbol,
            reason,
            qty,
        )

        corrected = self.executor.close_position_market(symbol, direction, qty)
        if not corrected:
            log.error(
                "[%s] %s: corrective exchange close failed for qty %.6f",
                symbol,
                reason,
                qty,
            )
            return False

        self._increment_metric("exchange_local_drift_corrected")

        residual_signed_qty = self._exchange_signed_quantity(symbol)
        if residual_signed_qty == 0.0:
            return True

        log.error(
            "[%s] %s: exchange quantity %.6f still open after corrective close",
            symbol,
            reason,
            abs(residual_signed_qty),
        )
        return False

    def _exchange_signed_quantity(self, symbol: str) -> float:
        try:
            positions = self.executor.get_open_positions([symbol])
        except Exception as exc:
            log.warning("[%s] failed to verify exchange flatten state: %s", symbol, exc)
            return 0.0

        remote = positions.get(symbol)
        if not remote:
            return 0.0
        try:
            return float(remote.get("quantity", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _increment_metric(self, name: str) -> None:
        increment = getattr(self.portfolio, "increment_metric", None)
        if callable(increment):
            increment(name)

    def _close_time_exit(self, symbol: str, pos: dict, price: float):
        log.warning("[%s] max hold time reached; forcing market exit", symbol)
        self._emergency_flatten(symbol, pos, price, reason="TIME_EXIT")

    def _get_decision_price(self, symbol: str) -> float | None:
        price: float | None = None
        source = "legacy"

        if hasattr(self.fetcher, "get_current_price_with_meta"):
            try:
                price, source = self.fetcher.get_current_price_with_meta(symbol)
            except Exception:
                price = self.fetcher.get_current_price(symbol)
                source = "legacy"
        else:
            price = self.fetcher.get_current_price(symbol)

        if price is None:
            return None

        # REST fallback snapshots can be stale during fast moves.
        # Confirm once before making TP/SL decisions.
        if source == "rest":
            confirmed = self.fetcher.get_current_price(symbol)
            if confirmed is None:
                log.debug("[%s] REST fallback price could not be confirmed; skipping check", symbol)
                return None
            return float(confirmed)

        return float(price)

    def _should_force_time_exit(self, pos: dict) -> bool:
        if self._max_hold_hours <= 0:
            return False

        opened_at = self._parse_open_time(pos)
        if opened_at is None:
            return False

        age_hours = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600.0
        return age_hours >= self._max_hold_hours

    @staticmethod
    def _parse_open_time(pos: dict) -> datetime | None:
        raw = pos.get("open_time")
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _breakeven_stop(pos: dict) -> float:
        entry = float(pos["entry_price"])
        tp1 = float(pos["take_profit_1"])
        direction = pos["direction"]
        if direction == Direction.LONG.value:
            return entry + 0.2 * (tp1 - entry)
        return entry - 0.2 * (entry - tp1)
