"""
main.py
Full paper-trading entrypoint for signal scan + position management.

Examples:
    # Single scan cycle (safe smoke test)
    python main.py --once --dry-run

    # Single cycle with real TESTNET order placement
    python main.py --once --live

    # Continuous loop every 5 minutes
    python main.py --analysis-interval 300
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import signal as os_signal
import time
from datetime import datetime
from typing import Iterable

from config import CONFIG
from core.ai_sentiment import AISentimentEngine
from core.control_plane import drain_commands, get_control_state
from core.exchange_factory import create_data_fetcher, create_executor
from core.portfolio import Portfolio
from core.position_monitor import PositionMonitor
from core.risk_manager import RiskManager
from core.signal import Direction
from strategies.adx_trend import ADXTrendStrategy
from strategies.breakout_momentum import BreakoutMomentumStrategy
from strategies.ema_adx_volume import EMAAdxVolumeStrategy
from strategies.ensemble import EnsembleStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.supertrend_rsi import SuperTrendRSIStrategy
from utils.logger import get_logger
from utils.notifier import notify_event, notify_signal, notify_stats, notify_trade_open

log = get_logger("Main")


STRATEGY_MAP = {
    "adx_trend": ADXTrendStrategy,
    "ensemble": EnsembleStrategy,
    "supertrend_rsi": SuperTrendRSIStrategy,
    "ema_adx_volume": EMAAdxVolumeStrategy,
    "breakout_momentum": BreakoutMomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
}


def _parse_symbols(raw: str) -> list[str]:
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    # Keep deterministic order while removing duplicates.
    return list(dict.fromkeys(symbols))


def _resolve_dry_run(cli_dry_run: bool, cli_live: bool) -> bool:
    if cli_live:
        return False
    if cli_dry_run:
        return True
    return os.getenv("DRY_RUN", "true").lower() == "true"


def _has_testnet_credentials() -> bool:
    key = os.getenv("BINANCE_TESTNET_API_KEY") or os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_TESTNET_SECRET") or os.getenv("BINANCE_API_SECRET")
    return bool(key and secret)


class TradingBot:
    def __init__(self, symbols: Iterable[str], strategy_name: str, dry_run: bool):
        if strategy_name not in STRATEGY_MAP:
            raise ValueError(f"Unsupported strategy: {strategy_name}")

        self.symbols = list(symbols)
        self.strategy_name = strategy_name
        self.strategy_cls = STRATEGY_MAP[strategy_name]
        self.strategy = self.strategy_cls()
        self.dry_run = dry_run
        self.exchange = CONFIG.exchange.name
        self._symbol_order = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        self._runtime_paused = False
        self._runtime_overrides: dict[str, float | int] = {}
        self._reconciliation_enabled = bool(getattr(CONFIG.trading, "reconciliation_enabled", True))
        self._reconciliation_interval_seconds = max(
            30.0,
            float(getattr(CONFIG.trading, "reconciliation_interval_seconds", 300.0)),
        )
        self._last_reconciliation_ts = 0.0
        self._no_signal_streaks: dict[str, int] = {}
        self._last_evaluated_signal_bar_close: dict[str, float] = {}

        self.fetcher = create_data_fetcher(self.exchange)
        self.executor = create_executor(self.exchange)
        self.portfolio = Portfolio()
        self.risk = RiskManager(self.portfolio)
        self.monitor = PositionMonitor(self.portfolio, self.executor, self.fetcher)
        self.ai_sentiment = AISentimentEngine()

        try:
            self.fetcher.start_price_stream(self.symbols)
            self._refresh_runtime_state()
            self._sync_positions_from_exchange()
            self._sync_balance_from_exchange()
            self._last_reconciliation_ts = time.time()
        except Exception as e:
            log.error("Startup initialization failed: %s", e, exc_info=True)
            raise

        log.info(
            "Bot initialized | exchange=%s | strategy=%s | dry_run=%s | symbols=%s",
            self.exchange,
            self.strategy_name,
            self.dry_run,
            ",".join(self.symbols),
        )

    def _sync_positions_from_exchange(self) -> None:
        """Reconcile local portfolio state with exchange positions at startup."""
        self._reconcile_positions_from_exchange(startup=True)

    def _run_periodic_reconciliation_if_due(self) -> None:
        if not self._reconciliation_enabled:
            return

        now = time.time()
        if now - self._last_reconciliation_ts < self._reconciliation_interval_seconds:
            return

        self._reconcile_positions_from_exchange(startup=False)
        self._last_reconciliation_ts = now

    def _reconcile_positions_from_exchange(self, startup: bool) -> None:
        phase = "Startup sync" if startup else "Periodic sync"
        close_reason = "SYNC_CLOSED_ON_EXCHANGE" if startup else "PERIODIC_SYNC_CLOSED_ON_EXCHANGE"

        exchange_positions = self.executor.get_open_positions(self.symbols)

        local_symbols = set(self.portfolio.open_positions.keys())
        exchange_symbols = set(exchange_positions.keys())

        changed = False

        # Local-only positions are stale after interruption or manual intervention.
        for symbol in sorted(local_symbols - exchange_symbols):
            exit_price = self.fetcher.get_current_price(symbol)
            if exit_price is None:
                exit_price = float(self.portfolio.open_positions[symbol].get("entry_price", 0))
            self.portfolio.close_position(symbol, float(exit_price), reason=close_reason)
            changed = True
            self._increment_metric("exchange_local_drift_detected")
            self._increment_metric("exchange_local_drift_corrected")
            log.warning("%s: closed stale local position %s", phase, symbol)

        # Exchange-only positions are force-closed by policy, or imported when disabled.
        force_close_untracked = bool(
            getattr(CONFIG.trading, "force_close_untracked_exchange_positions", True)
        )
        for symbol in sorted(exchange_symbols - local_symbols):
            self._increment_metric("exchange_local_drift_detected")
            if force_close_untracked:
                if self._force_close_untracked_exchange_position(symbol, exchange_positions[symbol], phase):
                    self._increment_metric("exchange_local_drift_corrected")
                continue

            if self._import_exchange_position(exchange_positions[symbol]):
                self._increment_metric("exchange_local_drift_corrected")
                changed = True

        # Shared positions are aligned on quantity/direction/entry when drift exists.
        for symbol in sorted(local_symbols & exchange_symbols):
            if self._align_local_position(symbol, exchange_positions[symbol]):
                self._increment_metric("exchange_local_drift_detected")
                self._increment_metric("exchange_local_drift_corrected")
                changed = True

        if changed:
            self.portfolio._save()

        log.info(
            "%s complete | local_open=%s | exchange_open=%s",
            phase,
            len(self.portfolio.open_positions),
            len(exchange_positions),
        )

    def _force_close_untracked_exchange_position(self, symbol: str, remote: dict, phase: str) -> bool:
        try:
            signed_qty = float(remote.get("quantity", 0.0) or 0.0)
        except Exception:
            signed_qty = 0.0

        qty = abs(signed_qty)
        if qty <= 0:
            return True

        direction = "LONG" if signed_qty > 0 else "SHORT"

        if self.dry_run:
            log.warning(
                "%s: dry-run mode; cannot force close untracked exchange position %s %s qty=%.6f",
                phase,
                symbol,
                direction,
                qty,
            )
            return False

        self.executor.cancel_all_open_orders(symbol)
        closed = self.executor.close_position_market(symbol, direction, qty)
        if not closed:
            log.error(
                "%s: failed to force close untracked exchange position %s %s qty=%.6f",
                phase,
                symbol,
                direction,
                qty,
            )
            return False

        residual = self.executor.get_open_positions([symbol])
        if symbol in residual:
            log.error(
                "%s: untracked exchange position %s still open after corrective close",
                phase,
                symbol,
            )
            return False

        log.critical(
            "%s: force-closed untracked exchange position %s %s qty=%.6f",
            phase,
            symbol,
            direction,
            qty,
        )
        return True

    def _increment_metric(self, name: str) -> None:
        increment = getattr(self.portfolio, "increment_metric", None)
        if callable(increment):
            increment(name)

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(value, (int, float)):
            return value != 0
        return False

    @staticmethod
    def _safe_positive_price(value) -> float | None:
        parsed = TradingBot._safe_float(value, default=0.0)
        return parsed if parsed > 0 else None

    def _derive_imported_protection(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
    ) -> dict[str, object]:
        sl_pct = float(os.getenv("STOP_LOSS_PCT", "0.03"))
        tp_pct = float(os.getenv("TAKE_PROFIT_PCT", "0.06"))

        if not (0 < sl_pct < 0.5):
            log.warning(
                "Startup sync [%s]: STOP_LOSS_PCT=%.4f out of range (0, 0.5); using 0.03",
                symbol,
                sl_pct,
            )
            sl_pct = 0.03
        if not (0 < tp_pct < 1.0):
            log.warning(
                "Startup sync [%s]: TAKE_PROFIT_PCT=%.4f out of range (0, 1.0); using 0.06",
                symbol,
                tp_pct,
            )
            tp_pct = 0.06

        if direction == "LONG":
            default_stop = entry_price * (1 - sl_pct)
            default_tp1 = entry_price * (1 + tp_pct * 0.5)
            default_tp2 = entry_price * (1 + tp_pct)
        else:
            default_stop = entry_price * (1 + sl_pct)
            default_tp1 = entry_price * (1 - tp_pct * 0.5)
            default_tp2 = entry_price * (1 - tp_pct)

        fetch_open_orders = getattr(self.executor, "get_open_orders", None)
        raw_orders = fetch_open_orders(symbol) if callable(fetch_open_orders) else []
        open_orders = [o for o in (raw_orders or []) if isinstance(o, dict)]

        close_side = "SELL" if direction == "LONG" else "BUY"
        stop_orders: list[tuple[float, dict]] = []
        tp_orders: list[tuple[float, dict]] = []

        for order in open_orders:
            order_side = str(order.get("side", "")).upper()
            if order_side and order_side != close_side:
                continue

            reduce_only = self._safe_bool(order.get("reduceOnly")) or self._safe_bool(order.get("closePosition"))
            if not reduce_only and order_side != close_side:
                continue

            order_type = str(order.get("type", "")).upper()
            stop_price = self._safe_positive_price(order.get("stopPrice"))
            limit_price = self._safe_positive_price(order.get("price"))
            activation_price = self._safe_positive_price(order.get("activationPrice"))

            if order_type in {"STOP", "STOP_MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT", "TRAILING_STOP_MARKET"}:
                px = stop_price or activation_price or limit_price
                if px is not None:
                    stop_orders.append((px, order))
                continue

            if order_type in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET", "TAKE_PROFIT_LIMIT"}:
                px = stop_price or limit_price
                if px is not None:
                    tp_orders.append((px, order))

        def _order_id(raw: dict) -> int | str | None:
            oid = raw.get("orderId")
            if oid is None:
                return None
            try:
                return int(oid)
            except Exception:
                return str(oid)

        selected_stop = None
        selected_stop_order = None
        if stop_orders:
            if direction == "LONG":
                valid = [(px, order) for px, order in stop_orders if px < entry_price]
                if valid:
                    selected_stop, selected_stop_order = max(valid, key=lambda item: item[0])
                else:
                    selected_stop, selected_stop_order = min(stop_orders, key=lambda item: abs(item[0] - entry_price))
            else:
                valid = [(px, order) for px, order in stop_orders if px > entry_price]
                if valid:
                    selected_stop, selected_stop_order = min(valid, key=lambda item: item[0])
                else:
                    selected_stop, selected_stop_order = min(stop_orders, key=lambda item: abs(item[0] - entry_price))

        selected_tps: list[tuple[float, dict]] = []
        if tp_orders:
            if direction == "LONG":
                valid_tps = [(px, order) for px, order in tp_orders if px > entry_price]
                valid_tps.sort(key=lambda item: item[0])
            else:
                valid_tps = [(px, order) for px, order in tp_orders if px < entry_price]
                valid_tps.sort(key=lambda item: item[0], reverse=True)

            if not valid_tps:
                valid_tps = sorted(tp_orders, key=lambda item: abs(item[0] - entry_price))

            selected_tps = valid_tps[:2]

        stop_loss = selected_stop if selected_stop is not None else default_stop

        if selected_tps:
            take_profit_1 = selected_tps[0][0]
            take_profit_2 = selected_tps[1][0] if len(selected_tps) > 1 else default_tp2
        else:
            take_profit_1 = default_tp1
            take_profit_2 = default_tp2

        stop_order_id = _order_id(selected_stop_order) if selected_stop_order else None
        tp1_order_id = _order_id(selected_tps[0][1]) if selected_tps else None
        tp2_order_id = _order_id(selected_tps[1][1]) if len(selected_tps) > 1 else None

        has_exchange_stop = stop_order_id is not None
        has_exchange_tp = tp1_order_id is not None or tp2_order_id is not None
        if has_exchange_stop and has_exchange_tp:
            protection_source = "exchange"
        elif has_exchange_stop or has_exchange_tp:
            protection_source = "mixed"
        else:
            protection_source = "synthetic"

        return {
            "stop_loss": float(stop_loss),
            "take_profit_1": float(take_profit_1),
            "take_profit_2": float(take_profit_2),
            "sl_order_id": stop_order_id,
            "tp1_order_id": tp1_order_id,
            "tp2_order_id": tp2_order_id,
            "protection_source": protection_source,
        }

    def _import_exchange_position(self, pos: dict) -> bool:
        symbol = pos["symbol"]
        quantity = abs(float(pos.get("quantity", 0)))
        if quantity <= 0:
            return False

        direction = "LONG" if float(pos.get("quantity", 0)) > 0 else "SHORT"

        entry_price = float(pos.get("entry_price", 0) or 0)
        mark_price = float(pos.get("mark_price", 0) or 0)
        if entry_price <= 0:
            entry_price = mark_price
        if entry_price <= 0:
            live_price = self.fetcher.get_current_price(symbol)
            if live_price is not None:
                entry_price = float(live_price)
        if entry_price <= 0:
            log.warning("Startup sync: could not infer entry price for %s; skipping import", symbol)
            return False

        protection = self._derive_imported_protection(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
        )

        stop_loss = float(protection["stop_loss"])
        take_profit_1 = float(protection["take_profit_1"])
        take_profit_2 = float(protection["take_profit_2"])
        protection_source = str(protection["protection_source"])

        leverage = max(1, int(float(pos.get("leverage", 1) or 1)))
        notional = abs(float(pos.get("notional", 0) or 0))
        if notional <= 0:
            notional = quantity * entry_price

        risk_usdt = abs(entry_price - stop_loss) * quantity

        self.portfolio.open_positions[symbol] = {
            "id": f"{symbol}_sync_{int(time.time())}",
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit_1": take_profit_1,
            "take_profit_2": take_profit_2,
            "quantity": quantity,
            "notional": notional,
            "risk_usdt": risk_usdt,
            "strategy": "startup_sync",
            "confidence": 1.0,
            "open_time": datetime.utcnow().isoformat(),
            "close_time": None,
            "exit_price": None,
            "pnl": 0.0,
            "status": "SYNCED_OPEN",
            "tp1_hit": False,
            "leverage": leverage,
            "protection_source": protection_source,
            "unprotected_sync": protection_source == "synthetic",
            "order_ids": {
                "entry": "SYNC_IMPORT",
                "sl": protection["sl_order_id"],
                "tp1": protection["tp1_order_id"],
                "tp2": protection["tp2_order_id"],
            },
        }

        if protection_source != "exchange":
            self._increment_metric("import_unprotected_sync")
            log.warning(
                "Startup sync: %s imported with %s protection source",
                symbol,
                protection_source,
            )

        log.warning(
            "Startup sync: imported exchange position %s %s qty=%s (protection=%s)",
            symbol,
            direction,
            quantity,
            protection_source,
        )
        return True

    def _align_local_position(self, symbol: str, remote: dict) -> bool:
        local = self.portfolio.open_positions.get(symbol)
        if not local:
            return False

        changed = False

        remote_qty = abs(float(remote.get("quantity", 0) or 0))
        local_qty = abs(float(local.get("quantity", 0) or 0))
        if remote_qty > 0 and abs(local_qty - remote_qty) / remote_qty > 0.001:
            local["quantity"] = remote_qty
            changed = True

        remote_dir = "LONG" if float(remote.get("quantity", 0) or 0) > 0 else "SHORT"
        if local.get("direction") != remote_dir:
            local["direction"] = remote_dir
            changed = True

        remote_entry = float(remote.get("entry_price", 0) or 0)
        if remote_entry <= 0:
            remote_entry = float(remote.get("mark_price", 0) or 0)
        local_entry = float(local.get("entry_price", 0) or 0)
        if remote_entry > 0 and (local_entry <= 0 or abs(local_entry - remote_entry) / remote_entry > 0.005):
            local["entry_price"] = remote_entry
            changed = True

        remote_notional = abs(float(remote.get("notional", 0) or 0))
        if remote_notional > 0:
            local_notional = abs(float(local.get("notional", 0) or 0))
            if local_notional <= 0 or abs(local_notional - remote_notional) / remote_notional > 0.01:
                local["notional"] = remote_notional
                changed = True

        if changed:
            self._refresh_local_position_risk(local)
            log.warning("Position sync: aligned local position %s with exchange state", symbol)
        return changed

    @staticmethod
    def _refresh_local_position_risk(local: dict) -> None:
        try:
            entry = float(local.get("entry_price", 0.0) or 0.0)
            stop = float(local.get("stop_loss", 0.0) or 0.0)
            qty = abs(float(local.get("quantity", 0.0) or 0.0))
        except Exception:
            return

        if entry <= 0 or stop <= 0 or qty <= 0:
            return

        local["risk_usdt"] = abs(entry - stop) * qty

    def _sync_balance_from_exchange(self) -> None:
        if self.dry_run:
            return

        quote_asset = str(CONFIG.binance.quote_asset or "USDT").upper()
        balance_info = self.executor.get_quote_asset_balance(quote_asset)
        if not balance_info:
            log.warning("Startup balance sync: exchange balance unavailable")
            return

        wallet_balance = float(balance_info.get("wallet_balance", 0.0))
        available_balance = float(balance_info.get("available_balance", wallet_balance))
        # Available balance excludes margin locked in open futures positions.
        # Use wallet balance as the strategy equity baseline.
        sync_balance = wallet_balance if wallet_balance > 0 else available_balance
        if sync_balance <= 0:
            log.warning(
                "Startup balance sync: non-positive balance for %s (wallet=%.4f available=%.4f)",
                quote_asset,
                wallet_balance,
                available_balance,
            )
            return

        reset_total = len(self.portfolio.closed_trades) == 0
        self.portfolio.sync_exchange_balance(sync_balance, reset_total_capital=reset_total)
        log.info(
            "Startup balance sync complete | asset=%s wallet=%.2f available=%.2f",
            quote_asset,
            wallet_balance,
            available_balance,
        )

    def _refresh_runtime_state(self) -> None:
        state = get_control_state()
        self._runtime_paused = bool(state.get("paused", False))

        overrides = state.get("overrides", {})
        parsed: dict[str, float | int] = {}
        if isinstance(overrides, dict):
            min_conf = overrides.get("min_confidence")
            if min_conf is not None:
                try:
                    parsed["min_confidence"] = max(0.0, min(1.0, float(min_conf)))
                except Exception:
                    pass

            corr_threshold = overrides.get("correlation_threshold")
            if corr_threshold is not None:
                try:
                    parsed["correlation_threshold"] = max(0.0, min(1.0, float(corr_threshold)))
                except Exception:
                    pass

            max_corr = overrides.get("max_correlated_positions")
            if max_corr is not None:
                try:
                    parsed["max_correlated_positions"] = max(1, int(max_corr))
                except Exception:
                    pass

        self._runtime_overrides = parsed

    def _effective_min_confidence(self) -> float:
        override = self._runtime_overrides.get("min_confidence")
        if override is None:
            return float(CONFIG.strategy.min_confidence)
        return float(override)

    def _effective_correlation_threshold(self) -> float:
        override = self._runtime_overrides.get("correlation_threshold")
        if override is None:
            return float(CONFIG.risk.correlation_threshold)
        return float(override)

    def _effective_max_correlated_positions(self) -> int:
        override = self._runtime_overrides.get("max_correlated_positions")
        if override is None:
            return max(1, int(CONFIG.risk.max_correlated_positions))
        return max(1, int(override))

    def _process_runtime_commands(self) -> None:
        commands = drain_commands()
        if not commands:
            return

        for command in commands:
            action = str(command.get("action", "")).strip().lower()
            payload = command.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}

            if action == "close_symbol":
                symbol = str(payload.get("symbol", "")).strip().upper()
                if symbol:
                    self._force_close_symbol(symbol, reason="DASHBOARD_CLOSE")
                continue

            if action == "close_all":
                symbols = sorted(self.portfolio.open_positions.keys())
                for symbol in symbols:
                    self._force_close_symbol(symbol, reason="DASHBOARD_CLOSE_ALL")
                continue

            log.warning("Ignoring unsupported runtime command: %s", action)

    def _force_close_symbol(self, symbol: str, reason: str) -> None:
        pos = self.portfolio.open_positions.get(symbol)
        if not pos:
            return

        self._cancel_protective_orders(symbol, pos)

        qty = float(pos.get("quantity", 0.0))
        if qty > 0 and not self.dry_run:
            self.executor.close_position_market(symbol, pos.get("direction", "LONG"), qty)

        exit_price = self.fetcher.get_current_price(symbol)
        if exit_price is None:
            exit_price = float(pos.get("entry_price", 0.0))
        self.portfolio.close_position(symbol, float(exit_price), reason=reason)
        notify_event("Dashboard Action", f"{reason}: closed {symbol}")

    def _cancel_protective_orders(self, symbol: str, pos: dict) -> None:
        order_ids = pos.get("order_ids", {})
        for key in ["sl", "tp1", "tp2", "trail"]:
            oid = order_ids.get(key)
            if oid and oid != "DRY_RUN":
                self.executor.cancel_order(symbol, oid)

    def run_cycle(self) -> None:
        self.run_position_management()
        self.run_signal_scan()

    def run_position_management(self) -> None:
        """Process local position stop/targets check and runtime/UI dashboard commands."""
        self.monitor.check_all()
        self._process_runtime_commands()
        self._refresh_runtime_state()
        self._run_periodic_reconciliation_if_due()

    def run_signal_scan(self) -> None:
        """Scan eligible symbols for new trading signals, size positions, and execute entries."""
        primary_tf = CONFIG.strategy.primary_tf
        htf_1 = CONFIG.strategy.htf_1
        htf_2 = CONFIG.strategy.htf_2
        min_volume_24h = CONFIG.trading.min_volume_24h

        cycle_stats = self.portfolio.stats()
        daily_loss_limit_pct = CONFIG.risk.max_daily_loss_pct * 100
        halt_new_entries = cycle_stats.get("return_pct", 0.0) <= -daily_loss_limit_pct
        if halt_new_entries:
            log.critical(
                "Daily loss guard active | return=%.2f%% <= -%.2f%% | pausing new entries",
                cycle_stats.get("return_pct", 0.0),
                daily_loss_limit_pct,
            )

        if self._runtime_paused:
            halt_new_entries = True
            log.warning("Runtime pause is active; skipping new entries this cycle")

        eligible_symbols: list[str] = []
        cycle_rejections: dict[str, str] = {}
        opened_symbols: set[str] = set()

        for symbol in self.symbols:
            if halt_new_entries:
                self._record_cycle_rejection(cycle_rejections, symbol, "global_halt_new_entries")
                continue

            if symbol in self.portfolio.open_positions:
                log.debug("[%s] position already open, skipping new entry", symbol)
                self._record_cycle_rejection(cycle_rejections, symbol, "already_open_position")
                continue

            if min_volume_24h > 0:
                quote_volume = self.fetcher.get_24h_quote_volume(symbol)
                if quote_volume is None:
                    log.warning("[%s] 24h quote volume unavailable, skipping", symbol)
                    self._record_cycle_rejection(cycle_rejections, symbol, "volume_unavailable")
                    continue
                if quote_volume < min_volume_24h:
                    log.info(
                        "[%s] skipped: 24h volume %.0f < min %.0f",
                        symbol,
                        quote_volume,
                        min_volume_24h,
                    )
                    self._record_cycle_rejection(
                        cycle_rejections,
                        symbol,
                        f"volume_below_min({quote_volume:.0f}<{min_volume_24h:.0f})",
                    )
                    continue

            eligible_symbols.append(symbol)

        multi_tf_by_symbol = self.fetcher.get_multi_tf_bulk(eligible_symbols) if eligible_symbols else {}
        min_confidence = self._effective_min_confidence()

        signal_candidates: list[dict] = []
        if eligible_symbols:
            max_workers = min(
                len(eligible_symbols),
                max(1, int(CONFIG.api.max_concurrent_requests)),
            )
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        self._generate_signal_candidate,
                        symbol,
                        multi_tf_by_symbol.get(symbol, {}),
                        primary_tf,
                        htf_1,
                        htf_2,
                        min_confidence,
                    ): symbol
                    for symbol in eligible_symbols
                }

                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        candidate = future.result()
                    except Exception as e:
                        log.error("[%s] signal generation failed: %s", symbol, e)
                        self._record_cycle_rejection(cycle_rejections, symbol, "signal_generation_error")
                        continue
                    if candidate is None:
                        self._record_cycle_rejection(cycle_rejections, symbol, "signal_generation_empty")
                        continue

                    rejection_reason = candidate.get("rejection_reason")
                    if rejection_reason:
                        self._record_cycle_rejection(cycle_rejections, symbol, str(rejection_reason))
                        continue

                    signal_candidates.append(candidate)

        signal_candidates.sort(key=lambda item: self._symbol_order.get(item["symbol"], 10**9))

        corr_matrix: dict[tuple[str, str], float] = {}
        if bool(CONFIG.risk.correlation_management_enabled) and signal_candidates:
            corr_symbols = [c["symbol"] for c in signal_candidates]
            corr_symbols.extend(self.portfolio.open_positions.keys())
            corr_matrix = self.fetcher.get_close_correlation_matrix(
                corr_symbols,
                primary_tf,
                int(CONFIG.risk.correlation_lookback),
            )

        for candidate in signal_candidates:
            symbol = candidate["symbol"]
            signal = candidate["signal"]
            df = candidate["df"]

            funding_threshold = CONFIG.trading.max_unfavorable_funding_rate
            if funding_threshold > 0:
                funding_rate = self.fetcher.get_funding_rate(symbol)
                if funding_rate is not None:
                    if signal.direction == Direction.LONG and funding_rate > funding_threshold:
                        log.info(
                            "[%s] skipped: funding %.5f too high for LONG (threshold %.5f)",
                            symbol,
                            funding_rate,
                            funding_threshold,
                        )
                        self._record_cycle_rejection(
                            cycle_rejections,
                            symbol,
                            f"funding_unfavorable_long({funding_rate:.5f}>{funding_threshold:.5f})",
                        )
                        continue
                    if signal.direction == Direction.SHORT and funding_rate < -funding_threshold:
                        log.info(
                            "[%s] skipped: funding %.5f too low for SHORT (threshold %.5f)",
                            symbol,
                            funding_rate,
                            funding_threshold,
                        )
                        self._record_cycle_rejection(
                            cycle_rejections,
                            symbol,
                            f"funding_unfavorable_short({funding_rate:.5f}<-{funding_threshold:.5f})",
                        )
                        continue

            correlation_reason = self._is_correlation_blocked(symbol, primary_tf, corr_matrix)
            if correlation_reason:
                self._record_cycle_rejection(cycle_rejections, symbol, correlation_reason)
                continue

            if self.ai_sentiment.enabled:
                context = self._ai_context(df)
                regime = str(signal.extra.get("regime", "UNKNOWN"))
                ai_adj = self.ai_sentiment.confidence_adjustment(symbol, signal, regime, context)
                if ai_adj != 0:
                    signal.confidence = max(0.0, min(1.0, signal.confidence + ai_adj))
                    signal.reason = f"{signal.reason} | ai_adj={ai_adj:+.2f}"
                    if not self._signal_is_tradeable(signal, min_confidence):
                        log.info("[%s] AI sentiment reduced confidence below threshold", symbol)
                        rejection_reason = self._signal_rejection_reason(signal, min_confidence)
                        if rejection_reason is None:
                            rejection_reason = "ai_adjustment_rejected"
                        self._record_cycle_rejection(cycle_rejections, symbol, rejection_reason)
                        continue

            log.info(
                "[%s] valid signal %s | conf=%.2f | rr=%.2f",
                symbol,
                signal.direction.value,
                signal.confidence,
                signal.risk_reward,
            )
            notify_signal(signal)

            exchange_info = self.fetcher.get_exchange_info(symbol)
            if not exchange_info:
                log.warning("[%s] exchange info unavailable, skipping", symbol)
                self._record_cycle_rejection(cycle_rejections, symbol, "exchange_info_unavailable")
                continue

            position = self.risk.size_position(signal, exchange_info)
            if position is None:
                self._record_cycle_rejection(cycle_rejections, symbol, "risk_sizing_rejected")
                continue

            if self.dry_run:
                order_ids = {
                    "entry": "DRY_RUN",
                    "sl": "DRY_RUN",
                    "tp1": "DRY_RUN",
                    "tp2": "DRY_RUN",
                }
                log.info("[DRY_RUN] [%s] order placement skipped", symbol)
            else:
                order_ids = self.executor.open_position(position)
                if not order_ids or not order_ids.get("entry"):
                    log.warning("[%s] order placement failed, skipping portfolio open", symbol)
                    self._record_cycle_rejection(cycle_rejections, symbol, "order_placement_failed")
                    continue

            self.portfolio.open_position(position, signal, order_ids)
            opened_symbols.add(symbol)
            notify_trade_open(
                symbol=symbol,
                direction=signal.direction.value,
                entry=position.entry_price,
                qty=position.quantity,
                notional=position.notional_usdt,
            )

        self._log_cycle_rejection_summary(cycle_rejections, opened_symbols)

        stats = self.portfolio.stats()
        log.info(
            "Cycle complete | balance=$%.2f | open=%s | trades=%s",
            stats.get("balance", 0.0),
            stats.get("open", 0),
            stats.get("trades", 0),
        )
        notify_stats(stats)

    def _generate_signal_candidate(
        self,
        symbol: str,
        multi_tf: dict,
        primary_tf: str,
        htf_1: str,
        htf_2: str,
        min_confidence: float,
    ) -> dict | None:
        df = multi_tf.get(primary_tf)
        if df is None or len(df) < 120:
            log.warning("[%s] not enough %s data to evaluate", symbol, primary_tf)
            observed = 0 if df is None else len(df)
            return {
                "symbol": symbol,
                "rejection_reason": f"insufficient_{primary_tf}_data({observed}<120)",
            }

        align_to_new_candle = bool(getattr(CONFIG.strategy, "align_signal_to_new_candle", True))
        signal_bar_close = self._signal_bar_close_epoch(df)
        if align_to_new_candle and signal_bar_close is not None:
            last_evaluated = self._last_evaluated_signal_bar_close.get(symbol)
            if last_evaluated is not None and signal_bar_close <= last_evaluated:
                return {
                    "symbol": symbol,
                    "rejection_reason": "same_bar_already_evaluated",
                }

        strategy = self.strategy_cls()
        if isinstance(strategy, EnsembleStrategy):
            signal = strategy.generate(
                symbol=symbol,
                df=df,
                htf_df=multi_tf.get(htf_1),
                htf_df2=multi_tf.get(htf_2),
                no_signal_streak=int(self._no_signal_streaks.get(symbol, 0)),
            )
        else:
            signal = strategy.generate(
                symbol=symbol,
                df=df,
                htf_df=multi_tf.get(htf_1),
                htf_df2=multi_tf.get(htf_2),
            )

        if signal_bar_close is not None:
            self._last_evaluated_signal_bar_close[symbol] = signal_bar_close

        if signal is None:
            skip_reason = str(getattr(strategy, "last_skip_reason", "")).strip()
            if skip_reason.startswith("no_substrategy_signal"):
                self._no_signal_streaks[symbol] = int(self._no_signal_streaks.get(symbol, 0)) + 1
            elif skip_reason != "same_bar_already_evaluated":
                self._no_signal_streaks[symbol] = 0
            return {
                "symbol": symbol,
                "rejection_reason": skip_reason or "strategy_no_signal",
            }

        self._no_signal_streaks[symbol] = 0

        effective_min_confidence = float(min_confidence)
        if bool((signal.extra or {}).get("fallback_activated")):
            relax = float(
                getattr(CONFIG.strategy, "no_signal_fallback_confidence_relaxation", 0.05)
            )
            effective_min_confidence = max(0.0, effective_min_confidence - max(0.0, relax))

        rejection_reason = self._signal_rejection_reason(signal, effective_min_confidence)
        if rejection_reason:
            log.info(
                "[%s] signal rejected | conf=%.2f | rr=%.2f | reason=%s",
                symbol,
                signal.confidence,
                signal.risk_reward,
                rejection_reason,
            )
            return {
                "symbol": symbol,
                "rejection_reason": rejection_reason,
            }

        return {
            "symbol": symbol,
            "signal": signal,
            "df": df,
        }

    @staticmethod
    def _signal_bar_close_epoch(df) -> float | None:
        if df is None or len(df) < 2:
            return None

        close_series = None
        if hasattr(df, "columns") and "close_time" in df.columns:
            close_series = df["close_time"]
        elif hasattr(df, "index"):
            close_series = df.index

        if close_series is None or len(close_series) < 2:
            return None

        raw_value = close_series.iloc[-2] if hasattr(close_series, "iloc") else close_series[-2]
        return TradingBot._as_epoch_seconds(raw_value)

    @staticmethod
    def _as_epoch_seconds(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value.timestamp())
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None

    def _signal_is_tradeable(self, signal, min_confidence: float) -> bool:
        return self._signal_rejection_reason(signal, min_confidence) is None

    @staticmethod
    def _signal_rejection_reason(signal, min_confidence: float) -> str | None:
        min_conf = float(min_confidence)
        min_rr = float(CONFIG.risk.min_rr_ratio)

        if signal.direction == Direction.FLAT:
            return "direction_flat"
        if signal.confidence < min_conf:
            return f"confidence_below_min({signal.confidence:.2f}<{min_conf:.2f})"
        if signal.risk_reward < min_rr:
            return f"rr_below_min({signal.risk_reward:.2f}<{min_rr:.2f})"
        if signal.stop_loss <= 0:
            return "invalid_stop_loss"
        if signal.take_profit_1 <= 0:
            return "invalid_take_profit_1"
        if signal.take_profit_2 <= 0:
            return "invalid_take_profit_2"
        return None

    @staticmethod
    def _record_cycle_rejection(cycle_rejections: dict[str, str], symbol: str, reason: str) -> None:
        if symbol not in cycle_rejections:
            cycle_rejections[symbol] = reason

    def _log_cycle_rejection_summary(self, cycle_rejections: dict[str, str], opened_symbols: set[str]) -> None:
        skipped_entries: list[str] = []
        for symbol in self.symbols:
            if symbol in opened_symbols:
                continue
            reason = cycle_rejections.get(symbol)
            if reason is None:
                reason = "no_entry_taken"
            skipped_entries.append(f"{symbol}:{reason}")

        if not skipped_entries:
            log.info("Cycle rejection summary | no skipped symbols")
            return

        log.info("Cycle rejection summary | %s", " | ".join(skipped_entries))

    def _is_correlation_blocked(
        self,
        candidate_symbol: str,
        interval: str,
        corr_matrix: dict[tuple[str, str], float] | None = None,
    ) -> str | None:
        if not bool(CONFIG.risk.correlation_management_enabled):
            return None

        open_symbols = [s for s in self.portfolio.open_positions.keys() if s != candidate_symbol]
        if not open_symbols:
            return None

        threshold = float(self._effective_correlation_threshold())
        lookback = int(CONFIG.risk.correlation_lookback)
        max_correlated = self._effective_max_correlated_positions()

        correlated = 0
        for open_symbol in open_symbols:
            corr = None
            if corr_matrix:
                corr = corr_matrix.get((candidate_symbol, open_symbol))
                if corr is None:
                    corr = corr_matrix.get((open_symbol, candidate_symbol))

            if corr is None:
                corr = self.fetcher.get_close_correlation(candidate_symbol, open_symbol, interval, lookback)

            if corr is None:
                continue
            if abs(corr) >= threshold:
                correlated += 1
                log.info(
                    "[%s] high correlation with %s: %.2f",
                    candidate_symbol,
                    open_symbol,
                    corr,
                )

        if correlated >= max_correlated:
            log.warning(
                "[%s] skipped due correlation cap | correlated=%s threshold=%.2f",
                candidate_symbol,
                correlated,
                threshold,
            )
            return (
                f"correlation_cap(correlated={correlated},"
                f"max={max_correlated},threshold={threshold:.2f})"
            )
        return None

    @staticmethod
    def _ai_context(df) -> dict:
        signal_idx = -2
        context = {
            "close": float(df["close"].iloc[signal_idx]),
            "volume": float(df["volume"].iloc[signal_idx]),
        }
        if "taker_ratio" in df.columns:
            context["taker_ratio"] = float(df["taker_ratio"].iloc[signal_idx])
        if "body" in df.columns:
            context["body"] = float(df["body"].iloc[signal_idx])
        return context

    def shutdown(self, close_positions: bool, reason: str = "shutdown") -> None:
        positions = dict(self.portfolio.open_positions)
        if not positions:
            self.fetcher.stop_price_stream()
            notify_event("AlphaBot Shutdown", f"No open positions. Reason: {reason}")
            return

        closed_count = 0
        for symbol, pos in positions.items():
            if close_positions and not self.dry_run:
                self._cancel_protective_orders(symbol, pos)

                qty = float(pos.get("quantity", 0.0))
                if qty > 0:
                    self.executor.close_position_market(symbol, pos.get("direction", "LONG"), qty)
                exit_price = self.fetcher.get_current_price(symbol) or float(pos.get("entry_price", 0.0))
                self.portfolio.close_position(symbol, float(exit_price), reason="SHUTDOWN_EXIT")
                closed_count += 1

        self.portfolio._save()
        self.fetcher.stop_price_stream()
        notify_event(
            "AlphaBot Shutdown",
            (
                f"Reason: {reason}. Open positions: {len(positions)}. "
                f"Closed on shutdown: {closed_count}."
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaBot paper-trading entrypoint")
    parser.add_argument(
        "--symbols",
        default=os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT"),
        help="Comma-separated symbols",
    )
    parser.add_argument(
        "--strategy",
        default=os.getenv("ACTIVE_STRATEGY", "adx_trend"),
        choices=list(STRATEGY_MAP.keys()),
        help="Strategy to run",
    )
    parser.add_argument(
        "--analysis-interval",
        type=int,
        default=int(os.getenv("ANALYSIS_INTERVAL", str(CONFIG.trading.analysis_interval))),
        help="Seconds between cycles when running continuously",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of cycles to run (0 = infinite)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and size trades without submitting testnet orders",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Submit real TESTNET orders",
    )
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise ValueError("No symbols provided")

    dry_run = _resolve_dry_run(args.dry_run, args.live)
    if not dry_run and not _has_testnet_credentials():
        log.warning("No Binance credentials found; switching to dry-run mode")
        dry_run = True

    cycle_target = 1 if args.once else args.cycles

    shutdown_state = {"requested": False, "reason": "manual"}

    def _request_shutdown(reason: str) -> None:
        if shutdown_state["requested"]:
            return
        shutdown_state["requested"] = True
        shutdown_state["reason"] = reason
        log.warning("Shutdown requested: %s", reason)
        notify_event("AlphaBot", f"Shutdown requested: {reason}")

    def _handle_signal(signum, _frame) -> None:
        try:
            name = os_signal.Signals(signum).name
        except Exception:
            name = str(signum)
        _request_shutdown(f"signal:{name}")

    if hasattr(os_signal, "SIGTERM"):
        os_signal.signal(os_signal.SIGTERM, _handle_signal)
    if hasattr(os_signal, "SIGINT"):
        os_signal.signal(os_signal.SIGINT, _handle_signal)

    bot = TradingBot(symbols=symbols, strategy_name=args.strategy, dry_run=dry_run)

    cycle = 0
    try:
        if cycle_target > 0:
            while cycle < cycle_target:
                if shutdown_state["requested"]:
                    break
                cycle += 1
                log.info("Starting cycle %s", cycle)
                bot.run_cycle()
        else:
            # Continuous mode: run position checks frequently, and signal scans every analysis_interval
            last_position_check = 0.0
            last_signal_scan = 0.0
            position_check_interval = max(
                2.0,
                float(getattr(CONFIG.trading, "position_check_interval_seconds", 30.0)),
            )
            analysis_interval = max(5, args.analysis_interval)

            while True:
                if shutdown_state["requested"]:
                    break

                now = time.time()

                # 1. Run position check if interval has elapsed
                if now - last_position_check >= position_check_interval:
                    log.info("Running position management check")
                    bot.run_position_management()
                    last_position_check = time.time()

                if shutdown_state["requested"]:
                    break

                # 2. Run signal scan if interval has elapsed
                if now - last_signal_scan >= analysis_interval:
                    cycle += 1
                    log.info("Starting signal scan cycle %s", cycle)
                    bot.run_signal_scan()
                    last_signal_scan = time.time()

                if shutdown_state["requested"]:
                    break

                # Sleep in short increments to allow responsiveness to shutdown signals
                time.sleep(1.0)
    except KeyboardInterrupt:
        _request_shutdown("keyboard_interrupt")

    bot.shutdown(close_positions=bool(CONFIG.trading.close_on_shutdown), reason=shutdown_state["reason"])

    final_stats = bot.portfolio.stats()
    log.info("Final stats: %s", final_stats)


if __name__ == "__main__":
    main()
