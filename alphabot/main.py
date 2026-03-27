"""
AlphaBot — Entry Point.
Starts all async services: data feed, strategy engine, position manager,
dashboard, and Telegram notifications.

Handles Ctrl+C gracefully: closes all positions, cancels orders, saves state.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import signal
import sys
from decimal import Decimal
from typing import Optional

from loguru import logger


async def main() -> None:
    """Main async entry point — wires all modules and starts the bot."""

    # ---- 1. Config & Logger ----
    from alphabot.config import settings
    from alphabot.utils.logger import setup_logger
    setup_logger()

    logger.info("=" * 60)
    logger.info("   ⚡ AlphaBot — Adaptive Algorithmic Trading System")
    logger.info(f"   Environment: {settings.environment.upper()}")
    logger.info(f"   Pairs: {', '.join(settings.trading_pairs)}")
    logger.info(f"   Timeframe: {settings.primary_timeframe}")
    logger.info(f"   Max Leverage: {settings.max_leverage}x")
    logger.info(f"   Risk Per Trade: {settings.risk_per_trade_pct}%")
    logger.info("=" * 60)

    # ---- 2. Database ----
    from alphabot.database.db import Database
    db = Database()
    logger.info("Database initialized")

    # ---- 3. Data Store ----
    from alphabot.data.data_store import DataStore
    data_store = DataStore()

    # ---- 4. Regime Detector ----
    from alphabot.regime.detector import RegimeDetector
    regime_detector = RegimeDetector(data_store)

    # ---- 5. Strategy Engine ----
    from alphabot.strategies.engine import StrategyEngine
    strategy_engine = StrategyEngine(data_store, regime_detector)

    # ---- 5b. Timeframe Manager (multi-timeframe routing) ----
    from alphabot.data.timeframe_manager import TimeframeManager
    tf_manager = TimeframeManager(data_store)
    tf_manager.configure_default_stack(
        entry_timeframes=list(settings.entry_timeframes) or [settings.primary_timeframe],
        bias_timeframes=list(settings.bias_timeframes) or ["1h", "4h"],
    )

    # ---- 6. Risk Manager ----
    from alphabot.risk.risk_manager import RiskManager
    risk_manager = RiskManager(db)

    # ---- 7. PnL Tracker ----
    from alphabot.positions.pnl_tracker import PnLTracker
    pnl_tracker = PnLTracker(db)

    # ---- 8. Exchange Client ----
    from alphabot.execution.testnet_client import BinanceTestnetClient
    client = BinanceTestnetClient()
    await client.connect()

    # Get initial account snapshot
    account_snapshot: dict = {}
    try:
        account_snapshot = await client.get_futures_account_snapshot()
        available = float(account_snapshot.get("availableBalance", 0) or 0)
        wallet = float(account_snapshot.get("walletBalance", 0) or 0)
        margin = float(account_snapshot.get("marginBalance", 0) or 0)
        unreal = float(account_snapshot.get("unrealizedProfit", 0) or 0)
        balance = available
        logger.info(
            f"Account snapshot (futures): available=${available:,.2f} wallet=${wallet:,.2f} "
            f"margin=${margin:,.2f} unrealized=${unreal:,.2f}"
        )
    except Exception as e:
        logger.warning(f"Could not fetch balance (using default): {e}")
        balance = 10000.0  # Testnet default

    risk_manager.initialize(Decimal(str(balance)), db=db)

    # Optional manual resume after a drawdown halt.
    # This is intentionally NOT automatic; it requires an explicit env var on restart.
    if os.getenv("ALPHABOT_MANUAL_RESUME", "").strip().lower() in {"1", "true", "yes"}:
        risk_manager.manual_resume()
        logger.warning("[Risk] ALPHABOT_MANUAL_RESUME enabled — drawdown halt cleared")

    async def refresh_balance() -> None:
        nonlocal balance
        nonlocal account_snapshot
        try:
            account_snapshot = await client.get_futures_account_snapshot()
            balance = float(account_snapshot.get("availableBalance", 0) or 0)
            risk_manager.initialize(Decimal(str(balance)), db=db, reset_runtime=False)
            logger.info(f"Balance refreshed: ${balance:,.2f}")
        except Exception as e:
            logger.error(f"Balance refresh failed: {e}")

    # ---- 9. Order Executor ----
    from alphabot.execution.order_executor import OrderExecutor
    order_executor = OrderExecutor(client)

    # ---- 10. Position Manager ----
    from alphabot.positions.position_manager import PositionManager
    position_manager = PositionManager(
        data_store=data_store,
        database=db,
        risk_manager=risk_manager,
        pnl_tracker=pnl_tracker,
        order_executor=order_executor,
        on_position_closed=refresh_balance,
    )

    # ---- 11. Telegram Notifier ----
    from alphabot.notifications.telegram_bot import TelegramNotifier
    notifier = TelegramNotifier()
    await notifier.start()
    position_manager.set_notifier(notifier)

    # ---- 12. Dashboard ----
    from alphabot.dashboard.terminal_ui import TerminalUI
    from alphabot.dashboard.api import DashboardServer
    terminal_ui = TerminalUI()
    await terminal_ui.start()

    start_time = datetime.datetime.now(datetime.UTC)

    def get_dashboard_state() -> dict:
        """Collect full bot state for dashboard."""
        stats = pnl_tracker.get_stats()
        risk_status = risk_manager.get_status()
        recent = db.get_trades(limit=20)
        trade_dicts = []
        for t in recent:
            trade_dicts.append({
                "symbol": t.symbol,
                "direction": t.direction,
                "net_pnl": t.net_pnl,
                "exit_reason": t.exit_reason,
                "duration_minutes": t.duration_minutes,
                "strategy_name": t.strategy_name,
            })

        bot_status = "ACTIVE"
        if risk_manager.is_halted:
            bot_status = "MAX_DRAWDOWN"
        elif risk_manager.is_daily_halted:
            bot_status = "DAILY_CAP"

        uptime = str(datetime.datetime.now(datetime.UTC) - start_time).split(".")[0]

        return {
            "bot_status": bot_status,
            "uptime": uptime,
            # Match Binance Futures UI semantics as closely as possible.
            # 'balance' stays as availableBalance for sizing, but we also expose wallet/margin/unrealized.
            "balance": balance,
            "available_balance": float(account_snapshot.get("availableBalance", 0) or 0),
            "wallet_balance": float(account_snapshot.get("walletBalance", 0) or 0),
            "margin_balance": float(account_snapshot.get("marginBalance", 0) or 0),
            "unrealized_pnl": float(account_snapshot.get("unrealizedProfit", 0) or 0),
            "daily_pnl": risk_status.get("daily_pnl", 0),
            "total_pnl": stats.get("total_pnl", 0),
            "drawdown": 0.0,
            "regimes": regime_detector.get_current_regimes() if hasattr(regime_detector, 'get_current_regimes') else regime_detector._last_regime,
            "open_positions": position_manager.open_positions_dicts,
            "recent_trades": trade_dicts,
            "stats": stats,
            "risk_status": risk_status,
            "last_signal_time": "N/A",
        }

    dashboard_server = DashboardServer(get_dashboard_state)
    await dashboard_server.start(settings.dashboard_host, settings.dashboard_port)

    # ---- 13. Candle Close Callback (routes to TFManager) ----
    async def on_candle_close(candle) -> None:
        await tf_manager.on_candle_close(candle)

    async def on_entry_signal_ready(symbol: str, timeframe: str) -> None:
        """Called only when an ENTRY timeframe candle closes and bias data is ready."""
        nonlocal balance

        logger.info(f"--- Entry eval: {symbol} {timeframe} ---")

        if risk_manager.is_halted or risk_manager.is_daily_halted:
            return

        bias_tf = tf_manager.get_bias_timeframe(timeframe)
        signal = strategy_engine.evaluate(symbol, timeframe, bias_timeframe=bias_tf)
        if signal is None:
            return

        open_pos_dicts = [p.to_dict() for p in position_manager.open_positions]
        total_exp = position_manager.total_exposure

        approved, reason, size_info = risk_manager.validate_signal(
            signal=signal,
            account_balance=Decimal(str(balance)),
            open_positions=open_pos_dicts,
            existing_exposure=total_exp,
        )

        if not approved:
            logger.info(f"Signal rejected for {symbol}: {reason}")
            return

        pos = await position_manager.open_position(signal, size_info)
        if pos:
            logger.info(f"Position opened: {pos.id} {pos.symbol} {pos.direction}")
            risk_manager.record_trade_opened(pos.symbol)

    tf_manager.register_callback(on_entry_signal_ready)

    # ---- 14. WebSocket Data Feed ----
    from alphabot.data.websocket_client import BinanceWebSocketClient
    ws_client = BinanceWebSocketClient(data_store, on_candle_close=on_candle_close)
    await ws_client.start()

    # ---- 15. Position Monitor ----
    await position_manager.start_monitor()

    # ---- 16. Terminal UI ----
    # Run terminal UI update loop
    async def ui_update_loop():
        while True:
            try:
                terminal_ui.update_state(get_dashboard_state())
            except Exception as e:
                logger.error(f"UI update error: {e}")
            await asyncio.sleep(2)

    ui_task = asyncio.create_task(ui_update_loop())

    # ---- 17. Daily Reset Scheduler ----
    async def daily_reset_loop():
        """Reset daily limits at UTC midnight."""
        while True:
            now = datetime.datetime.now(datetime.UTC)
            tomorrow = (now + datetime.timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            sleep_seconds = (tomorrow - now).total_seconds()
            await asyncio.sleep(sleep_seconds)
            risk_manager.reset_daily()
            logger.info("Daily limits reset at UTC midnight")
            # Send daily summary
            try:
                stats = pnl_tracker.get_stats()
                await notifier.notify_daily_summary(stats)
            except Exception as e:
                logger.error(f"Daily summary error: {e}")

    daily_task = asyncio.create_task(daily_reset_loop())

    # ---- 18. Stale Order Cleanup Scheduler ----
    async def stale_order_cleanup():
        """Clean up stale orders every 5 minutes."""
        while True:
            await asyncio.sleep(300)
            for symbol in settings.trading_pairs:
                try:
                    protected: list[str] = []
                    for pos in position_manager.open_positions:
                        if pos.symbol == symbol:
                            protected.extend(getattr(pos, "tp_order_ids", []))
                            protected.extend(getattr(pos, "sl_order_ids", []))

                    await order_executor.cleanup_stale_orders(
                        symbol,
                        settings.stale_order_minutes,
                        protected_ids=protected,
                    )
                except Exception as e:
                    logger.error(f"Stale order cleanup error for {symbol}: {e}")

    cleanup_task = asyncio.create_task(stale_order_cleanup())

    # ---- 19. Graceful Shutdown Handler ----
    shutdown_event = asyncio.Event()

    def handle_shutdown(sig, frame):
        logger.warning(f"Received signal {sig} — initiating graceful shutdown...")
        shutdown_event.set()

    # Register signal handlers
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, lambda: shutdown_event.set())
        loop.add_signal_handler(signal.SIGTERM, lambda: shutdown_event.set())
    else:
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("🚀 AlphaBot is running! Press Ctrl+C to stop.")

    # ---- 20. Main Wait ----
    try:
        await shutdown_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    # ---- 21. Graceful Shutdown ----
    logger.info("Shutting down gracefully...")

    async def log_session_summary() -> None:
        stats = pnl_tracker.get_stats()
        risk = risk_manager.get_status()
        uptime = str(datetime.datetime.now(datetime.UTC) - start_time).split(".")[0]
        logger.info("=" * 50)
        logger.info(f"SESSION SUMMARY — uptime: {uptime}")
        logger.info(
            f"Trades: {stats.get('total_trades', 0)} | "
            f"W:{stats.get('wins', 0)} L:{stats.get('losses', 0)} | "
            f"WR:{stats.get('win_rate', 0):.1f}%"
        )
        logger.info(
            f"PnL: ${stats.get('total_pnl', 0):.2f} | "
            f"PF:{stats.get('profit_factor', 0):.2f}"
        )
        logger.info(
            f"Daily PnL: ${risk.get('daily_pnl', 0):.2f} | "
            f"Consecutive losses: {risk.get('consecutive_losses', 0)}"
        )
        logger.info("=" * 50)

    await log_session_summary()

    # Close all positions
    if position_manager.open_positions:
        logger.info(f"Closing {len(position_manager.open_positions)} open positions...")
        await position_manager.close_all_positions(reason="BOT_SHUTDOWN")

    # Cancel all outstanding orders
    for symbol in settings.trading_pairs:
        try:
            await order_executor.cancel_all_orders(symbol)
        except Exception as e:
            logger.error(f"Error cancelling orders for {symbol}: {e}")

    # Stop components
    await position_manager.stop_monitor()
    await ws_client.stop()
    await dashboard_server.stop()
    await terminal_ui.stop()
    await notifier.stop()
    await client.close()

    # Cancel background tasks
    for task in [ui_task, daily_task, cleanup_task]:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    logger.info("✅ AlphaBot shut down cleanly. Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAlphaBot stopped.")
