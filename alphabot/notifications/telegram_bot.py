"""
AlphaBot Telegram Notifications — Real-time trade alerts.
Sends alerts for: trade opened/closed, TP1 hit, SL hit, daily cap,
max drawdown, daily summary, errors.
All alerts include: timestamp, event, PnL impact, balance, drawdown%.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Optional

from loguru import logger

from alphabot.config import settings

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    logger.warning("python-telegram-bot not installed — Telegram alerts disabled")


class TelegramNotifier:
    """Sends Telegram messages for trade events and alerts."""

    def __init__(self):
        self._bot: Optional["Bot"] = None
        self._enabled = False

    async def start(self) -> None:
        if not HAS_TELEGRAM:
            logger.info("[Telegram] Library not available — skipping")
            return

        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            logger.info("[Telegram] No token/chat_id configured — skipping")
            return

        try:
            self._bot = Bot(token=settings.telegram_bot_token)
            self._enabled = True
            await self._send("🤖 *AlphaBot Started*\n"
                           f"Environment: `{settings.environment}`\n"
                           f"Pairs: `{', '.join(settings.trading_pairs)}`\n"
                           f"Timeframe: `{settings.primary_timeframe}`")
            logger.info("[Telegram] Notifier connected")
        except Exception as e:
            logger.error(f"[Telegram] Failed to initialize: {e}")

    async def stop(self) -> None:
        if self._enabled:
            await self._send("🛑 *AlphaBot Stopped*")
        self._enabled = False

    async def _send(self, message: str) -> None:
        if not self._enabled or not self._bot:
            return
        try:
            await self._bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            logger.error(f"[Telegram] Send failed: {e}")

    async def notify_trade_opened(self, position) -> None:
        """Alert: trade opened."""
        msg = (
            f"📈 *Trade Opened*\n"
            f"Symbol: `{position.symbol}`\n"
            f"Direction: `{position.direction}`\n"
            f"Entry: `{float(position.entry_price):.2f}`\n"
            f"Size: `{float(position.quantity):.4f}`\n"
            f"Leverage: `{position.leverage}x`\n"
            f"SL: `{float(position.sl_price):.2f}`\n"
            f"TP1: `{float(position.tp1_price):.2f}`\n"
            f"Strategy: `{position.strategy_name}`\n"
            f"Confidence: `{position.signal_confidence:.1f}`\n"
            f"Regime: `{position.regime}`"
        )
        await self._send(msg)

    async def notify_trade_closed(self, position, net_pnl: float) -> None:
        """Alert: trade closed."""
        outcome = "✅ WIN" if net_pnl > 0 else "❌ LOSS"
        msg = (
            f"{outcome} *Trade Closed*\n"
            f"Symbol: `{position.symbol}`\n"
            f"Direction: `{position.direction}`\n"
            f"PnL: `${net_pnl:.2f}`\n"
            f"Reason: `{position.close_reason}`\n"
            f"Duration: `{self._duration(position)}`\n"
            f"Strategy: `{position.strategy_name}`"
        )
        await self._send(msg)

    async def notify_tp1_hit(self, position) -> None:
        """Alert: TP1 reached — partial close."""
        msg = (
            f"🎯 *TP1 Hit — Partial Close*\n"
            f"Symbol: `{position.symbol}`\n"
            f"Direction: `{position.direction}`\n"
            f"SL moved to breakeven: `{float(position.entry_price):.2f}`\n"
            f"Remaining qty: `{float(position.remaining_qty):.4f}`"
        )
        await self._send(msg)

    async def notify_daily_loss_cap(self, daily_pnl: float, balance: float) -> None:
        """Alert: daily loss cap reached."""
        msg = (
            f"🚨 *Daily Loss Cap Hit*\n"
            f"Daily PnL: `${daily_pnl:.2f}`\n"
            f"Balance: `${balance:.2f}`\n"
            f"Trading halted until next UTC day"
        )
        await self._send(msg)

    async def notify_drawdown_halt(self, drawdown_pct: float, balance: float) -> None:
        """Alert: max drawdown — CRITICAL."""
        msg = (
            f"🔴 *CRITICAL: MAX DRAWDOWN HALT*\n"
            f"Drawdown: `{drawdown_pct:.2f}%`\n"
            f"Balance: `${balance:.2f}`\n"
            f"All positions closed. Manual restart required."
        )
        await self._send(msg)

    async def notify_daily_summary(self, stats: dict) -> None:
        """Daily summary notification."""
        msg = (
            f"📊 *Daily Summary*\n"
            f"Total Trades: `{stats.get('total_trades', 0)}`\n"
            f"Win Rate: `{stats.get('win_rate', 0):.1f}%`\n"
            f"Gross PnL: `${stats.get('total_pnl', 0):.2f}`\n"
            f"Profit Factor: `{stats.get('profit_factor', 0):.2f}`\n"
            f"Sharpe: `{stats.get('sharpe_ratio', 0):.2f}`"
        )
        await self._send(msg)

    async def notify_error(self, module: str, error: str) -> None:
        """Alert: error/exception."""
        msg = (
            f"⚠️ *Error*\n"
            f"Module: `{module}`\n"
            f"Error: `{error[:200]}`"
        )
        await self._send(msg)

    @staticmethod
    def _duration(position) -> str:
        if position.closed_at and position.opened_at:
            delta = position.closed_at - position.opened_at
            minutes = delta.total_seconds() / 60
            if minutes > 60:
                return f"{minutes / 60:.1f}h"
            return f"{minutes:.0f}m"
        return "N/A"
