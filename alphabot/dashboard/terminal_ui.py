"""
AlphaBot Terminal UI — Rich-based live dashboard.
Displays: Bot Status, Account Snapshot, Open Positions, Recent Trades,
Performance Stats, Risk Gauges, and Live Log Feed.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Optional

from loguru import logger

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    logger.warning("Rich not installed — terminal UI disabled")


class TerminalUI:
    """
    Real-time terminal dashboard using Rich Live display.
    Updates every 2 seconds with bot state.
    """

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._state: dict = {}
        self._console = Console() if HAS_RICH else None

    def update_state(self, state: dict) -> None:
        """Update the dashboard state (called from main loop)."""
        self._state = state

    async def start(self) -> None:
        if not HAS_RICH:
            logger.info("[Dashboard] Rich not available — terminal UI skipped")
            return
        self._running = True
        self._task = asyncio.create_task(self._render_loop())
        logger.info("[Dashboard] Terminal UI started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _render_loop(self) -> None:
        """Continuous render loop with Rich Live."""
        try:
            with Live(self._build_layout(), console=self._console,
                       refresh_per_second=0.5, screen=False) as live:
                while self._running:
                    live.update(self._build_layout())
                    await asyncio.sleep(2)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[Dashboard] Render error: {e}")

    def _build_layout(self) -> Table:
        """Build the full dashboard layout."""
        s = self._state

        # Master table
        grid = Table.grid(expand=True)
        grid.add_row(self._bot_status_panel(s))
        grid.add_row(self._account_panel(s))
        grid.add_row(self._positions_panel(s))
        grid.add_row(self._recent_trades_panel(s))
        grid.add_row(self._performance_panel(s))
        grid.add_row(self._risk_panel(s))
        return grid

    def _bot_status_panel(self, s: dict) -> Panel:
        status = s.get("bot_status", "UNKNOWN")
        regime_info = s.get("regimes", {})
        uptime = s.get("uptime", "0:00:00")
        last_signal = s.get("last_signal_time", "N/A")

        text = Text()
        color = "green" if status == "ACTIVE" else "red"
        text.append(f"  Status: ", style="bold")
        text.append(f"{status}", style=f"bold {color}")
        text.append(f"  |  Uptime: {uptime}")
        text.append(f"  |  Last Signal: {last_signal}")

        for pair, regime in regime_info.items():
            text.append(f"\n  {pair}: {regime}", style="cyan")

        return Panel(text, title="[bold]🤖 AlphaBot Status[/bold]", border_style="blue")

    def _account_panel(self, s: dict) -> Panel:
        balance = s.get("balance", 0)
        daily_pnl = s.get("daily_pnl", 0)
        total_pnl = s.get("total_pnl", 0)
        drawdown = s.get("drawdown", 0)

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="bold")
        table.add_column("Value")

        pnl_color = "green" if daily_pnl >= 0 else "red"
        table.add_row("Balance", f"${balance:,.2f}")
        table.add_row("Daily PnL", f"[{pnl_color}]${daily_pnl:,.2f}[/{pnl_color}]")
        table.add_row("Total PnL", f"${total_pnl:,.2f}")
        table.add_row("Drawdown", f"{drawdown:.2f}%")

        return Panel(table, title="[bold]💰 Account Snapshot[/bold]", border_style="green")

    def _positions_panel(self, s: dict) -> Panel:
        positions = s.get("open_positions", [])

        table = Table(show_lines=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Dir")
        table.add_column("Qty")
        table.add_column("Entry")
        table.add_column("Current")
        table.add_column("PnL", justify="right")
        table.add_column("SL")
        table.add_column("TP1")
        table.add_column("R-Mult")
        table.add_column("Strategy")

        for p in positions:
            pnl = p.get("unrealized_pnl", 0)
            pnl_style = "green" if pnl >= 0 else "red"
            dir_style = "green" if p.get("direction") == "LONG" else "red"
            table.add_row(
                p.get("symbol", ""),
                f"[{dir_style}]{p.get('direction', '')}[/{dir_style}]",
                f"{p.get('quantity', 0):.4f}",
                f"{p.get('entry_price', 0):.2f}",
                f"{p.get('current_price', 0):.2f}",
                f"[{pnl_style}]${pnl:.2f}[/{pnl_style}]",
                f"{p.get('sl_price', 0):.2f}",
                f"{p.get('tp1_price', 0):.2f}",
                f"{p.get('r_multiple', 0):.1f}R",
                p.get("strategy", ""),
            )

        if not positions:
            table.add_row("—", "—", "—", "—", "—", "—", "—", "—", "—", "—")

        return Panel(table, title=f"[bold]📊 Open Positions ({len(positions)})[/bold]",
                     border_style="yellow")

    def _recent_trades_panel(self, s: dict) -> Panel:
        trades = s.get("recent_trades", [])

        table = Table(show_lines=True)
        table.add_column("Symbol")
        table.add_column("Dir")
        table.add_column("PnL", justify="right")
        table.add_column("Exit Reason")
        table.add_column("Duration")
        table.add_column("Strategy")

        for t in trades[:10]:
            pnl = t.get("net_pnl", 0)
            pnl_style = "green" if pnl >= 0 else "red"
            table.add_row(
                t.get("symbol", ""),
                t.get("direction", ""),
                f"[{pnl_style}]${pnl:.2f}[/{pnl_style}]",
                t.get("exit_reason", ""),
                f"{t.get('duration_minutes', 0):.0f}m",
                t.get("strategy_name", ""),
            )

        return Panel(table, title="[bold]📜 Recent Trades[/bold]", border_style="magenta")

    def _performance_panel(self, s: dict) -> Panel:
        stats = s.get("stats", {})

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
        table.add_row("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")
        table.add_row("Avg Win", f"${stats.get('avg_win', 0):.2f}")
        table.add_row("Avg Loss", f"${stats.get('avg_loss', 0):.2f}")
        table.add_row("Sharpe Ratio", f"{stats.get('sharpe_ratio', 0):.2f}")
        table.add_row("Total Trades", f"{stats.get('total_trades', 0)}")

        return Panel(table, title="[bold]📈 Performance[/bold]", border_style="cyan")

    def _risk_panel(self, s: dict) -> Panel:
        risk = s.get("risk_status", {})

        daily_used = abs(risk.get("daily_pnl", 0))
        daily_cap = float(s.get("balance", 1)) * float(settings_import().daily_loss_cap_pct) / 100
        daily_pct = (daily_used / daily_cap * 100) if daily_cap > 0 else 0

        text = Text()
        text.append(f"  Daily Loss: {daily_pct:.1f}% of cap")
        text.append(f"  |  Consecutive Losses: {risk.get('consecutive_losses', 0)}")
        text.append(f"  |  Halted: {'YES' if risk.get('halted') else 'No'}")

        color = "green" if daily_pct < 50 else ("yellow" if daily_pct < 80 else "red")
        return Panel(text, title=f"[bold]⚠️ Risk Gauges[/bold]",
                     border_style=color)


def settings_import():
    """Lazy import to avoid circular."""
    from alphabot.config import settings
    return settings
