"""
AlphaBot PnL Tracker — Records all closed trades, computes cumulative stats.
Writes to SQLite + CSV journal.
"""

from __future__ import annotations

import csv
import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from alphabot.database.db import Database
from alphabot.database.models import TradeRecord


_JOURNAL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "trade_journal.csv"

_JOURNAL_HEADERS = [
    "trade_id", "symbol", "side", "entry_price", "exit_price",
    "quantity", "leverage", "gross_pnl", "fees", "net_pnl",
    "pnl_percent", "duration_minutes", "strategy", "regime",
    "signal_confidence", "close_reason", "daily_pnl_after",
    "session_drawdown_after",
]


class PnLTracker:
    """Tracks all closed trades and maintains running statistics."""

    def __init__(self, database: Database):
        self.db = database
        self._total_pnl: Decimal = Decimal("0")
        self._total_trades: int = 0
        self._wins: int = 0
        self._losses: int = 0
        self._gross_profit: Decimal = Decimal("0")
        self._gross_loss: Decimal = Decimal("0")
        self._returns: List[float] = []
        self._ensure_journal()

    def _ensure_journal(self) -> None:
        """Create CSV journal file with headers if it doesn't exist."""
        _JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not _JOURNAL_PATH.exists():
            with open(_JOURNAL_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(_JOURNAL_HEADERS)
            logger.info(f"Trade journal created: {_JOURNAL_PATH}")

    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        leverage: int,
        fees: float,
        strategy_name: str,
        regime: str,
        signal_confidence: float,
        exit_reason: str,
        open_time: datetime.datetime,
        close_time: datetime.datetime,
        daily_pnl: float = 0.0,
        session_drawdown: float = 0.0,
    ) -> TradeRecord:
        """Record a closed trade to DB and CSV journal."""
        # Calculate PnL
        if direction == "LONG":
            gross_pnl = (exit_price - entry_price) * quantity * leverage
        else:
            gross_pnl = (entry_price - exit_price) * quantity * leverage

        net_pnl = gross_pnl - fees
        pnl_pct = (net_pnl / (entry_price * quantity)) * 100 if entry_price * quantity > 0 else 0
        duration = (close_time - open_time).total_seconds() / 60.0

        # Update running stats
        self._total_trades += 1
        self._total_pnl += Decimal(str(net_pnl))
        self._returns.append(net_pnl)

        if net_pnl > 0:
            self._wins += 1
            self._gross_profit += Decimal(str(gross_pnl))
        else:
            self._losses += 1
            self._gross_loss += abs(Decimal(str(gross_pnl)))

        # Save to database
        trade = TradeRecord(
            id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            leverage=leverage,
            gross_pnl=gross_pnl,
            fees=fees,
            net_pnl=net_pnl,
            pnl_percent=pnl_pct,
            duration_minutes=duration,
            strategy_name=strategy_name,
            regime_at_entry=regime,
            signal_confidence=signal_confidence,
            exit_reason=exit_reason,
            open_timestamp=open_time,
            close_timestamp=close_time,
            daily_pnl_after=daily_pnl,
            session_drawdown_after=session_drawdown,
        )
        self.db.save_trade(trade)

        # Append to CSV journal
        self._write_journal_row(trade)

        logger.info(
            f"[PnL] Trade recorded: {symbol} {direction} "
            f"PnL=${net_pnl:.2f} ({pnl_pct:.2f}%) reason={exit_reason} "
            f"duration={duration:.1f}min"
        )
        return trade

    def _write_journal_row(self, trade: TradeRecord) -> None:
        """Append trade to CSV journal."""
        try:
            with open(_JOURNAL_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade.id, trade.symbol, trade.direction,
                    trade.entry_price, trade.exit_price,
                    trade.quantity, trade.leverage,
                    trade.gross_pnl, trade.fees, trade.net_pnl,
                    trade.pnl_percent, trade.duration_minutes,
                    trade.strategy_name, trade.regime_at_entry,
                    trade.signal_confidence, trade.exit_reason,
                    trade.daily_pnl_after, trade.session_drawdown_after,
                ])
        except Exception as e:
            logger.error(f"[PnL] Failed to write journal: {e}")

    @property
    def win_rate(self) -> float:
        if self._total_trades == 0:
            return 0.0
        return (self._wins / self._total_trades) * 100

    @property
    def profit_factor(self) -> float:
        if self._gross_loss == 0:
            return float("inf") if self._gross_profit > 0 else 0.0
        return float(self._gross_profit / self._gross_loss)

    @property
    def avg_win(self) -> float:
        if self._wins == 0:
            return 0.0
        return float(self._gross_profit / self._wins)

    @property
    def avg_loss(self) -> float:
        if self._losses == 0:
            return 0.0
        return float(self._gross_loss / self._losses)

    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe ratio (annualized for 252 trading days/year on 15m)."""
        import numpy as np
        if len(self._returns) < 2:
            return 0.0
        arr = np.array(self._returns)
        mean_ret = np.mean(arr)
        std_ret = np.std(arr, ddof=1)
        if std_ret == 0:
            return 0.0
        # Annualize: ~252 trading days/year, ~96 trades/day for 15m = ~24192 trades/year
        # Annualization_factor = sqrt(252 * 96) ≈ sqrt(24192) ≈ 155.6
        annualization_factor = (252 * 96) ** 0.5  # Approx 155.6
        return float(mean_ret / std_ret * annualization_factor)

    def get_stats(self) -> dict:
        """Return performance statistics for dashboard."""
        return {
            "total_trades": self._total_trades,
            "wins": self._wins,
            "losses": self._losses,
            "win_rate": round(self.win_rate, 2),
            "total_pnl": float(self._total_pnl),
            "gross_profit": float(self._gross_profit),
            "gross_loss": float(self._gross_loss),
            "profit_factor": round(self.profit_factor, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
        }
