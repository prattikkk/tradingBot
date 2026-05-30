"""
core/portfolio.py
In-memory portfolio for paper trading P&L tracking.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from core.signal import Direction
from config import CONFIG
from utils.logger import get_logger

log = get_logger("Portfolio")


@dataclass
class Trade:
    id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    quantity: float
    notional: float
    risk_usdt: float
    strategy: str
    confidence: float
    open_time: str
    close_time: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = "OPEN"       # OPEN | TP1_HIT | CLOSED | SL_HIT
    tp1_hit: bool = False
    leverage: int = 5


class Portfolio:
    DATA_FILE = Path("data/portfolio.json")

    def __init__(self):
        self.total_capital: float = CONFIG.risk.initial_capital
        self._balance: float = CONFIG.risk.initial_capital
        self.open_positions: dict[str, dict] = {}   # symbol → position dict
        self.closed_trades: list[dict] = []
        self.telemetry: dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Capital
    # ------------------------------------------------------------------ #

    def available_capital(self) -> float:
        reserved = sum(
            p.get("notional", 0) / p.get("leverage", 5)
            for p in self.open_positions.values()
        )
        return max(0.0, self._balance - reserved)

    def sync_exchange_balance(self, new_balance: float, reset_total_capital: bool = False) -> None:
        """Align runtime portfolio cash with exchange-reported wallet/available balance."""
        try:
            balance = float(new_balance)
        except Exception:
            return

        if balance <= 0:
            return

        old_balance = self._balance
        self._balance = balance
        if reset_total_capital:
            self.total_capital = balance

        self._save()
        log.info(
            "Synced portfolio balance from exchange | old=$%.2f new=$%.2f",
            old_balance,
            self._balance,
        )

    def equity(self) -> float:
        return float(self._balance)

    def increment_metric(self, name: str, amount: int = 1, persist: bool = False) -> int:
        key = str(name).strip()
        if not key:
            return 0

        current = int(self.telemetry.get(key, 0))
        updated = current + int(amount)
        self.telemetry[key] = updated

        if persist:
            self._save()
        return updated

    # ------------------------------------------------------------------ #
    # Position management
    # ------------------------------------------------------------------ #

    def open_position(self, position_size, signal, order_ids: dict) -> Trade:
        trade_id = f"{signal.symbol}_{int(time.time())}"
        trade = Trade(
            id=trade_id,
            symbol=signal.symbol,
            direction=signal.direction.value,
            entry_price=position_size.entry_price,
            stop_loss=position_size.stop_loss,
            take_profit_1=position_size.take_profit_1,
            take_profit_2=position_size.take_profit_2,
            quantity=position_size.quantity,
            notional=position_size.notional_usdt,
            risk_usdt=position_size.risk_usdt,
            strategy=signal.strategy,
            confidence=signal.confidence,
            open_time=datetime.utcnow().isoformat(),
            leverage=position_size.leverage,
        )
        self.open_positions[signal.symbol] = {
            **asdict(trade),
            "order_ids": order_ids,
        }
        self._save()
        log.info(f"📂 Opened: {trade.direction} {trade.symbol} @ {trade.entry_price:.4f}")
        return trade

    def update_tp1(self, symbol: str, exit_price: float) -> Optional[float]:
        """Partial exit at TP1 — close 50% of position."""
        pos = self.open_positions.get(symbol)
        if not pos or pos.get("tp1_hit"):
            return None

        qty_exit = pos["quantity"] * CONFIG.risk.partial_exit_pct
        pnl = self._calc_pnl(pos["direction"], pos["entry_price"], exit_price, qty_exit)

        pos["tp1_hit"] = True
        pos["quantity"] -= qty_exit
        pos["status"] = "TP1_HIT"
        pos["pnl"] = pos.get("pnl", 0) + pnl

        try:
            risk_per_unit = abs(float(pos.get("entry_price", 0.0)) - float(pos.get("stop_loss", 0.0)))
            pos["risk_usdt"] = max(0.0, float(pos.get("quantity", 0.0)) * risk_per_unit)
        except Exception:
            pass

        self._balance += pnl
        self._save()
        log.info(f"🎯 TP1 hit [{symbol}] | pnl=${pnl:+.2f} | remaining qty={pos['quantity']:.4f}")
        return pnl

    def close_position(self, symbol: str, exit_price: float, reason: str = "TP2") -> Optional[float]:
        pos = self.open_positions.pop(symbol, None)
        if not pos:
            return None

        pnl = self._calc_pnl(pos["direction"], pos["entry_price"], exit_price, pos["quantity"])
        pnl += pos.get("pnl", 0)   # add TP1 partial

        pos.update({
            "exit_price": exit_price,
            "close_time": datetime.utcnow().isoformat(),
            "pnl": pnl,
            "status": reason,
        })
        self.closed_trades.append(pos)
        self._balance += pnl
        self._save()

        emoji = "✅" if pnl > 0 else "❌"
        log.info(f"{emoji} Closed [{symbol}] {reason} | pnl=${pnl:+.2f} | balance=${self._balance:.2f}")
        return pnl

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        if not self.closed_trades:
            return {
                "trades": 0,
                "open": len(self.open_positions),
                "balance": self._balance,
                "total_capital": self.total_capital,
                "return_pct": round((self._balance - self.total_capital) / self.total_capital * 100, 2),
            }

        pnls = [t["pnl"] for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0

        avg_win  = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        loss_sum = sum(losses)
        profit_factor = abs(sum(wins) / loss_sum) if loss_sum != 0 else float("inf")

        return {
            "trades": len(pnls),
            "open": len(self.open_positions),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 3),
            "total_pnl": round(sum(pnls), 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "balance": round(self._balance, 2),
            "total_capital": round(self.total_capital, 2),
            "return_pct": round((self._balance - self.total_capital) / self.total_capital * 100, 2),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _save(self):
        self.DATA_FILE.parent.mkdir(exist_ok=True)
        tmp_file = self.DATA_FILE.with_suffix(".tmp")
        try:
            with open(tmp_file, "w") as f:
                json.dump({
                    "balance": self._balance,
                    "total_capital": self.total_capital,
                    "telemetry": self.telemetry,
                    "open_positions": self.open_positions,
                    "closed_trades": self.closed_trades,
                }, f, indent=2, default=str)
            os.replace(tmp_file, self.DATA_FILE)
        except Exception as e:
            log.error(f"Portfolio save failed: {e}")
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

    def _load(self):
        if not self.DATA_FILE.exists():
            return
        try:
            with open(self.DATA_FILE) as f:
                data = json.load(f)
            self._balance        = data.get("balance", self._balance)
            self.total_capital   = data.get("total_capital", self.total_capital)
            telemetry = data.get("telemetry", {})
            self.telemetry      = dict(telemetry) if isinstance(telemetry, dict) else {}
            self.open_positions  = data.get("open_positions", {})
            self.closed_trades   = data.get("closed_trades", [])
            log.info(f"📊 Loaded portfolio | balance=${self._balance:.2f} | "
                     f"open={len(self.open_positions)} | closed={len(self.closed_trades)}")
        except Exception as e:
            log.error(f"Portfolio load failed: {e}")

    @staticmethod
    def _calc_pnl(direction: str, entry: float, exit_p: float, qty: float) -> float:
        if direction == Direction.LONG.value:
            return (exit_p - entry) * qty
        else:
            return (entry - exit_p) * qty
