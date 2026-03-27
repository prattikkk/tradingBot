#!/usr/bin/env python3
"""EC2 bot status summary.

Prints a single report combining:
- Live bot state via dashboard API (localhost:8080)
- Trading performance from SQLite DB (alphabot_data.db)

Designed to run on the EC2 host:
  python3 ec2_bot_status.py

Or inside the container:
  python /app/ec2_bot_status.py

No external dependencies beyond what's already in the image.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


UTC = timezone.utc


def _iso_now() -> datetime:
    return datetime.now(UTC)


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _fmt_money(v: Any) -> str:
    f = _safe_float(v)
    if f is None:
        return "n/a"
    return f"{f:.2f}"


def _fmt_pct(v: Any) -> str:
    f = _safe_float(v)
    if f is None:
        return "n/a"
    return f"{f * 100:.1f}%"


def _http_json(url: str, timeout: float = 3.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None


@dataclass
class DbTradeStats:
    trades_total: int = 0
    pnl_total: Optional[float] = None
    win_rate_total: Optional[float] = None

    trades_24h: int = 0
    pnl_24h: Optional[float] = None
    win_rate_24h: Optional[float] = None

    last_trades: List[dict] = None


def _find_table(cur: sqlite3.Cursor, preferred: Iterable[str], contains: Optional[str] = None) -> Optional[str]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]

    for name in preferred:
        if name in tables:
            return name

    if contains:
        for t in tables:
            if contains.lower() in t.lower():
                return t

    return None


def _table_cols(cur: sqlite3.Cursor, table: str) -> List[str]:
    cur.execute("PRAGMA table_info(" + table + ")")
    return [r[1] for r in cur.fetchall()]


def _pick_col(cols: List[str], *candidates: str) -> Optional[str]:
    lowered = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def _db_trade_stats(db_path: str) -> Tuple[Optional[str], DbTradeStats, Dict[str, str]]:
    stats = DbTradeStats(last_trades=[])
    state: Dict[str, str] = {}

    if not os.path.exists(db_path):
        return None, stats, state

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    trade_table = _find_table(cur, ["trades", "trade", "trade_records", "trade_record"], contains="trade")
    state_table = _find_table(cur, ["bot_state", "state", "botstate"], contains="state")

    now = _iso_now()
    window_start = now - timedelta(hours=24)

    if trade_table:
        cols = _table_cols(cur, trade_table)
        pnl_col = _pick_col(cols, "pnl", "realized_pnl", "profit", "profit_usd", "pnl_usd")
        ts_col = _pick_col(cols, "closed_at", "timestamp", "ts", "time", "created_at")
        sym_col = _pick_col(cols, "symbol", "pair")
        reason_col = _pick_col(cols, "reason", "close_reason", "exit_reason")

        if pnl_col:
            q = (
                "SELECT COUNT(*) n, "
                "SUM(" + pnl_col + ") pnl_sum, "
                "SUM(CASE WHEN " + pnl_col + " > 0 THEN 1 ELSE 0 END) wins "
                "FROM " + trade_table
            )
            n, pnl_sum, wins = cur.execute(q).fetchone()
            stats.trades_total = int(n or 0)
            stats.pnl_total = _safe_float(pnl_sum)
            stats.win_rate_total = (float(wins) / float(n)) if n else None

        if pnl_col and ts_col:
            q = (
                "SELECT COUNT(*) n, "
                "SUM(" + pnl_col + ") pnl_sum, "
                "SUM(CASE WHEN " + pnl_col + " > 0 THEN 1 ELSE 0 END) wins "
                "FROM " + trade_table + " WHERE " + ts_col + " >= ?"
            )
            n, pnl_sum, wins = cur.execute(q, (window_start.isoformat(),)).fetchone()
            stats.trades_24h = int(n or 0)
            stats.pnl_24h = _safe_float(pnl_sum)
            stats.win_rate_24h = (float(wins) / float(n)) if n else None

        order_col = ts_col if ts_col else "rowid"
        select_cols = [c for c in [sym_col, pnl_col, ts_col, reason_col] if c] or ["rowid"]
        q = (
            "SELECT " + ", ".join(select_cols) +
            " FROM " + trade_table +
            " ORDER BY " + order_col + " DESC LIMIT 10"
        )
        for row in cur.execute(q).fetchall():
            stats.last_trades.append({select_cols[i]: row[i] for i in range(len(select_cols))})

    if state_table:
        try:
            for k, v in cur.execute("SELECT key, value FROM " + state_table).fetchall():
                state[str(k)] = str(v)
        except Exception:
            pass

    conn.close()
    return trade_table, stats, state


def main() -> int:
    now = _iso_now()

    # Try common paths (host vs container)
    db_candidates = [
        os.environ.get("ALPHABOT_DB_PATH"),
        "/app/alphabot_data.db",
        os.path.join(os.getcwd(), "alphabot_data.db"),
        os.path.join(os.path.dirname(__file__), "alphabot_data.db"),
    ]
    db_path = next((p for p in db_candidates if p and os.path.exists(p)), db_candidates[1])

    status = _http_json("http://127.0.0.1:8080/api/status")

    trade_table, db_stats, state = _db_trade_stats(db_path)

    print("AlphaBot EC2 Status")
    print("utc_now:", now.isoformat())
    print("db_path:", db_path)
    print("trade_table:", trade_table or "n/a")

    if status is None:
        print("dashboard_status: unavailable (api/status failed)")
    else:
        bot_status = status.get("bot_status")
        balance = status.get("balance")
        daily_pnl = status.get("daily_pnl")
        total_pnl = status.get("total_pnl")
        open_positions = status.get("open_positions", [])

        print("\nLive")
        print("bot_status:", bot_status)
        print("balance:", _fmt_money(balance))
        print("daily_pnl:", _fmt_money(daily_pnl))
        print("total_pnl:", _fmt_money(total_pnl))
        print("open_positions:", len(open_positions) if isinstance(open_positions, list) else "n/a")

    print("\nDB performance")
    print("trades_total:", db_stats.trades_total)
    print("pnl_total:", _fmt_money(db_stats.pnl_total))
    print("win_rate_total:", _fmt_pct(db_stats.win_rate_total))
    print("trades_24h:", db_stats.trades_24h)
    print("pnl_24h:", _fmt_money(db_stats.pnl_24h))
    print("win_rate_24h:", _fmt_pct(db_stats.win_rate_24h))

    if state:
        print("\nRisk/state")
        for k in ("daily_loss_halt", "daily_pnl", "total_pnl", "consecutive_losses", "last_reset_date"):
            if k in state:
                print(f"{k}:", state[k])

    if db_stats.last_trades:
        print("\nLast trades (most recent first)")
        for t in db_stats.last_trades[:5]:
            print(t)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
