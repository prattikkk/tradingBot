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


def _fmt_num(v: Any) -> str:
    f = _safe_float(v)
    if f is None:
        return "n/a"
    return f"{f:.4f}" if abs(f) < 100 else f"{f:.2f}"


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


@dataclass
class DbIntrospection:
    tables: List[str]
    table_cols: Dict[str, List[str]]
    warnings: List[str]


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


def _introspect_db(cur: sqlite3.Cursor) -> DbIntrospection:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    table_cols: Dict[str, List[str]] = {}
    warnings: List[str] = []

    for t in tables:
        try:
            table_cols[t] = _table_cols(cur, t)
        except Exception as e:
            warnings.append(f"failed PRAGMA table_info({t}): {e}")
            table_cols[t] = []

    return DbIntrospection(tables=tables, table_cols=table_cols, warnings=warnings)


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

    introspection = _introspect_db(cur)

    trade_table = _find_table(cur, ["trades", "trade", "trade_records", "trade_record"], contains="trade")
    state_table = _find_table(cur, ["bot_state", "state", "botstate"], contains="state")

    now = _iso_now()
    window_start = now - timedelta(hours=24)

    if trade_table:
        cols = introspection.table_cols.get(trade_table) or _table_cols(cur, trade_table)
        pnl_col = _pick_col(cols, "pnl", "realized_pnl", "profit", "profit_usd", "pnl_usd")
        ts_col = _pick_col(cols, "closed_at", "timestamp", "ts", "time", "created_at")
        sym_col = _pick_col(cols, "symbol", "pair")
        reason_col = _pick_col(cols, "reason", "close_reason", "exit_reason")

        # If we can't find a clear PnL column, try to derive it.
        # Common patterns: entry_price/exit_price/qty/side/leverage.
        if pnl_col is None:
            entry_col = _pick_col(cols, "entry_price", "entry")
            exit_col = _pick_col(cols, "exit_price", "exit")
            qty_col = _pick_col(cols, "qty", "quantity", "size", "amount")
            side_col = _pick_col(cols, "side", "direction")
            lev_col = _pick_col(cols, "leverage")
            fee_col = _pick_col(cols, "fees", "fee", "fees_paid")

            if entry_col and exit_col and qty_col and side_col:
                # We'll compute pnl as:
                # LONG: (exit-entry)*qty*lev ; SHORT: (entry-exit)*qty*lev
                # If leverage missing, default to 1.
                lev_expr = lev_col if lev_col else "1"
                side_expr = "LOWER(" + side_col + ")"
                fee_expr = fee_col if fee_col else "0"
                pnl_expr = (
                    "(CASE WHEN " + side_expr + " IN ('long','buy') THEN "
                    "(CAST(" + exit_col + " AS REAL) - CAST(" + entry_col + " AS REAL)) "
                    "ELSE (CAST(" + entry_col + " AS REAL) - CAST(" + exit_col + " AS REAL)) END) "
                    "* CAST(" + qty_col + " AS REAL) * CAST(" + lev_expr + " AS REAL) - CAST(" + fee_expr + " AS REAL)"
                )
                pnl_col = pnl_expr

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
        base_cols = [c for c in [sym_col, ts_col, reason_col] if c]
        select_cols = base_cols or ["rowid"]

        # For derived PnL expression we can't select it by name reliably across DBs.
        if pnl_col and pnl_col in cols:
            select_cols = [sym_col, pnl_col, ts_col, reason_col]
            select_cols = [c for c in select_cols if c]
            q = (
                "SELECT " + ", ".join(select_cols) +
                " FROM " + trade_table +
                " ORDER BY " + order_col + " DESC LIMIT 10"
            )
            for row in cur.execute(q).fetchall():
                stats.last_trades.append({select_cols[i]: row[i] for i in range(len(select_cols))})
        else:
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
    # Attach introspection for printing by returning it through state under a reserved key.
    state["__introspection_tables"] = ",".join(introspection.tables)
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

        # Always show what tables exist (useful when stats are missing).
        tables_csv = state.get("__introspection_tables")
        if tables_csv:
            print("db_tables:", tables_csv)

    if db_stats.last_trades:
        print("\nLast trades (most recent first)")
        for t in db_stats.last_trades[:5]:
            print(t)

    # Balance drop explanation (best-effort)
    if status is not None:
        print("\nBalance reconciliation (best-effort)")
        live_balance = _safe_float(status.get("balance"))
        live_total_pnl = _safe_float(status.get("total_pnl"))
        db_total_pnl = db_stats.pnl_total
        if live_balance is None:
            print("live_balance: n/a")
        else:
            print("live_balance:", _fmt_money(live_balance))
        print("live_total_pnl:", _fmt_money(live_total_pnl))
        print("db_pnl_total:", _fmt_money(db_total_pnl))
        if live_total_pnl is not None and db_total_pnl is not None:
            print("pnl_gap (live - db):", _fmt_money(live_total_pnl - db_total_pnl))
            print("note: gap usually = fees/untracked trades or schema mismatch")
        print("note: balance can change due to realized PnL + fees + funding + manual testnet wallet actions")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
