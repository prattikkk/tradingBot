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


def _http_json_list(url: str, timeout: float = 3.0) -> Optional[list]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        return data if isinstance(data, list) else None
    except Exception:
        return None


def _is_isoish_timestamp(s: str) -> bool:
    # Very small heuristic: ISO timestamps usually contain 'T' and ':'
    return isinstance(s, str) and ("T" in s) and (":" in s)


def _coerce_to_datetime(v: Any) -> Optional[datetime]:
    """Best-effort conversion to aware UTC datetime.

    Supports:
    - ISO strings (YYYY-MM-DDTHH:MM:SS[.fff][Z|+00:00])
    - epoch seconds (int/float or numeric strings)
    - epoch milliseconds
    """
    if v is None:
        return None

    # Already a datetime
    if isinstance(v, datetime):
        return v.astimezone(UTC) if v.tzinfo else v.replace(tzinfo=UTC)

    # ISO-ish string
    if isinstance(v, str) and _is_isoish_timestamp(v):
        try:
            s = v.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
        except Exception:
            return None

    # Numeric epoch
    try:
        f = float(v)
    except Exception:
        return None

    # Heuristic: ms epochs are much larger.
    # 10^12 ms ~ 2001-09-09; 10^10 s ~ 2286.
    if f > 1e12:
        f = f / 1000.0
    if f < 0:
        return None
    try:
        return datetime.fromtimestamp(f, tz=UTC)
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


@dataclass
class DbDeepDive:
    introspection: DbIntrospection
    trades_table: Optional[str]
    trades_cols: List[str]
    trades_ts_col: Optional[str]
    trades_ts_mode: str
    trades_fee_cols: List[str]
    trades_pnl_col: Optional[str]
    last_trade_rows: List[dict]
    candidate_ledger_tables: List[str]
    ledger_summaries: Dict[str, dict]


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


def _query_one(cur: sqlite3.Cursor, q: str, params: Tuple[Any, ...] = ()) -> Optional[Tuple[Any, ...]]:
    try:
        cur.execute(q, params)
        return cur.fetchone()
    except Exception:
        return None


def _query_all(cur: sqlite3.Cursor, q: str, params: Tuple[Any, ...] = ()) -> List[Tuple[Any, ...]]:
    try:
        cur.execute(q, params)
        rows = cur.fetchall()
        return rows if rows else []
    except Exception:
        return []


def _detect_ts_mode(cur: sqlite3.Cursor, table: str, ts_col: str) -> str:
    """Return one of: iso, epoch_s, epoch_ms, unknown."""
    row = _query_one(cur, f"SELECT {ts_col} FROM {table} WHERE {ts_col} IS NOT NULL ORDER BY {ts_col} DESC LIMIT 1")
    if not row:
        return "unknown"
    v = row[0]
    if isinstance(v, str) and _is_isoish_timestamp(v):
        return "iso"
    try:
        f = float(v)
    except Exception:
        return "unknown"
    return "epoch_ms" if f > 1e12 else "epoch_s"


def _window_filter_sql(ts_mode: str, ts_col: str) -> Tuple[str, Any]:
    """Return (predicate_sql, bound_value) for last 24h filter."""
    window_start = _iso_now() - timedelta(hours=24)
    if ts_mode == "iso":
        return f"{ts_col} >= ?", window_start.isoformat()
    if ts_mode == "epoch_ms":
        return f"CAST({ts_col} AS REAL) >= ?", window_start.timestamp() * 1000.0
    if ts_mode == "epoch_s":
        return f"CAST({ts_col} AS REAL) >= ?", window_start.timestamp()
    # Fallback: attempt ISO compare
    return f"{ts_col} >= ?", window_start.isoformat()


def _summarize_ledger_table(cur: sqlite3.Cursor, table: str, cols: List[str]) -> Optional[dict]:
    # Look for typical columns in an accounting/ledger table
    amount_col = _pick_col(cols, "amount", "qty", "value", "delta", "pnl", "realized_pnl", "profit")
    ts_col = _pick_col(cols, "timestamp", "ts", "time", "created_at", "recorded_at")
    type_col = _pick_col(cols, "type", "event", "kind", "reason", "category")
    if not amount_col:
        return None

    summary: Dict[str, Any] = {"table": table, "amount_col": amount_col, "ts_col": ts_col, "type_col": type_col}
    one = _query_one(cur, f"SELECT COUNT(*), SUM(CAST({amount_col} AS REAL)), MIN(CAST({amount_col} AS REAL)), MAX(CAST({amount_col} AS REAL)) FROM {table}")
    if one:
        summary.update({"rows": int(one[0] or 0), "sum": _safe_float(one[1]), "min": _safe_float(one[2]), "max": _safe_float(one[3])})

    if ts_col:
        ts_mode = _detect_ts_mode(cur, table, ts_col)
        pred, bound = _window_filter_sql(ts_mode, ts_col)
        one24 = _query_one(cur, f"SELECT COUNT(*), SUM(CAST({amount_col} AS REAL)) FROM {table} WHERE {pred}", (bound,))
        if one24:
            summary.update({"rows_24h": int(one24[0] or 0), "sum_24h": _safe_float(one24[1]), "ts_mode": ts_mode})

    # Last few rows (lightweight)
    order_col = ts_col if ts_col else "rowid"
    select_cols = [c for c in [ts_col, type_col, amount_col] if c]
    if not select_cols:
        select_cols = ["rowid", amount_col]
    rows = _query_all(cur, f"SELECT {', '.join(select_cols)} FROM {table} ORDER BY {order_col} DESC LIMIT 5")
    summary["sample"] = [{select_cols[i]: r[i] for i in range(len(select_cols))} for r in rows]
    return summary


def _db_trade_stats(db_path: str) -> Tuple[Optional[str], DbTradeStats, Dict[str, str], Optional[DbDeepDive]]:
    stats = DbTradeStats(last_trades=[])
    state: Dict[str, str] = {}

    deep_dive: Optional[DbDeepDive] = None

    if not os.path.exists(db_path):
        return None, stats, state, deep_dive

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

        fee_cols = [c for c in (
            _pick_col(cols, "fee"),
            _pick_col(cols, "fees"),
            _pick_col(cols, "commission"),
            _pick_col(cols, "commission_usd"),
            _pick_col(cols, "funding"),
            _pick_col(cols, "funding_fee"),
        ) if c]

        ts_mode = _detect_ts_mode(cur, trade_table, ts_col) if ts_col else "unknown"

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
            pred, bound = _window_filter_sql(ts_mode, ts_col)
            q = (
                "SELECT COUNT(*) n, "
                "SUM(" + pnl_col + ") pnl_sum, "
                "SUM(CASE WHEN " + pnl_col + " > 0 THEN 1 ELSE 0 END) wins "
                "FROM " + trade_table + " WHERE " + pred
            )
            n, pnl_sum, wins = cur.execute(q, (bound,)).fetchone()
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

        # Deep dive: print-friendly schema + sample rows + candidate ledger tables
        candidate_ledger_tables: List[str] = []
        ledger_summaries: Dict[str, dict] = {}
        for t in introspection.tables:
            name_l = t.lower()
            if any(k in name_l for k in ("ledger", "fees", "fee", "fund", "funding", "wallet", "balance", "account", "trans")):
                candidate_ledger_tables.append(t)

        # Summarize candidates (lightweight)
        for t in candidate_ledger_tables[:10]:
            cols_t = introspection.table_cols.get(t) or []
            s = _summarize_ledger_table(cur, t, cols_t)
            if s:
                ledger_summaries[t] = s

        # Grab last raw rows from trades for exactness
        last_trade_rows: List[dict] = []
        select_raw = cols[:]
        if select_raw:
            qraw = f"SELECT {', '.join(select_raw)} FROM {trade_table} ORDER BY {order_col} DESC LIMIT 5"
            for r in _query_all(cur, qraw):
                last_trade_rows.append({select_raw[i]: r[i] for i in range(len(select_raw))})

        deep_dive = DbDeepDive(
            introspection=introspection,
            trades_table=trade_table,
            trades_cols=cols,
            trades_ts_col=ts_col,
            trades_ts_mode=ts_mode,
            trades_fee_cols=fee_cols,
            trades_pnl_col=pnl_col if isinstance(pnl_col, str) and pnl_col in cols else None,
            last_trade_rows=last_trade_rows,
            candidate_ledger_tables=candidate_ledger_tables,
            ledger_summaries=ledger_summaries,
        )

    if state_table:
        try:
            for k, v in cur.execute("SELECT key, value FROM " + state_table).fetchall():
                state[str(k)] = str(v)
        except Exception:
            pass

    conn.close()
    # Attach introspection for printing by returning it through state under a reserved key.
    state["__introspection_tables"] = ",".join(introspection.tables)
    return trade_table, stats, state, deep_dive


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
    api_trades = _http_json_list("http://127.0.0.1:8080/api/trades")
    api_stats = _http_json("http://127.0.0.1:8080/api/stats")

    trade_table, db_stats, state, deep = _db_trade_stats(db_path)

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

    if api_stats is not None:
        print("\nDashboard stats (in-memory, from /api/stats)")
        for k in ("total_trades", "win_rate", "profit_factor", "sharpe_ratio", "avg_win", "avg_loss"):
            if k in api_stats:
                v = api_stats.get(k)
                if k in ("win_rate",):
                    # Dashboard uses percent already
                    print(f"{k}:", _fmt_num(v))
                else:
                    print(f"{k}:", v)

    if api_trades is not None:
        print("\nDashboard recent trades (in-memory, from /api/trades)")
        print("api_trades_count:", len(api_trades))
        for t in api_trades[:10]:
            if isinstance(t, dict):
                # print key fields similar to UI table
                print({
                    "symbol": t.get("symbol"),
                    "direction": t.get("direction"),
                    "net_pnl": t.get("net_pnl"),
                    "exit_reason": t.get("exit_reason"),
                    "duration_minutes": t.get("duration_minutes"),
                    "strategy_name": t.get("strategy_name"),
                })

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

    if deep is not None:
        print("\nDB deep dive")
        if deep.introspection.warnings:
            print("introspection_warnings:")
            for w in deep.introspection.warnings[:5]:
                print("-", w)
        if deep.trades_table:
            print("trades_columns:", ",".join(deep.trades_cols))
            print("trades_timestamp_column:", deep.trades_ts_col or "n/a")
            print("trades_timestamp_mode:", deep.trades_ts_mode)
            if deep.trades_fee_cols:
                print("trades_fee_columns:", ",".join(sorted(set(deep.trades_fee_cols))))

            if deep.last_trade_rows:
                print("\nLast raw trade rows (exact columns)")
                for r in deep.last_trade_rows:
                    # keep it one line each
                    print(r)

        if deep.candidate_ledger_tables:
            print("\nCandidate ledger/accounting tables:")
            print(",".join(deep.candidate_ledger_tables[:20]))
        if deep.ledger_summaries:
            print("\nLedger summaries (best-effort)")
            for _, s in list(deep.ledger_summaries.items())[:10]:
                print(s)

    if db_stats.last_trades:
        print("\nLast trades (most recent first)")
        for t in db_stats.last_trades[:5]:
            print(t)

    # Explicitly call out API-vs-DB mismatch (this is what the screenshot shows)
    if api_stats is not None:
        api_total_trades = api_stats.get("total_trades")
        try:
            api_total_trades_n = int(api_total_trades)
        except Exception:
            api_total_trades_n = None
        if api_total_trades_n is not None and api_total_trades_n != db_stats.trades_total:
            print("\nDiscrepancy")
            print("dashboard_total_trades:", api_total_trades_n)
            print("db_trades_total:", db_stats.trades_total)
            print("explanation: dashboard uses in-memory state (recent_trades/stats), DB query reads alphabot_data.db")
            print("likely causes: different DB path/volume, DB not being written, or dashboard including trades from another source")

    # Balance drop explanation (best-effort)
    if status is not None:
        print("\nBalance reconciliation (best-effort)")
        live_balance = _safe_float(status.get("balance"))
        live_total_pnl = _safe_float(status.get("total_pnl"))
        db_total_pnl = db_stats.pnl_total
        starting_balance = None
        for k in ("starting_balance", "start_balance", "initial_balance", "starting_equity"):
            if k in state:
                starting_balance = _safe_float(state.get(k))
                if starting_balance is not None:
                    break
        if live_balance is None:
            print("live_balance: n/a")
        else:
            print("live_balance:", _fmt_money(live_balance))
        if starting_balance is not None:
            print("starting_balance (from db state):", _fmt_money(starting_balance))
        print("live_total_pnl:", _fmt_money(live_total_pnl))
        print("db_pnl_total:", _fmt_money(db_total_pnl))
        if live_total_pnl is not None and db_total_pnl is not None:
            print("pnl_gap (live - db):", _fmt_money(live_total_pnl - db_total_pnl))
            print("note: gap usually = fees/funding/untracked trades or schema mismatch")

        # Try reconstructing balance from starting_balance + DB pnl_total
        if starting_balance is not None and db_total_pnl is not None and live_balance is not None:
            expected = starting_balance + db_total_pnl
            print("expected_balance (start + db_pnl):", _fmt_money(expected))
            print("balance_gap (live - expected):", _fmt_money(live_balance - expected))

        print("note: balance can change due to realized PnL + fees + funding + manual wallet actions")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
