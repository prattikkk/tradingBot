#!/usr/bin/env python3
"""Backfill missing TradeRecord rows into SQLite.

Why this exists:
- The dashboard shows trades from the SQLite `trades` table.
- Historical trade details are reliably present in the CSV trade journal.
- Timestamps required by the DB schema are available in structured JSON logs.

This script joins:
- CSV journal rows (numeric trade details)
- Log events (open/close timestamps per trade_id)

…then INSERTs any missing trade IDs into the `trades` table.

Design goals:
- Idempotent: safe to re-run; never overwrites existing trades.
- Safe: optional DB backup before writing.
- No third-party deps: uses only the Python standard library.

Typical usage (repo root):
  python backfill_trades.py --dry-run
  python backfill_trades.py

On EC2 with docker-compose volume-mounted DB, run on the host in the repo
directory so it edits the same `./alphabot_data.db` the container uses.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import json
import os
import re
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


UTC = _dt.timezone.utc


OPEN_RE = re.compile(r"Position opened:\s+(?P<id>[0-9a-fA-F-]{8,64})\b")
CLOSE_RE = re.compile(r"Position closed:\s+(?P<id>[0-9a-fA-F-]{8,64})\b")


REQUIRED_TRADE_FIELDS = [
    "trade_id",
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "quantity",
    "leverage",
    "gross_pnl",
    "net_pnl",
    "pnl_percent",
    "duration_minutes",
    "strategy",
    "regime",
    "close_reason",
]


TRADE_COLS = [
    "id",
    "symbol",
    "direction",
    "entry_price",
    "exit_price",
    "quantity",
    "leverage",
    "gross_pnl",
    "fees",
    "net_pnl",
    "pnl_percent",
    "duration_minutes",
    "strategy_name",
    "regime_at_entry",
    "signal_confidence",
    "exit_reason",
    "open_timestamp",
    "close_timestamp",
    "daily_pnl_after",
    "session_drawdown_after",
]


INSERT_SQL = (
    "INSERT INTO trades ("
    + ",".join(TRADE_COLS)
    + ") VALUES ("
    + ",".join(["?"] * len(TRADE_COLS))
    + ")"
)


@dataclass(frozen=True)
class JournalTrade:
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    gross_pnl: float
    fees: float
    net_pnl: float
    pnl_percent: float
    duration_minutes: float
    strategy: str
    regime: str
    signal_confidence: Optional[float]
    close_reason: str
    daily_pnl_after: Optional[float]
    session_drawdown_after: Optional[float]


def _parse_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _dt_from_epoch(ts: float) -> _dt.datetime:
    return _dt.datetime.fromtimestamp(float(ts), tz=UTC)


def _dt_to_sqlite_str(dt: _dt.datetime) -> str:
    """Match SQLAlchemy/SQLite default DateTime storage format.

    Observed format in this repo's DB: "YYYY-MM-DD HH:MM:SS.ffffff" (UTC).
    """
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=UTC)
    else:
        dt_utc = dt.astimezone(UTC)
    dt_naive = dt_utc.replace(tzinfo=None)
    return dt_naive.strftime("%Y-%m-%d %H:%M:%S.%f")


def _iter_log_files(log_dir: Path) -> List[Path]:
    if not log_dir.exists():
        return []

    paths: List[Path] = []
    for p in log_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".log") or name.endswith(".log.gz"):
            paths.append(p)

    # Sort by filename, which in this repo includes YYYY-MM-DD.
    paths.sort(key=lambda x: x.name)
    return paths


def _open_text_maybe_gzip(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _iter_log_records(log_path: Path) -> Iterator[Tuple[float, str]]:
    """Yield (epoch_seconds, message) from loguru JSON serialized logs."""
    with _open_text_maybe_gzip(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            record = obj.get("record") or {}
            msg = record.get("message") or obj.get("text")
            if not isinstance(msg, str):
                continue
            time_obj = record.get("time") or {}
            ts_raw = time_obj.get("timestamp")
            ts_f: Optional[float]
            if ts_raw is None:
                ts_f = None
            else:
                try:
                    ts_f = float(ts_raw)
                except Exception:
                    ts_f = None

            if ts_f is None:
                # Fallback to repr if timestamp missing
                repr_s = time_obj.get("repr")
                if not isinstance(repr_s, str):
                    continue
                try:
                    # Example: 2026-03-25 01:00:03.657116+05:30
                    dt = _dt.datetime.fromisoformat(repr_s)
                    ts_f = dt.timestamp()
                except Exception:
                    continue
            yield ts_f, msg


def _extract_position_events(log_dir: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (open_ts_by_id, close_ts_by_id) from logs."""
    open_ts: Dict[str, float] = {}
    close_ts: Dict[str, float] = {}

    for p in _iter_log_files(log_dir):
        for ts, msg in _iter_log_records(p):
            m = OPEN_RE.search(msg)
            if m:
                tid = m.group("id")
                prev = open_ts.get(tid)
                open_ts[tid] = ts if prev is None else min(prev, ts)
                continue

            m = CLOSE_RE.search(msg)
            if m:
                tid = m.group("id")
                prev = close_ts.get(tid)
                close_ts[tid] = ts if prev is None else max(prev, ts)

    return open_ts, close_ts


def _read_journal_paths(explicit: Optional[List[str]]) -> List[Path]:
    if explicit:
        return [Path(p) for p in explicit]

    candidates = [
        Path("data") / "trade_journal.csv",
        Path("trade_journal.csv"),
    ]
    return [p for p in candidates if p.exists()]


def _load_journal_rows(journal_paths: List[Path]) -> Dict[str, JournalTrade]:
    """Load journal CSV rows into dict keyed by trade_id.

    If a trade_id appears multiple times, the last encountered row wins.
    """
    trades: Dict[str, JournalTrade] = {}

    for path in journal_paths:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                trade_id = (row.get("trade_id") or "").strip()
                if not trade_id:
                    continue

                missing_required = [k for k in REQUIRED_TRADE_FIELDS if not str(row.get(k, "")).strip()]
                if missing_required:
                    # Skip malformed rows
                    continue

                entry_price_f = _parse_float(row.get("entry_price"))
                exit_price_f = _parse_float(row.get("exit_price"))
                quantity_f = _parse_float(row.get("quantity"))
                leverage_i = _parse_int(row.get("leverage"))
                gross_pnl_f = _parse_float(row.get("gross_pnl"))
                fees_f = _parse_float(row.get("fees"))
                net_pnl_f = _parse_float(row.get("net_pnl"))
                pnl_percent_f = _parse_float(row.get("pnl_percent"))
                duration_minutes_f = _parse_float(row.get("duration_minutes"))

                if (
                    entry_price_f is None
                    or exit_price_f is None
                    or quantity_f is None
                    or leverage_i is None
                    or gross_pnl_f is None
                    or net_pnl_f is None
                    or pnl_percent_f is None
                    or duration_minutes_f is None
                ):
                    continue

                entry_price = float(entry_price_f)
                exit_price = float(exit_price_f)
                quantity = float(quantity_f)
                leverage = int(leverage_i)
                gross_pnl = float(gross_pnl_f)
                fees = float(fees_f) if fees_f is not None else 0.0
                net_pnl = float(net_pnl_f)
                pnl_percent = float(pnl_percent_f)
                duration_minutes = float(duration_minutes_f)

                trades[trade_id] = JournalTrade(
                    trade_id=trade_id,
                    symbol=str(row.get("symbol", "")).strip(),
                    side=str(row.get("side", "")).strip(),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    leverage=leverage,
                    gross_pnl=gross_pnl,
                    fees=fees,
                    net_pnl=net_pnl,
                    pnl_percent=pnl_percent,
                    duration_minutes=duration_minutes,
                    strategy=str(row.get("strategy", "")).strip(),
                    regime=str(row.get("regime", "")).strip(),
                    signal_confidence=_parse_float(row.get("signal_confidence")),
                    close_reason=str(row.get("close_reason", "")).strip(),
                    daily_pnl_after=_parse_float(row.get("daily_pnl_after")),
                    session_drawdown_after=_parse_float(row.get("session_drawdown_after")),
                )

    return trades


def _ensure_trades_table(con: sqlite3.Connection) -> None:
    """Create the `trades` table if missing.

    This matches the current ORM schema closely enough for backfill.
    """
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
    if cur.fetchone():
        return

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            quantity REAL NOT NULL,
            leverage INTEGER NOT NULL,
            gross_pnl REAL NOT NULL,
            fees REAL,
            net_pnl REAL NOT NULL,
            pnl_percent REAL NOT NULL,
            duration_minutes REAL NOT NULL,
            strategy_name TEXT NOT NULL,
            regime_at_entry TEXT NOT NULL,
            signal_confidence REAL,
            exit_reason TEXT NOT NULL,
            open_timestamp DATETIME NOT NULL,
            close_timestamp DATETIME NOT NULL,
            daily_pnl_after REAL,
            session_drawdown_after REAL
        )
        """
    )
    con.commit()


def _trade_exists(con: sqlite3.Connection, trade_id: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT 1 FROM trades WHERE id = ? LIMIT 1", (trade_id,))
    return cur.fetchone() is not None


def _maybe_backup_db(db_path: Path, enabled: bool) -> Optional[Path]:
    if not enabled:
        return None
    if not db_path.exists():
        return None

    ts = _dt.datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}-{ts}.bak")
    shutil.copy2(db_path, backup_path)
    return backup_path


def _resolve_trade_timestamps(
    trade_id: str,
    journal: JournalTrade,
    open_ts_by_id: Dict[str, float],
    close_ts_by_id: Dict[str, float],
) -> Tuple[Optional[_dt.datetime], Optional[_dt.datetime]]:
    open_ts = open_ts_by_id.get(trade_id)
    close_ts = close_ts_by_id.get(trade_id)

    open_dt = _dt_from_epoch(open_ts) if open_ts is not None else None
    close_dt = _dt_from_epoch(close_ts) if close_ts is not None else None

    if open_dt and close_dt:
        return open_dt, close_dt

    # Fallback: infer missing side using duration from journal.
    dur = journal.duration_minutes
    if dur is None or dur <= 0:
        return open_dt, close_dt

    delta = _dt.timedelta(minutes=float(dur))
    if open_dt is not None and close_dt is None:
        return open_dt, open_dt + delta
    if close_dt is not None and open_dt is None:
        return close_dt - delta, close_dt

    return open_dt, close_dt


def _insert_trade(con: sqlite3.Connection, journal: JournalTrade, open_dt: _dt.datetime, close_dt: _dt.datetime) -> None:
    values = (
        journal.trade_id,
        journal.symbol,
        journal.side,
        journal.entry_price,
        journal.exit_price,
        journal.quantity,
        journal.leverage,
        journal.gross_pnl,
        journal.fees,
        journal.net_pnl,
        journal.pnl_percent,
        journal.duration_minutes,
        journal.strategy,
        journal.regime,
        journal.signal_confidence,
        journal.close_reason,
        _dt_to_sqlite_str(open_dt),
        _dt_to_sqlite_str(close_dt),
        journal.daily_pnl_after,
        journal.session_drawdown_after,
    )

    cur = con.cursor()
    cur.execute(INSERT_SQL, values)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Backfill missing trades from CSV journal + logs")
    p.add_argument("--db", default="alphabot_data.db", help="Path to SQLite DB (default: alphabot_data.db)")
    p.add_argument("--logs", default="logs", help="Directory containing JSON logs (default: logs)")
    p.add_argument(
        "--journal",
        action="append",
        help="CSV journal path. Can be specified multiple times. Defaults to data/trade_journal.csv and/or trade_journal.csv if present.",
    )
    p.add_argument("--dry-run", action="store_true", help="Compute inserts but do not write to DB")
    p.add_argument("--no-backup", action="store_true", help="Do not create a DB backup before writing")
    p.add_argument("--limit", type=int, default=0, help="Limit number of inserts (0 = no limit)")
    p.add_argument("--verbose", action="store_true", help="Print each inserted/skipped trade_id")

    args = p.parse_args(argv)

    db_path = Path(args.db)
    log_dir = Path(args.logs)

    journal_paths = _read_journal_paths(args.journal)
    if not journal_paths:
        print("No journal files found. Provide --journal ...")
        return 2

    print(f"DB: {db_path}")
    print(f"Logs: {log_dir}")
    print("Journals:")
    for jp in journal_paths:
        print(f"  - {jp}")

    journal_trades = _load_journal_rows(journal_paths)
    print(f"Loaded {len(journal_trades)} unique journal trade_ids")

    open_ts_by_id, close_ts_by_id = _extract_position_events(log_dir)
    print(f"Log open events: {len(open_ts_by_id)} | close events: {len(close_ts_by_id)}")

    if args.dry_run:
        con = sqlite3.connect(":memory:")
    else:
        if not db_path.exists():
            print(f"DB file not found: {db_path}")
            return 2

        backup = _maybe_backup_db(db_path, enabled=not args.no_backup)
        if backup:
            print(f"DB backup created: {backup}")

        con = sqlite3.connect(db_path)

    try:
        _ensure_trades_table(con)

        inserted = 0
        skipped_existing = 0
        skipped_missing_ts = 0
        skipped_other = 0

        con.execute("BEGIN")

        for trade_id, jt in journal_trades.items():
            if not args.dry_run and _trade_exists(con, trade_id):
                skipped_existing += 1
                if args.verbose:
                    print(f"SKIP existing: {trade_id}")
                continue

            open_dt, close_dt = _resolve_trade_timestamps(trade_id, jt, open_ts_by_id, close_ts_by_id)
            if open_dt is None or close_dt is None:
                skipped_missing_ts += 1
                if args.verbose:
                    print(f"SKIP missing timestamps: {trade_id}")
                continue

            if close_dt < open_dt:
                skipped_other += 1
                if args.verbose:
                    print(f"SKIP invalid timestamps (close<open): {trade_id}")
                continue

            try:
                _insert_trade(con, jt, open_dt, close_dt)
            except sqlite3.IntegrityError:
                skipped_existing += 1
                if args.verbose:
                    print(f"SKIP duplicate (IntegrityError): {trade_id}")
                continue
            except Exception as e:
                skipped_other += 1
                if args.verbose:
                    print(f"SKIP error: {trade_id} ({e})")
                continue

            inserted += 1
            if args.verbose:
                print(f"INSERT: {trade_id}")

            if args.limit and inserted >= args.limit:
                break

        if args.dry_run:
            con.rollback()
        else:
            con.commit()

        print("\nSummary")
        print(f"  inserted: {inserted}")
        print(f"  skipped_existing: {skipped_existing}")
        print(f"  skipped_missing_timestamps: {skipped_missing_ts}")
        print(f"  skipped_other: {skipped_other}")

    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
