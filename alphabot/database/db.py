"""
AlphaBot Database — SQLite connection and helpers.
All writes are synchronous and wrapped in transactions.
On startup, creates tables if they don't exist.
On restart, reads open positions for recovery.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from alphabot.database.models import Base, PositionRecord, TradeRecord, SignalLog

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "alphabot_data.db"


class Database:
    """Thin wrapper around SQLite via SQLAlchemy."""

    def __init__(self, db_path: str | Path | None = None):
        path = db_path or _DB_PATH
        self.engine = create_engine(
            f"sqlite:///{path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self._SessionFactory = sessionmaker(bind=self.engine)

    def session(self) -> Session:
        return self._SessionFactory()

    # ---- Position CRUD ----
    def save_position(self, pos: PositionRecord) -> None:
        with self.session() as s:
            s.merge(pos)
            s.commit()

    def get_open_positions(self) -> List[PositionRecord]:
        with self.session() as s:
            return (
                s.query(PositionRecord)
                .filter(PositionRecord.status.in_(["OPEN", "PARTIAL"]))
                .all()
            )

    def close_position(self, position_id: str, exit_price: float,
                       realized_pnl: float, fees: float,
                       exit_reason: str) -> None:
        with self.session() as s:
            pos = s.query(PositionRecord).filter_by(id=position_id).first()
            if pos:
                pos.status = "CLOSED"
                pos.realized_pnl = realized_pnl
                pos.fees_paid = fees
                pos.exit_reason = exit_reason
                pos.current_price = exit_price
                pos.close_timestamp = datetime.datetime.now(datetime.UTC)
                s.commit()

    def update_position(self, position_id: str, **kwargs) -> None:
        with self.session() as s:
            pos = s.query(PositionRecord).filter_by(id=position_id).first()
            if pos:
                for k, v in kwargs.items():
                    if hasattr(pos, k):
                        setattr(pos, k, v)
                s.commit()

    # ---- Trade Journal ----
    def save_trade(self, trade: TradeRecord) -> None:
        with self.session() as s:
            s.merge(trade)
            s.commit()

    def get_trades(self, limit: int = 100) -> List[TradeRecord]:
        with self.session() as s:
            return (
                s.query(TradeRecord)
                .order_by(TradeRecord.close_timestamp.desc())
                .limit(limit)
                .all()
            )

    def get_trades_since(self, since: datetime.datetime) -> List[TradeRecord]:
        with self.session() as s:
            return (
                s.query(TradeRecord)
                .filter(TradeRecord.close_timestamp >= since)
                .all()
            )

    # ---- Signal Audit Log ----
    def log_signal(self, sig: SignalLog) -> None:
        with self.session() as s:
            s.add(sig)
            s.commit()

    # ---- Aggregations ----
    def total_trades(self) -> int:
        with self.session() as s:
            return s.query(TradeRecord).count()

    def winning_trades(self) -> int:
        with self.session() as s:
            return s.query(TradeRecord).filter(TradeRecord.net_pnl > 0).count()

    def total_pnl(self) -> float:
        with self.session() as s:
            result = s.query(TradeRecord.net_pnl).all()
            return sum(r[0] for r in result) if result else 0.0
