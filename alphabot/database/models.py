"""
AlphaBot Database Models — SQLAlchemy ORM definitions.
Tables: positions, trades (closed), signals_log.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class PositionRecord(Base):
    """Persisted position — for crash recovery."""
    __tablename__ = "positions"

    id = Column(String(64), primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(5), nullable=False)  # LONG / SHORT
    status = Column(String(10), nullable=False, default="OPEN")  # OPEN / PARTIAL / CLOSED
    size_usdt = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    leverage = Column(Integer, nullable=False, default=5)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=False)
    tp1_price = Column(Float, nullable=True)
    tp2_price = Column(Float, nullable=True)
    trailing_stop_price = Column(Float, nullable=True)
    trailing_stop_active = Column(Integer, default=0)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    fees_paid = Column(Float, default=0.0)
    strategy_name = Column(String(50), nullable=False)
    regime_at_entry = Column(String(20), nullable=False)
    signal_confidence = Column(Float, nullable=True)
    open_timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    close_timestamp = Column(DateTime, nullable=True)
    exit_reason = Column(String(30), nullable=True)
    order_ids = Column(Text, nullable=True)  # JSON list of exchange order IDs
    tp_order_ids = Column(Text, nullable=True)  # JSON list of TP order IDs
    sl_order_ids = Column(Text, nullable=True)  # JSON list of SL order IDs


class TradeRecord(Base):
    """Closed trade record for PnL tracking and journal."""
    __tablename__ = "trades"

    id = Column(String(64), primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(5), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    leverage = Column(Integer, nullable=False)
    gross_pnl = Column(Float, nullable=False)
    fees = Column(Float, default=0.0)
    net_pnl = Column(Float, nullable=False)
    pnl_percent = Column(Float, nullable=False)
    duration_minutes = Column(Float, nullable=False)
    strategy_name = Column(String(50), nullable=False)
    regime_at_entry = Column(String(20), nullable=False)
    signal_confidence = Column(Float, nullable=True)
    exit_reason = Column(String(30), nullable=False)
    open_timestamp = Column(DateTime, nullable=False)
    close_timestamp = Column(DateTime, nullable=False)
    daily_pnl_after = Column(Float, nullable=True)
    session_drawdown_after = Column(Float, nullable=True)


class SignalLog(Base):
    """Audit trail for every signal generated."""
    __tablename__ = "signals_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    symbol = Column(String(20), nullable=False)
    direction = Column(String(5), nullable=False)
    confidence = Column(Float, nullable=False)
    strategy_name = Column(String(50), nullable=False)
    regime = Column(String(20), nullable=False)
    entry_price = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)
    tp_price = Column(Float, nullable=True)
    approved = Column(Integer, nullable=False, default=0)  # 0 = rejected, 1 = approved
    rejection_reason = Column(String(200), nullable=True)


class BotState(Base):
    """Simple key-value state store for persistent bot runtime state."""
    __tablename__ = "bot_state"

    key = Column(String(50), primary_key=True)
    value = Column(String(200), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
