"""
AlphaBot Data Models — Pydantic models for Candle, Ticker, OrderBook.
All price / monetary fields use Decimal for precision.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field


class Candle(BaseModel):
    """Single OHLCV candlestick."""
    symbol: str
    timeframe: str
    open_time: datetime.datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: datetime.datetime
    is_closed: bool = False

    @property
    def ohlcv_tuple(self):
        return (
            float(self.open),
            float(self.high),
            float(self.low),
            float(self.close),
            float(self.volume),
        )


class Ticker(BaseModel):
    """Real-time price ticker."""
    symbol: str
    price: Decimal
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class OrderBookLevel(BaseModel):
    """Single level in the order book."""
    price: Decimal
    quantity: Decimal


class OrderBook(BaseModel):
    """Top-N order book snapshot."""
    symbol: str
    bids: List[OrderBookLevel] = []
    asks: List[OrderBookLevel] = []
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class FundingRate(BaseModel):
    """Funding rate data."""
    symbol: str
    rate: Decimal
    next_funding_time: datetime.datetime
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
