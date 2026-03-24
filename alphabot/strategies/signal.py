"""
AlphaBot Signal Model — Pydantic model for trading signals + confidence scoring.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class Signal(BaseModel):
    """
    Trading signal emitted by a strategy.
    Includes entry, SL, TP levels, confidence score, and metadata.
    """
    symbol: str
    direction: SignalDirection
    confidence: float = Field(ge=0, le=100)
    entry_price: Decimal
    stop_loss: Decimal
    take_profit_1: Decimal
    take_profit_2: Optional[Decimal] = None
    strategy_name: str
    regime: str
    timeframe: str
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))

    # Scoring breakdown
    regime_alignment_score: float = 0.0
    primary_indicator_score: float = 0.0
    confirmation_score: float = 0.0
    volume_score: float = 0.0
    higher_tf_score: float = 0.0

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate R:R ratio."""
        sl_distance = abs(float(self.entry_price) - float(self.stop_loss))
        tp_distance = abs(float(self.take_profit_1) - float(self.entry_price))
        if sl_distance == 0:
            return 0.0
        return tp_distance / sl_distance

    @property
    def sl_distance_pct(self) -> float:
        """Stop loss distance as percentage of entry price."""
        return abs(float(self.entry_price) - float(self.stop_loss)) / float(self.entry_price) * 100

    def passes_minimum_confidence(self, min_conf: float) -> bool:
        return self.confidence >= min_conf

    def passes_minimum_rr(self, min_rr: float) -> bool:
        return self.risk_reward_ratio >= min_rr


def compute_confidence(
    regime_alignment: float,
    primary_indicator: float,
    confirmation: float,
    volume: float,
    higher_tf: float,
    weights: dict | None = None,
) -> float:
    """
    Compute signal confidence score (0-100).
    Weights from PRD:
      regime_alignment: 30
      primary_indicator: 25
      confirmation: 20
      volume: 15
      higher_tf: 10
    Each input should be 0.0 to 1.0 (fraction of alignment).
    """
    w = weights or {
        "regime": 30,
        "primary": 25,
        "confirmation": 20,
        "volume": 15,
        "higher_tf": 10,
    }
    score = (
        regime_alignment * w["regime"]
        + primary_indicator * w["primary"]
        + confirmation * w["confirmation"]
        + volume * w["volume"]
        + higher_tf * w["higher_tf"]
    )
    return min(max(score, 0), 100)
