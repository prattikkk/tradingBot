"""
AlphaBot Base Strategy — Abstract base class that all strategies implement.
"""

from __future__ import annotations

import abc
from typing import Optional

import pandas as pd

from alphabot.strategies.signal import Signal


class BaseStrategy(abc.ABC):
    """
    Abstract strategy interface.
    Each sub-strategy must implement `generate_signal`.
    """

    name: str = "base"

    @abc.abstractmethod
    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str,
        timeframe: str,
        higher_tf_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        """
        Evaluate the DataFrame and produce a Signal or None.

        Args:
            symbol: Trading pair (e.g. BTCUSDT)
            df: OHLCV + indicators DataFrame (closed candles only)
            regime: Current market regime string
            timeframe: Primary timeframe
            higher_tf_df: Optional higher-timeframe DataFrame for confirmation

        Returns:
            Signal if conditions are met, None otherwise.
        """
        ...
