"""Base strategy interface for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict

import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for all strategies.

    The engine/backtester will:
    - call `generate_target_weights` on each rebalance date
    - pass in the full price history up to that date
    - then take your target weights and turn them into orders via Portfolio

    Every concrete strategy (momentum, mean reversion, equal-weight, etc.)
    should inherit from this class.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        date : datetime
            Rebalance date (end-of-day).
        price_history : pd.DataFrame
            Price history for the entire universe up to `date`.
            index   = DateTimeIndex
            columns = symbols (e.g. "SPY", "QQQ")
            values  = prices (usually Close)

        Returns
        -------
        Dict[str, float]
            Target portfolio weights for each symbol.
            Example:
                {"SPY": 0.5, "QQQ": 0.5, "TLT": 0.0}
        """
        ...

    