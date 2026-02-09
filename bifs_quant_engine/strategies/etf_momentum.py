from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd

from .base import Strategy


class ETFMomentumStrategy(Strategy):
    """
    Simple ETF momentum strategy:

    - Look back `lookback_days` trading days
    - Compute total return over that window
    - Rank ETFs by performance
    - Select top `top_n` ETFs with positive return
    - Equal-weight those winners
    """

    def __init__(
        self,
        universe: List[str],
        lookback_days: int = 126,
        top_n: int = 3,
        name: str = "etf_momentum",
    ):
        super().__init__(name)
        self.universe = universe
        self.lookback_days = lookback_days
        self.top_n = top_n

    def generate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> Dict[str, float]:
        # price_history: all prices up to `date` (Backtest passes this in)

        # Not enough data → stay in cash
        if len(price_history) < self.lookback_days:
            return {s: 0.0 for s in self.universe}

        # Ensure we only use data up to `date`
        price_history = price_history.loc[price_history.index <= date]

        # Take last `lookback_days` rows
        window = price_history.iloc[-self.lookback_days:]

        start_prices = window.iloc[0]
        end_prices = window.iloc[-1]

        total_returns = end_prices / start_prices - 1.0

        # Rank by performance, best at top
        ranked = total_returns.sort_values(ascending=False)

        # Winners: positive return, top_n
        winners = ranked[ranked > 0].head(self.top_n).index.tolist()

        # Build weight dict
        weights: Dict[str, float] = {s: 0.0 for s in self.universe}

        if not winners:
            return weights

        equal_w = 1.0 / len(winners)
        for s in winners:
            weights[s] = equal_w

        return weights
    