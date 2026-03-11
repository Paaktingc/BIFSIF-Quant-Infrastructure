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
        use_trend_filter: bool = True,
        trend_lookback: int = 200,
        use_vol_weighting: bool = True,
    ):
        super().__init__(name)
        self.universe = universe
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.use_trend_filter = use_trend_filter
        self.trend_lookback = trend_lookback
        self.use_vol_weighting = use_vol_weighting

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

        # Absolute momentum filter: only keep winners above their SMA
        if self.use_trend_filter and len(price_history) >= self.trend_lookback:
            sma = price_history.iloc[-self.trend_lookback:].mean()
            winners = [s for s in winners if end_prices[s] > sma[s]]

        # Build weight dict
        weights: Dict[str, float] = {s: 0.0 for s in self.universe}

        if not winners:
            return weights

        # Inverse-volatility weighting for better risk-adjusted returns
        if self.use_vol_weighting:
            vols = window.pct_change().std()
            inv_vols = {}
            for s in winners:
                v = vols.get(s, 0)
                if v > 0:
                    inv_vols[s] = 1.0 / v
            total_inv = sum(inv_vols.values())
            if total_inv > 0:
                for s in winners:
                    weights[s] = inv_vols.get(s, 0) / total_inv
            else:
                equal_w = 1.0 / len(winners)
                for s in winners:
                    weights[s] = equal_w
        else:
            equal_w = 1.0 / len(winners)
            for s in winners:
                weights[s] = equal_w

        return weights
    