from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd

from .base import Strategy


class ExampleEqualWeightStrategy(Strategy):
    """Very small example strategy for new developers.

    - Accepts a `universe` list on construction
    - On each rebalance returns equal weights across the universe
    - Returns 0 for missing symbols

    Use this as a template for new strategies.
    """

    def __init__(self, universe: List[str], name: str = "example_equal_weight"):
        super().__init__(name)
        self.universe = universe

    def generate_target_weights(
        self, date: datetime, price_history: pd.DataFrame
    ) -> Dict[str, float]:
        # Minimal contract: return weights for all symbols in the universe
        weights: Dict[str, float] = {s: 0.0 for s in self.universe}

        # If we have no data, stay in cash
        if price_history is None or price_history.empty:
            return weights

        # Simple equal-weight across all available symbols
        n = len(self.universe)
        if n == 0:
            return weights

        w = 1.0 / n
        for s in self.universe:
            weights[s] = w

        return weights
