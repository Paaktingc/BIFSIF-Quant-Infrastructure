"""Market-neutral strategies package.

Provides strategies for market-neutral trading:
- Pairs trading
- Statistical arbitrage
- Cointegration utilities
"""

from .cointegration import (
    check_cointegration,
    calculate_hedge_ratio,
    calculate_spread,
    calculate_zscore,
)
from .pairs_trading import PairsTradingStrategy
from .stat_arb import StatisticalArbitrageStrategy

__all__ = [
    "check_cointegration",
    "calculate_hedge_ratio",
    "calculate_spread",
    "calculate_zscore",
    "PairsTradingStrategy",
    "StatisticalArbitrageStrategy",
]
