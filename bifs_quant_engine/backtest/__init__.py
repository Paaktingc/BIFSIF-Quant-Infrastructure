"""Backtest package.

Provides production-grade backtesting with realistic transaction costs,
slippage modeling, and performance analytics.
"""

from .realistic_backtest import RealisticBacktest, BacktestConfig, BacktestResult
from .performance_analytics import PerformanceAnalytics

__all__ = [
    "RealisticBacktest",
    "BacktestConfig",
    "BacktestResult",
    "PerformanceAnalytics",
]
