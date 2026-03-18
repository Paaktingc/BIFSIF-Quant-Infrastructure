"""Strategy protocol and orchestration.

Provides multi-strategy support with:
- StrategyProtocol for type hints
- StrategyOrchestrator for managing multiple strategies
- Capital allocation and signal aggregation
"""

from .strategy_protocol import StrategyProtocol, Signal
from .orchestrator import StrategyOrchestrator, AllocationMethod
from .sp500_momentum import SP500MomentumStrategy, MomentumConfig

__all__ = [
    "StrategyProtocol",
    "Signal",
    "StrategyOrchestrator",
    "AllocationMethod",
    "SP500MomentumStrategy",
    "MomentumConfig",
]
