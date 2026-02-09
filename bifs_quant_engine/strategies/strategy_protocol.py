"""Strategy protocol definitions.

Defines the interface for trading strategies that can be used with
the backtester and orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Protocol
from uuid import UUID, uuid4

import pandas as pd

from bifs_quant_engine.core.enums import OrderSide
from bifs_quant_engine.core.models import Fill, PortfolioSnapshot


class SignalType(Enum):
    """Type of trading signal."""
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal from a strategy.
    
    Attributes:
        strategy_id: Identifier of the generating strategy
        symbol: Target symbol
        signal_type: Type of signal (long/short/close)
        weight: Target weight (0.0 to 1.0 for longs, -1.0 to 0.0 for shorts)
        confidence: Signal confidence (0.0 to 1.0)
        timestamp: When signal was generated
        metadata: Optional additional data
    """
    strategy_id: str
    symbol: str
    signal_type: SignalType
    weight: float = 0.0
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now())
    metadata: Dict = field(default_factory=dict)
    signal_id: UUID = field(default_factory=uuid4)


class StrategyProtocol(Protocol):
    """Protocol defining the strategy interface.
    
    Strategies must implement this interface to be compatible with
    the backtester and orchestrator.
    """
    
    @property
    def name(self) -> str:
        """Unique strategy identifier."""
        ...
    
    @property
    def universe(self) -> List[str]:
        """List of symbols this strategy trades."""
        ...
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: Dict[str, Decimal],
    ) -> List[Signal]:
        """Generate trading signals from market data.
        
        Args:
            market_data: Historical price data
            current_positions: Current position quantities by symbol
            
        Returns:
            List of trading signals
        """
        ...
    
    def generate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """Generate target portfolio weights.
        
        This is the simpler interface for backtesting.
        
        Args:
            date: Current date
            price_history: Historical prices up to date
            
        Returns:
            Dict mapping symbol to target weight
        """
        ...
    
    def on_fill(self, fill: Fill) -> None:
        """Handle fill notification.
        
        Called when an order is filled, allows strategy to track execution.
        
        Args:
            fill: Fill details
        """
        ...


class BaseStrategy:
    """Base class for strategies implementing StrategyProtocol.
    
    Provides default implementations and common utility methods.
    """
    
    def __init__(self, name: str, universe: Optional[List[str]] = None) -> None:
        """Initialize strategy.
        
        Args:
            name: Strategy identifier
            universe: List of symbols to trade
        """
        self._name = name
        self._universe = universe or []
        self._fills: List[Fill] = []
        self._pnl = Decimal("0")
    
    @property
    def name(self) -> str:
        """Strategy identifier."""
        return self._name
    
    @property
    def universe(self) -> List[str]:
        """Symbols traded by this strategy."""
        return self._universe
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: Dict[str, Decimal],
    ) -> List[Signal]:
        """Generate signals - override in subclass."""
        return []
    
    def generate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """Generate target weights - override in subclass."""
        return {}
    
    def on_fill(self, fill: Fill) -> None:
        """Handle fill notification."""
        self._fills.append(fill)
    
    def get_fills(self) -> List[Fill]:
        """Get all fills for this strategy."""
        return list(self._fills)
    
    def reset(self) -> None:
        """Reset strategy state."""
        self._fills = []
        self._pnl = Decimal("0")
