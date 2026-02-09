"""Strategy orchestrator for multi-strategy support.

Manages multiple strategies with:
- Capital allocation (equal weight, risk parity, custom)
- Signal aggregation into net positions
- Strategy-level P&L tracking
- Position limit enforcement per strategy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Callable, Dict, List, Optional
import logging

import pandas as pd

from bifs_quant_engine.core.models import Fill, Position
from bifs_quant_engine.strategies.strategy_protocol import (
    StrategyProtocol, 
    Signal, 
    SignalType,
)

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Capital allocation method across strategies."""
    EQUAL = "equal"  # Equal weight to all strategies
    RISK_PARITY = "risk_parity"  # Allocate inversely to volatility
    CUSTOM = "custom"  # User-defined weights


@dataclass
class StrategyAllocation:
    """Allocation for a single strategy.
    
    Attributes:
        strategy: The strategy instance
        weight: Capital allocation weight (0.0 to 1.0)
        max_position_pct: Maximum position size for this strategy
        active: Whether strategy is currently active
    """
    strategy: StrategyProtocol
    weight: Decimal = Decimal("1.0")
    max_position_pct: Decimal = Decimal("0.10")
    active: bool = True


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy.
    
    Attributes:
        strategy_name: Strategy identifier
        pnl: Total P&L
        pnl_today: Today's P&L
        positions: Current positions
        fills_count: Number of fills
    """
    strategy_name: str
    pnl: Decimal = Decimal("0")
    pnl_today: Decimal = Decimal("0")
    positions: Dict[str, Position] = field(default_factory=dict)
    fills_count: int = 0


class StrategyOrchestrator:
    """Orchestrates multiple trading strategies.
    
    Features:
    - Register/deregister strategies
    - Capital allocation (equal weight, risk parity, custom)
    - Aggregate signals into net positions
    - Strategy-level P&L tracking
    - Position limit enforcement per strategy
    
    Example:
        orchestrator = StrategyOrchestrator(total_capital=Decimal("1000000"))
        orchestrator.register_strategy(momentum_strategy, weight=0.5)
        orchestrator.register_strategy(mean_reversion_strategy, weight=0.5)
        
        signals = orchestrator.generate_signals(market_data, current_positions)
        weights = orchestrator.aggregate_target_weights(date, price_history)
    """
    
    def __init__(
        self,
        total_capital: Decimal = Decimal("1000000"),
        allocation_method: AllocationMethod = AllocationMethod.EQUAL,
        max_gross_exposure: Decimal = Decimal("2.0"),
    ) -> None:
        """Initialize orchestrator.
        
        Args:
            total_capital: Total capital to allocate
            allocation_method: How to allocate capital
            max_gross_exposure: Maximum gross exposure (long + |short|) / capital
        """
        self._total_capital = total_capital
        self._allocation_method = allocation_method
        self._max_gross_exposure = max_gross_exposure
        
        # Strategy tracking
        self._strategies: Dict[str, StrategyAllocation] = {}
        self._metrics: Dict[str, StrategyMetrics] = {}
        
        # Position tracking per strategy
        self._strategy_positions: Dict[str, Dict[str, Position]] = {}
        
    def register_strategy(
        self,
        strategy: StrategyProtocol,
        weight: Optional[Decimal] = None,
        max_position_pct: Decimal = Decimal("0.10"),
    ) -> None:
        """Register a strategy.
        
        Args:
            strategy: Strategy to register
            weight: Capital allocation weight (auto-calculated if None)
            max_position_pct: Maximum position size for this strategy
        """
        name = strategy.name
        
        if name in self._strategies:
            logger.warning(f"Strategy {name} already registered, replacing")
        
        self._strategies[name] = StrategyAllocation(
            strategy=strategy,
            weight=weight or Decimal("1"),
            max_position_pct=max_position_pct,
        )
        
        self._metrics[name] = StrategyMetrics(strategy_name=name)
        self._strategy_positions[name] = {}
        
        # Rebalance weights if using equal allocation
        if self._allocation_method == AllocationMethod.EQUAL:
            self._rebalance_weights()
        
        logger.info(f"Registered strategy: {name}")
    
    def deregister_strategy(self, name: str) -> bool:
        """Remove a strategy.
        
        Args:
            name: Strategy name to remove
            
        Returns:
            True if removed, False if not found
        """
        if name not in self._strategies:
            return False
        
        del self._strategies[name]
        del self._metrics[name]
        del self._strategy_positions[name]
        
        if self._allocation_method == AllocationMethod.EQUAL:
            self._rebalance_weights()
        
        logger.info(f"Deregistered strategy: {name}")
        return True
    
    def set_strategy_active(self, name: str, active: bool) -> None:
        """Enable or disable a strategy.
        
        Args:
            name: Strategy name
            active: Whether strategy should be active
        """
        if name in self._strategies:
            self._strategies[name].active = active
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: Dict[str, Decimal],
    ) -> List[Signal]:
        """Generate signals from all active strategies.
        
        Args:
            market_data: Historical price data
            current_positions: Current portfolio positions
            
        Returns:
            Aggregated list of signals from all strategies
        """
        all_signals = []
        
        for name, allocation in self._strategies.items():
            if not allocation.active:
                continue
            
            strategy = allocation.strategy
            
            try:
                # Get strategy's current positions
                strategy_positions = self._strategy_positions.get(name, {})
                strategy_pos_qty = {
                    sym: pos.quantity 
                    for sym, pos in strategy_positions.items()
                }
                
                signals = strategy.generate_signals(
                    market_data,
                    strategy_pos_qty,
                )
                
                # Scale signals by strategy weight
                for signal in signals:
                    signal.weight *= float(allocation.weight)
                    all_signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signals from {name}: {e}")
        
        return all_signals
    
    def aggregate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """Aggregate target weights from all active strategies.
        
        Args:
            date: Current date
            price_history: Historical prices
            
        Returns:
            Combined target weights (can exceed 1.0 for leverage)
        """
        combined_weights: Dict[str, float] = {}
        
        for name, allocation in self._strategies.items():
            if not allocation.active:
                continue
            
            strategy = allocation.strategy
            
            try:
                weights = strategy.generate_target_weights(date, price_history)
                
                # Scale by strategy allocation
                for symbol, weight in weights.items():
                    scaled = weight * float(allocation.weight)
                    
                    # Apply position limit
                    max_weight = float(allocation.max_position_pct)
                    scaled = max(-max_weight, min(max_weight, scaled))
                    
                    if symbol in combined_weights:
                        combined_weights[symbol] += scaled
                    else:
                        combined_weights[symbol] = scaled
                        
            except Exception as e:
                logger.error(f"Error getting weights from {name}: {e}")
        
        # Apply gross exposure limit
        combined_weights = self._apply_exposure_limit(combined_weights)
        
        return combined_weights
    
    def on_fill(self, fill: Fill, strategy_name: str) -> None:
        """Handle fill for a specific strategy.
        
        Args:
            fill: Fill details
            strategy_name: Which strategy the fill belongs to
        """
        if strategy_name not in self._strategies:
            logger.warning(f"Fill for unknown strategy: {strategy_name}")
            return
        
        # Update strategy metrics
        metrics = self._metrics[strategy_name]
        metrics.fills_count += 1
        
        # Update strategy's positions
        self._update_strategy_position(strategy_name, fill)
        
        # Notify strategy
        self._strategies[strategy_name].strategy.on_fill(fill)
    
    def get_strategy_metrics(self, name: str) -> Optional[StrategyMetrics]:
        """Get metrics for a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Metrics or None if not found
        """
        return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies."""
        return dict(self._metrics)
    
    def get_active_strategies(self) -> List[str]:
        """Get names of active strategies."""
        return [
            name for name, alloc in self._strategies.items()
            if alloc.active
        ]
    
    def get_total_weight(self) -> Decimal:
        """Get sum of all strategy weights."""
        return sum(
            alloc.weight 
            for alloc in self._strategies.values()
            if alloc.active
        )
    
    def reset_daily(self) -> None:
        """Reset daily metrics for all strategies."""
        for metrics in self._metrics.values():
            metrics.pnl_today = Decimal("0")
    
    def _rebalance_weights(self) -> None:
        """Rebalance weights equally across active strategies."""
        active = [a for a in self._strategies.values() if a.active]
        if not active:
            return
        
        equal_weight = Decimal("1") / Decimal(len(active))
        for alloc in active:
            alloc.weight = equal_weight
    
    def _apply_exposure_limit(
        self, 
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply gross exposure limit to weights.
        
        Args:
            weights: Target weights
            
        Returns:
            Scaled weights respecting exposure limit
        """
        gross_exposure = sum(abs(w) for w in weights.values())
        max_exposure = float(self._max_gross_exposure)
        
        if gross_exposure > max_exposure:
            scale = max_exposure / gross_exposure
            return {s: w * scale for s, w in weights.items()}
        
        return weights
    
    def _update_strategy_position(
        self, 
        strategy_name: str, 
        fill: Fill,
    ) -> None:
        """Update a strategy's position after a fill.
        
        Args:
            strategy_name: Strategy identifier
            fill: Fill details
        """
        positions = self._strategy_positions.setdefault(strategy_name, {})
        current = positions.get(fill.symbol)
        
        # Calculate new quantity
        if fill.side.value in ("buy", "cover"):
            delta = fill.quantity
        else:
            delta = -fill.quantity
        
        current_qty = current.quantity if current else 0
        new_qty = current_qty + delta
        
        if new_qty == 0:
            if fill.symbol in positions:
                del positions[fill.symbol]
        else:
            # Update or create position
            current_cost = current.avg_cost if current else Decimal("0")
            
            if (current_qty >= 0 and delta > 0) or (current_qty <= 0 and delta < 0):
                # Adding to position
                total_cost = abs(current_qty) * current_cost + abs(delta) * fill.price
                new_cost = total_cost / abs(new_qty)
            else:
                new_cost = current_cost
            
            positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=new_qty,
                avg_cost=new_cost,
            )
        
        # Update metrics
        self._metrics[strategy_name].positions = positions
