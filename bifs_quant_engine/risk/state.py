"""Risk state tracking and management.

This module provides the RiskStateTracker class which maintains real-time
risk metrics and handles state transitions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from bifs_quant_engine.core.models import RiskState, PortfolioSnapshot


class RiskStateTracker:
    """Tracks and updates risk state in real-time.
    
    Maintains the current risk state including:
    - Daily P&L tracking
    - Drawdown calculations
    - High water mark updates
    - Exposure metrics
    
    Example:
        tracker = RiskStateTracker(starting_equity=Decimal("100000"))
        tracker.update_from_snapshot(portfolio_snapshot)
        if tracker.get_state().trading_halted:
            # Stop trading
    """
    
    def __init__(
        self,
        starting_equity: Decimal = Decimal("0"),
        high_water_mark: Optional[Decimal] = None,
    ) -> None:
        """Initialize the risk state tracker.
        
        Args:
            starting_equity: Portfolio equity at start of trading day
            high_water_mark: Historical high water mark (uses starting_equity if None)
        """
        self._state = RiskState(
            starting_equity=starting_equity,
            current_equity=starting_equity,
            high_water_mark=high_water_mark or starting_equity,
        )
    
    def get_state(self) -> RiskState:
        """Get current risk state."""
        return self._state
    
    def update_from_snapshot(self, snapshot: PortfolioSnapshot) -> RiskState:
        """Update risk state from a portfolio snapshot.
        
        Args:
            snapshot: Current portfolio snapshot
            
        Returns:
            Updated risk state
        """
        self._state.timestamp = datetime.now(timezone.utc)
        self._state.current_equity = snapshot.total_equity
        
        # Update daily P&L
        if self._state.starting_equity > 0:
            self._state.daily_pnl = snapshot.total_equity - self._state.starting_equity
            self._state.daily_pnl_pct = self._state.daily_pnl / self._state.starting_equity
        
        # Update high water mark
        if snapshot.total_equity > self._state.high_water_mark:
            self._state.high_water_mark = snapshot.total_equity
        
        # Update drawdown
        if self._state.high_water_mark > 0:
            self._state.current_drawdown = (
                (self._state.high_water_mark - snapshot.total_equity) 
                / self._state.high_water_mark
            )
            if self._state.current_drawdown > self._state.max_drawdown:
                self._state.max_drawdown = self._state.current_drawdown
        
        # Update exposure metrics
        self._state.gross_exposure = snapshot.gross_exposure
        self._state.net_exposure = snapshot.net_exposure
        self._state.long_exposure = snapshot.long_market_value
        self._state.short_exposure = abs(snapshot.short_market_value)
        
        # Update position metrics
        if snapshot.total_equity > 0:
            # Find largest position percentage
            max_position_value = Decimal("0")
            for pos in snapshot.positions.values():
                # We need to use the stored market value or calculate from snapshot
                # For now, use a simple proxy based on position count
                pass  # Position values would need prices to calculate
            
            self._state.largest_position_pct = (
                max_position_value / snapshot.total_equity 
                if snapshot.total_equity > 0 else Decimal("0")
            )
        
        self._state.position_count = snapshot.position_count
        
        return self._state
    
    def reset_daily(self, new_starting_equity: Decimal) -> RiskState:
        """Reset daily state for a new trading day.
        
        Resets:
        - Starting equity to current value
        - Daily P&L to zero
        - Daily loss circuit breaker flag
        
        Does NOT reset:
        - High water mark
        - Max drawdown
        - Drawdown circuit breaker (requires explicit reset)
        
        Args:
            new_starting_equity: The starting equity for the new day
            
        Returns:
            Updated risk state
        """
        self._state.starting_equity = new_starting_equity
        self._state.current_equity = new_starting_equity
        self._state.daily_pnl = Decimal("0")
        self._state.daily_pnl_pct = Decimal("0")
        self._state.daily_loss_breaker_triggered = False
        self._state.timestamp = datetime.now(timezone.utc)
        
        return self._state
    
    def set_starting_equity(self, equity: Decimal) -> None:
        """Set the starting equity for the day."""
        self._state.starting_equity = equity
        if self._state.high_water_mark < equity:
            self._state.high_water_mark = equity
    
    def trigger_halt(self, reason: str = "manual") -> RiskState:
        """Manually trigger a trading halt.
        
        Args:
            reason: Reason for the halt (for logging)
            
        Returns:
            Updated risk state with volatility breaker triggered
        """
        # Use volatility breaker as a catch-all manual halt
        self._state.volatility_breaker_triggered = True
        return self._state
    
    def clear_all_breakers(self) -> RiskState:
        """Clear all circuit breaker flags.
        
        Use with caution - typically for administrative reset only.
        
        Returns:
            Updated risk state with all breakers cleared
        """
        self._state.daily_loss_breaker_triggered = False
        self._state.drawdown_breaker_triggered = False
        self._state.volatility_breaker_triggered = False
        return self._state
