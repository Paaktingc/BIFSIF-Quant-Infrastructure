"""Circuit breaker implementations.

Circuit breakers halt trading when risk thresholds are exceeded to
prevent catastrophic losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from bifs_quant_engine.core.models import RiskState


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers.
    
    Attributes:
        daily_loss_limit_pct: Halt when daily P&L < -X% (e.g., 0.02 = 2%)
        drawdown_limit_pct: Halt when drawdown > Y% (e.g., 0.10 = 10%)
        enabled: Whether circuit breakers are active
    """
    daily_loss_limit_pct: Decimal = Decimal("0.02")
    drawdown_limit_pct: Decimal = Decimal("0.10")
    enabled: bool = True


def check_daily_loss_breaker(
    risk_state: RiskState,
    config: CircuitBreakerConfig,
) -> bool:
    """Check if daily loss circuit breaker should trigger.
    
    Args:
        risk_state: Current risk state with daily P&L
        config: Circuit breaker configuration
        
    Returns:
        True if breaker should trigger (daily loss exceeded)
    """
    if not config.enabled:
        return False
    
    if risk_state.starting_equity <= 0:
        return False
    
    # Calculate daily loss percentage
    daily_loss_pct = -risk_state.daily_pnl_pct
    
    # Trigger if loss exceeds limit
    return daily_loss_pct >= config.daily_loss_limit_pct


def check_drawdown_breaker(
    risk_state: RiskState,
    config: CircuitBreakerConfig,
) -> bool:
    """Check if drawdown circuit breaker should trigger.
    
    Args:
        risk_state: Current risk state with drawdown info
        config: Circuit breaker configuration
        
    Returns:
        True if breaker should trigger (drawdown exceeded)
    """
    if not config.enabled:
        return False
    
    if risk_state.high_water_mark <= 0:
        return False
    
    # Current drawdown is already calculated in risk_state
    return risk_state.current_drawdown >= config.drawdown_limit_pct


def check_circuit_breakers(
    risk_state: RiskState,
    config: CircuitBreakerConfig,
) -> RiskState:
    """Check all circuit breakers and update risk state.
    
    This function checks each circuit breaker and updates the corresponding
    flags in the risk state. Once a breaker is triggered, it stays triggered
    until explicitly reset.
    
    Args:
        risk_state: Current risk state (will be modified)
        config: Circuit breaker configuration
        
    Returns:
        Updated risk state with breaker flags set
    """
    if not config.enabled:
        return risk_state
    
    # Check daily loss breaker
    if check_daily_loss_breaker(risk_state, config):
        risk_state.daily_loss_breaker_triggered = True
    
    # Check drawdown breaker
    if check_drawdown_breaker(risk_state, config):
        risk_state.drawdown_breaker_triggered = True
    
    return risk_state


def reset_circuit_breakers(risk_state: RiskState) -> RiskState:
    """Reset all circuit breaker flags.
    
    Typically called at the start of a new trading day.
    
    Args:
        risk_state: Risk state to reset
        
    Returns:
        Risk state with all breaker flags cleared
    """
    risk_state.daily_loss_breaker_triggered = False
    risk_state.drawdown_breaker_triggered = False
    risk_state.volatility_breaker_triggered = False
    return risk_state
