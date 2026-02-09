"""Unit tests for circuit breakers."""

import pytest
from decimal import Decimal

from bifs_quant_engine.core.models import RiskState
from bifs_quant_engine.risk.circuit_breakers import (
    CircuitBreakerConfig,
    check_daily_loss_breaker,
    check_drawdown_breaker,
    check_circuit_breakers,
    reset_circuit_breakers,
)


class TestDailyLossBreaker:
    """Tests for daily loss circuit breaker."""
    
    def test_not_triggered_when_profitable(self):
        """Breaker should not trigger when day is profitable."""
        state = RiskState(
            starting_equity=Decimal("100000"),
            current_equity=Decimal("101000"),
            daily_pnl=Decimal("1000"),
            daily_pnl_pct=Decimal("0.01"),  # +1%
        )
        config = CircuitBreakerConfig(daily_loss_limit_pct=Decimal("0.02"))
        
        assert check_daily_loss_breaker(state, config) is False
    
    def test_not_triggered_below_limit(self):
        """Breaker should not trigger when loss is below limit."""
        state = RiskState(
            starting_equity=Decimal("100000"),
            current_equity=Decimal("99000"),
            daily_pnl=Decimal("-1000"),
            daily_pnl_pct=Decimal("-0.01"),  # -1% (below 2% limit)
        )
        config = CircuitBreakerConfig(daily_loss_limit_pct=Decimal("0.02"))
        
        assert check_daily_loss_breaker(state, config) is False
    
    def test_triggered_at_limit(self):
        """Breaker should trigger when loss equals limit."""
        state = RiskState(
            starting_equity=Decimal("100000"),
            current_equity=Decimal("98000"),
            daily_pnl=Decimal("-2000"),
            daily_pnl_pct=Decimal("-0.02"),  # -2% (equals limit)
        )
        config = CircuitBreakerConfig(daily_loss_limit_pct=Decimal("0.02"))
        
        assert check_daily_loss_breaker(state, config) is True
    
    def test_triggered_beyond_limit(self):
        """Breaker should trigger when loss exceeds limit."""
        state = RiskState(
            starting_equity=Decimal("100000"),
            current_equity=Decimal("95000"),
            daily_pnl=Decimal("-5000"),
            daily_pnl_pct=Decimal("-0.05"),  # -5% (exceeds 2% limit)
        )
        config = CircuitBreakerConfig(daily_loss_limit_pct=Decimal("0.02"))
        
        assert check_daily_loss_breaker(state, config) is True
    
    def test_disabled_breaker(self):
        """Disabled breaker should not trigger."""
        state = RiskState(
            starting_equity=Decimal("100000"),
            current_equity=Decimal("50000"),
            daily_pnl=Decimal("-50000"),
            daily_pnl_pct=Decimal("-0.50"),  # -50%
        )
        config = CircuitBreakerConfig(enabled=False)
        
        assert check_daily_loss_breaker(state, config) is False


class TestDrawdownBreaker:
    """Tests for drawdown circuit breaker."""
    
    def test_not_triggered_no_drawdown(self):
        """Breaker should not trigger with no drawdown."""
        state = RiskState(
            high_water_mark=Decimal("100000"),
            current_equity=Decimal("100000"),
            current_drawdown=Decimal("0"),
        )
        config = CircuitBreakerConfig(drawdown_limit_pct=Decimal("0.10"))
        
        assert check_drawdown_breaker(state, config) is False
    
    def test_not_triggered_below_limit(self):
        """Breaker should not trigger when drawdown is below limit."""
        state = RiskState(
            high_water_mark=Decimal("100000"),
            current_equity=Decimal("95000"),
            current_drawdown=Decimal("0.05"),  # 5% (below 10% limit)
        )
        config = CircuitBreakerConfig(drawdown_limit_pct=Decimal("0.10"))
        
        assert check_drawdown_breaker(state, config) is False
    
    def test_triggered_at_limit(self):
        """Breaker should trigger when drawdown equals limit."""
        state = RiskState(
            high_water_mark=Decimal("100000"),
            current_equity=Decimal("90000"),
            current_drawdown=Decimal("0.10"),  # 10% (equals limit)
        )
        config = CircuitBreakerConfig(drawdown_limit_pct=Decimal("0.10"))
        
        assert check_drawdown_breaker(state, config) is True
    
    def test_triggered_beyond_limit(self):
        """Breaker should trigger when drawdown exceeds limit."""
        state = RiskState(
            high_water_mark=Decimal("100000"),
            current_equity=Decimal("80000"),
            current_drawdown=Decimal("0.20"),  # 20% (exceeds 10% limit)
        )
        config = CircuitBreakerConfig(drawdown_limit_pct=Decimal("0.10"))
        
        assert check_drawdown_breaker(state, config) is True


class TestCheckAllBreakers:
    """Tests for combined circuit breaker check."""
    
    def test_updates_state_flags(self):
        """check_circuit_breakers should update state flags."""
        state = RiskState(
            starting_equity=Decimal("100000"),
            current_equity=Decimal("85000"),
            daily_pnl=Decimal("-15000"),
            daily_pnl_pct=Decimal("-0.15"),  # -15% daily
            high_water_mark=Decimal("100000"),
            current_drawdown=Decimal("0.15"),  # 15% drawdown
        )
        config = CircuitBreakerConfig(
            daily_loss_limit_pct=Decimal("0.02"),
            drawdown_limit_pct=Decimal("0.10"),
        )
        
        result = check_circuit_breakers(state, config)
        
        assert result.daily_loss_breaker_triggered is True
        assert result.drawdown_breaker_triggered is True
        assert result.trading_halted is True
    
    def test_no_breakers_triggered(self):
        """Normal state should have no breakers triggered."""
        state = RiskState(
            starting_equity=Decimal("100000"),
            current_equity=Decimal("101000"),
            daily_pnl=Decimal("1000"),
            daily_pnl_pct=Decimal("0.01"),
            high_water_mark=Decimal("101000"),
            current_drawdown=Decimal("0"),
        )
        config = CircuitBreakerConfig()
        
        result = check_circuit_breakers(state, config)
        
        assert result.daily_loss_breaker_triggered is False
        assert result.drawdown_breaker_triggered is False
        assert result.trading_halted is False


class TestResetBreakers:
    """Tests for circuit breaker reset."""
    
    def test_reset_clears_all_flags(self):
        """reset_circuit_breakers should clear all flags."""
        state = RiskState(
            daily_loss_breaker_triggered=True,
            drawdown_breaker_triggered=True,
            volatility_breaker_triggered=True,
        )
        
        result = reset_circuit_breakers(state)
        
        assert result.daily_loss_breaker_triggered is False
        assert result.drawdown_breaker_triggered is False
        assert result.volatility_breaker_triggered is False
        assert result.trading_halted is False
