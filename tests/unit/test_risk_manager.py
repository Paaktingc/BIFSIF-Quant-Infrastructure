"""Unit tests for ProductionRiskManager."""

import pytest
from decimal import Decimal

from bifs_quant_engine.core.enums import OrderSide, RiskAction
from bifs_quant_engine.core.models import Order, Position, PortfolioSnapshot
from bifs_quant_engine.risk.limits import PositionLimits, ExposureLimits
from bifs_quant_engine.risk.circuit_breakers import CircuitBreakerConfig
from bifs_quant_engine.risk.manager import ProductionRiskManager


class TestRiskManagerValidation:
    """Tests for order validation."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create a test risk manager."""
        return ProductionRiskManager(
            position_limits=PositionLimits(
                max_position_pct=Decimal("0.10"),
                max_position_notional=Decimal("50000"),
            ),
            exposure_limits=ExposureLimits(),
            circuit_breaker_config=CircuitBreakerConfig(
                daily_loss_limit_pct=Decimal("0.02"),
                drawdown_limit_pct=Decimal("0.10"),
            ),
            starting_equity=Decimal("100000"),
        )
    
    def test_validate_order_allow(self, risk_manager):
        """Valid order should be allowed."""
        # Set up prices
        risk_manager.update_prices({"AAPL": Decimal("150")})
        
        # 50 shares * $150 = $7,500 = 7.5% of portfolio (under 10% limit)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=50)
        decision = risk_manager.validate_order(order)
        
        assert decision.action == RiskAction.ALLOW
        assert decision.approved_quantity == 50
    
    def test_validate_order_reject_invalid_quantity(self, risk_manager):
        """Order with invalid quantity should be rejected."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=0)
        decision = risk_manager.validate_order(order)
        
        assert decision.action == RiskAction.REJECT
    
    def test_validate_order_reject_empty_symbol(self, risk_manager):
        """Order with empty symbol should be rejected."""
        order = Order(symbol="", side=OrderSide.BUY, quantity=100)
        decision = risk_manager.validate_order(order)
        
        assert decision.action == RiskAction.REJECT
    
    def test_validate_orders_batch(self, risk_manager):
        """Multiple orders should be validated."""
        risk_manager.update_prices({
            "AAPL": Decimal("150"),
            "GOOG": Decimal("100"),
        })
        
        # Small orders that stay within 10% limit
        orders = [
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=50),  # $7,500 = 7.5%
            Order(symbol="GOOG", side=OrderSide.BUY, quantity=50),  # $5,000 = 5%
        ]
        
        decisions = risk_manager.validate_orders(orders)
        
        assert len(decisions) == 2
        assert all(d.action == RiskAction.ALLOW for d in decisions)


class TestRiskManagerCircuitBreaker:
    """Tests for circuit breaker integration."""
    
    def test_validate_order_halt_on_breaker(self):
        """Order should be halted when circuit breaker is triggered."""
        risk_manager = ProductionRiskManager(
            starting_equity=Decimal("100000"),
        )
        
        # Trigger circuit breaker by updating with large loss
        snapshot = PortfolioSnapshot(
            total_equity=Decimal("85000"),  # 15% loss
            cash=Decimal("85000"),
            daily_pnl=Decimal("-15000"),
            gross_exposure=Decimal("0"),
            net_exposure=Decimal("0"),
        )
        risk_manager.update_state(snapshot)
        
        # Try to submit order
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        decision = risk_manager.validate_order(order)
        
        assert decision.action == RiskAction.HALT
        assert "circuit breaker" in decision.reason.lower()
    
    def test_can_trade_true_normal_state(self):
        """can_trade should return True when no breakers triggered."""
        risk_manager = ProductionRiskManager(
            starting_equity=Decimal("100000"),
        )
        
        assert risk_manager.can_trade() is True
    
    def test_can_trade_false_after_breaker(self):
        """can_trade should return False after breaker triggers."""
        risk_manager = ProductionRiskManager(
            starting_equity=Decimal("100000"),
            circuit_breaker_config=CircuitBreakerConfig(
                daily_loss_limit_pct=Decimal("0.02"),
            ),
        )
        
        # Trigger breaker
        snapshot = PortfolioSnapshot(
            total_equity=Decimal("95000"),
            cash=Decimal("95000"),
            daily_pnl=Decimal("-5000"),
            gross_exposure=Decimal("0"),
            net_exposure=Decimal("0"),
        )
        risk_manager.update_state(snapshot)
        
        assert risk_manager.can_trade() is False


class TestRiskManagerState:
    """Tests for risk state management."""
    
    def test_update_state(self):
        """update_state should update risk metrics."""
        risk_manager = ProductionRiskManager(
            starting_equity=Decimal("100000"),
        )
        
        snapshot = PortfolioSnapshot(
            total_equity=Decimal("105000"),
            cash=Decimal("20000"),
            long_market_value=Decimal("85000"),
            gross_exposure=Decimal("85000"),
            net_exposure=Decimal("85000"),
            daily_pnl=Decimal("5000"),
        )
        
        state = risk_manager.update_state(snapshot)
        
        assert state.current_equity == Decimal("105000")
        assert state.daily_pnl == Decimal("5000")
    
    def test_reset_daily_state(self):
        """reset_daily_state should reset daily metrics."""
        risk_manager = ProductionRiskManager(
            starting_equity=Decimal("100000"),
        )
        
        # Apply some P&L
        snapshot = PortfolioSnapshot(
            total_equity=Decimal("105000"),
            cash=Decimal("105000"),
            daily_pnl=Decimal("5000"),
            gross_exposure=Decimal("0"),
            net_exposure=Decimal("0"),
        )
        risk_manager.update_state(snapshot)
        
        # Reset for new day
        risk_manager.reset_daily_state()
        
        state = risk_manager.get_risk_state()
        assert state.daily_pnl == Decimal("0")
    
    def test_get_risk_state(self):
        """get_risk_state should return current state."""
        risk_manager = ProductionRiskManager(
            starting_equity=Decimal("100000"),
        )
        
        state = risk_manager.get_risk_state()
        
        assert state.starting_equity == Decimal("100000")
        assert state.trading_halted is False
