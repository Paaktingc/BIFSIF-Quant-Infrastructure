"""Unit tests for risk limit validators."""

import pytest
from decimal import Decimal

from bifs_quant_engine.core.enums import OrderSide, RiskAction
from bifs_quant_engine.core.models import Order, Position, PortfolioSnapshot
from bifs_quant_engine.risk.limits import (
    PositionLimits,
    ExposureLimits,
    check_position_limit,
    check_exposure_limits,
)


# ─────────────────────────────────────────────────────────────────────────────
# Position Limit Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPositionLimits:
    """Tests for position limit checks."""
    
    def test_order_within_limits_allowed(self):
        """Order within all limits should be allowed."""
        # 50 shares * $150 = $7,500 = 7.5% of $100,000 (under 10% limit)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=50)
        limits = PositionLimits(
            max_position_pct=Decimal("0.10"),
            max_position_notional=Decimal("50000"),
        )
        
        decision = check_position_limit(
            order=order,
            current_positions={},
            current_prices={"AAPL": Decimal("150")},
            portfolio_equity=Decimal("100000"),
            limits=limits,
        )
        
        assert decision.action == RiskAction.ALLOW
        assert decision.approved_quantity == 50
    
    def test_order_exceeds_notional_rejected(self):
        """Order exceeding notional limit should be rejected or reduced."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1000)
        limits = PositionLimits(
            max_position_notional=Decimal("10000"),
        )
        
        decision = check_position_limit(
            order=order,
            current_positions={},
            current_prices={"AAPL": Decimal("150")},  # 1000 * 150 = 150,000 > 10,000
            portfolio_equity=Decimal("100000"),
            limits=limits,
        )
        
        assert decision.action in (RiskAction.REDUCE, RiskAction.REJECT)
        assert decision.approved_quantity < 1000
    
    def test_order_exceeds_percentage_reduced(self):
        """Order exceeding percentage limit should be reduced."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1000)
        limits = PositionLimits(
            max_position_pct=Decimal("0.10"),  # 10% of 100,000 = 10,000
            max_position_notional=Decimal("1000000"),  # High enough not to trigger
        )
        
        decision = check_position_limit(
            order=order,
            current_positions={},
            current_prices={"AAPL": Decimal("150")},  # 1000 * 150 = 150,000 > 10,000
            portfolio_equity=Decimal("100000"),
            limits=limits,
        )
        
        assert decision.action == RiskAction.REDUCE
        assert decision.approved_quantity < 1000
    
    def test_order_with_existing_position(self):
        """Order should consider existing position."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        existing_position = Position(symbol="AAPL", quantity=300, avg_cost=Decimal("145"))
        
        limits = PositionLimits(
            max_position_notional=Decimal("50000"),
        )
        
        decision = check_position_limit(
            order=order,
            current_positions={"AAPL": existing_position},
            current_prices={"AAPL": Decimal("150")},  # (300 + 100) * 150 = 60,000 > 50,000
            portfolio_equity=Decimal("100000"),
            limits=limits,
        )
        
        assert decision.action in (RiskAction.REDUCE, RiskAction.REJECT)
    
    def test_order_no_price_rejected(self):
        """Order with no price available should be rejected."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        limits = PositionLimits()
        
        decision = check_position_limit(
            order=order,
            current_positions={},
            current_prices={},  # No price for AAPL
            portfolio_equity=Decimal("100000"),
            limits=limits,
        )
        
        assert decision.action == RiskAction.REJECT
        assert "price" in decision.reason.lower()
    
    def test_shares_per_order_limit(self):
        """Order exceeding shares per order should be reduced."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=15000)
        limits = PositionLimits(
            max_shares_per_order=10000,
            max_position_notional=Decimal("10000000"),  # Very high
        )
        
        decision = check_position_limit(
            order=order,
            current_positions={},
            current_prices={"AAPL": Decimal("1")},
            portfolio_equity=Decimal("10000000"),
            limits=limits,
        )
        
        assert decision.action == RiskAction.REDUCE
        assert decision.approved_quantity == 10000


# ─────────────────────────────────────────────────────────────────────────────
# Exposure Limit Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExposureLimits:
    """Tests for exposure limit checks."""
    
    def test_exposure_within_limits(self):
        """Exposure within limits should be allowed."""
        snapshot = PortfolioSnapshot(
            total_equity=Decimal("100000"),
            long_market_value=Decimal("80000"),
            short_market_value=Decimal("-20000"),
            gross_exposure=Decimal("100000"),  # 80k + 20k
            net_exposure=Decimal("60000"),     # 80k - 20k
        )
        
        limits = ExposureLimits(
            max_gross_exposure=Decimal("2.0"),
            max_net_exposure=Decimal("1.0"),
        )
        
        decision = check_exposure_limits(snapshot, limits)
        
        assert decision.action == RiskAction.ALLOW
    
    def test_gross_exposure_exceeded(self):
        """Gross exposure exceeding limit should be rejected."""
        snapshot = PortfolioSnapshot(
            total_equity=Decimal("100000"),
            long_market_value=Decimal("150000"),
            short_market_value=Decimal("-100000"),
            gross_exposure=Decimal("250000"),  # 2.5x
            net_exposure=Decimal("50000"),
        )
        
        limits = ExposureLimits(
            max_gross_exposure=Decimal("2.0"),
            max_net_exposure=Decimal("1.0"),
        )
        
        decision = check_exposure_limits(snapshot, limits)
        
        assert decision.action == RiskAction.REJECT
        assert "max_gross_exposure" in decision.violated_limits
    
    def test_net_exposure_exceeded(self):
        """Net exposure exceeding limit should be rejected."""
        snapshot = PortfolioSnapshot(
            total_equity=Decimal("100000"),
            long_market_value=Decimal("90000"),
            short_market_value=Decimal("-10000"),
            gross_exposure=Decimal("100000"),
            net_exposure=Decimal("80000"),  # 80% net long
        )
        
        limits = ExposureLimits(
            max_gross_exposure=Decimal("2.0"),
            max_net_exposure=Decimal("0.20"),  # 20% max
        )
        
        decision = check_exposure_limits(snapshot, limits)
        
        assert decision.action == RiskAction.REJECT
        assert "max_net_exposure" in decision.violated_limits
    
    def test_zero_equity(self):
        """Zero equity should be rejected."""
        snapshot = PortfolioSnapshot(
            total_equity=Decimal("0"),
            gross_exposure=Decimal("0"),
            net_exposure=Decimal("0"),
        )
        
        limits = ExposureLimits()
        
        decision = check_exposure_limits(snapshot, limits)
        
        assert decision.action == RiskAction.REJECT
