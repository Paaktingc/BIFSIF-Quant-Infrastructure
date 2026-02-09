"""Unit tests for core data models."""

import pytest
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from bifs_quant_engine.core.enums import OrderSide, OrderType, OrderStatus, RiskAction
from bifs_quant_engine.core.models import (
    Order,
    Fill,
    Position,
    PortfolioSnapshot,
    RiskState,
    RiskDecision,
)
from bifs_quant_engine.core.protocols import can_transition, VALID_TRANSITIONS


# ─────────────────────────────────────────────────────────────────────────────
# Order Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOrder:
    """Tests for the Order dataclass."""
    
    def test_order_creation_defaults(self):
        """Order should have sensible defaults."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0
        assert order.order_id is not None
        assert order.client_order_id == str(order.order_id)
    
    def test_order_is_terminal(self):
        """Order should correctly identify terminal states."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        
        order.status = OrderStatus.PENDING
        assert order.is_terminal is False
        
        order.status = OrderStatus.SUBMITTED
        assert order.is_terminal is False
        
        order.status = OrderStatus.FILLED
        assert order.is_terminal is True
        
        order.status = OrderStatus.REJECTED
        assert order.is_terminal is True
        
        order.status = OrderStatus.CANCELLED
        assert order.is_terminal is True
    
    def test_order_remaining_quantity(self):
        """remaining_quantity should track unfilled shares."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        
        assert order.remaining_quantity == 100
        
        order.filled_quantity = 30
        assert order.remaining_quantity == 70
        
        order.filled_quantity = 100
        assert order.remaining_quantity == 0
    
    def test_order_side_properties(self):
        """Order should correctly identify buy/sell side."""
        buy = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        assert buy.is_buy_side is True
        assert buy.is_sell_side is False
        
        sell = Order(symbol="AAPL", side=OrderSide.SELL, quantity=100)
        assert sell.is_sell_side is True
        assert sell.is_buy_side is False
        
        short = Order(symbol="AAPL", side=OrderSide.SHORT, quantity=100)
        assert short.is_sell_side is True
        
        cover = Order(symbol="AAPL", side=OrderSide.COVER, quantity=100)
        assert cover.is_buy_side is True


# ─────────────────────────────────────────────────────────────────────────────
# Fill Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFill:
    """Tests for the Fill dataclass."""
    
    def test_fill_notional(self):
        """notional should be quantity * price."""
        fill = Fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal("150.00"),
        )
        assert fill.notional == Decimal("15000.00")
    
    def test_fill_total_cost(self):
        """total_cost should include commission."""
        fill = Fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )
        assert fill.total_cost == Decimal("15001.00")


# ─────────────────────────────────────────────────────────────────────────────
# Position Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPosition:
    """Tests for the Position dataclass."""
    
    def test_position_long(self):
        """Positive quantity is a long position."""
        pos = Position(symbol="AAPL", quantity=100, avg_cost=Decimal("150.00"))
        
        assert pos.is_long is True
        assert pos.is_short is False
        assert pos.is_flat is False
        assert pos.abs_quantity == 100
    
    def test_position_short(self):
        """Negative quantity is a short position."""
        pos = Position(symbol="AAPL", quantity=-100, avg_cost=Decimal("150.00"))
        
        assert pos.is_long is False
        assert pos.is_short is True
        assert pos.is_flat is False
        assert pos.abs_quantity == 100
    
    def test_position_flat(self):
        """Zero quantity is a flat position."""
        pos = Position(symbol="AAPL", quantity=0)
        
        assert pos.is_long is False
        assert pos.is_short is False
        assert pos.is_flat is True
    
    def test_position_market_value_long(self):
        """Long position has positive market value."""
        pos = Position(symbol="AAPL", quantity=100, avg_cost=Decimal("150.00"))
        mv = pos.market_value(Decimal("160.00"))
        
        assert mv == Decimal("16000.00")
    
    def test_position_market_value_short(self):
        """Short position has negative market value (liability)."""
        pos = Position(symbol="AAPL", quantity=-100, avg_cost=Decimal("150.00"))
        mv = pos.market_value(Decimal("160.00"))
        
        assert mv == Decimal("-16000.00")
    
    def test_position_unrealized_pnl_long_profit(self):
        """Long position profit when price rises."""
        pos = Position(symbol="AAPL", quantity=100, avg_cost=Decimal("150.00"))
        pnl = pos.unrealized_pnl(Decimal("160.00"))
        
        assert pnl == Decimal("1000.00")  # 100 * (160 - 150)
    
    def test_position_unrealized_pnl_long_loss(self):
        """Long position loss when price falls."""
        pos = Position(symbol="AAPL", quantity=100, avg_cost=Decimal("150.00"))
        pnl = pos.unrealized_pnl(Decimal("140.00"))
        
        assert pnl == Decimal("-1000.00")  # 100 * (140 - 150)
    
    def test_position_unrealized_pnl_short_profit(self):
        """Short position profit when price falls."""
        pos = Position(symbol="AAPL", quantity=-100, avg_cost=Decimal("150.00"))
        pnl = pos.unrealized_pnl(Decimal("140.00"))
        
        assert pnl == Decimal("1000.00")  # -100 * (140 - 150)
    
    def test_position_unrealized_pnl_short_loss(self):
        """Short position loss when price rises."""
        pos = Position(symbol="AAPL", quantity=-100, avg_cost=Decimal("150.00"))
        pnl = pos.unrealized_pnl(Decimal("160.00"))
        
        assert pnl == Decimal("-1000.00")  # -100 * (160 - 150)


# ─────────────────────────────────────────────────────────────────────────────
# RiskState Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRiskState:
    """Tests for the RiskState dataclass."""
    
    def test_trading_halted_none(self):
        """No breakers triggered means trading allowed."""
        state = RiskState()
        assert state.trading_halted is False
    
    def test_trading_halted_daily_loss(self):
        """Daily loss breaker halts trading."""
        state = RiskState(daily_loss_breaker_triggered=True)
        assert state.trading_halted is True
    
    def test_trading_halted_drawdown(self):
        """Drawdown breaker halts trading."""
        state = RiskState(drawdown_breaker_triggered=True)
        assert state.trading_halted is True
    
    def test_trading_halted_multiple(self):
        """Multiple breakers still halt trading."""
        state = RiskState(
            daily_loss_breaker_triggered=True,
            drawdown_breaker_triggered=True,
        )
        assert state.trading_halted is True


# ─────────────────────────────────────────────────────────────────────────────
# RiskDecision Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRiskDecision:
    """Tests for the RiskDecision dataclass."""
    
    def test_allow_is_approved(self):
        """ALLOW action is approved."""
        decision = RiskDecision(action=RiskAction.ALLOW)
        assert decision.is_approved is True
        assert decision.is_blocked is False
    
    def test_reduce_is_approved(self):
        """REDUCE action is still approved."""
        decision = RiskDecision(action=RiskAction.REDUCE)
        assert decision.is_approved is True
        assert decision.is_blocked is False
    
    def test_reject_is_blocked(self):
        """REJECT action is blocked."""
        decision = RiskDecision(action=RiskAction.REJECT)
        assert decision.is_approved is False
        assert decision.is_blocked is True
    
    def test_halt_is_blocked(self):
        """HALT action is blocked."""
        decision = RiskDecision(action=RiskAction.HALT)
        assert decision.is_approved is False
        assert decision.is_blocked is True


# ─────────────────────────────────────────────────────────────────────────────
# Order State Machine Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOrderStateMachine:
    """Tests for order state transitions."""
    
    def test_pending_to_submitted(self):
        """PENDING can transition to SUBMITTED."""
        assert can_transition(OrderStatus.PENDING, OrderStatus.SUBMITTED) is True
    
    def test_pending_to_rejected(self):
        """PENDING can transition to REJECTED (pre-trade block)."""
        assert can_transition(OrderStatus.PENDING, OrderStatus.REJECTED) is True
    
    def test_pending_to_filled_invalid(self):
        """PENDING cannot transition directly to FILLED."""
        assert can_transition(OrderStatus.PENDING, OrderStatus.FILLED) is False
    
    def test_submitted_to_filled(self):
        """SUBMITTED can transition to FILLED."""
        assert can_transition(OrderStatus.SUBMITTED, OrderStatus.FILLED) is True
    
    def test_filled_is_terminal(self):
        """FILLED has no valid transitions."""
        assert VALID_TRANSITIONS[OrderStatus.FILLED] == []
        assert can_transition(OrderStatus.FILLED, OrderStatus.PENDING) is False
        assert can_transition(OrderStatus.FILLED, OrderStatus.CANCELLED) is False
    
    def test_partial_fill_to_filled(self):
        """PARTIAL_FILL can transition to FILLED."""
        assert can_transition(OrderStatus.PARTIAL_FILL, OrderStatus.FILLED) is True
    
    def test_partial_fill_to_cancelled(self):
        """PARTIAL_FILL can be cancelled (cancel remainder)."""
        assert can_transition(OrderStatus.PARTIAL_FILL, OrderStatus.CANCELLED) is True
