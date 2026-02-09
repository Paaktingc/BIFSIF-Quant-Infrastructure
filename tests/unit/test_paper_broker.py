"""Unit tests for PaperBroker."""

import pytest
from decimal import Decimal

from bifs_quant_engine.core.enums import OrderSide, OrderStatus, OrderType
from bifs_quant_engine.core.models import Order
from bifs_quant_engine.brokers.paper_broker import PaperBroker, PaperBrokerConfig


class TestPaperBrokerConnection:
    """Tests for broker connection lifecycle."""
    
    def test_initial_state_disconnected(self):
        """Broker should start disconnected."""
        broker = PaperBroker()
        assert broker.is_connected is False
    
    def test_connect(self):
        """Broker should connect successfully."""
        broker = PaperBroker()
        broker.connect()
        assert broker.is_connected is True
    
    def test_disconnect(self):
        """Broker should disconnect successfully."""
        broker = PaperBroker()
        broker.connect()
        broker.disconnect()
        assert broker.is_connected is False
    
    def test_name(self):
        """Broker name should be 'paper'."""
        broker = PaperBroker()
        assert broker.name == "paper"


class TestPaperBrokerOrderExecution:
    """Tests for order submission and execution."""
    
    @pytest.fixture
    def broker(self):
        """Create connected broker with quotes."""
        config = PaperBrokerConfig(
            initial_cash=Decimal("100000"),
            latency_ms=0,  # No delay for tests
            slippage_bps=0,  # No slippage for predictable tests
        )
        broker = PaperBroker(config)
        broker.connect()
        broker.update_quotes({
            "AAPL": {"bid": 150.0, "ask": 150.0, "last": 150.0},
            "GOOG": {"bid": 100.0, "ask": 100.0, "last": 100.0},
        })
        return broker
    
    def test_submit_order_not_connected(self):
        """Should raise when submitting to disconnected broker."""
        broker = PaperBroker()
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        
        with pytest.raises(RuntimeError, match="not connected"):
            broker.submit_order(order)
    
    def test_submit_market_buy_order(self, broker):
        """Market buy order should fill immediately."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        result = broker.submit_order(order)
        
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 100
        assert result.avg_fill_price == Decimal("150.00")
    
    def test_submit_market_sell_order(self, broker):
        """Market sell order should fill after establishing position."""
        # First buy
        buy_order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(buy_order)
        
        # Then sell
        sell_order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=50)
        result = broker.submit_order(sell_order)
        
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 50
    
    def test_submit_order_no_quote(self, broker):
        """Order for symbol without quote should reject."""
        order = Order(symbol="UNKNOWN", side=OrderSide.BUY, quantity=100)
        result = broker.submit_order(order)
        
        assert result.status == OrderStatus.REJECTED
        assert "No quote" in result.reject_reason
    
    def test_submit_limit_order_favorable(self, broker):
        """Limit order at favorable price should fill."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("160.00"),  # Above ask
        )
        result = broker.submit_order(order)
        
        assert result.status == OrderStatus.FILLED
    
    def test_submit_limit_order_unfavorable(self, broker):
        """Limit order at unfavorable price should not fill."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("140.00"),  # Below ask
        )
        result = broker.submit_order(order)
        
        assert result.status == OrderStatus.REJECTED


class TestPaperBrokerPositionTracking:
    """Tests for position management."""
    
    @pytest.fixture
    def broker(self):
        """Create connected broker with quotes."""
        config = PaperBrokerConfig(
            initial_cash=Decimal("100000"),
            latency_ms=0,
            slippage_bps=0,
            commission_per_share=Decimal("0"),
            min_commission=Decimal("0"),
        )
        broker = PaperBroker(config)
        broker.connect()
        broker.update_quotes({
            "AAPL": {"bid": 150.0, "ask": 150.0, "last": 150.0},
        })
        return broker
    
    def test_no_initial_positions(self, broker):
        """Should start with no positions."""
        positions = broker.get_positions()
        assert len(positions) == 0
    
    def test_position_after_buy(self, broker):
        """Position should exist after buy."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(order)
        
        positions = broker.get_positions()
        assert "AAPL" in positions
        assert positions["AAPL"].quantity == 100
        assert positions["AAPL"].avg_cost == Decimal("150.00")
    
    def test_position_after_partial_sell(self, broker):
        """Position should reduce after partial sell."""
        # Buy 100
        buy_order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(buy_order)
        
        # Sell 40
        sell_order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=40)
        broker.submit_order(sell_order)
        
        positions = broker.get_positions()
        assert positions["AAPL"].quantity == 60
    
    def test_position_closed_after_full_sell(self, broker):
        """Position should be removed after selling all."""
        # Buy 100
        buy_order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(buy_order)
        
        # Sell 100
        sell_order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=100)
        broker.submit_order(sell_order)
        
        positions = broker.get_positions()
        assert "AAPL" not in positions  # Flat positions excluded
    
    def test_short_position(self, broker):
        """Short selling should create negative position."""
        order = Order(symbol="AAPL", side=OrderSide.SHORT, quantity=100)
        broker.submit_order(order)
        
        positions = broker.get_positions()
        assert positions["AAPL"].quantity == -100


class TestPaperBrokerCashManagement:
    """Tests for cash balance tracking."""
    
    @pytest.fixture
    def broker(self):
        """Create connected broker with quotes."""
        config = PaperBrokerConfig(
            initial_cash=Decimal("100000"),
            latency_ms=0,
            slippage_bps=0,
            commission_per_share=Decimal("0.01"),
            min_commission=Decimal("1.00"),
        )
        broker = PaperBroker(config)
        broker.connect()
        broker.update_quotes({
            "AAPL": {"bid": 150.0, "ask": 150.0, "last": 150.0},
        })
        return broker
    
    def test_initial_cash(self, broker):
        """Should have configured initial cash."""
        assert broker.get_cash_balance() == Decimal("100000")
    
    def test_cash_after_buy(self, broker):
        """Cash should decrease after buy."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(order)
        
        # 100 * $150 = $15,000 + $1.00 commission
        expected_cash = Decimal("100000") - Decimal("15000") - Decimal("1.00")
        assert broker.get_cash_balance() == expected_cash
    
    def test_cash_after_sell(self, broker):
        """Cash should increase after sell."""
        # Buy first
        buy_order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(buy_order)
        
        # Then sell
        sell_order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=100)
        broker.submit_order(sell_order)
        
        # $15,000 out - $1 commission, then $15,000 in - $1 commission
        expected_cash = Decimal("100000") - Decimal("2.00")
        assert broker.get_cash_balance() == expected_cash


class TestPaperBrokerSlippage:
    """Tests for slippage simulation."""
    
    def test_buy_slippage(self):
        """Buy orders should slip up."""
        config = PaperBrokerConfig(
            latency_ms=0,
            slippage_bps=100,  # 1% slippage
            commission_per_share=Decimal("0"),
            min_commission=Decimal("0"),
        )
        broker = PaperBroker(config)
        broker.connect()
        broker.update_quotes({
            "AAPL": {"bid": 100.0, "ask": 100.0, "last": 100.0},
        })
        
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        result = broker.submit_order(order)
        
        # 1% slippage on $100 = $101
        assert result.avg_fill_price == Decimal("101.00")
    
    def test_sell_slippage(self):
        """Sell orders should slip down."""
        config = PaperBrokerConfig(
            latency_ms=0,
            slippage_bps=100,  # 1% slippage
            commission_per_share=Decimal("0"),
            min_commission=Decimal("0"),
        )
        broker = PaperBroker(config)
        broker.connect()
        broker.update_quotes({
            "AAPL": {"bid": 100.0, "ask": 100.0, "last": 100.0},
        })
        
        # Buy first
        broker.submit_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=100))
        
        # Then sell
        order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=100)
        result = broker.submit_order(order)
        
        # 1% slippage on $100 = $99
        assert result.avg_fill_price == Decimal("99.00")


class TestPaperBrokerOrderManagement:
    """Tests for order tracking and cancellation."""
    
    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = PaperBroker(PaperBrokerConfig(latency_ms=0))
        broker.connect()
        broker.update_quotes({
            "AAPL": {"bid": 150.0, "ask": 150.0, "last": 150.0},
        })
        return broker
    
    def test_get_order_status(self, broker):
        """Should return order status."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(order)
        
        status = broker.get_order_status(order.order_id)
        assert status == OrderStatus.FILLED
    
    def test_get_order_status_unknown(self, broker):
        """Unknown order should return None."""
        from uuid import uuid4
        status = broker.get_order_status(uuid4())
        assert status is None
    
    def test_get_open_orders(self, broker):
        """Should return only open orders."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(order)  # Will fill immediately
        
        open_orders = broker.get_open_orders()
        assert len(open_orders) == 0  # All filled
    
    def test_get_fills(self, broker):
        """Should return fills."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(order)
        
        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].symbol == "AAPL"
        assert fills[0].quantity == 100
