"""Unit tests for ExecutionEngineImpl."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from bifs_quant_engine.core.enums import OrderSide, OrderStatus
from bifs_quant_engine.core.models import Order, Fill
from bifs_quant_engine.brokers.paper_broker import PaperBroker, PaperBrokerConfig
from bifs_quant_engine.execution.execution_engine import ExecutionEngineImpl, ExecutionConfig


@pytest.fixture
def broker():
    """Create a connected paper broker."""
    config = PaperBrokerConfig(
        initial_cash=Decimal("100000"),
        latency_ms=0,
        slippage_bps=0,
    )
    broker = PaperBroker(config)
    broker.connect()
    broker.update_quotes({
        "AAPL": {"bid": 150.0, "ask": 150.0, "last": 150.0},
        "GOOG": {"bid": 100.0, "ask": 100.0, "last": 100.0},
    })
    return broker


@pytest.fixture
def engine(broker):
    """Create execution engine with paper broker."""
    config = ExecutionConfig(max_retries=0)  # No retries for tests
    return ExecutionEngineImpl(broker, config)


class TestExecutionEngineSubmission:
    """Tests for order submission."""
    
    def test_submit_single_order(self, engine):
        """Should submit and fill a single order."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        results = engine.submit_orders([order])
        
        assert len(results) == 1
        assert results[0].status == OrderStatus.FILLED
    
    def test_submit_multiple_orders(self, engine):
        """Should submit multiple orders."""
        orders = [
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=100),
            Order(symbol="GOOG", side=OrderSide.BUY, quantity=50),
        ]
        results = engine.submit_orders(orders)
        
        assert len(results) == 2
        assert all(r.status == OrderStatus.FILLED for r in results)


class TestExecutionEngineIdempotency:
    """Tests for idempotent order submission."""
    
    def test_resubmit_same_order_returns_existing(self, engine):
        """Resubmitting same order should return existing."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        
        # Submit first time
        result1 = engine.submit_orders([order])
        
        # Submit again
        result2 = engine.submit_orders([order])
        
        assert result1[0].order_id == result2[0].order_id
        assert result1[0].status == result2[0].status


class TestExecutionEngineOrderTracking:
    """Tests for order tracking."""
    
    def test_get_order(self, engine):
        """Should retrieve submitted order."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        engine.submit_orders([order])
        
        retrieved = engine.get_order(order.order_id)
        
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
    
    def test_get_order_unknown(self, engine):
        """Unknown order should return None."""
        result = engine.get_order(uuid4())
        assert result is None
    
    def test_get_open_orders(self, engine):
        """Should return only open orders."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        engine.submit_orders([order])
        
        open_orders = engine.get_open_orders()
        
        # Order was filled immediately, so no open orders
        assert len(open_orders) == 0


class TestExecutionEngineCancellation:
    """Tests for order cancellation."""
    
    def test_cancel_unknown_order(self, engine):
        """Cancelling unknown order should return False."""
        result = engine.cancel_order(uuid4())
        assert result is False
    
    def test_cancel_filled_order(self, engine):
        """Cancelling filled order should return False."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        engine.submit_orders([order])
        
        result = engine.cancel_order(order.order_id)
        assert result is False  # Already filled
    
    def test_cancel_all_orders(self, engine):
        """cancel_all_orders should return count of cancelled."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        engine.submit_orders([order])
        
        count = engine.cancel_all_orders()
        # All orders filled, so none to cancel
        assert count == 0


class TestExecutionEngineFills:
    """Tests for fill tracking."""
    
    def test_get_fills(self, engine):
        """Should return fills."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        engine.submit_orders([order])
        
        fills = engine.get_fills()
        
        assert len(fills) == 1
        assert fills[0].symbol == "AAPL"
        assert fills[0].quantity == 100
    
    def test_get_fills_since(self, engine):
        """Should filter fills by timestamp."""
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        engine.submit_orders([order])
        
        fills = engine.get_fills(since=old_time)
        
        assert len(fills) == 1  # Recent fill included
    
    def test_add_fill_updates_order(self, engine):
        """Adding fill should update order state."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        order.status = OrderStatus.ACKNOWLEDGED
        engine._orders[order.order_id] = order
        engine._client_order_ids.add(order.order_id)
        
        fill = Fill(
            order_id=order.order_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal("150.00"),
        )
        engine.add_fill(fill)
        
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100


class TestExecutionEngineReconciliation:
    """Tests for position reconciliation."""
    
    def test_sync_positions(self, engine):
        """Should sync positions from broker."""
        # Submit order to create position
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        engine.submit_orders([order])
        
        positions = engine.sync_positions()
        
        assert "AAPL" in positions
        assert positions["AAPL"].quantity == 100
    
    def test_reconcile(self, engine):
        """Should reconcile with broker."""
        # Submit order to create position
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        engine.submit_orders([order])
        
        report = engine.reconcile()
        
        assert "position_discrepancies" in report
        assert "order_discrepancies" in report
        assert "corrective_actions" in report
        assert "timestamp" in report
