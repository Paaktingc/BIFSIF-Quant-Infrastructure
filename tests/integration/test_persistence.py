"""Integration tests for persistence layer."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from bifs_quant_engine.core.enums import OrderSide, OrderType, OrderStatus
from bifs_quant_engine.core.models import Order, Fill, Position, PortfolioSnapshot
from bifs_quant_engine.persistence.database import Database
from bifs_quant_engine.persistence.repositories.orders import OrderRepository
from bifs_quant_engine.persistence.repositories.fills import FillRepository
from bifs_quant_engine.persistence.repositories.positions import PositionRepository


@pytest.fixture
def database():
    """Create in-memory test database."""
    db = Database(":memory:")
    db.run_migrations()
    yield db
    db.close()


@pytest.fixture
def order_repo(database):
    """Create order repository."""
    return OrderRepository(database)


@pytest.fixture
def fill_repo(database, order_repo):
    """Create fill repository with access to order repo."""
    return FillRepository(database), order_repo


@pytest.fixture
def position_repo(database):
    """Create position repository."""
    return PositionRepository(database)


class TestDatabaseMigrations:
    """Tests for database migration system."""
    
    def test_migrations_run(self, database):
        """Migrations should create required tables."""
        with database.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row["name"] for row in cursor.fetchall()]
        
        assert "orders" in tables
        assert "fills" in tables
        assert "positions" in tables
        assert "portfolio_snapshots" in tables
        assert "risk_states" in tables
        assert "audit_log" in tables
    
    def test_migrations_idempotent(self, database):
        """Running migrations twice should not fail."""
        # First run already happened in fixture
        count = database.run_migrations()
        assert count == 0  # No new migrations


class TestOrderRepository:
    """Tests for order persistence."""
    
    def test_save_and_get(self, order_repo):
        """Should save and retrieve an order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        
        order_repo.save(order)
        retrieved = order_repo.get(order.order_id)
        
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.side == OrderSide.BUY
        assert retrieved.quantity == 100
    
    def test_get_by_status(self, order_repo):
        """Should filter orders by status."""
        order1 = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        order2 = Order(symbol="GOOG", side=OrderSide.BUY, quantity=50)
        order2.status = OrderStatus.FILLED
        
        order_repo.save(order1)
        order_repo.save(order2)
        
        pending = order_repo.get_by_status(OrderStatus.PENDING)
        filled = order_repo.get_by_status(OrderStatus.FILLED)
        
        assert len(pending) == 1
        assert pending[0].symbol == "AAPL"
        assert len(filled) == 1
        assert filled[0].symbol == "GOOG"
    
    def test_get_open_orders(self, order_repo):
        """Should return non-terminal orders."""
        order1 = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        order2 = Order(symbol="GOOG", side=OrderSide.BUY, quantity=50)
        order2.status = OrderStatus.FILLED
        
        order_repo.save(order1)
        order_repo.save(order2)
        
        open_orders = order_repo.get_open_orders()
        
        assert len(open_orders) == 1
        assert open_orders[0].symbol == "AAPL"
    
    def test_update_status(self, order_repo):
        """Should update order status."""
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        order_repo.save(order)
        
        order_repo.update_status(
            order.order_id,
            OrderStatus.FILLED,
            filled_quantity=100,
            avg_fill_price=Decimal("150.50"),
        )
        
        updated = order_repo.get(order.order_id)
        assert updated.status == OrderStatus.FILLED
        assert updated.filled_quantity == 100
        assert updated.avg_fill_price == Decimal("150.50")


class TestFillRepository:
    """Tests for fill persistence."""
    
    def test_save_and_get(self, fill_repo):
        """Should save and retrieve a fill."""
        fill_repository, order_repo = fill_repo
        # Create parent order first (FK constraint)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        order_repo.save(order)
        
        fill = Fill(
            order_id=order.order_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal("150.25"),
            commission=Decimal("1.00"),
        )
        
        fill_repository.save(fill)
        retrieved = fill_repository.get(fill.fill_id)
        
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.quantity == 100
        assert retrieved.price == Decimal("150.25")
    
    def test_get_by_order(self, fill_repo):
        """Should get all fills for an order."""
        fill_repository, order_repo = fill_repo
        # Create parent order first
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        order_repo.save(order)
        
        fill1 = Fill(order_id=order.order_id, symbol="AAPL", side=OrderSide.BUY,
                     quantity=50, price=Decimal("150"))
        fill2 = Fill(order_id=order.order_id, symbol="AAPL", side=OrderSide.BUY,
                     quantity=50, price=Decimal("151"))
        
        fill_repository.save(fill1)
        fill_repository.save(fill2)
        
        fills = fill_repository.get_by_order(order.order_id)
        
        assert len(fills) == 2
        assert sum(f.quantity for f in fills) == 100


class TestPositionRepository:
    """Tests for position and snapshot persistence."""
    
    def test_save_and_get_position(self, position_repo):
        """Should save and retrieve a position."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_cost=Decimal("150.00"),
        )
        
        position_repo.save_position(position)
        retrieved = position_repo.get_position("AAPL")
        
        assert retrieved is not None
        assert retrieved.quantity == 100
        assert retrieved.avg_cost == Decimal("150.00")
    
    def test_get_all_positions(self, position_repo):
        """Should get all non-flat positions."""
        position_repo.save_position(Position(symbol="AAPL", quantity=100, avg_cost=Decimal("150")))
        position_repo.save_position(Position(symbol="GOOG", quantity=-50, avg_cost=Decimal("100")))
        position_repo.save_position(Position(symbol="MSFT", quantity=0, avg_cost=Decimal("300")))
        
        positions = position_repo.get_all_positions()
        
        assert len(positions) == 2  # MSFT excluded (flat)
        assert "AAPL" in positions
        assert "GOOG" in positions
    
    def test_save_and_get_snapshot(self, position_repo):
        """Should save and retrieve portfolio snapshot."""
        snapshot = PortfolioSnapshot(
            cash=Decimal("50000"),
            total_equity=Decimal("100000"),
            long_market_value=Decimal("50000"),
            gross_exposure=Decimal("50000"),
        )
        
        position_repo.save_snapshot(snapshot)
        retrieved = position_repo.get_snapshot(snapshot.snapshot_id)
        
        assert retrieved is not None
        assert retrieved.cash == Decimal("50000")
        assert retrieved.total_equity == Decimal("100000")
    
    def test_get_latest_snapshot(self, position_repo):
        """Should get most recent snapshot."""
        snapshot1 = PortfolioSnapshot(total_equity=Decimal("100000"), cash=Decimal("100000"))
        snapshot2 = PortfolioSnapshot(total_equity=Decimal("105000"), cash=Decimal("105000"))
        
        position_repo.save_snapshot(snapshot1)
        position_repo.save_snapshot(snapshot2)
        
        latest = position_repo.get_latest_snapshot()
        
        assert latest is not None
        assert latest.total_equity == Decimal("105000")
