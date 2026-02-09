"""Order repository for persistence.

Provides CRUD operations for Order entities.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from bifs_quant_engine.core.enums import OrderSide, OrderType, OrderStatus
from bifs_quant_engine.core.models import Order
from bifs_quant_engine.persistence.database import Database


class OrderRepository:
    """Repository for persisting and querying orders.
    
    Example:
        repo = OrderRepository(database)
        repo.save(order)
        order = repo.get(order_id)
        pending_orders = repo.get_by_status(OrderStatus.PENDING)
    """
    
    def __init__(self, db: Database) -> None:
        """Initialize with database connection.
        
        Args:
            db: Database instance
        """
        self._db = db
    
    def save(self, order: Order) -> None:
        """Save or update an order.
        
        Args:
            order: Order to save
        """
        with self._db.connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO orders (
                    order_id, client_order_id, broker_order_id, symbol, side,
                    order_type, quantity, limit_price, stop_price, status,
                    filled_quantity, avg_fill_price, strategy_id, parent_order_id,
                    reject_reason, created_at, submitted_at, filled_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(order.order_id),
                order.client_order_id or None,
                order.broker_order_id,
                order.symbol,
                order.side.value,
                order.order_type.value,
                order.quantity,
                float(order.limit_price) if order.limit_price else None,
                float(order.stop_price) if order.stop_price else None,
                order.status.name.lower(),
                order.filled_quantity,
                float(order.avg_fill_price) if order.avg_fill_price else None,
                order.strategy_id or None,
                str(order.parent_order_id) if order.parent_order_id else None,
                order.reject_reason,
                order.created_at.isoformat(),
                order.submitted_at.isoformat() if order.submitted_at else None,
                order.filled_at.isoformat() if order.filled_at else None,
                datetime.now(timezone.utc).isoformat(),
            ))
    
    def get(self, order_id: UUID) -> Optional[Order]:
        """Get an order by ID.
        
        Args:
            order_id: UUID of the order
            
        Returns:
            Order if found, None otherwise
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (str(order_id),)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_order(row)
    
    def get_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get an order by client order ID.
        
        Args:
            client_order_id: Client order ID (idempotency key)
            
        Returns:
            Order if found, None otherwise
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM orders WHERE client_order_id = ?",
                (client_order_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_order(row)
    
    def get_by_status(self, status: OrderStatus) -> List[Order]:
        """Get all orders with given status.
        
        Args:
            status: Order status to filter by
            
        Returns:
            List of matching orders
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM orders WHERE status = ? ORDER BY created_at DESC",
                (status.name.lower(),)
            )
            return [self._row_to_order(row) for row in cursor.fetchall()]
    
    def get_open_orders(self) -> List[Order]:
        """Get all non-terminal orders."""
        terminal_statuses = ("filled", "rejected", "cancelled", "expired")
        placeholders = ",".join("?" * len(terminal_statuses))
        with self._db.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM orders WHERE status NOT IN ({placeholders}) ORDER BY created_at DESC",
                terminal_statuses
            )
            return [self._row_to_order(row) for row in cursor.fetchall()]
    
    def get_by_symbol(self, symbol: str, limit: int = 100) -> List[Order]:
        """Get recent orders for a symbol.
        
        Args:
            symbol: Ticker symbol
            limit: Maximum number of orders to return
            
        Returns:
            List of orders for the symbol
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM orders WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                (symbol.upper(), limit)
            )
            return [self._row_to_order(row) for row in cursor.fetchall()]
    
    def update_status(
        self,
        order_id: UUID,
        status: OrderStatus,
        filled_quantity: Optional[int] = None,
        avg_fill_price: Optional[Decimal] = None,
        reject_reason: Optional[str] = None,
    ) -> None:
        """Update order status and related fields.
        
        Args:
            order_id: Order to update
            status: New status
            filled_quantity: Updated fill quantity
            avg_fill_price: Updated average fill price
            reject_reason: Reason if rejected
        """
        updates = ["status = ?", "updated_at = ?"]
        params = [status.name.lower(), datetime.now(timezone.utc).isoformat()]
        
        if filled_quantity is not None:
            updates.append("filled_quantity = ?")
            params.append(filled_quantity)
        
        if avg_fill_price is not None:
            updates.append("avg_fill_price = ?")
            params.append(float(avg_fill_price))
        
        if reject_reason is not None:
            updates.append("reject_reason = ?")
            params.append(reject_reason)
        
        if status == OrderStatus.SUBMITTED:
            updates.append("submitted_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())
        elif status == OrderStatus.FILLED:
            updates.append("filled_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())
        
        params.append(str(order_id))
        
        with self._db.connection() as conn:
            conn.execute(
                f"UPDATE orders SET {', '.join(updates)} WHERE order_id = ?",
                tuple(params)
            )
    
    def _row_to_order(self, row) -> Order:
        """Convert database row to Order object."""
        return Order(
            order_id=UUID(row["order_id"]),
            client_order_id=row["client_order_id"] or "",
            broker_order_id=row["broker_order_id"],
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            order_type=OrderType(row["order_type"]),
            quantity=row["quantity"],
            limit_price=Decimal(str(row["limit_price"])) if row["limit_price"] else None,
            stop_price=Decimal(str(row["stop_price"])) if row["stop_price"] else None,
            status=OrderStatus[row["status"].upper()],
            filled_quantity=row["filled_quantity"] or 0,
            avg_fill_price=Decimal(str(row["avg_fill_price"])) if row["avg_fill_price"] else None,
            strategy_id=row["strategy_id"] or "",
            parent_order_id=UUID(row["parent_order_id"]) if row["parent_order_id"] else None,
            reject_reason=row["reject_reason"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(timezone.utc),
            submitted_at=datetime.fromisoformat(row["submitted_at"]) if row["submitted_at"] else None,
            filled_at=datetime.fromisoformat(row["filled_at"]) if row["filled_at"] else None,
        )
