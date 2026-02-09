"""Fill repository for persistence.

Provides CRUD operations for Fill entities (trade executions).
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from bifs_quant_engine.core.enums import OrderSide
from bifs_quant_engine.core.models import Fill
from bifs_quant_engine.persistence.database import Database


class FillRepository:
    """Repository for persisting and querying fills.
    
    Example:
        repo = FillRepository(database)
        repo.save(fill)
        fills = repo.get_by_order(order_id)
    """
    
    def __init__(self, db: Database) -> None:
        """Initialize with database connection.
        
        Args:
            db: Database instance
        """
        self._db = db
    
    def save(self, fill: Fill) -> None:
        """Save a fill record.
        
        Args:
            fill: Fill to save
        """
        with self._db.connection() as conn:
            conn.execute("""
                INSERT INTO fills (
                    fill_id, order_id, broker_fill_id, broker_order_id,
                    symbol, side, quantity, price, commission, slippage, filled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(fill.fill_id),
                str(fill.order_id),
                fill.broker_fill_id,
                fill.broker_order_id,
                fill.symbol,
                fill.side.value,
                fill.quantity,
                float(fill.price),
                float(fill.commission),
                float(fill.slippage),
                fill.filled_at.isoformat(),
            ))
    
    def get(self, fill_id: UUID) -> Optional[Fill]:
        """Get a fill by ID.
        
        Args:
            fill_id: UUID of the fill
            
        Returns:
            Fill if found, None otherwise
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM fills WHERE fill_id = ?",
                (str(fill_id),)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_fill(row)
    
    def get_by_order(self, order_id: UUID) -> List[Fill]:
        """Get all fills for an order.
        
        Args:
            order_id: UUID of the order
            
        Returns:
            List of fills for the order
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM fills WHERE order_id = ? ORDER BY filled_at",
                (str(order_id),)
            )
            return [self._row_to_fill(row) for row in cursor.fetchall()]
    
    def get_since(self, since: datetime) -> List[Fill]:
        """Get all fills since a given time.
        
        Args:
            since: Start datetime
            
        Returns:
            List of fills since the datetime
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM fills WHERE filled_at >= ? ORDER BY filled_at",
                (since.isoformat(),)
            )
            return [self._row_to_fill(row) for row in cursor.fetchall()]
    
    def get_by_symbol(self, symbol: str, limit: int = 100) -> List[Fill]:
        """Get recent fills for a symbol.
        
        Args:
            symbol: Ticker symbol
            limit: Maximum number of fills to return
            
        Returns:
            List of fills for the symbol
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM fills WHERE symbol = ? ORDER BY filled_at DESC LIMIT ?",
                (symbol.upper(), limit)
            )
            return [self._row_to_fill(row) for row in cursor.fetchall()]
    
    def get_today(self) -> List[Fill]:
        """Get all fills from today."""
        today = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return self.get_since(today)
    
    def _row_to_fill(self, row) -> Fill:
        """Convert database row to Fill object."""
        return Fill(
            fill_id=UUID(row["fill_id"]),
            order_id=UUID(row["order_id"]),
            broker_fill_id=row["broker_fill_id"],
            broker_order_id=row["broker_order_id"],
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            quantity=row["quantity"],
            price=Decimal(str(row["price"])),
            commission=Decimal(str(row["commission"])),
            slippage=Decimal(str(row["slippage"])),
            filled_at=datetime.fromisoformat(row["filled_at"]),
        )
