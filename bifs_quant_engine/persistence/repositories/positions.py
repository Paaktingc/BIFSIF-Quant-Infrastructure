"""Position and snapshot repository for persistence.

Provides operations for persisting portfolio snapshots and position history.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID

from bifs_quant_engine.core.models import Position, PortfolioSnapshot
from bifs_quant_engine.persistence.database import Database


class PositionRepository:
    """Repository for persisting positions and portfolio snapshots.
    
    Example:
        repo = PositionRepository(database)
        repo.save_snapshot(snapshot)
        latest = repo.get_latest_snapshot()
    """
    
    def __init__(self, db: Database) -> None:
        """Initialize with database connection.
        
        Args:
            db: Database instance
        """
        self._db = db
    
    def save_position(self, position: Position) -> None:
        """Save or update a position.
        
        Args:
            position: Position to save
        """
        with self._db.connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions (
                    symbol, quantity, avg_cost, realized_pnl,
                    borrow_rate, is_hard_to_borrow, last_fill_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.symbol,
                position.quantity,
                float(position.avg_cost),
                float(position.realized_pnl),
                float(position.borrow_rate),
                1 if position.is_hard_to_borrow else 0,
                position.last_fill_at.isoformat() if position.last_fill_at else None,
                datetime.now(timezone.utc).isoformat(),
            ))
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Position if exists, None otherwise
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM positions WHERE symbol = ?",
                (symbol.upper(),)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_position(row)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions.
        
        Returns:
            Dict mapping symbol to Position
        """
        with self._db.connection() as conn:
            cursor = conn.execute("SELECT * FROM positions WHERE quantity != 0")
            positions = {}
            for row in cursor.fetchall():
                pos = self._row_to_position(row)
                positions[pos.symbol] = pos
            return positions
    
    def delete_position(self, symbol: str) -> None:
        """Delete a position (e.g., when flat).
        
        Args:
            symbol: Symbol to delete
        """
        with self._db.connection() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol.upper(),))
    
    def save_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Save a portfolio snapshot.
        
        Args:
            snapshot: Snapshot to save
        """
        with self._db.connection() as conn:
            conn.execute("""
                INSERT INTO portfolio_snapshots (
                    snapshot_id, timestamp, cash, long_market_value, short_market_value,
                    gross_exposure, net_exposure, total_equity, daily_pnl,
                    realized_pnl, unrealized_pnl, leverage, position_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(snapshot.snapshot_id),
                snapshot.timestamp.isoformat(),
                float(snapshot.cash),
                float(snapshot.long_market_value),
                float(snapshot.short_market_value),
                float(snapshot.gross_exposure),
                float(snapshot.net_exposure),
                float(snapshot.total_equity),
                float(snapshot.daily_pnl),
                float(snapshot.realized_pnl),
                float(snapshot.unrealized_pnl),
                float(snapshot.leverage),
                snapshot.position_count,
            ))
    
    def get_snapshot(self, snapshot_id: UUID) -> Optional[PortfolioSnapshot]:
        """Get a snapshot by ID.
        
        Args:
            snapshot_id: UUID of the snapshot
            
        Returns:
            Snapshot if found, None otherwise
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM portfolio_snapshots WHERE snapshot_id = ?",
                (str(snapshot_id),)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_snapshot(row)
    
    def get_latest_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Get the most recent snapshot.
        
        Returns:
            Latest snapshot, or None if no snapshots exist
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_snapshot(row)
    
    def get_snapshots_since(self, since: datetime) -> List[PortfolioSnapshot]:
        """Get all snapshots since a given time.
        
        Args:
            since: Start datetime
            
        Returns:
            List of snapshots
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM portfolio_snapshots WHERE timestamp >= ? ORDER BY timestamp",
                (since.isoformat(),)
            )
            return [self._row_to_snapshot(row) for row in cursor.fetchall()]
    
    def get_snapshot_history(self, days: int = 30) -> List[PortfolioSnapshot]:
        """Get snapshot history for past N days.
        
        Args:
            days: Number of days of history
            
        Returns:
            List of snapshots
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        return self.get_snapshots_since(since)
    
    def _row_to_position(self, row) -> Position:
        """Convert database row to Position object."""
        return Position(
            symbol=row["symbol"],
            quantity=row["quantity"],
            avg_cost=Decimal(str(row["avg_cost"])),
            realized_pnl=Decimal(str(row["realized_pnl"])),
            borrow_rate=Decimal(str(row["borrow_rate"])),
            is_hard_to_borrow=bool(row["is_hard_to_borrow"]),
            last_fill_at=datetime.fromisoformat(row["last_fill_at"]) if row["last_fill_at"] else None,
        )
    
    def _row_to_snapshot(self, row) -> PortfolioSnapshot:
        """Convert database row to PortfolioSnapshot object."""
        return PortfolioSnapshot(
            snapshot_id=UUID(row["snapshot_id"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            cash=Decimal(str(row["cash"])),
            long_market_value=Decimal(str(row["long_market_value"])),
            short_market_value=Decimal(str(row["short_market_value"])),
            gross_exposure=Decimal(str(row["gross_exposure"])),
            net_exposure=Decimal(str(row["net_exposure"])),
            total_equity=Decimal(str(row["total_equity"])),
            daily_pnl=Decimal(str(row["daily_pnl"])),
            realized_pnl=Decimal(str(row["realized_pnl"])),
            unrealized_pnl=Decimal(str(row["unrealized_pnl"])),
            leverage=Decimal(str(row["leverage"])),
            positions={},  # Positions need separate query if needed
        )
