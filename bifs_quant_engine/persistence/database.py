"""SQLite database connection and migration management.

This module provides the Database class for managing SQLite connections
and running schema migrations.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional
import threading


class Database:
    """SQLite database connection manager.
    
    Provides:
    - Connection pooling via thread-local storage
    - Context manager for transactions
    - Migration runner for schema updates
    
    Example:
        db = Database("data/bifs.db")
        db.run_migrations()
        
        with db.connection() as conn:
            cursor = conn.execute("SELECT * FROM orders")
            orders = cursor.fetchall()
        
        db.close()
    """
    
    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory
        """
        self._db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        
        # Create parent directory if needed
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def path(self) -> str:
        """Get database file path."""
        return self._db_path
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection, creating if needed."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(self._db_path)
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            # Return rows as dictionaries
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Get a connection with automatic commit/rollback.
        
        Yields:
            SQLite connection object
            
        Example:
            with db.connection() as conn:
                conn.execute("INSERT INTO orders ...")
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        """Get a cursor with automatic commit/rollback.
        
        Yields:
            SQLite cursor object
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            yield cursor
    
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL and return cursor.
        
        Args:
            sql: SQL statement to execute
            params: Parameters for the statement
            
        Returns:
            Cursor with results
        """
        conn = self._get_connection()
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor
    
    def executemany(self, sql: str, params_list: list) -> sqlite3.Cursor:
        """Execute SQL for multiple parameter sets.
        
        Args:
            sql: SQL statement to execute
            params_list: List of parameter tuples
            
        Returns:
            Cursor
        """
        conn = self._get_connection()
        cursor = conn.executemany(sql, params_list)
        conn.commit()
        return cursor
    
    def run_migrations(self, migrations_dir: Optional[str] = None) -> int:
        """Run pending database migrations.
        
        Migrations are SQL files in the migrations directory, named with
        a numeric prefix (e.g., 001_initial.sql, 002_add_index.sql).
        
        Args:
            migrations_dir: Path to migrations directory, defaults to package location
            
        Returns:
            Number of migrations run
        """
        if migrations_dir is None:
            migrations_dir = str(
                Path(__file__).parent / "migrations"
            )
        
        migrations_path = Path(migrations_dir)
        if not migrations_path.exists():
            return 0
        
        # Create migrations tracking table if not exists
        with self.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _migrations (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        # Get already-applied migrations
        with self.connection() as conn:
            cursor = conn.execute("SELECT name FROM _migrations")
            applied = {row["name"] for row in cursor.fetchall()}
        
        # Find and run pending migrations
        migrations_run = 0
        migration_files = sorted(migrations_path.glob("*.sql"))
        
        for migration_file in migration_files:
            if migration_file.name in applied:
                continue
            
            with open(migration_file) as f:
                sql = f.read()
            
            with self.connection() as conn:
                # Execute migration
                conn.executescript(sql)
                # Record migration
                conn.execute(
                    "INSERT INTO _migrations (name) VALUES (?)",
                    (migration_file.name,),
                )
            
            migrations_run += 1
        
        return migrations_run
    
    def close(self) -> None:
        """Close all connections."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    def __enter__(self) -> "Database":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
