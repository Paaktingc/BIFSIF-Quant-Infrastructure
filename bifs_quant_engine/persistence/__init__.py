"""Persistence module for BIFS Quant Engine.

This module provides SQLite-based persistence for:
- Orders and order history
- Fills (trade executions)
- Portfolio snapshots
- Risk state history
- Audit logs
"""

from bifs_quant_engine.persistence.database import Database

__all__ = ["Database"]
