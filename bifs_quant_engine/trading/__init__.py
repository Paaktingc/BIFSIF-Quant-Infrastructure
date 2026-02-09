"""Trading package for live/paper trading sessions.

Provides:
- TradingSession for orchestrating trading
- Scheduler for timed strategy execution
- Market data management
"""

from .session import TradingSession, SessionConfig, SessionState

__all__ = [
    "TradingSession",
    "SessionConfig",
    "SessionState",
]
