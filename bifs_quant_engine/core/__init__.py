"""Core modules for the BIFS Quant Engine."""

# Enums
from .enums import OrderSide, OrderType, OrderStatus, RiskAction

# Data models
from .models import (
    Order,
    Fill,
    Position,
    PortfolioSnapshot,
    RiskState,
    RiskDecision,
)

# Protocols and interfaces
from .protocols import (
    BrokerAdapter,
    ExecutionEngine,
    RiskManager,
    can_transition,
    VALID_TRANSITIONS,
)

# Exceptions
from .exceptions import (
    BIFSError,
    OrderError,
    OrderSubmissionError,
    DuplicateOrderError,
    InvalidOrderStateTransition,
    RiskError,
    RiskLimitViolation,
    CircuitBreakerTriggered,
    BrokerError,
    BrokerConnectionError,
    ReconciliationError,
)

__all__ = [
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "RiskAction",
    # Models
    "Order",
    "Fill",
    "Position",
    "PortfolioSnapshot",
    "RiskState",
    "RiskDecision",
    # Protocols
    "BrokerAdapter",
    "ExecutionEngine",
    "RiskManager",
    "can_transition",
    "VALID_TRANSITIONS",
    # Exceptions
    "BIFSError",
    "OrderError",
    "OrderSubmissionError",
    "DuplicateOrderError",
    "InvalidOrderStateTransition",
    "RiskError",
    "RiskLimitViolation",
    "CircuitBreakerTriggered",
    "BrokerError",
    "BrokerConnectionError",
    "ReconciliationError",
]
