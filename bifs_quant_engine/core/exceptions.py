"""Domain-specific exceptions for BIFS Quant Engine."""

from __future__ import annotations


class BIFSError(Exception):
    """Base exception for all BIFS Quant Engine errors."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Order & Execution Errors
# ─────────────────────────────────────────────────────────────────────────────


class OrderError(BIFSError):
    """Base exception for order-related errors."""
    pass


class OrderSubmissionError(OrderError):
    """Failed to submit order to broker."""
    pass


class OrderCancellationError(OrderError):
    """Failed to cancel order."""
    pass


class DuplicateOrderError(OrderError):
    """Attempted to submit duplicate order (idempotency violation)."""
    pass


class InvalidOrderStateTransition(OrderError):
    """Invalid order status transition attempted."""
    
    def __init__(self, from_status, to_status):
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(f"Invalid transition: {from_status} → {to_status}")


# ─────────────────────────────────────────────────────────────────────────────
# Risk Errors
# ─────────────────────────────────────────────────────────────────────────────


class RiskError(BIFSError):
    """Base exception for risk-related errors."""
    pass


class RiskLimitViolation(RiskError):
    """Order violates risk limits."""
    
    def __init__(self, limit_name: str, current_value, limit_value):
        self.limit_name = limit_name
        self.current_value = current_value
        self.limit_value = limit_value
        super().__init__(
            f"Risk limit '{limit_name}' violated: {current_value} exceeds {limit_value}"
        )


class CircuitBreakerTriggered(RiskError):
    """Trading halted due to circuit breaker."""
    
    def __init__(self, breaker_name: str):
        self.breaker_name = breaker_name
        super().__init__(f"Circuit breaker triggered: {breaker_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Broker Errors
# ─────────────────────────────────────────────────────────────────────────────


class BrokerError(BIFSError):
    """Base exception for broker-related errors."""
    pass


class BrokerConnectionError(BrokerError):
    """Failed to connect to broker."""
    pass


class BrokerDisconnectedError(BrokerError):
    """Broker connection lost."""
    pass


class InsufficientFundsError(BrokerError):
    """Not enough cash/buying power for order."""
    pass


class PositionNotFoundError(BrokerError):
    """Position does not exist."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Data Errors
# ─────────────────────────────────────────────────────────────────────────────


class DataError(BIFSError):
    """Base exception for data-related errors."""
    pass


class DataNotAvailableError(DataError):
    """Requested data not available."""
    pass


class ReconciliationError(BIFSError):
    """Position reconciliation failed."""
    
    def __init__(self, discrepancies: list):
        self.discrepancies = discrepancies
        super().__init__(f"Reconciliation failed: {len(discrepancies)} discrepancies found")
