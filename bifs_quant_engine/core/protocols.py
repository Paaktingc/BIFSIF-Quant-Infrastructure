"""Core protocols and interfaces for the BIFS Quant Engine.

This module defines the abstract interfaces that must be implemented
by concrete broker adapters, execution engines, and risk managers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import UUID

from .models import Fill, Order, Position, PortfolioSnapshot, RiskDecision, RiskState
from .enums import OrderStatus


# ─────────────────────────────────────────────────────────────────────────────
# ORDER STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────


VALID_TRANSITIONS: Dict[OrderStatus, List[OrderStatus]] = {
    OrderStatus.PENDING: [
        OrderStatus.SUBMITTED,
        OrderStatus.REJECTED,
    ],
    OrderStatus.SUBMITTED: [
        OrderStatus.ACKNOWLEDGED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
        OrderStatus.PARTIAL_FILL,  # Some brokers skip ACK
        OrderStatus.FILLED,        # Some brokers skip ACK
    ],
    OrderStatus.ACKNOWLEDGED: [
        OrderStatus.PARTIAL_FILL,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
    ],
    OrderStatus.PARTIAL_FILL: [
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
    ],
    # Terminal states - no transitions
    OrderStatus.FILLED: [],
    OrderStatus.REJECTED: [],
    OrderStatus.CANCELLED: [],
    OrderStatus.EXPIRED: [],
}


def can_transition(from_status: OrderStatus, to_status: OrderStatus) -> bool:
    """Check if a status transition is valid."""
    return to_status in VALID_TRANSITIONS.get(from_status, [])


# ─────────────────────────────────────────────────────────────────────────────
# BROKER ADAPTER PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class BrokerAdapter(Protocol):
    """
    Abstract interface for broker integrations.
    
    Implementations: SimulatedBroker, AlpacaBroker, IBKRBroker
    
    Design principles:
    - Broker is source-of-truth for positions
    - All methods should be idempotent where possible
    - Async-ready (can be sync wrappers initially)
    
    Example:
        >>> class MyBroker:
        ...     @property
        ...     def name(self) -> str: return "my_broker"
        ...     @property
        ...     def is_connected(self) -> bool: return True
        ...     # ... implement all required methods
        >>> broker: BrokerAdapter = MyBroker()
    """
    
    @property
    def name(self) -> str:
        """Broker identifier (e.g., 'alpaca', 'ibkr', 'sim')."""
        ...
    
    @property
    def is_connected(self) -> bool:
        """Returns True if connected to broker."""
        ...
    
    def connect(self) -> None:
        """Establish connection to broker."""
        ...
    
    def disconnect(self) -> None:
        """Gracefully disconnect from broker."""
        ...
    
    # ─── Order Management ───
    
    def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the broker.
        
        Returns the Order with updated status and broker_order_id.
        Raises: OrderSubmissionError if submission fails.
        """
        ...
    
    def cancel_order(self, order_id: UUID) -> bool:
        """
        Cancel an open order.
        
        Returns True if cancellation succeeded.
        """
        ...
    
    def get_order_status(self, order_id: UUID) -> Order:
        """Get current status of an order from broker."""
        ...
    
    def get_open_orders(self) -> List[Order]:
        """Get all open/pending orders."""
        ...
    
    # ─── Position & Account ───
    
    def get_positions(self) -> Dict[str, Position]:
        """
        Get current positions from broker (source of truth).
        
        Returns dict mapping symbol to Position.
        """
        ...
    
    def get_cash_balance(self) -> Decimal:
        """Get current cash balance."""
        ...
    
    def get_buying_power(self) -> Decimal:
        """Get available buying power (accounts for margin)."""
        ...
    
    # ─── Market Data ───
    
    def get_quote(self, symbol: str) -> Dict[str, Decimal]:
        """
        Get current quote for symbol.
        
        Returns dict with 'bid', 'ask', 'last' keys.
        """
        ...
    
    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Decimal]]:
        """Batch quote request."""
        ...
    
    # ─── Short Selling ───
    
    def get_shortable_shares(self, symbol: str) -> int:
        """
        Get number of shares available to short.
        
        Returns 0 if not shortable.
        """
        ...
    
    def get_borrow_rate(self, symbol: str) -> Decimal:
        """Get annualized borrow rate for shorting."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION ENGINE INTERFACE
# ─────────────────────────────────────────────────────────────────────────────


class ExecutionEngine(ABC):
    """
    Manages order lifecycle and execution.
    
    Responsibilities:
    - Idempotent order submission (tracks by client_order_id)
    - Order state machine management
    - Retry logic for transient failures
    - Partial fill handling
    - Reconciliation with broker
    
    Example implementation pattern:
        >>> class MyExecutionEngine(ExecutionEngine):
        ...     def __init__(self, broker: BrokerAdapter):
        ...         self.broker = broker
        ...         self._orders: Dict[UUID, Order] = {}
        ...         self._fills: List[Fill] = []
    """
    
    @abstractmethod
    def submit_orders(self, orders: List[Order]) -> List[Order]:
        """
        Submit a batch of orders.
        
        Returns orders with updated status.
        Idempotent: resubmitting same client_order_id returns existing order.
        """
        ...
    
    @abstractmethod
    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel a specific order."""
        ...
    
    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        ...
    
    @abstractmethod
    def get_order(self, order_id: UUID) -> Optional[Order]:
        """Get order by ID from internal tracking."""
        ...
    
    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Get all non-terminal orders."""
        ...
    
    @abstractmethod
    def get_fills(self, since: Optional[datetime] = None) -> List[Fill]:
        """Get fills, optionally since a timestamp."""
        ...
    
    @abstractmethod
    def reconcile(self) -> Dict[str, Any]:
        """
        Reconcile internal state with broker.
        
        Returns reconciliation report:
        {
            'position_discrepancies': [...],
            'order_discrepancies': [...],
            'corrective_actions': [...]
        }
        """
        ...
    
    @abstractmethod
    def sync_positions(self) -> Dict[str, Position]:
        """
        Fetch positions from broker and update internal state.
        
        Returns current positions (broker is source of truth).
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER INTERFACE
# ─────────────────────────────────────────────────────────────────────────────


class RiskManager(ABC):
    """
    Manages risk controls and circuit breakers.
    
    Two main functions:
    1. Pre-trade validation: Check orders before submission
    2. Circuit breakers: Monitor and halt trading when limits hit
    
    Example implementation pattern:
        >>> class MyRiskManager(RiskManager):
        ...     def __init__(self, config: dict):
        ...         self.limits = config['limits']
        ...         self._state = RiskState()
    """
    
    @abstractmethod
    def validate_order(self, order: Order) -> RiskDecision:
        """
        Validate a single order against risk limits.
        
        Checks:
        - Position limits (% of portfolio, $ notional)
        - Concentration limits
        - Gross/net exposure limits
        - Order size limits
        - Symbol restrictions
        """
        ...
    
    @abstractmethod
    def validate_orders(self, orders: List[Order]) -> List[RiskDecision]:
        """Validate a batch of orders (considers aggregate impact)."""
        ...
    
    @abstractmethod
    def check_circuit_breakers(self) -> RiskState:
        """
        Check all circuit breakers and update RiskState.
        
        Circuit breakers:
        - Daily loss limit (e.g., -2% of NAV)
        - Drawdown limit (e.g., -10% from HWM)
        - Volatility spike (optional)
        """
        ...
    
    @abstractmethod
    def update_state(self, snapshot: PortfolioSnapshot) -> RiskState:
        """Update risk state with new portfolio snapshot."""
        ...
    
    @abstractmethod
    def get_risk_state(self) -> RiskState:
        """Get current risk state."""
        ...
    
    @abstractmethod
    def reset_daily_state(self) -> None:
        """Reset daily counters (call at start of trading day)."""
        ...
    
    @abstractmethod
    def can_trade(self) -> bool:
        """Returns True if trading is allowed (no breakers triggered)."""
        ...
