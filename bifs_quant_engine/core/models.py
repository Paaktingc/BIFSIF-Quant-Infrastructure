"""Core data models for the BIFS Quant Engine.

This module defines the fundamental data structures used throughout the system:
- Order: Trading order with lifecycle tracking
- Fill: Executed trade record
- Position: Single security position (long/short)
- PortfolioSnapshot: Point-in-time portfolio state
- RiskState: Real-time risk metrics
- RiskDecision: Risk check result
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from .enums import OrderSide, OrderType, OrderStatus, RiskAction


# ─────────────────────────────────────────────────────────────────────────────
# ORDER & FILL MODELS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Order:
    """
    Represents a trading order with full lifecycle tracking.
    
    Attributes:
        symbol: Ticker symbol (e.g., "AAPL")
        side: Order side (BUY, SELL, SHORT, COVER)
        quantity: Number of shares
        order_type: Execution type (MARKET, LIMIT, etc.)
        order_id: Unique internal identifier
        client_order_id: Idempotency key for duplicate prevention
        status: Current lifecycle status
    """
    
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    
    # Lifecycle
    order_id: UUID = field(default_factory=uuid4)
    client_order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Fill tracking
    filled_quantity: int = 0
    avg_fill_price: Optional[Decimal] = None
    
    # Context
    strategy_id: str = ""
    parent_order_id: Optional[UUID] = None
    
    # Broker response
    broker_order_id: Optional[str] = None
    reject_reason: Optional[str] = None
    
    @property
    def is_terminal(self) -> bool:
        """Returns True if order is in a final state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        )
    
    @property
    def remaining_quantity(self) -> int:
        """Returns unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_buy_side(self) -> bool:
        """Returns True for BUY or COVER (adding to long)."""
        return self.side in (OrderSide.BUY, OrderSide.COVER)
    
    @property
    def is_sell_side(self) -> bool:
        """Returns True for SELL or SHORT (reducing long or opening short)."""
        return self.side in (OrderSide.SELL, OrderSide.SHORT)
    
    def __post_init__(self):
        # Generate client_order_id if not provided (for idempotency)
        if not self.client_order_id:
            self.client_order_id = str(self.order_id)


@dataclass
class Fill:
    """
    Represents an executed trade (partial or complete fill).
    
    Attributes:
        fill_id: Unique identifier for this fill
        order_id: Reference to parent order
        quantity: Number of shares filled
        price: Execution price
        commission: Trading commission
        slippage: Estimated slippage cost
    """
    
    fill_id: UUID = field(default_factory=uuid4)
    order_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    price: Decimal = Decimal("0")
    
    # Costs
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    
    # Timestamps
    filled_at: datetime = field(default_factory=datetime.utcnow)
    
    # Broker info
    broker_fill_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    
    @property
    def notional(self) -> Decimal:
        """Gross trade value (quantity * price)."""
        return Decimal(self.quantity) * self.price
    
    @property
    def total_cost(self) -> Decimal:
        """Total cost including commission."""
        return self.notional + self.commission


# ─────────────────────────────────────────────────────────────────────────────
# POSITION MODEL
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Position:
    """
    Represents a position in a single security.
    
    Supports both long (positive quantity) and short (negative quantity) positions.
    
    Attributes:
        symbol: Ticker symbol
        quantity: Signed quantity (positive=long, negative=short)
        avg_cost: Average entry cost per share
        borrow_rate: Annualized borrow rate for shorts
        realized_pnl: Cumulative realized P&L from closed trades
    """
    
    symbol: str
    quantity: int  # Positive = long, Negative = short
    avg_cost: Decimal = Decimal("0")
    
    # Short selling
    borrow_rate: Decimal = Decimal("0")  # Annualized
    is_hard_to_borrow: bool = False
    
    # P&L tracking
    realized_pnl: Decimal = Decimal("0")
    
    # Metadata
    last_fill_at: Optional[datetime] = None
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0
    
    @property
    def abs_quantity(self) -> int:
        """Absolute position size."""
        return abs(self.quantity)
    
    def market_value(self, current_price: Decimal) -> Decimal:
        """
        Calculate market value given current price.
        
        For longs: positive value
        For shorts: negative value (liability)
        """
        return Decimal(self.quantity) * current_price
    
    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """
        Calculate unrealized P&L given current price.
        
        For longs: positive when price > avg_cost
        For shorts: positive when price < avg_cost
        """
        if self.quantity == 0:
            return Decimal("0")
        return Decimal(self.quantity) * (current_price - self.avg_cost)


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PortfolioSnapshot:
    """
    Point-in-time snapshot of portfolio state for persistence and audit.
    
    Captures all key metrics at a specific moment for:
    - Historical analysis
    - Equity curve construction
    - Risk monitoring
    - Audit trail
    """
    
    snapshot_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Cash and positions
    cash: Decimal = Decimal("0")
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # Exposure metrics
    long_market_value: Decimal = Decimal("0")
    short_market_value: Decimal = Decimal("0")   # Absolute value
    gross_exposure: Decimal = Decimal("0")       # Long + |Short|
    net_exposure: Decimal = Decimal("0")         # Long - |Short|
    
    # Value metrics
    total_equity: Decimal = Decimal("0")         # Cash + Net positions
    
    # P&L metrics
    daily_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    
    # Leverage
    leverage: Decimal = Decimal("1")             # Gross / Equity
    
    @property
    def nav(self) -> Decimal:
        """Net Asset Value (alias for total_equity)."""
        return self.total_equity
    
    @property
    def position_count(self) -> int:
        """Number of positions (excluding flat)."""
        return sum(1 for p in self.positions.values() if not p.is_flat)


# ─────────────────────────────────────────────────────────────────────────────
# RISK MODELS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RiskState:
    """
    Current state of risk metrics for real-time monitoring.
    
    Updated continuously as positions and prices change.
    Used to evaluate circuit breakers and provide status.
    """
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Equity tracking
    starting_equity: Decimal = Decimal("0")   # Start of day
    current_equity: Decimal = Decimal("0")
    high_water_mark: Decimal = Decimal("0")
    
    # P&L
    daily_pnl: Decimal = Decimal("0")
    daily_pnl_pct: Decimal = Decimal("0")
    
    # Drawdown
    current_drawdown: Decimal = Decimal("0")  # Pct from HWM
    max_drawdown: Decimal = Decimal("0")      # Max historical
    
    # Exposure
    gross_exposure: Decimal = Decimal("0")
    net_exposure: Decimal = Decimal("0")
    long_exposure: Decimal = Decimal("0")
    short_exposure: Decimal = Decimal("0")
    
    # Concentration
    largest_position_pct: Decimal = Decimal("0")
    position_count: int = 0
    
    # Circuit breaker state
    daily_loss_breaker_triggered: bool = False
    drawdown_breaker_triggered: bool = False
    volatility_breaker_triggered: bool = False
    
    @property
    def trading_halted(self) -> bool:
        """Returns True if any circuit breaker is triggered."""
        return (
            self.daily_loss_breaker_triggered or
            self.drawdown_breaker_triggered or
            self.volatility_breaker_triggered
        )
    
    @property
    def exposure_ratio(self) -> Decimal:
        """Net exposure as a ratio of gross."""
        if self.gross_exposure == 0:
            return Decimal("0")
        return self.net_exposure / self.gross_exposure


@dataclass
class RiskDecision:
    """
    Result of a risk check on a proposed order or action.
    
    Returned by RiskManager.validate_order() to indicate whether
    an order should be allowed, rejected, or modified.
    """
    
    action: RiskAction
    order_id: Optional[UUID] = None
    original_quantity: int = 0
    approved_quantity: int = 0
    
    # Explanation
    reason: str = ""
    violated_limits: List[str] = field(default_factory=list)
    
    # Audit
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_approved(self) -> bool:
        """Returns True if order can proceed (possibly reduced)."""
        return self.action in (RiskAction.ALLOW, RiskAction.REDUCE)
    
    @property
    def is_blocked(self) -> bool:
        """Returns True if order is blocked."""
        return self.action in (RiskAction.REJECT, RiskAction.HALT)
