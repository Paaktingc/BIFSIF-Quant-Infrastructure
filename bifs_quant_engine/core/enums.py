"""Core enums for the BIFS Quant Engine."""

from enum import Enum, auto


class OrderSide(Enum):
    """Side of a trading order."""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"   # Open short position
    COVER = "cover"   # Close short position


class OrderType(Enum):
    """Type of order execution."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    FOK = "fok"              # Fill Or Kill
    GTD = "gtd"              # Good Till Date


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = auto()       # Created, not yet submitted
    SUBMITTED = auto()     # Sent to broker
    ACKNOWLEDGED = auto()  # Broker confirmed receipt
    PARTIAL_FILL = auto()  # Partially filled
    FILLED = auto()        # Fully filled
    REJECTED = auto()      # Broker rejected
    CANCELLED = auto()     # Cancelled by system/user
    EXPIRED = auto()       # Time-based expiration


class RiskAction(Enum):
    """Risk manager decision on an order."""
    ALLOW = auto()   # Order approved
    REJECT = auto()  # Order blocked
    REDUCE = auto()  # Order size reduced
    HALT = auto()    # Circuit breaker triggered
