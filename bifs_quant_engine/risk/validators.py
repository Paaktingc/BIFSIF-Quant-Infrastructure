"""Pre-trade order validators.

These validators perform basic checks on orders before they are submitted
to the risk manager for limit checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from bifs_quant_engine.core.enums import OrderSide, OrderStatus, RiskAction
from bifs_quant_engine.core.models import Order, RiskDecision


@dataclass
class ValidationConfig:
    """Configuration for order validation.
    
    Attributes:
        min_order_quantity: Minimum shares per order
        max_order_quantity: Maximum shares per order
        min_order_notional: Minimum order value in dollars
        max_order_notional: Maximum order value in dollars
        allowed_symbols: If set, only these symbols can be traded
        blocked_symbols: Symbols that cannot be traded
    """
    min_order_quantity: int = 1
    max_order_quantity: int = 100000
    min_order_notional: Decimal = Decimal("100")
    max_order_notional: Decimal = Decimal("1000000")
    allowed_symbols: Optional[List[str]] = None
    blocked_symbols: List[str] = None
    
    def __post_init__(self):
        if self.blocked_symbols is None:
            self.blocked_symbols = []


class OrderValidator:
    """Validates orders for basic correctness before risk checks.
    
    Performs checks that don't require portfolio state:
    - Valid quantity (positive, within bounds)
    - Valid symbol (not empty, not blocked)
    - Order not already in terminal state
    
    Example:
        validator = OrderValidator(config)
        decision = validator.validate(order)
        if decision.action == RiskAction.REJECT:
            print(f"Order rejected: {decision.reason}")
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        """Initialize validator with configuration.
        
        Args:
            config: Validation configuration, uses defaults if None
        """
        self.config = config or ValidationConfig()
    
    def validate(self, order: Order) -> RiskDecision:
        """Validate a single order.
        
        Args:
            order: The order to validate
            
        Returns:
            RiskDecision indicating if order passes validation
        """
        # Check order is not already terminal
        if order.is_terminal:
            return RiskDecision(
                action=RiskAction.REJECT,
                order_id=order.order_id,
                original_quantity=order.quantity,
                approved_quantity=0,
                reason=f"Order already in terminal state: {order.status.name}",
                violated_limits=["terminal_state"],
            )
        
        # Check symbol is not empty
        if not order.symbol or not order.symbol.strip():
            return RiskDecision(
                action=RiskAction.REJECT,
                order_id=order.order_id,
                original_quantity=order.quantity,
                approved_quantity=0,
                reason="Order symbol is empty",
                violated_limits=["empty_symbol"],
            )
        
        # Normalize symbol
        symbol = order.symbol.strip().upper()
        
        # Check if symbol is blocked
        if symbol in self.config.blocked_symbols:
            return RiskDecision(
                action=RiskAction.REJECT,
                order_id=order.order_id,
                original_quantity=order.quantity,
                approved_quantity=0,
                reason=f"Symbol {symbol} is blocked from trading",
                violated_limits=["blocked_symbol"],
            )
        
        # Check if symbol is in allowed list (if configured)
        if self.config.allowed_symbols is not None:
            if symbol not in self.config.allowed_symbols:
                return RiskDecision(
                    action=RiskAction.REJECT,
                    order_id=order.order_id,
                    original_quantity=order.quantity,
                    approved_quantity=0,
                    reason=f"Symbol {symbol} not in allowed universe",
                    violated_limits=["symbol_not_allowed"],
                )
        
        # Check quantity is positive
        if order.quantity <= 0:
            return RiskDecision(
                action=RiskAction.REJECT,
                order_id=order.order_id,
                original_quantity=order.quantity,
                approved_quantity=0,
                reason=f"Order quantity must be positive, got {order.quantity}",
                violated_limits=["invalid_quantity"],
            )
        
        # Check quantity bounds
        if order.quantity < self.config.min_order_quantity:
            return RiskDecision(
                action=RiskAction.REJECT,
                order_id=order.order_id,
                original_quantity=order.quantity,
                approved_quantity=0,
                reason=f"Order quantity {order.quantity} below minimum {self.config.min_order_quantity}",
                violated_limits=["min_order_quantity"],
            )
        
        if order.quantity > self.config.max_order_quantity:
            return RiskDecision(
                action=RiskAction.REDUCE,
                order_id=order.order_id,
                original_quantity=order.quantity,
                approved_quantity=self.config.max_order_quantity,
                reason=f"Order quantity reduced from {order.quantity} to {self.config.max_order_quantity}",
                violated_limits=["max_order_quantity"],
            )
        
        # All basic checks passed
        return RiskDecision(
            action=RiskAction.ALLOW,
            order_id=order.order_id,
            original_quantity=order.quantity,
            approved_quantity=order.quantity,
            reason="Order passed basic validation",
        )
    
    def validate_event_contract(self, order: Order) -> RiskDecision:
        """Validate an event contract order.

        Checks event-contract-specific constraints on top of basic validation:
        - Price must be a valid probability (0.01-0.99)
        - Limit orders must have a limit_price

        Args:
            order: The order to validate.

        Returns:
            RiskDecision indicating if order passes validation.
        """
        # Run basic validation first
        basic = self.validate(order)
        if basic.action == RiskAction.REJECT:
            return basic

        # Event contracts require a limit price
        if order.limit_price is None:
            return RiskDecision(
                action=RiskAction.REJECT,
                order_id=order.order_id,
                original_quantity=order.quantity,
                approved_quantity=0,
                reason="Event contract orders require a limit price (probability)",
                violated_limits=["missing_limit_price"],
            )

        # Price must be a valid probability
        if order.limit_price < Decimal("0.01") or order.limit_price > Decimal("0.99"):
            return RiskDecision(
                action=RiskAction.REJECT,
                order_id=order.order_id,
                original_quantity=order.quantity,
                approved_quantity=0,
                reason=f"Price {order.limit_price} not a valid probability [0.01, 0.99]",
                violated_limits=["invalid_probability"],
            )

        return RiskDecision(
            action=RiskAction.ALLOW,
            order_id=order.order_id,
            original_quantity=order.quantity,
            approved_quantity=order.quantity,
            reason="Event contract order passed validation",
        )

    def validate_batch(self, orders: List[Order]) -> List[RiskDecision]:
        """Validate multiple orders.

        Args:
            orders: List of orders to validate

        Returns:
            List of RiskDecisions, one per order
        """
        return [self.validate(order) for order in orders]
