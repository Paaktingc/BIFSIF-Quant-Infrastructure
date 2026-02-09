"""Position and exposure limit validators.

These validators check orders and portfolio state against configured limits
to prevent excessive risk taking.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional

from bifs_quant_engine.core.enums import RiskAction
from bifs_quant_engine.core.models import Order, Position, RiskDecision, PortfolioSnapshot


@dataclass
class PositionLimits:
    """Configuration for position-level limits.
    
    Attributes:
        max_position_pct: Maximum position size as percentage of portfolio (0.10 = 10%)
        max_position_notional: Maximum position size in dollars
        max_shares_per_order: Maximum shares in a single order
    """
    max_position_pct: Decimal = Decimal("0.10")
    max_position_notional: Decimal = Decimal("50000")
    max_shares_per_order: int = 10000


@dataclass
class ExposureLimits:
    """Configuration for portfolio-level exposure limits.
    
    Attributes:
        max_gross_exposure: Maximum (long + |short|) / equity ratio
        max_net_exposure: Maximum |long - |short|| / equity ratio
        max_long_exposure: Maximum long / equity ratio
        max_short_exposure: Maximum |short| / equity ratio
    """
    max_gross_exposure: Decimal = Decimal("2.0")
    max_net_exposure: Decimal = Decimal("0.20")
    max_long_exposure: Decimal = Decimal("1.5")
    max_short_exposure: Decimal = Decimal("0.5")


def check_position_limit(
    order: Order,
    current_positions: Dict[str, Position],
    current_prices: Dict[str, Decimal],
    portfolio_equity: Decimal,
    limits: PositionLimits,
) -> RiskDecision:
    """Check if an order would violate position limits.
    
    Args:
        order: The order to validate
        current_positions: Current portfolio positions
        current_prices: Current market prices
        portfolio_equity: Current portfolio equity
        limits: Position limit configuration
        
    Returns:
        RiskDecision indicating if order is allowed, reduced, or rejected
    """
    # Get current position for symbol
    current_pos = current_positions.get(order.symbol)
    current_qty = current_pos.quantity if current_pos else 0
    
    # Get current price
    price = current_prices.get(order.symbol)
    if price is None or price <= 0:
        return RiskDecision(
            action=RiskAction.REJECT,
            order_id=order.order_id,
            original_quantity=order.quantity,
            approved_quantity=0,
            reason=f"No valid price for {order.symbol}",
            violated_limits=["price_unavailable"],
        )
    
    # Calculate what position would be after order
    # BUY/COVER adds to position, SELL/SHORT subtracts
    if order.is_buy_side:
        new_qty = current_qty + order.quantity
    else:
        new_qty = current_qty - order.quantity
    
    # Check shares per order limit
    if order.quantity > limits.max_shares_per_order:
        return RiskDecision(
            action=RiskAction.REDUCE,
            order_id=order.order_id,
            original_quantity=order.quantity,
            approved_quantity=limits.max_shares_per_order,
            reason=f"Order exceeds max shares per order ({limits.max_shares_per_order})",
            violated_limits=["max_shares_per_order"],
        )
    
    # Calculate new position notional
    new_notional = abs(Decimal(new_qty) * price)
    
    # Check notional limit
    if new_notional > limits.max_position_notional:
        # Calculate max allowed quantity
        max_qty = int(limits.max_position_notional / price)
        if order.is_buy_side:
            allowed_qty = max(0, max_qty - current_qty)
        else:
            allowed_qty = max(0, current_qty + max_qty)
        
        if allowed_qty <= 0:
            return RiskDecision(
                action=RiskAction.REJECT,
                order_id=order.order_id,
                original_quantity=order.quantity,
                approved_quantity=0,
                reason=f"Position notional would exceed ${limits.max_position_notional}",
                violated_limits=["max_position_notional"],
            )
        
        return RiskDecision(
            action=RiskAction.REDUCE,
            order_id=order.order_id,
            original_quantity=order.quantity,
            approved_quantity=allowed_qty,
            reason=f"Position notional limited to ${limits.max_position_notional}",
            violated_limits=["max_position_notional"],
        )
    
    # Check percentage limit
    if portfolio_equity > 0:
        new_position_pct = new_notional / portfolio_equity
        if new_position_pct > limits.max_position_pct:
            # Calculate max allowed quantity based on percentage
            max_notional = portfolio_equity * limits.max_position_pct
            max_qty = int(max_notional / price)
            if order.is_buy_side:
                allowed_qty = max(0, max_qty - current_qty)
            else:
                allowed_qty = max(0, current_qty + max_qty)
            
            if allowed_qty <= 0:
                return RiskDecision(
                    action=RiskAction.REJECT,
                    order_id=order.order_id,
                    original_quantity=order.quantity,
                    approved_quantity=0,
                    reason=f"Position would exceed {limits.max_position_pct*100}% of portfolio",
                    violated_limits=["max_position_pct"],
                )
            
            return RiskDecision(
                action=RiskAction.REDUCE,
                order_id=order.order_id,
                original_quantity=order.quantity,
                approved_quantity=allowed_qty,
                reason=f"Position limited to {limits.max_position_pct*100}% of portfolio",
                violated_limits=["max_position_pct"],
            )
    
    # All checks passed
    return RiskDecision(
        action=RiskAction.ALLOW,
        order_id=order.order_id,
        original_quantity=order.quantity,
        approved_quantity=order.quantity,
        reason="Order within position limits",
    )


def check_exposure_limits(
    snapshot: PortfolioSnapshot,
    limits: ExposureLimits,
) -> RiskDecision:
    """Check if current portfolio exposure violates limits.
    
    Args:
        snapshot: Current portfolio snapshot with exposure metrics
        limits: Exposure limit configuration
        
    Returns:
        RiskDecision indicating if exposure is acceptable
    """
    violated = []
    
    if snapshot.total_equity <= 0:
        return RiskDecision(
            action=RiskAction.REJECT,
            reason="Portfolio equity is zero or negative",
            violated_limits=["zero_equity"],
        )
    
    # Calculate exposure ratios
    gross_ratio = snapshot.gross_exposure / snapshot.total_equity
    net_ratio = abs(snapshot.net_exposure) / snapshot.total_equity
    long_ratio = snapshot.long_market_value / snapshot.total_equity
    short_ratio = abs(snapshot.short_market_value) / snapshot.total_equity
    
    # Check each limit
    if gross_ratio > limits.max_gross_exposure:
        violated.append("max_gross_exposure")
    
    if net_ratio > limits.max_net_exposure:
        violated.append("max_net_exposure")
    
    if long_ratio > limits.max_long_exposure:
        violated.append("max_long_exposure")
    
    if short_ratio > limits.max_short_exposure:
        violated.append("max_short_exposure")
    
    if violated:
        return RiskDecision(
            action=RiskAction.REJECT,
            reason=f"Exposure limits violated: {', '.join(violated)}",
            violated_limits=violated,
        )
    
    return RiskDecision(
        action=RiskAction.ALLOW,
        reason="Exposure within limits",
    )
