"""Paper trading broker for Polymarket simulation.

Simulates Polymarket event contract and crypto short-duration market
trading with binary settlement, configurable taker fees, and probability-
based pricing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID

from bifs_quant_engine.core.enums import OrderSide, OrderStatus, OrderType
from bifs_quant_engine.core.models import Fill, Order, Position

logger = logging.getLogger(__name__)


@dataclass
class PolymarketPaperConfig:
    """Configuration for Polymarket paper broker.

    Attributes:
        initial_cash: Starting USDC balance.
        taker_fee_bps: Taker fee in basis points (0 for event, ~156 for crypto).
        slippage_bps: Simulated slippage in basis points.
        fill_probability: Probability an order fills (0.0 to 1.0).
    """

    initial_cash: Decimal = Decimal("10000")
    taker_fee_bps: int = 0
    slippage_bps: int = 0
    fill_probability: float = 1.0


class PolymarketPaperBroker:
    """Simulated Polymarket broker for paper trading and backtesting.

    Supports binary outcome settlement where contracts resolve to $1.00
    (correct) or $0.00 (incorrect). Probability prices range from $0.01
    to $0.99.

    Example:
        config = PolymarketPaperConfig(initial_cash=Decimal("5000"))
        broker = PolymarketPaperBroker(config)
        broker.connect()

        # Set prices for a YES token
        broker.update_quotes({"token_yes_abc": {"bid": 0.24, "ask": 0.26, "last": 0.25}})

        order = Order(
            symbol="token_yes_abc",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.26"),
        )
        filled = broker.submit_order(order)

        # Settle the market
        broker.resolve_market("token_yes_abc", won=True)
    """

    def __init__(self, config: Optional[PolymarketPaperConfig] = None) -> None:
        """Initialize paper broker.

        Args:
            config: Broker configuration.
        """
        self._config = config or PolymarketPaperConfig()
        self._connected = False

        # Account state
        self._cash = self._config.initial_cash
        self._positions: Dict[str, Position] = {}

        # Order tracking
        self._orders: Dict[UUID, Order] = {}
        self._fills: List[Fill] = []

        # Market data
        self._quotes: Dict[str, Dict[str, Decimal]] = {}

        # Settled markets: token_id -> won (True/False)
        self._settlements: Dict[str, bool] = {}

    @property
    def name(self) -> str:
        """Broker identifier."""
        return "polymarket_paper"

    @property
    def is_connected(self) -> bool:
        """Returns True if connected."""
        return self._connected

    def connect(self) -> None:
        """Establish connection (no-op for paper)."""
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect (no-op for paper)."""
        self._connected = False

    def update_quotes(self, quotes: Dict[str, Dict[str, float]]) -> None:
        """Update market quotes for tokens.

        Args:
            quotes: Dict mapping token_id to quote dict with bid/ask/last.
        """
        for token_id, quote in quotes.items():
            self._quotes[token_id] = {
                "bid": Decimal(str(quote.get("bid", 0))),
                "ask": Decimal(str(quote.get("ask", 0))),
                "last": Decimal(str(quote.get("last", 0))),
            }

    def submit_order(self, order: Order) -> Order:
        """Submit an order for paper execution.

        Args:
            order: Order to submit.

        Returns:
            Order with updated status.
        """
        self._check_connected()
        self._orders[order.order_id] = order
        order.status = OrderStatus.SUBMITTED

        # Get fill price
        fill_price = self._calculate_fill_price(order)

        if fill_price is None:
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"No quote for {order.symbol}"
            return order

        # Validate price is in valid probability range
        if fill_price < Decimal("0.01") or fill_price > Decimal("0.99"):
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"Price {fill_price} outside valid range [0.01, 0.99]"
            return order

        # Check sufficient cash for buys
        if order.is_buy_side:
            total_cost = fill_price * order.quantity
            fee = self._calculate_fee(fill_price, order.quantity)
            if total_cost + fee > self._cash:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "Insufficient USDC balance"
                return order

        self._execute_fill(order, fill_price)
        return order

    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel an open order.

        Args:
            order_id: ID of order to cancel.

        Returns:
            True if cancelled.
        """
        order = self._orders.get(order_id)
        if order is None or order.is_terminal:
            return False
        order.status = OrderStatus.CANCELLED
        return True

    def get_order_status(self, order_id: UUID) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        """Get all non-terminal orders."""
        return [o for o in self._orders.values() if not o.is_terminal]

    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return {
            token_id: pos
            for token_id, pos in self._positions.items()
            if pos.quantity != 0
        }

    def get_cash_balance(self) -> Decimal:
        """Get current USDC cash balance."""
        return self._cash

    def get_buying_power(self) -> Decimal:
        """Get available buying power."""
        return self._cash

    def get_quote(self, symbol: str) -> Optional[Dict[str, Decimal]]:
        """Get current quote for a token.

        Args:
            symbol: Token ID.

        Returns:
            Quote dict or None.
        """
        return self._quotes.get(symbol)

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Decimal]]:
        """Batch quote request."""
        return {s: self._quotes[s] for s in symbols if s in self._quotes}

    def get_shortable_shares(self, symbol: str) -> int:
        """Not applicable for event contracts."""
        return 0

    def get_borrow_rate(self, symbol: str) -> Decimal:
        """Not applicable for event contracts."""
        return Decimal("0")

    def resolve_market(self, token_id: str, won: bool) -> Decimal:
        """Settle a binary outcome market.

        If the position won, each share pays $1.00. If lost, pays $0.00.
        Position is closed and cash is updated.

        Args:
            token_id: The token that is being resolved.
            won: True if this token's outcome occurred.

        Returns:
            Settlement P&L for this position.
        """
        self._settlements[token_id] = won
        position = self._positions.get(token_id)

        if position is None or position.quantity == 0:
            return Decimal("0")

        qty = position.quantity
        avg_cost = position.avg_cost

        if won:
            # Each share pays $1.00
            payout = Decimal("1.00") * abs(qty)
            pnl = payout - (avg_cost * abs(qty))
            self._cash += payout
        else:
            # Shares expire worthless
            pnl = -(avg_cost * abs(qty))

        # Close the position
        self._positions[token_id] = Position(
            symbol=token_id,
            quantity=0,
            avg_cost=Decimal("0"),
            realized_pnl=pnl,
            last_fill_at=datetime.now(timezone.utc),
        )

        logger.info(
            "Market resolved: %s %s — P&L: $%.2f",
            token_id[:12],
            "WON" if won else "LOST",
            pnl,
        )

        return pnl

    def _check_connected(self) -> None:
        """Raise if not connected."""
        if not self._connected:
            raise RuntimeError("Paper broker not connected")

    def _calculate_fee(self, price: Decimal, quantity: int) -> Decimal:
        """Calculate taker fee for a trade.

        Args:
            price: Fill price (probability).
            quantity: Number of shares.

        Returns:
            Fee amount in USDC.
        """
        if self._config.taker_fee_bps == 0:
            return Decimal("0")

        fee_rate = Decimal(self._config.taker_fee_bps) / Decimal("10000")
        return (price * quantity * fee_rate).quantize(Decimal("0.01"))

    def _calculate_fill_price(self, order: Order) -> Optional[Decimal]:
        """Calculate fill price with optional slippage.

        Args:
            order: The order being filled.

        Returns:
            Fill price or None if no quote.
        """
        quote = self._quotes.get(order.symbol)
        if quote is None:
            return None

        if order.is_buy_side:
            base_price = quote.get("ask") or quote.get("last")
        else:
            base_price = quote.get("bid") or quote.get("last")

        if base_price is None or base_price <= 0:
            return None

        # Apply slippage
        if self._config.slippage_bps > 0:
            slippage = Decimal(self._config.slippage_bps) / Decimal("10000")
            if order.is_buy_side:
                base_price = base_price * (1 + slippage)
            else:
                base_price = base_price * (1 - slippage)

        # Limit order check
        if order.order_type == OrderType.LIMIT and order.limit_price:
            if order.is_buy_side and base_price > order.limit_price:
                return None
            if order.is_sell_side and base_price < order.limit_price:
                return None
            base_price = order.limit_price

        # Clamp to valid probability range
        base_price = max(Decimal("0.01"), min(Decimal("0.99"), base_price))
        return base_price.quantize(Decimal("0.01"))

    def _execute_fill(self, order: Order, fill_price: Decimal) -> None:
        """Execute order fill and update positions/cash.

        Args:
            order: Order being filled.
            fill_price: Price of the fill.
        """
        quantity = order.quantity
        fee = self._calculate_fee(fill_price, quantity)

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=fill_price,
            commission=fee,
            filled_at=datetime.now(timezone.utc),
        )
        self._fills.append(fill)

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = quantity
        order.avg_fill_price = fill_price
        order.filled_at = datetime.now(timezone.utc)

        # Update cash
        notional = fill_price * quantity
        if order.is_buy_side:
            self._cash -= notional + fee
        else:
            self._cash += notional - fee

        # Update position
        self._update_position(order.symbol, order.side, quantity, fill_price)

    def _update_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Decimal,
    ) -> None:
        """Update position after a fill.

        Args:
            symbol: Token ID.
            side: Order side.
            quantity: Number of shares.
            price: Fill price.
        """
        current = self._positions.get(symbol)

        if current is None:
            new_qty = quantity if side in (OrderSide.BUY, OrderSide.COVER) else -quantity
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_cost=price,
                last_fill_at=datetime.now(timezone.utc),
            )
            return

        delta = quantity if side in (OrderSide.BUY, OrderSide.COVER) else -quantity
        new_qty = current.quantity + delta

        if new_qty == 0:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_cost=Decimal("0"),
                last_fill_at=datetime.now(timezone.utc),
            )
        elif (current.quantity >= 0 and delta > 0) or (
            current.quantity <= 0 and delta < 0
        ):
            total_cost = abs(current.quantity) * current.avg_cost + abs(delta) * price
            new_avg = total_cost / abs(new_qty)
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_cost=new_avg.quantize(Decimal("0.0001")),
                last_fill_at=datetime.now(timezone.utc),
            )
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_cost=current.avg_cost,
                last_fill_at=datetime.now(timezone.utc),
            )
