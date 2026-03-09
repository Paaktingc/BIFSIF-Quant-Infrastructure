"""Paper trading broker implementation.

Provides a realistic simulated broker for paper trading and backtesting:
- Order book simulation with fill logic
- Position tracking
- Cash and buying power management
- Latency simulation
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID

from bifs_quant_engine.core.enums import OrderSide, OrderStatus, OrderType
from bifs_quant_engine.core.models import Order, Fill, Position
from bifs_quant_engine.core.protocols import BrokerAdapter


@dataclass
class PaperBrokerConfig:
    """Configuration for paper broker behavior.
    
    Attributes:
        initial_cash: Starting cash balance
        latency_ms: Simulated network latency in milliseconds
        fill_probability: Probability of order fill (0.0 to 1.0)
        slippage_bps: Default slippage in basis points
        commission_per_share: Commission charged per share
        min_commission: Minimum commission per order
    """
    initial_cash: Decimal = Decimal("100000")
    latency_ms: int = 50
    fill_probability: float = 1.0
    slippage_bps: int = 5
    commission_per_share: Decimal = Decimal("0.005")
    min_commission: Decimal = Decimal("1.00")


class PaperBroker(BrokerAdapter):
    """Simulated paper trading broker.
    
    Implements the BrokerAdapter protocol for paper trading with realistic
    simulation of order execution, position tracking, and cash management.
    
    Features:
    - Market and limit order support
    - Configurable slippage and latency
    - Position tracking with average cost basis
    - Commission calculation
    - Short selling support
    
    Example:
        config = PaperBrokerConfig(initial_cash=Decimal("100000"))
        broker = PaperBroker(config)
        broker.connect()
        
        # Set market prices
        broker.update_quotes({"AAPL": {"bid": 149.50, "ask": 150.50, "last": 150.00}})
        
        # Submit order
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        filled_order = broker.submit_order(order)
    """
    
    def __init__(self, config: Optional[PaperBrokerConfig] = None) -> None:
        """Initialize paper broker.
        
        Args:
            config: Broker configuration, uses defaults if not provided
        """
        self._config = config or PaperBrokerConfig()
        self._connected = False
        
        # Account state
        self._cash = self._config.initial_cash
        self._positions: Dict[str, Position] = {}
        
        # Order tracking
        self._orders: Dict[UUID, Order] = {}
        self._fills: List[Fill] = []
        
        # Market data
        self._quotes: Dict[str, Dict[str, Decimal]] = {}
        
    @property
    def name(self) -> str:
        """Broker identifier."""
        return "paper"
    
    @property
    def is_connected(self) -> bool:
        """Returns True if connected to broker."""
        return self._connected
    
    def connect(self) -> None:
        """Establish connection to broker."""
        self._connected = True
        
    def disconnect(self) -> None:
        """Gracefully disconnect from broker."""
        self._connected = False
        
    def update_quotes(self, quotes: Dict[str, Dict[str, float]]) -> None:
        """Update market quotes for symbols.
        
        Args:
            quotes: Dict mapping symbol to quote dict with bid/ask/last
        """
        for symbol, quote in quotes.items():
            self._quotes[symbol] = {
                "bid": Decimal(str(quote.get("bid", 0))),
                "ask": Decimal(str(quote.get("ask", 0))),
                "last": Decimal(str(quote.get("last", 0))),
            }
    
    def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker.
        
        Simulates order execution with configurable latency and slippage.
        
        Args:
            order: The order to submit
            
        Returns:
            Order with updated status and fill information
        """
        self._check_connected()
        
        # Simulate network latency
        if self._config.latency_ms > 0:
            time.sleep(self._config.latency_ms / 1000.0)
        
        # Store order
        self._orders[order.order_id] = order
        order.status = OrderStatus.SUBMITTED
        
        # Attempt to fill
        fill_price = self._calculate_fill_price(order)
        
        if fill_price is None:
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"No quote available for {order.symbol}"
            return order
        
        # Check if order should fill (based on fill probability)
        if random.random() > self._config.fill_probability:
            order.status = OrderStatus.ACKNOWLEDGED
            return order
        
        # Execute the fill
        self._execute_fill(order, fill_price)
        
        return order
    
    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation succeeded
        """
        self._check_connected()
        
        order = self._orders.get(order_id)
        if order is None:
            return False
            
        if order.is_terminal:
            return False
            
        order.status = OrderStatus.CANCELLED
        return True
    
    def get_order_status(self, order_id: UUID) -> Optional[OrderStatus]:
        """Get current status of an order.
        
        Args:
            order_id: ID of order to look up
            
        Returns:
            Order status or None if not found
        """
        order = self._orders.get(order_id)
        return order.status if order else None
    
    def get_open_orders(self) -> List[Order]:
        """Get all open/pending orders."""
        return [o for o in self._orders.values() if not o.is_terminal]
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions (source of truth during paper trading)."""
        return {
            symbol: pos 
            for symbol, pos in self._positions.items() 
            if pos.quantity != 0
        }
    
    def get_cash_balance(self) -> Decimal:
        """Get current cash balance."""
        return self._cash
    
    def get_buying_power(self) -> Decimal:
        """Get available buying power.
        
        For paper trading, this equals cash balance (no margin).
        """
        return self._cash
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Decimal]]:
        """Get current quote for symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Dict with bid, ask, last prices or None
        """
        return self._quotes.get(symbol)
    
    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Decimal]]:
        """Batch quote request.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dict mapping symbol to quote dict
        """
        return {s: self._quotes[s] for s in symbols if s in self._quotes}
    
    def get_shortable_shares(self, symbol: str) -> int:
        """Get number of shares available to short.
        
        Paper broker allows unlimited shorting.
        """
        return 1_000_000  # Virtually unlimited for paper trading
    
    def get_borrow_rate(self, symbol: str) -> Decimal:
        """Get annualized borrow rate for shorting.
        
        Paper broker uses a flat 2% rate.
        """
        return Decimal("0.02")
    
    def get_fills(self, since: Optional[datetime] = None) -> List[Fill]:
        """Get fills, optionally since a timestamp.
        
        Args:
            since: Optional cutoff timestamp
            
        Returns:
            List of fills
        """
        if since is None:
            return list(self._fills)
        return [f for f in self._fills if f.filled_at >= since]
    
    def _check_connected(self) -> None:
        """Raise if not connected."""
        if not self._connected:
            raise RuntimeError("Broker not connected")
    
    def _calculate_fill_price(self, order: Order) -> Optional[Decimal]:
        """Calculate fill price with slippage.
        
        Args:
            order: The order being filled
            
        Returns:
            Fill price or None if no quote available
        """
        quote = self._quotes.get(order.symbol)
        if quote is None:
            return None
        
        # Use appropriate side of spread
        if order.is_buy_side:
            base_price = quote.get("ask") or quote.get("last")
        else:
            base_price = quote.get("bid") or quote.get("last")
        
        if base_price is None or base_price <= 0:
            return None
        
        # Apply slippage
        slippage_pct = Decimal(self._config.slippage_bps) / Decimal("10000")
        if order.is_buy_side:
            fill_price = base_price * (1 + slippage_pct)
        else:
            fill_price = base_price * (1 - slippage_pct)
        
        # For limit orders, check if fill is possible
        if order.order_type == OrderType.LIMIT and order.limit_price:
            if order.is_buy_side and fill_price > order.limit_price:
                return None  # Can't fill above limit
            if order.is_sell_side and fill_price < order.limit_price:
                return None  # Can't fill below limit
            # Fill at limit price if favorable
            fill_price = order.limit_price
        
        return fill_price.quantize(Decimal("0.01"))
    
    def _execute_fill(self, order: Order, fill_price: Decimal) -> None:
        """Execute order fill and update positions.
        
        Args:
            order: Order being filled
            fill_price: Price of the fill
        """
        quantity = order.quantity
        
        # Calculate commission
        commission = max(
            self._config.min_commission,
            self._config.commission_per_share * quantity
        )
        
        # Create fill record
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=fill_price,
            commission=commission,
            filled_at=datetime.now(timezone.utc),
        )
        self._fills.append(fill)
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = quantity
        order.avg_fill_price = fill_price
        
        # Update cash
        notional = fill_price * quantity
        if order.is_buy_side:
            self._cash -= notional + commission
        else:
            self._cash += notional - commission
        
        # Update position
        self._update_position(order.symbol, order.side, quantity, fill_price)
    
    def _update_position(
        self, 
        symbol: str, 
        side: OrderSide, 
        quantity: int, 
        price: Decimal
    ) -> None:
        """Update position after a fill.
        
        Args:
            symbol: Ticker symbol
            side: Order side
            quantity: Number of shares
            price: Fill price
        """
        current = self._positions.get(symbol)
        
        if current is None:
            # New position
            if side in (OrderSide.BUY, OrderSide.COVER):
                new_qty = quantity
            else:
                new_qty = -quantity
                
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_cost=price,
            )
            return
        
        # Update existing position
        current_qty = current.quantity
        current_cost = current.avg_cost
        
        if side in (OrderSide.BUY, OrderSide.COVER):
            delta_qty = quantity
        else:
            delta_qty = -quantity
        
        new_qty = current_qty + delta_qty
        
        if new_qty == 0:
            # Position closed
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_cost=Decimal("0"),
            )
        elif (current_qty >= 0 and delta_qty > 0) or (current_qty <= 0 and delta_qty < 0):
            # Adding to position - recalculate avg cost
            total_cost = abs(current_qty) * current_cost + abs(delta_qty) * price
            new_avg_cost = total_cost / abs(new_qty)
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_cost=new_avg_cost.quantize(Decimal("0.0001")),
            )

        elif (current_qty > 0 and delta_qty < 0) or (current_qty < 0 and delta_qty > 0):
            # Reducing position - keep original cost basis
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_cost=current_cost,
            )

        else:
            # Reducing position - keep original cost basis
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_cost=current_cost,
            )

        
