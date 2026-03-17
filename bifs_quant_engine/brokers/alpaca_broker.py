"""Alpaca broker adapter for live trading.

Provides integration with Alpaca's trading API for live and paper trading.
Implements the BrokerAdapter protocol.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID

from bifs_quant_engine.core.enums import OrderSide, OrderStatus, OrderType
from bifs_quant_engine.core.models import Order, Fill, Position
from bifs_quant_engine.core.protocols import BrokerAdapter


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca broker.
    
    Attributes:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        base_url: API base URL (paper or live)
        paper: Whether this is a paper trading account
    """
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    paper: bool = True
    
    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        """Create config from environment variables."""
        return cls(
            api_key=os.environ.get("ALPACA_API_KEY", ""),
            api_secret=os.environ.get("ALPACA_API_SECRET", ""),
            base_url=os.environ.get(
                "ALPACA_BASE_URL", 
                "https://paper-api.alpaca.markets"
            ),
            paper="paper" in os.environ.get("ALPACA_BASE_URL", "paper").lower(),
        )


class AlpacaBroker(BrokerAdapter):
    """Alpaca trading broker adapter.
    
    Implements the BrokerAdapter protocol for trading via Alpaca's API.
    Supports both paper and live trading accounts.
    
    Requires alpaca-trade-api package:
        pip install alpaca-trade-api
    
    Example:
        config = AlpacaConfig.from_env()
        broker = AlpacaBroker(config)
        broker.connect()1
        
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        result = broker.submit_order(order)
    """
    
    def __init__(self, config: Optional[AlpacaConfig] = None) -> None:
        """Initialize Alpaca broker.
        
        Args:
            config: Broker configuration, attempts env vars if not provided
        """
        self._config = config or AlpacaConfig.from_env()
        self._api = None
        self._connected = False
        
        # Local order tracking for mapping
        self._order_map: Dict[UUID, str] = {}  # local_id -> alpaca_id
        self._reverse_map: Dict[str, UUID] = {}  # alpaca_id -> local_id
        
    @property
    def name(self) -> str:
        """Broker identifier."""
        return "alpaca"
    
    @property
    def is_connected(self) -> bool:
        """Returns True if connected to broker."""
        return self._connected and self._api is not None
    
    def connect(self) -> None:
        """Establish connection to Alpaca API."""
        try:
            # Lazy import to avoid dependency issues
            import alpaca_trade_api as tradeapi
            
            self._api = tradeapi.REST(
                key_id=self._config.api_key,
                secret_key=self._config.api_secret,
                base_url=self._config.base_url,
            )
            
            # Verify connection
            self._api.get_account()
            self._connected = True
            
        except ImportError:
            raise ImportError(
                "alpaca-trade-api package required. "
                "Install with: pip install alpaca-trade-api"
            )
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Alpaca: {e}")
    
    def disconnect(self) -> None:
        """Gracefully disconnect from broker."""
        self._api = None
        self._connected = False
        
    def submit_order(self, order: Order) -> Order:
        """Submit an order to Alpaca.
        
        Args:
            order: The order to submit
            
        Returns:
            Order with updated status and broker_order_id
        """
        self._check_connected()
        
        try:
            # Map order side
            side = "buy" if order.is_buy_side else "sell"
            
            # Map order type
            if order.order_type == OrderType.MARKET:
                order_type = "market"
                limit_price = None
            elif order.order_type == OrderType.LIMIT:
                order_type = "limit"
                limit_price = float(order.limit_price) if order.limit_price else None
            else:
                order_type = "market"
                limit_price = None
            
            # Submit to Alpaca
            alpaca_order = self._api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                type=order_type,
                time_in_force="day",
                limit_price=limit_price,
                client_order_id=str(order.order_id),
            )
            
            # Track mapping
            self._order_map[order.order_id] = alpaca_order.id
            self._reverse_map[alpaca_order.id] = order.order_id
            
            # Update order
            order.broker_order_id = alpaca_order.id
            order.status = self._map_status(alpaca_order.status)
            
            # Update fill info if filled
            if alpaca_order.filled_qty:
                order.filled_quantity = int(alpaca_order.filled_qty)
            if alpaca_order.filled_avg_price:
                order.avg_fill_price = Decimal(str(alpaca_order.filled_avg_price))
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(e)
            return order
    
    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation succeeded
        """
        self._check_connected()
        
        alpaca_id = self._order_map.get(order_id)
        if alpaca_id is None:
            return False
            
        try:
            self._api.cancel_order(alpaca_id)
            return True
        except Exception:
            return False
    
    def get_order_status(self, order_id: UUID) -> Optional[OrderStatus]:
        """Get current status of an order from Alpaca."""
        self._check_connected()
        
        alpaca_id = self._order_map.get(order_id)
        if alpaca_id is None:
            return None
            
        try:
            alpaca_order = self._api.get_order(alpaca_id)
            return self._map_status(alpaca_order.status)
        except Exception:
            return None
    
    def get_open_orders(self) -> List[Order]:
        """Get all open/pending orders from Alpaca."""
        self._check_connected()
        
        try:
            alpaca_orders = self._api.list_orders(status="open")
            return [self._convert_order(ao) for ao in alpaca_orders]
        except Exception:
            return []
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions from Alpaca (source of truth)."""
        self._check_connected()
        
        try:
            alpaca_positions = self._api.list_positions()
            return {
                p.symbol: Position(
                    symbol=p.symbol,
                    quantity=int(p.qty),
                    avg_cost=Decimal(str(p.avg_entry_price)),
                )
                for p in alpaca_positions
            }
        except Exception:
            return {}
    
    def get_cash_balance(self) -> Decimal:
        """Get current cash balance from Alpaca."""
        self._check_connected()
        
        try:
            account = self._api.get_account()
            return Decimal(str(account.cash))
        except Exception:
            return Decimal("0")
    
    def get_buying_power(self) -> Decimal:
        """Get available buying power from Alpaca."""
        self._check_connected()
        
        try:
            account = self._api.get_account()
            return Decimal(str(account.buying_power))
        except Exception:
            return Decimal("0")
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Decimal]]:
        """Get current quote for symbol from Alpaca."""
        self._check_connected()
        
        try:
            quote = self._api.get_latest_quote(symbol)
            return {
                "bid": Decimal(str(quote.bp)),
                "ask": Decimal(str(quote.ap)),
                "last": Decimal(str((quote.bp + quote.ap) / 2)),
            }
        except Exception:
            return None
    
    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Decimal]]:
        """Batch quote request from Alpaca."""
        self._check_connected()
        
        result = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                result[symbol] = quote
        return result
    
    def get_shortable_shares(self, symbol: str) -> int:
        """Get number of shares available to short.
        
        Note: Alpaca doesn't provide this directly, returns 0.
        """
        return 0
    
    def get_borrow_rate(self, symbol: str) -> Decimal:
        """Get annualized borrow rate for shorting.
        
        Note: Alpaca doesn't provide this directly, returns 0.
        """
        return Decimal("0")
    
    def _check_connected(self) -> None:
        """Raise if not connected."""
        if not self.is_connected:
            raise RuntimeError("Broker not connected")
    
    def _map_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to internal status."""
        status_map = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.ACKNOWLEDGED,
            "pending_new": OrderStatus.PENDING,
            "accepted_for_bidding": OrderStatus.ACKNOWLEDGED,
            "stopped": OrderStatus.ACKNOWLEDGED,
            "rejected": OrderStatus.REJECTED,
            "suspended": OrderStatus.ACKNOWLEDGED,
            "calculated": OrderStatus.ACKNOWLEDGED,
            "held": OrderStatus.ACKNOWLEDGED,
            "filled": OrderStatus.FILLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "done_for_day": OrderStatus.CANCELLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "replaced": OrderStatus.CANCELLED,
            "pending_cancel": OrderStatus.ACKNOWLEDGED,
            "pending_replace": OrderStatus.ACKNOWLEDGED,
        }
        return status_map.get(alpaca_status.lower(), OrderStatus.PENDING)
    
    def _convert_order(self, alpaca_order) -> Order:
        """Convert Alpaca order to internal Order model."""
        # Determine side
        if alpaca_order.side == "buy":
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL
        
        # Determine type
        if alpaca_order.type == "limit":
            order_type = OrderType.LIMIT
        else:
            order_type = OrderType.MARKET
        
        order = Order(
            symbol=alpaca_order.symbol,
            side=side,
            quantity=int(alpaca_order.qty),
            order_type=order_type,
        )
        
        order.broker_order_id = alpaca_order.id
        order.status = self._map_status(alpaca_order.status)
        
        if alpaca_order.filled_qty:
            order.filled_quantity = int(alpaca_order.filled_qty)
        if alpaca_order.filled_avg_price:
            order.avg_fill_price = Decimal(str(alpaca_order.filled_avg_price))
        if alpaca_order.limit_price:
            order.limit_price = Decimal(str(alpaca_order.limit_price))
        
        # Track in our mapping
        self._order_map[order.order_id] = alpaca_order.id
        self._reverse_map[alpaca_order.id] = order.order_id
        
        return order
