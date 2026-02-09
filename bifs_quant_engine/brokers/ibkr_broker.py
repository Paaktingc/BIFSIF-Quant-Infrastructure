"""Interactive Brokers adapter using ib_insync.

Provides integration with Interactive Brokers via TWS or IB Gateway.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID
import os

from bifs_quant_engine.core.enums import OrderSide, OrderStatus, OrderType
from bifs_quant_engine.core.models import Order, Position
from bifs_quant_engine.core.protocols import BrokerAdapter


@dataclass
class IBKRConfig:
    """Configuration for Interactive Brokers connection.
    
    Attributes:
        host: Host where TWS/Gateway is running
        port: Port where TWS/Gateway is listening (7497=paper, 7496=live)
        client_id: Unique client ID for this connection
    """
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1

    @classmethod
    def from_env(cls) -> "IBKRConfig":
        """Create config from environment variables."""
        return cls(
            host=os.environ.get("IBKR_HOST", "127.0.0.1"),
            port=int(os.environ.get("IBKR_PORT", "7497")),
            client_id=int(os.environ.get("IBKR_CLIENT_ID", "1")),
        )


class IBKRBroker(BrokerAdapter):
    """Interactive Brokers adapter using ib_insync.
    
    Requires functionality TWS or IB Gateway to be running and configured
    to accept API connections.
    """

    def __init__(self, config: Optional[IBKRConfig] = None) -> None:
        """Initialize IBKR broker.
        
        Args:
            config: Broker configuration
        """
        self._config = config or IBKRConfig.from_env()
        self._ib = None
        self._connected = False
        
        # Local order tracking
        self._order_map: Dict[UUID, int] = {}  # local_id -> ib_perm_id
        self._reverse_map: Dict[int, UUID] = {}  # ib_perm_id -> local_id

    @property
    def name(self) -> str:
        return "ibkr"

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ib is not None and self._ib.isConnected()

    def connect(self) -> None:
        """Connect to TWS/Gateway."""
        try:
            import ib_insync
        except ImportError:
            raise ImportError(
                "ib_insync package required. Install with: pip install ib_insync"
            )

        self._ib = ib_insync.IB()
        try:
            self._ib.connect(
                self._config.host,
                self._config.port,
                clientId=self._config.client_id
            )
            self._connected = True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to IBKR: {e}")

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib:
            self._ib.disconnect()
        self._connected = False
        self._ib = None

    def submit_order(self, order: Order) -> Order:
        """Submit an order to IBKR."""
        self._check_connected()
        from ib_insync import Order as IBOrder
        from ib_insync import Stock, MarketOrder, LimitOrder

        # Create IB contract (assuming US Stocks for now)
        contract = Stock(order.symbol, 'SMART', 'USD')
        
        # Create IB Order
        action = 'BUY' if order.is_buy_side else 'SELL'
        quantity = order.quantity
        
        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, quantity)
        elif order.order_type == OrderType.LIMIT:
            if not order.limit_price:
                raise ValueError("Limit price required for LIMIT order")
            ib_order = LimitOrder(action, quantity, float(order.limit_price))
        else:
            raise NotImplementedError(f"Order type {order.order_type} not supported")

        # Set reference
        ib_order.orderRef = str(order.order_id)

        try:
            trade = self._ib.placeOrder(contract, ib_order)
            
            # Map IDs
            # Note: IB has multiple IDs (orderId, permId). permId is permanent.
            # However, trade.order.orderId is what's used for tracking initially.
            # We'll use orderRef to match back if needed.
            
            # For simplicity in this sync wrapper, we might not get immediate fill
            # status updates without the event loop running processing. 
            # ib_insync is designed for async.
            self._ib.sleep(0.1)  # Allow a brief moment for update
            
            # Map status
            order.status = self._map_status(trade.orderStatus.status)
            if trade.orderStatus.filled:
                order.filled_quantity = int(trade.orderStatus.filled)
            if trade.orderStatus.avgFillPrice:
                order.avg_fill_price = Decimal(str(trade.orderStatus.avgFillPrice))
                
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(e)
            return order

    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel an order."""
        self._check_connected()
        
        # We need to find the open trade/order
        # Since we might not have the IB order object cached, we search open orders
        open_trades = self._ib.openTrades()
        for trade in open_trades:
            if trade.order.orderRef == str(order_id):
                self._ib.cancelOrder(trade.order)
                return True
        return False

    def get_order_status(self, order_id: UUID) -> Optional[OrderStatus]:
        self._check_connected()
        
        # Check open trades first
        open_trades = self._ib.openTrades()
        for trade in open_trades:
            if trade.order.orderRef == str(order_id):
                return self._map_status(trade.orderStatus.status)
        
        # TODO: Implement historical order lookup if needed
        return None

    def get_open_orders(self) -> List[Order]:
        self._check_connected()
        
        orders = []
        for trade in self._ib.openTrades():
            order = self._convert_trade(trade)
            orders.append(order)
        return orders

    def get_positions(self) -> Dict[str, Position]:
        self._check_connected()
        
        positions = {}
        for p in self._ib.positions():
            if p.contract.secType != 'STK':
                continue
                
            scale = 1  # 1 for long, -1 for short? 
            # IB reports size signed. 
            
            positions[p.contract.symbol] = Position(
                symbol=p.contract.symbol,
                quantity=int(p.position),
                avg_cost=Decimal(str(p.avgCost))
            )
        return positions

    def get_cash_balance(self) -> Decimal:
        self._check_connected()
        
        # Account summary
        tags = self._ib.accountSummary()
        for tag in tags:
            if tag.tag == 'TotalCashValue' and tag.currency == 'USD':
                 return Decimal(tag.value)
        return Decimal("0")

    def get_buying_power(self) -> Decimal:
        self._check_connected()
        
        tags = self._ib.accountSummary()
        for tag in tags:
             if tag.tag == 'BuyingPower' and tag.currency == 'USD':
                 return Decimal(tag.value)
        return Decimal("0")

    def get_quote(self, symbol: str) -> Optional[Dict[str, Decimal]]:
        self._check_connected()
        from ib_insync import Stock
        
        contract = Stock(symbol, 'SMART', 'USD')
        self._ib.qualifyContracts(contract)
        
        ticker = self._ib.reqMktData(contract, '', True, False)
        self._ib.sleep(0.5)  # Wait for data
        
        if ticker.hasBidAsk():
             return {
                "bid": Decimal(str(ticker.bid)),
                "ask": Decimal(str(ticker.ask)),
                "last": Decimal(str(ticker.last if ticker.last else (ticker.bid + ticker.ask)/2))
             }
        return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Decimal]]:
        results = {}
        for symbol in symbols:
            q = self.get_quote(symbol)
            if q:
                results[symbol] = q
        return results

    def get_shortable_shares(self, symbol: str) -> int:
        return 0  # Not implemented yet

    def get_borrow_rate(self, symbol: str) -> Decimal:
        return Decimal("0")  # Not implemented yet

    def _check_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("IBKR Broker not connected")

    def _map_status(self, status: str) -> OrderStatus:
        # Map ib_insync status strings
        # Api: PendingSubmit, PendingCancel, PreSubmitted, Submitted, ApiCancelled, Cancelled, Filled, Inactive
        s = status.lower()
        if s in ('pendingsubmit', 'presubmitted'):
            return OrderStatus.PENDING
        elif s == 'submitted':
            return OrderStatus.ACKNOWLEDGED
        elif s == 'filled':
            return OrderStatus.FILLED
        elif s in ('cancelled', 'apicancelled'):
            return OrderStatus.CANCELLED
        elif s == 'inactive':
            return OrderStatus.REJECTED
        return OrderStatus.PENDING

    def _convert_trade(self, trade) -> Order:
        # Convert ib_insync Trade object to local Order
        side = OrderSide.BUY if trade.order.action == 'BUY' else OrderSide.SELL
        order_type = OrderType.MARKET if trade.order.orderType == 'MKT' else OrderType.LIMIT
        
        o = Order(
            symbol=trade.contract.symbol,
            side=side,
            quantity=int(trade.order.totalQuantity),
            order_type=order_type
        )
        # Re-attach ID if we can parse it from ref, else new
        try:
            o.order_id = UUID(trade.order.orderRef)
        except:
            pass
            
        o.status = self._map_status(trade.orderStatus.status)
        return o
