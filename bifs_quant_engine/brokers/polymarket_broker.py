"""Polymarket broker adapter.

Implements the BrokerAdapter protocol for Polymarket's CLOB (Central Limit
Order Book) via the py-clob-client SDK. Supports order submission, position
tracking, and quote retrieval for event contracts and crypto short-duration
markets.

Requires environment variables:
    POLYMARKET_PRIVATE_KEY: Polygon wallet private key
    POLYMARKET_FUNDER: Funder (proxy) address for Magic wallets
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID

from bifs_quant_engine.core.enums import OrderSide, OrderStatus, OrderType
from bifs_quant_engine.core.models import Fill, Order, Position

logger = logging.getLogger(__name__)


@dataclass
class PolymarketConfig:
    """Configuration for the Polymarket broker.

    Attributes:
        host: CLOB API endpoint.
        chain_id: Polygon chain ID (137 for mainnet).
        private_key: Wallet private key (loaded from env).
        funder: Funder/proxy address (loaded from env).
        signature_type: 0 for EOA wallets, 1 for Magic/email wallets.
    """

    host: str = "https://clob.polymarket.com"
    chain_id: int = 137
    private_key: str = ""
    funder: str = ""
    signature_type: int = 0


class PolymarketBroker:
    """Polymarket CLOB broker adapter.

    Wraps py-clob-client to expose the same interface as other broker
    adapters (PaperBroker, IBKRBroker). Uses USDC on Polygon for
    settlement.

    Example:
        config = PolymarketConfig()
        broker = PolymarketBroker(config)
        broker.connect()

        order = Order(
            symbol="<token_id>",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.25"),
            strategy_id="tz_info_arb",
        )
        filled = broker.submit_order(order)
    """

    def __init__(self, config: Optional[PolymarketConfig] = None) -> None:
        """Initialize Polymarket broker.

        Args:
            config: Broker configuration. Loads secrets from env if not set.
        """
        self._config = config or PolymarketConfig()
        self._connected = False
        self._client = None

        # Account state
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[UUID, Order] = {}
        self._fills: List[Fill] = []
        self._cash = Decimal("0")

        # Load secrets from environment if not provided
        if not self._config.private_key:
            self._config.private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
        if not self._config.funder:
            self._config.funder = os.environ.get("POLYMARKET_FUNDER", "")

    @property
    def name(self) -> str:
        """Broker identifier."""
        return "polymarket"

    @property
    def is_connected(self) -> bool:
        """Returns True if connected to Polymarket CLOB."""
        return self._connected

    def connect(self) -> None:
        """Establish connection to Polymarket CLOB API.

        Derives API credentials from the wallet private key and verifies
        connectivity.

        Raises:
            RuntimeError: If private key is not configured.
            ConnectionError: If CLOB API is unreachable.
        """
        if not self._config.private_key:
            raise RuntimeError(
                "POLYMARKET_PRIVATE_KEY not set. "
                "Set via environment variable or config."
            )

        try:
            from py_clob_client.client import ClobClient

            self._client = ClobClient(
                self._config.host,
                key=self._config.private_key,
                chain_id=self._config.chain_id,
                funder=self._config.funder or None,
                signature_type=self._config.signature_type,
            )

            # Derive or create API credentials
            self._client.set_api_creds(self._client.create_or_derive_api_creds())

            logger.info("Connected to Polymarket CLOB at %s", self._config.host)
            self._connected = True

        except ImportError:
            raise RuntimeError(
                "py-clob-client not installed. Run: pip install py-clob-client"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Polymarket: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from Polymarket."""
        self._client = None
        self._connected = False
        logger.info("Disconnected from Polymarket CLOB")

    def submit_order(self, order: Order) -> Order:
        """Submit an order to Polymarket CLOB.

        Maps internal Order to py-clob-client order format.
        symbol = Polymarket token_id, limit_price = probability (0.01-0.99),
        quantity = number of outcome shares.

        Args:
            order: Order to submit.

        Returns:
            Order with updated status and broker_order_id.
        """
        self._check_connected()
        self._orders[order.order_id] = order

        try:
            from py_clob_client.order_builder.constants import BUY, SELL

            clob_side = BUY if order.is_buy_side else SELL
            price = float(order.limit_price) if order.limit_price else 0.5
            size = order.quantity

            # Map order type
            if order.order_type == OrderType.FOK:
                time_in_force = "FOK"
            elif order.order_type == OrderType.GTD:
                time_in_force = "GTD"
            else:
                time_in_force = "GTC"

            signed_order = self._client.create_order(
                {
                    "token_id": order.symbol,
                    "price": price,
                    "size": size,
                    "side": clob_side,
                    "fee_rate_bps": 0,
                }
            )

            response = self._client.post_order(
                signed_order, order_type=time_in_force
            )

            if response and response.get("orderID"):
                order.broker_order_id = response["orderID"]
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.now(timezone.utc)
                logger.info(
                    "Order submitted: %s %s %d @ %s (broker_id=%s)",
                    "BUY" if order.is_buy_side else "SELL",
                    order.symbol[:12],
                    order.quantity,
                    order.limit_price,
                    order.broker_order_id,
                )
            else:
                order.status = OrderStatus.REJECTED
                order.reject_reason = f"CLOB rejected order: {response}"
                logger.warning("Order rejected: %s", order.reject_reason)

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(e)
            logger.error("Order submission failed: %s", e)

        return order

    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel an open order on Polymarket.

        Args:
            order_id: Internal order ID.

        Returns:
            True if cancellation succeeded.
        """
        self._check_connected()

        order = self._orders.get(order_id)
        if order is None or order.is_terminal:
            return False

        if not order.broker_order_id:
            order.status = OrderStatus.CANCELLED
            return True

        try:
            self._client.cancel(order.broker_order_id)
            order.status = OrderStatus.CANCELLED
            logger.info("Order cancelled: %s", order.broker_order_id)
            return True
        except Exception as e:
            logger.error("Cancel failed for %s: %s", order.broker_order_id, e)
            return False

    def get_order_status(self, order_id: UUID) -> Order:
        """Get current status of an order.

        Args:
            order_id: Internal order ID.

        Returns:
            Order with current status.

        Raises:
            KeyError: If order not found.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")

        # Sync with CLOB if order has a broker ID and is not terminal
        if order.broker_order_id and not order.is_terminal:
            try:
                clob_order = self._client.get_order(order.broker_order_id)
                if clob_order:
                    self._sync_order_status(order, clob_order)
            except Exception as e:
                logger.warning("Failed to sync order %s: %s", order_id, e)

        return order

    def get_open_orders(self) -> List[Order]:
        """Get all open/pending orders."""
        return [o for o in self._orders.values() if not o.is_terminal]

    def get_positions(self) -> Dict[str, Position]:
        """Get current positions (token holdings).

        Returns:
            Dict mapping token_id to Position.
        """
        return {
            token_id: pos
            for token_id, pos in self._positions.items()
            if pos.quantity != 0
        }

    def get_cash_balance(self) -> Decimal:
        """Get USDC balance on Polygon."""
        if not self._connected:
            return self._cash

        try:
            # py-clob-client doesn't expose balance directly;
            # query via web3 or track locally
            return self._cash
        except Exception as e:
            logger.warning("Failed to fetch balance: %s", e)
            return self._cash

    def get_buying_power(self) -> Decimal:
        """Get available buying power (USDC balance)."""
        return self.get_cash_balance()

    def get_quote(self, symbol: str) -> Dict[str, Decimal]:
        """Get current order book quote for a token.

        Args:
            symbol: Polymarket token_id.

        Returns:
            Dict with 'bid', 'ask', 'last' probability prices.
        """
        self._check_connected()

        try:
            book = self._client.get_order_book(symbol)
            best_bid = Decimal("0")
            best_ask = Decimal("1")

            if book and book.get("bids"):
                best_bid = Decimal(str(book["bids"][0]["price"]))
            if book and book.get("asks"):
                best_ask = Decimal(str(book["asks"][0]["price"]))

            mid = (best_bid + best_ask) / 2

            return {
                "bid": best_bid,
                "ask": best_ask,
                "last": mid,
            }
        except Exception as e:
            logger.warning("Failed to get quote for %s: %s", symbol[:12], e)
            return {
                "bid": Decimal("0"),
                "ask": Decimal("1"),
                "last": Decimal("0.5"),
            }

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Decimal]]:
        """Batch quote request for multiple tokens.

        Args:
            symbols: List of token_ids.

        Returns:
            Dict mapping token_id to quote dict.
        """
        return {s: self.get_quote(s) for s in symbols}

    def get_shortable_shares(self, symbol: str) -> int:
        """Not applicable for event contracts."""
        return 0

    def get_borrow_rate(self, symbol: str) -> Decimal:
        """Not applicable for event contracts."""
        return Decimal("0")

    def update_cash(self, amount: Decimal) -> None:
        """Manually set the tracked cash balance.

        Args:
            amount: USDC balance to set.
        """
        self._cash = amount

    def record_fill(self, order: Order, price: Decimal, quantity: int) -> Fill:
        """Record a fill from CLOB trade confirmation.

        Args:
            order: The parent order.
            price: Fill price (probability).
            quantity: Number of shares filled.

        Returns:
            The created Fill record.
        """
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=Decimal("0"),  # Event contracts are fee-free
            filled_at=datetime.now(timezone.utc),
        )
        self._fills.append(fill)

        # Update order state
        order.filled_quantity += quantity
        order.avg_fill_price = price
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now(timezone.utc)
        else:
            order.status = OrderStatus.PARTIAL_FILL

        # Update position
        self._update_position(order.symbol, order.side, quantity, price)

        # Update cash
        notional = price * quantity
        if order.is_buy_side:
            self._cash -= notional
        else:
            self._cash += notional

        return fill

    def _check_connected(self) -> None:
        """Raise if not connected."""
        if not self._connected:
            raise RuntimeError("Polymarket broker not connected")

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
            # Adding to position — recalculate avg cost
            total_cost = abs(current.quantity) * current.avg_cost + abs(delta) * price
            new_avg = total_cost / abs(new_qty)
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_cost=new_avg.quantize(Decimal("0.0001")),
                last_fill_at=datetime.now(timezone.utc),
            )
        else:
            # Reducing position — keep original cost basis
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_cost=current.avg_cost,
                last_fill_at=datetime.now(timezone.utc),
            )

    def _sync_order_status(self, order: Order, clob_order: dict) -> None:
        """Sync internal order state with CLOB response.

        Args:
            order: Internal order to update.
            clob_order: Response from CLOB API.
        """
        status = clob_order.get("status", "").upper()
        if status == "MATCHED":
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
        elif status == "CANCELLED":
            order.status = OrderStatus.CANCELLED
        elif status == "LIVE":
            order.status = OrderStatus.ACKNOWLEDGED
