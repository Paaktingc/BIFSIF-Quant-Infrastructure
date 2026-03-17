"""Tests for the Polymarket paper trading broker."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bifs_quant_engine.brokers.polymarket_paper_broker import (
    PolymarketPaperBroker,
    PolymarketPaperConfig,
)
from bifs_quant_engine.core.enums import OrderSide, OrderStatus, OrderType
from bifs_quant_engine.core.models import Order


@pytest.fixture
def broker() -> PolymarketPaperBroker:
    config = PolymarketPaperConfig(initial_cash=Decimal("10000"))
    b = PolymarketPaperBroker(config)
    b.connect()
    return b


@pytest.fixture
def broker_with_fees() -> PolymarketPaperBroker:
    config = PolymarketPaperConfig(
        initial_cash=Decimal("10000"),
        taker_fee_bps=156,  # 1.56% max for crypto markets
    )
    b = PolymarketPaperBroker(config)
    b.connect()
    return b


class TestPaperBrokerBasics:
    def test_name(self, broker: PolymarketPaperBroker) -> None:
        assert broker.name == "polymarket_paper"

    def test_connect_disconnect(self) -> None:
        b = PolymarketPaperBroker()
        assert not b.is_connected
        b.connect()
        assert b.is_connected
        b.disconnect()
        assert not b.is_connected

    def test_initial_cash(self, broker: PolymarketPaperBroker) -> None:
        assert broker.get_cash_balance() == Decimal("10000")
        assert broker.get_buying_power() == Decimal("10000")

    def test_submit_raises_when_disconnected(self) -> None:
        b = PolymarketPaperBroker()
        order = Order(symbol="t", side=OrderSide.BUY, quantity=10)
        with pytest.raises(RuntimeError, match="not connected"):
            b.submit_order(order)


class TestPaperBrokerOrderExecution:
    def test_buy_limit_order(self, broker: PolymarketPaperBroker) -> None:
        broker.update_quotes({"token_yes": {"bid": 0.24, "ask": 0.26, "last": 0.25}})

        order = Order(
            symbol="token_yes",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.26"),
        )
        result = broker.submit_order(order)

        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 100
        assert result.avg_fill_price == Decimal("0.26")

    def test_buy_deducts_cash(self, broker: PolymarketPaperBroker) -> None:
        broker.update_quotes({"t": {"bid": 0.20, "ask": 0.22, "last": 0.21}})
        order = Order(
            symbol="t",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.22"),
        )
        broker.submit_order(order)
        # 10000 - 0.22 * 100 = 9978
        assert broker.get_cash_balance() == Decimal("9978.00")

    def test_sell_adds_cash(self, broker: PolymarketPaperBroker) -> None:
        broker.update_quotes({"t": {"bid": 0.20, "ask": 0.22, "last": 0.21}})

        # Buy first
        buy = Order(symbol="t", side=OrderSide.BUY, quantity=100,
                    order_type=OrderType.LIMIT, limit_price=Decimal("0.22"))
        broker.submit_order(buy)

        # Update price and sell
        broker.update_quotes({"t": {"bid": 0.50, "ask": 0.52, "last": 0.51}})
        sell = Order(symbol="t", side=OrderSide.SELL, quantity=100,
                     order_type=OrderType.LIMIT, limit_price=Decimal("0.50"))
        broker.submit_order(sell)

        # 10000 - 22 + 50 = 10028
        assert broker.get_cash_balance() == Decimal("10028.00")

    def test_reject_no_quote(self, broker: PolymarketPaperBroker) -> None:
        order = Order(symbol="unknown", side=OrderSide.BUY, quantity=10)
        result = broker.submit_order(order)
        assert result.status == OrderStatus.REJECTED
        assert "No quote" in result.reject_reason

    def test_reject_insufficient_cash(self, broker: PolymarketPaperBroker) -> None:
        broker.update_quotes({"t": {"bid": 0.90, "ask": 0.92, "last": 0.91}})
        # Try to buy way more than cash allows
        order = Order(
            symbol="t",
            side=OrderSide.BUY,
            quantity=100000,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.92"),
        )
        result = broker.submit_order(order)
        assert result.status == OrderStatus.REJECTED
        assert "Insufficient" in result.reject_reason

    def test_limit_order_not_filled_if_price_exceeds(
        self, broker: PolymarketPaperBroker
    ) -> None:
        broker.update_quotes({"t": {"bid": 0.50, "ask": 0.55, "last": 0.52}})
        order = Order(
            symbol="t",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.40"),  # Below ask
        )
        result = broker.submit_order(order)
        assert result.status == OrderStatus.REJECTED


class TestPaperBrokerPositions:
    def test_position_created_on_buy(self, broker: PolymarketPaperBroker) -> None:
        broker.update_quotes({"t": {"bid": 0.30, "ask": 0.32, "last": 0.31}})
        order = Order(symbol="t", side=OrderSide.BUY, quantity=50,
                      order_type=OrderType.LIMIT, limit_price=Decimal("0.32"))
        broker.submit_order(order)

        positions = broker.get_positions()
        assert "t" in positions
        assert positions["t"].quantity == 50
        assert positions["t"].avg_cost == Decimal("0.32")

    def test_position_closed_on_full_sell(self, broker: PolymarketPaperBroker) -> None:
        broker.update_quotes({"t": {"bid": 0.30, "ask": 0.32, "last": 0.31}})
        buy = Order(symbol="t", side=OrderSide.BUY, quantity=50,
                    order_type=OrderType.LIMIT, limit_price=Decimal("0.32"))
        broker.submit_order(buy)

        sell = Order(symbol="t", side=OrderSide.SELL, quantity=50,
                     order_type=OrderType.LIMIT, limit_price=Decimal("0.30"))
        broker.submit_order(sell)

        assert "t" not in broker.get_positions()


class TestBinarySettlement:
    def test_resolve_market_win(self, broker: PolymarketPaperBroker) -> None:
        broker.update_quotes({"yes_token": {"bid": 0.20, "ask": 0.22, "last": 0.21}})
        order = Order(
            symbol="yes_token",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.22"),
        )
        broker.submit_order(order)
        cash_after_buy = broker.get_cash_balance()

        pnl = broker.resolve_market("yes_token", won=True)

        # Payout: 100 * $1.00 = $100. Cost was 100 * $0.22 = $22. PnL = $78
        assert pnl == Decimal("78.00")
        assert broker.get_cash_balance() == cash_after_buy + Decimal("100")
        assert "yes_token" not in broker.get_positions()

    def test_resolve_market_loss(self, broker: PolymarketPaperBroker) -> None:
        broker.update_quotes({"yes_token": {"bid": 0.20, "ask": 0.22, "last": 0.21}})
        order = Order(
            symbol="yes_token",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.22"),
        )
        broker.submit_order(order)
        cash_after_buy = broker.get_cash_balance()

        pnl = broker.resolve_market("yes_token", won=False)

        # Shares expire worthless. PnL = -$22
        assert pnl == Decimal("-22.00")
        assert broker.get_cash_balance() == cash_after_buy  # No payout
        assert "yes_token" not in broker.get_positions()

    def test_resolve_no_position(self, broker: PolymarketPaperBroker) -> None:
        pnl = broker.resolve_market("nonexistent", won=True)
        assert pnl == Decimal("0")


class TestTakerFees:
    def test_fee_applied_on_buy(self, broker_with_fees: PolymarketPaperBroker) -> None:
        broker_with_fees.update_quotes({"t": {"bid": 0.49, "ask": 0.51, "last": 0.50}})
        order = Order(
            symbol="t",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.51"),
        )
        broker_with_fees.submit_order(order)

        # Notional: 0.51 * 100 = 51.00
        # Fee: 51.00 * 0.0156 = 0.7956 -> 0.80
        # Cash: 10000 - 51.00 - 0.80 = 9948.20
        assert broker_with_fees.get_cash_balance() == Decimal("9948.20")

    def test_zero_fee_for_event_contracts(
        self, broker: PolymarketPaperBroker
    ) -> None:
        broker.update_quotes({"t": {"bid": 0.49, "ask": 0.51, "last": 0.50}})
        order = Order(
            symbol="t",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.51"),
        )
        broker.submit_order(order)

        # No fee: 10000 - 51.00 = 9949.00
        assert broker.get_cash_balance() == Decimal("9949.00")


class TestOrderManagement:
    def test_cancel_order(self, broker: PolymarketPaperBroker) -> None:
        broker.update_quotes({"t": {"bid": 0.30, "ask": 0.32, "last": 0.31}})
        order = Order(
            symbol="t",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.20"),  # Won't fill
        )
        result = broker.submit_order(order)
        # Order rejected because limit < ask, so can't cancel
        assert result.status == OrderStatus.REJECTED

    def test_cancel_nonexistent(self, broker: PolymarketPaperBroker) -> None:
        import uuid
        assert broker.cancel_order(uuid.uuid4()) is False

    def test_short_selling_disabled(self, broker: PolymarketPaperBroker) -> None:
        assert broker.get_shortable_shares("t") == 0
        assert broker.get_borrow_rate("t") == Decimal("0")
