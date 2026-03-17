"""Tests for the Polymarket live broker adapter."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bifs_quant_engine.brokers.polymarket_broker import (
    PolymarketBroker,
    PolymarketConfig,
)
from bifs_quant_engine.core.enums import OrderSide, OrderStatus, OrderType
from bifs_quant_engine.core.models import Order


@pytest.fixture
def config() -> PolymarketConfig:
    return PolymarketConfig(
        host="https://clob.polymarket.com",
        chain_id=137,
        private_key="0x" + "a" * 64,
        funder="0x" + "b" * 40,
    )


@pytest.fixture
def broker(config: PolymarketConfig) -> PolymarketBroker:
    return PolymarketBroker(config)


class TestPolymarketBrokerInit:
    def test_name(self, broker: PolymarketBroker) -> None:
        assert broker.name == "polymarket"

    def test_not_connected_initially(self, broker: PolymarketBroker) -> None:
        assert not broker.is_connected

    def test_loads_private_key_from_env(self) -> None:
        with patch.dict("os.environ", {"POLYMARKET_PRIVATE_KEY": "0xtest123"}):
            b = PolymarketBroker(PolymarketConfig())
            assert b._config.private_key == "0xtest123"

    def test_raises_on_connect_without_key(self) -> None:
        b = PolymarketBroker(PolymarketConfig(private_key=""))
        with pytest.raises(RuntimeError, match="POLYMARKET_PRIVATE_KEY"):
            b.connect()


class TestPolymarketBrokerNotConnected:
    def test_submit_order_raises(self, broker: PolymarketBroker) -> None:
        order = Order(
            symbol="token_abc",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.25"),
        )
        with pytest.raises(RuntimeError, match="not connected"):
            broker.submit_order(order)

    def test_get_quote_raises(self, broker: PolymarketBroker) -> None:
        with pytest.raises(RuntimeError, match="not connected"):
            broker.get_quote("token_abc")

    def test_cancel_order_raises(self, broker: PolymarketBroker) -> None:
        with pytest.raises(RuntimeError, match="not connected"):
            broker.cancel_order(Order(symbol="x", side=OrderSide.BUY, quantity=1).order_id)


class TestPolymarketBrokerPositions:
    def test_update_cash(self, broker: PolymarketBroker) -> None:
        broker.update_cash(Decimal("5000"))
        assert broker.get_cash_balance() == Decimal("5000")
        assert broker.get_buying_power() == Decimal("5000")

    def test_record_fill_creates_position(self, broker: PolymarketBroker) -> None:
        order = Order(
            symbol="token_yes",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.25"),
        )
        broker.update_cash(Decimal("1000"))
        fill = broker.record_fill(order, Decimal("0.25"), 100)

        assert fill.quantity == 100
        assert fill.price == Decimal("0.25")
        assert fill.commission == Decimal("0")

        positions = broker.get_positions()
        assert "token_yes" in positions
        assert positions["token_yes"].quantity == 100
        assert positions["token_yes"].avg_cost == Decimal("0.25")

    def test_record_fill_updates_cash(self, broker: PolymarketBroker) -> None:
        broker.update_cash(Decimal("100"))
        order = Order(symbol="t", side=OrderSide.BUY, quantity=10)
        broker.record_fill(order, Decimal("0.50"), 10)
        assert broker.get_cash_balance() == Decimal("95")  # 100 - 0.50*10

    def test_record_fill_sell_adds_cash(self, broker: PolymarketBroker) -> None:
        broker.update_cash(Decimal("100"))
        # First buy
        buy = Order(symbol="t", side=OrderSide.BUY, quantity=10)
        broker.record_fill(buy, Decimal("0.30"), 10)
        # Then sell
        sell = Order(symbol="t", side=OrderSide.SELL, quantity=10)
        broker.record_fill(sell, Decimal("0.60"), 10)
        # Cash: 100 - 3.00 + 6.00 = 103.00
        assert broker.get_cash_balance() == Decimal("103")

    def test_get_positions_excludes_flat(self, broker: PolymarketBroker) -> None:
        buy = Order(symbol="t", side=OrderSide.BUY, quantity=10)
        broker.record_fill(buy, Decimal("0.20"), 10)
        sell = Order(symbol="t", side=OrderSide.SELL, quantity=10)
        broker.record_fill(sell, Decimal("0.30"), 10)
        assert "t" not in broker.get_positions()

    def test_short_selling_returns_zero(self, broker: PolymarketBroker) -> None:
        assert broker.get_shortable_shares("any") == 0
        assert broker.get_borrow_rate("any") == Decimal("0")


class TestPolymarketBrokerOrderStatus:
    def test_get_order_status_not_found(self, broker: PolymarketBroker) -> None:
        import uuid
        with pytest.raises(KeyError):
            broker.get_order_status(uuid.uuid4())

    def test_get_open_orders_empty(self, broker: PolymarketBroker) -> None:
        assert broker.get_open_orders() == []
