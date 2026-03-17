"""End-to-end integration tests for Polymarket strategies.

Tests the full pipeline from signal generation through order execution
using the paper broker. No external API calls are made.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest

from bifs_quant_engine.brokers.polymarket_paper_broker import (
    PolymarketPaperBroker,
    PolymarketPaperConfig,
)
from bifs_quant_engine.core.enums import OrderSide, OrderType, OrderStatus, RiskAction
from bifs_quant_engine.core.models import Order
from bifs_quant_engine.data.binance_feed import BinanceFeed, BinanceFeedConfig, Candle
from bifs_quant_engine.data.news_monitor import FeedSource
from bifs_quant_engine.risk.limits import EventContractLimits, check_event_contract_limit
from bifs_quant_engine.risk.validators import OrderValidator
from bifs_quant_engine.strategies.strategy_protocol import SignalType
from bifs_quant_engine.strategies.tz_info_arb import TZInfoArbConfig, TZInfoArbStrategy
from bifs_quant_engine.strategies.vol_straddle import (
    VolStraddleConfig,
    VolStraddleStrategy,
    StraddlePosition,
)
from bifs_quant_engine.trading.polymarket_session import (
    PolymarketSession,
    PolymarketSessionConfig,
    SessionState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def paper_broker() -> PolymarketPaperBroker:
    config = PolymarketPaperConfig(initial_cash=Decimal("5000"))
    broker = PolymarketPaperBroker(config)
    broker.connect()
    return broker


@pytest.fixture
def binance_feed() -> BinanceFeed:
    config = BinanceFeedConfig(symbol="BTCUSDT", intervals=["1m"])
    return BinanceFeed(config)


@pytest.fixture
def mock_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def tz_strategy(mock_api: MagicMock) -> TZInfoArbStrategy:
    config = TZInfoArbConfig(
        max_position_per_contract=Decimal("500"),
        min_edge_threshold=0.05,
        active_hours_utc=[(0, 24)],
        max_simultaneous_positions=5,
        sentiment_threshold=0.0,
        similarity_threshold=0.5,
    )
    sources = [FeedSource(name="test", url="http://example.com/rss", feed_type="rss")]
    return TZInfoArbStrategy(config, mock_api, feed_sources=sources)


@pytest.fixture
def vol_strategy(binance_feed: BinanceFeed) -> VolStraddleStrategy:
    config = VolStraddleConfig(
        vol_lookback_minutes=5,
        vol_threshold_ratio=0.5,
        historical_vol_lookback=20,
        max_position_usd=Decimal("300"),
        holding_period_minutes=60,
        taker_fee_bps=156,
        max_straddles_per_day=3,
    )
    return VolStraddleStrategy(config, binance_feed)


# ---------------------------------------------------------------------------
# Paper Broker Integration
# ---------------------------------------------------------------------------

class TestPaperBrokerOrderFlow:
    """Test order submission through paper broker."""

    def _set_quote(
        self, broker: PolymarketPaperBroker, token: str, price: float
    ) -> None:
        broker.update_quotes({
            token: {"bid": price, "ask": price, "last": price}
        })

    def test_submit_and_fill_order(self, paper_broker: PolymarketPaperBroker) -> None:
        self._set_quote(paper_broker, "tok_yes_123", 0.30)
        order = Order(
            symbol="tok_yes_123",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.30"),
            strategy_id="tz_info_arb",
        )
        result = paper_broker.submit_order(order)
        assert result.status == OrderStatus.FILLED

        positions = paper_broker.get_positions()
        assert "tok_yes_123" in positions
        assert positions["tok_yes_123"].quantity == 100

    def test_buy_then_sell(self, paper_broker: PolymarketPaperBroker) -> None:
        self._set_quote(paper_broker, "tok_abc", 0.40)
        buy_order = Order(
            symbol="tok_abc",
            side=OrderSide.BUY,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.40"),
            strategy_id="vol_straddle",
        )
        paper_broker.submit_order(buy_order)

        self._set_quote(paper_broker, "tok_abc", 0.60)
        sell_order = Order(
            symbol="tok_abc",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.60"),
            strategy_id="vol_straddle",
        )
        paper_broker.submit_order(sell_order)

        positions = paper_broker.get_positions()
        if "tok_abc" in positions:
            assert positions["tok_abc"].quantity == 0

    def test_settlement_win(self, paper_broker: PolymarketPaperBroker) -> None:
        self._set_quote(paper_broker, "tok_win", 0.25)
        order = Order(
            symbol="tok_win",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.25"),
            strategy_id="tz_info_arb",
        )
        paper_broker.submit_order(order)
        initial_cash = paper_broker.get_cash_balance()

        paper_broker.resolve_market("tok_win", won=True)
        final_cash = paper_broker.get_cash_balance()

        # Won: receive $1 × 100 shares = $100
        assert final_cash > initial_cash

    def test_settlement_loss(self, paper_broker: PolymarketPaperBroker) -> None:
        self._set_quote(paper_broker, "tok_lose", 0.25)
        order = Order(
            symbol="tok_lose",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.25"),
            strategy_id="tz_info_arb",
        )
        paper_broker.submit_order(order)
        cash_after_buy = paper_broker.get_cash_balance()

        paper_broker.resolve_market("tok_lose", won=False)
        final_cash = paper_broker.get_cash_balance()

        # Lost: shares worth $0, no payout
        assert final_cash == cash_after_buy


# ---------------------------------------------------------------------------
# Risk Validation Integration
# ---------------------------------------------------------------------------

class TestEventContractRiskIntegration:
    """Test risk limits with event contract orders."""

    def test_order_within_limits(self, paper_broker: PolymarketPaperBroker) -> None:
        limits = EventContractLimits(
            max_position_per_contract=Decimal("500"),
            max_total_event_exposure=Decimal("2500"),
        )
        order = Order(
            symbol="tok_1",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.30"),
        )
        decision = check_event_contract_limit(
            order=order,
            current_positions={},
            portfolio_equity=Decimal("5000"),
            limits=limits,
        )
        assert decision.action == RiskAction.ALLOW

    def test_order_exceeds_per_contract(self) -> None:
        limits = EventContractLimits(max_position_per_contract=Decimal("50"))
        order = Order(
            symbol="tok_1",
            side=OrderSide.BUY,
            quantity=200,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.50"),
        )
        decision = check_event_contract_limit(
            order=order,
            current_positions={},
            portfolio_equity=Decimal("5000"),
            limits=limits,
        )
        assert decision.action in (RiskAction.REDUCE, RiskAction.REJECT)

    def test_invalid_probability_rejected(self) -> None:
        limits = EventContractLimits()
        order = Order(
            symbol="tok_1",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("1.50"),
        )
        decision = check_event_contract_limit(
            order=order,
            current_positions={},
            portfolio_equity=Decimal("5000"),
            limits=limits,
        )
        assert decision.action == RiskAction.REJECT

    def test_event_contract_validator(self) -> None:
        validator = OrderValidator()
        order = Order(
            symbol="tok_1",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0.40"),
        )
        decision = validator.validate_event_contract(order)
        assert decision.action == RiskAction.ALLOW

    def test_event_contract_no_price_rejected(self) -> None:
        validator = OrderValidator()
        order = Order(
            symbol="tok_1",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        decision = validator.validate_event_contract(order)
        assert decision.action == RiskAction.REJECT


# ---------------------------------------------------------------------------
# Strategy → Signal → Order Pipeline
# ---------------------------------------------------------------------------

class TestTZInfoArbPipeline:
    """Test TZ Info Arb strategy signal-to-order flow."""

    def test_exit_signal_produces_close_order(
        self, tz_strategy: TZInfoArbStrategy, paper_broker: PolymarketPaperBroker
    ) -> None:
        from bifs_quant_engine.strategies.tz_info_arb import OpenPosition

        # Set up an expired position
        tz_strategy._open_positions["tok_old"] = OpenPosition(
            token_id="tok_old",
            condition_id="cond_1",
            entry_price=Decimal("0.25"),
            quantity=100,
            entry_time=datetime.now(timezone.utc) - timedelta(hours=13),
            entry_edge=Decimal("0.10"),
            direction="yes",
        )

        signals = tz_strategy.generate_signals(pd.DataFrame(), {})
        close_signals = [s for s in signals if s.signal_type == SignalType.CLOSE]
        assert len(close_signals) >= 1
        assert close_signals[0].symbol == "tok_old"


class TestVolStraddlePipeline:
    """Test Vol Straddle strategy signal-to-order flow."""

    def test_exit_signals_for_expired_straddle(
        self,
        vol_strategy: VolStraddleStrategy,
        binance_feed: BinanceFeed,
        paper_broker: PolymarketPaperBroker,
    ) -> None:
        # Add candle data
        for _ in range(10):
            binance_feed.add_candle(Candle(
                timestamp=datetime.now(timezone.utc),
                open=85000, high=85010, low=84990,
                close=85000, volume=100, interval="1m",
            ))

        # Set up an expired straddle
        vol_strategy._active_straddle = StraddlePosition(
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            yes_price=Decimal("0.30"),
            no_price=Decimal("0.30"),
            shares_per_leg=100,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=61),
            entry_vol=0.001,
        )

        signals = vol_strategy.generate_signals(pd.DataFrame(), {})
        close_signals = [s for s in signals if s.signal_type == SignalType.CLOSE]
        assert len(close_signals) == 2


# ---------------------------------------------------------------------------
# Session Lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    """Test PolymarketSession state transitions."""

    def test_initial_state(self, paper_broker: PolymarketPaperBroker) -> None:
        session = PolymarketSession(
            broker=paper_broker,
            strategies=[],
        )
        assert session.state == SessionState.IDLE

    def test_session_start_and_stop(
        self,
        paper_broker: PolymarketPaperBroker,
        vol_strategy: VolStraddleStrategy,
    ) -> None:
        session = PolymarketSession(
            broker=paper_broker,
            strategies=[vol_strategy],
            config=PolymarketSessionConfig(
                signal_check_interval=1,
                position_check_interval=1,
                enable_binance_feed=False,
            ),
        )

        async def run_briefly():
            task = asyncio.create_task(session.run())
            await asyncio.sleep(0.5)
            await session.stop()
            await task

        asyncio.run(run_briefly())
        assert session.state == SessionState.STOPPED

    def test_session_registers_strategies(
        self,
        paper_broker: PolymarketPaperBroker,
        tz_strategy: TZInfoArbStrategy,
        vol_strategy: VolStraddleStrategy,
    ) -> None:
        session = PolymarketSession(
            broker=paper_broker,
            strategies=[tz_strategy, vol_strategy],
        )
        assert "tz_info_arb" in session.strategy_names
        assert "vol_straddle" in session.strategy_names
