"""Tests for the Timezone Information Arbitrage strategy."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bifs_quant_engine.data.news_monitor import FeedSource, NewsItem
from bifs_quant_engine.data.polymarket_api import (
    PolymarketDataAPI,
    PolymarketMarket,
    PolymarketToken,
)
from bifs_quant_engine.strategies.tz_info_arb import (
    TZInfoArbConfig,
    TZInfoArbStrategy,
    OpenPosition,
)
from bifs_quant_engine.strategies.strategy_protocol import SignalType


@pytest.fixture
def mock_api() -> MagicMock:
    return MagicMock(spec=PolymarketDataAPI)


@pytest.fixture
def config() -> TZInfoArbConfig:
    return TZInfoArbConfig(
        max_position_per_contract=Decimal("500"),
        min_edge_threshold=0.05,
        active_hours_utc=[(0, 24)],  # Always active for testing
        max_simultaneous_positions=5,
        sentiment_threshold=0.0,  # Low threshold for testing
        similarity_threshold=0.5,
    )


@pytest.fixture
def strategy(config: TZInfoArbConfig, mock_api: MagicMock) -> TZInfoArbStrategy:
    sources = [
        FeedSource(name="test", url="http://example.com/rss", feed_type="rss"),
    ]
    return TZInfoArbStrategy(config, mock_api, feed_sources=sources)


class TestStrategyInit:
    def test_name(self, strategy: TZInfoArbStrategy) -> None:
        assert strategy.name == "tz_info_arb"

    def test_empty_universe(self, strategy: TZInfoArbStrategy) -> None:
        assert strategy.universe == []

    def test_no_open_positions(self, strategy: TZInfoArbStrategy) -> None:
        assert strategy.open_position_count == 0


class TestActiveHours:
    def test_always_active_config(self, strategy: TZInfoArbStrategy) -> None:
        assert strategy._is_active_hours()

    def test_overnight_window(self, mock_api: MagicMock) -> None:
        config = TZInfoArbConfig(active_hours_utc=[(22, 6)])
        strat = TZInfoArbStrategy(config, mock_api)

        # Mock time to 3AM UTC (within window)
        with patch("bifs_quant_engine.strategies.tz_info_arb.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 15, 3, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert strat._is_active_hours()

    def test_outside_window(self, mock_api: MagicMock) -> None:
        config = TZInfoArbConfig(active_hours_utc=[(22, 6)])
        strat = TZInfoArbStrategy(config, mock_api)

        # Mock time to 2PM UTC (outside window)
        with patch("bifs_quant_engine.strategies.tz_info_arb.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 15, 14, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert not strat._is_active_hours()


class TestPositionTracking:
    def test_record_entry(self, strategy: TZInfoArbStrategy) -> None:
        strategy.record_entry(
            token_id="tok_yes",
            condition_id="cond_1",
            price=Decimal("0.25"),
            quantity=100,
            direction="yes",
            edge=Decimal("0.10"),
        )
        assert strategy.open_position_count == 1
        assert "tok_yes" in strategy.universe

    def test_record_exit(self, strategy: TZInfoArbStrategy) -> None:
        strategy.record_entry(
            token_id="tok_yes",
            condition_id="cond_1",
            price=Decimal("0.25"),
            quantity=100,
            direction="yes",
            edge=Decimal("0.10"),
        )
        strategy.record_exit("tok_yes")
        assert strategy.open_position_count == 0
        assert "tok_yes" not in strategy.universe

    def test_exit_nonexistent_is_safe(self, strategy: TZInfoArbStrategy) -> None:
        strategy.record_exit("nonexistent")  # Should not raise


class TestExitConditions:
    def test_max_holding_time_exit(self, strategy: TZInfoArbStrategy) -> None:
        # Create a position opened 13 hours ago (exceeds 12h max)
        strategy._open_positions["tok_old"] = OpenPosition(
            token_id="tok_old",
            condition_id="cond_1",
            entry_price=Decimal("0.25"),
            quantity=100,
            entry_time=datetime.now(timezone.utc) - timedelta(hours=13),
            entry_edge=Decimal("0.10"),
            direction="yes",
        )

        exits = strategy._check_exits({})
        assert len(exits) == 1
        assert exits[0].signal_type == SignalType.CLOSE
        assert exits[0].symbol == "tok_old"
        assert "max_holding_hours" in exits[0].metadata["exit_reason"]

    def test_no_exit_within_holding_time(self, strategy: TZInfoArbStrategy) -> None:
        strategy._open_positions["tok_new"] = OpenPosition(
            token_id="tok_new",
            condition_id="cond_1",
            entry_price=Decimal("0.25"),
            quantity=100,
            entry_time=datetime.now(timezone.utc) - timedelta(hours=1),
            entry_edge=Decimal("0.10"),
            direction="yes",
        )

        exits = strategy._check_exits({})
        assert len(exits) == 0


class TestMaxPositions:
    def test_no_signals_when_max_reached(
        self, strategy: TZInfoArbStrategy
    ) -> None:
        # Fill up max positions
        for i in range(5):
            strategy.record_entry(
                token_id=f"tok_{i}",
                condition_id=f"cond_{i}",
                price=Decimal("0.25"),
                quantity=100,
                direction="yes",
                edge=Decimal("0.10"),
            )

        # Generate signals should not produce entry signals
        import pandas as pd
        signals = strategy.generate_signals(pd.DataFrame(), {})
        entry_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(entry_signals) == 0
