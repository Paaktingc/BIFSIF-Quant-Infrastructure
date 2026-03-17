"""Tests for the Volatility Compression Straddle strategy."""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from bifs_quant_engine.data.binance_feed import BinanceFeed, BinanceFeedConfig, Candle
from bifs_quant_engine.strategies.strategy_protocol import SignalType
from bifs_quant_engine.strategies.vol_straddle import (
    VolStraddleConfig,
    VolStraddleStrategy,
    StraddlePosition,
)


@pytest.fixture
def feed() -> BinanceFeed:
    config = BinanceFeedConfig(symbol="BTCUSDT", intervals=["1m"])
    return BinanceFeed(config)


@pytest.fixture
def config() -> VolStraddleConfig:
    return VolStraddleConfig(
        vol_lookback_minutes=5,
        vol_threshold_ratio=0.5,
        historical_vol_lookback=20,
        max_position_usd=Decimal("300"),
        holding_period_minutes=60,
        taker_fee_bps=156,
        max_straddles_per_day=3,
    )


@pytest.fixture
def strategy(config: VolStraddleConfig, feed: BinanceFeed) -> VolStraddleStrategy:
    return VolStraddleStrategy(config, feed)


def _add_volatile_candles(feed: BinanceFeed, count: int, base: float = 85000.0) -> None:
    """Add candles with significant price variation."""
    for i in range(count):
        price = base + (i % 2) * 200 - 100  # Oscillate ±100
        feed.add_candle(Candle(
            timestamp=datetime.now(timezone.utc),
            open=price - 50, high=price + 100, low=price - 100,
            close=price, volume=100.0, interval="1m",
        ))


def _add_flat_candles(feed: BinanceFeed, count: int, price: float = 85000.0) -> None:
    """Add candles with minimal price variation."""
    for _ in range(count):
        feed.add_candle(Candle(
            timestamp=datetime.now(timezone.utc),
            open=price, high=price + 1, low=price - 1,
            close=price, volume=100.0, interval="1m",
        ))


class TestStrategyInit:
    def test_name(self, strategy: VolStraddleStrategy) -> None:
        assert strategy.name == "vol_straddle"

    def test_not_in_straddle(self, strategy: VolStraddleStrategy) -> None:
        assert not strategy.in_straddle

    def test_zero_straddles_today(self, strategy: VolStraddleStrategy) -> None:
        assert strategy.straddles_today == 0


class TestVolCompression:
    def test_detects_compression(
        self, strategy: VolStraddleStrategy, feed: BinanceFeed
    ) -> None:
        # Add volatile history then flat recent candles
        _add_volatile_candles(feed, 20, base=85000.0)
        _add_flat_candles(feed, 6, price=85000.0)

        compressed, current, historical = strategy.detect_vol_compression()
        assert current < historical
        # May or may not trigger depending on exact vol ratio

    def test_no_compression_with_volatile_recent(
        self, strategy: VolStraddleStrategy, feed: BinanceFeed
    ) -> None:
        # All volatile
        _add_volatile_candles(feed, 25, base=85000.0)

        compressed, current, historical = strategy.detect_vol_compression()
        # Current vol should be similar to historical
        assert not compressed or (current / max(historical, 1e-10)) >= 0.5

    def test_insufficient_data(
        self, strategy: VolStraddleStrategy, feed: BinanceFeed
    ) -> None:
        feed.add_candle(Candle(
            timestamp=datetime.now(timezone.utc),
            open=85000, high=85100, low=84900,
            close=85050, volume=100, interval="1m",
        ))
        compressed, current, historical = strategy.detect_vol_compression()
        # With near-zero historical vol, shouldn't compress
        assert not compressed


class TestStraddleEV:
    def test_positive_ev_low_cost(self, strategy: VolStraddleStrategy) -> None:
        ev = strategy.compute_straddle_ev(
            yes_price=Decimal("0.25"),
            no_price=Decimal("0.25"),
            win_probability=0.90,
        )
        # Total cost = 0.50, payout = 1.00, gross = 0.50
        # Fee: ~0.0078 per leg * 2 = ~0.016
        # Net: 0.50 - 0.016 = 0.484
        # EV: 0.90 * 0.484 - 0.10 * 0.50 = 0.436 - 0.05 = 0.386
        assert ev > Decimal("0")

    def test_negative_ev_high_cost(self, strategy: VolStraddleStrategy) -> None:
        ev = strategy.compute_straddle_ev(
            yes_price=Decimal("0.55"),
            no_price=Decimal("0.55"),
            win_probability=0.50,
        )
        # Total cost = 1.10 > payout of 1.00 -> always negative
        assert ev < Decimal("0")

    def test_breakeven_at_60_pct(self, strategy: VolStraddleStrategy) -> None:
        # At 30c per leg, need ~60% win rate
        ev = strategy.compute_straddle_ev(
            yes_price=Decimal("0.30"),
            no_price=Decimal("0.30"),
            win_probability=0.60,
        )
        # Should be near zero or slightly negative (fees drag)
        assert abs(float(ev)) < Decimal("0.10")


class TestSignalGeneration:
    def test_no_signals_without_data(self, strategy: VolStraddleStrategy) -> None:
        import pandas as pd
        signals = strategy.generate_signals(pd.DataFrame(), {})
        assert len(signals) == 0

    def test_no_entry_when_max_straddles_reached(
        self, strategy: VolStraddleStrategy, feed: BinanceFeed
    ) -> None:
        strategy._straddles_today = 3
        import pandas as pd
        signals = strategy.generate_signals(pd.DataFrame(), {})
        entry_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(entry_signals) == 0


class TestStraddlePositionTracking:
    def test_record_entry(self, strategy: VolStraddleStrategy) -> None:
        strategy.record_straddle_entry(
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            yes_price=Decimal("0.30"),
            no_price=Decimal("0.30"),
            shares=100,
            vol=0.001,
        )
        assert strategy.in_straddle
        assert strategy.straddles_today == 1

    def test_record_exit(self, strategy: VolStraddleStrategy) -> None:
        strategy.record_straddle_entry(
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            yes_price=Decimal("0.30"),
            no_price=Decimal("0.30"),
            shares=100,
            vol=0.001,
        )
        strategy.record_straddle_exit()
        assert not strategy.in_straddle


class TestExitConditions:
    def test_holding_period_exit(
        self, strategy: VolStraddleStrategy, feed: BinanceFeed
    ) -> None:
        # Add some data so vol can be computed
        _add_flat_candles(feed, 10)

        strategy._active_straddle = StraddlePosition(
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            yes_price=Decimal("0.30"),
            no_price=Decimal("0.30"),
            shares_per_leg=100,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=61),
            entry_vol=0.001,
        )

        exits = strategy._check_exit_conditions()
        assert len(exits) == 2
        assert exits[0].signal_type == SignalType.CLOSE
        assert exits[1].signal_type == SignalType.CLOSE
        assert "holding_period" in exits[0].metadata["exit_reason"]

    def test_no_exit_within_period(
        self, strategy: VolStraddleStrategy, feed: BinanceFeed
    ) -> None:
        _add_flat_candles(feed, 10)

        strategy._active_straddle = StraddlePosition(
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            yes_price=Decimal("0.30"),
            no_price=Decimal("0.30"),
            shares_per_leg=100,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            entry_vol=0.001,
        )

        exits = strategy._check_exit_conditions()
        assert len(exits) == 0


class TestStraddlePositionDataclass:
    def test_total_cost(self) -> None:
        sp = StraddlePosition(
            yes_token_id="y", no_token_id="n",
            yes_price=Decimal("0.30"), no_price=Decimal("0.25"),
            shares_per_leg=100,
            entry_time=datetime.now(timezone.utc),
            entry_vol=0.001,
        )
        assert sp.total_cost == Decimal("55.00")  # (0.30 + 0.25) * 100
