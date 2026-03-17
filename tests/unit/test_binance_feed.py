"""Tests for the Binance WebSocket data feed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from bifs_quant_engine.data.binance_feed import (
    BinanceFeed,
    BinanceFeedConfig,
    Candle,
)


@pytest.fixture
def feed() -> BinanceFeed:
    config = BinanceFeedConfig(symbol="BTCUSDT", intervals=["1m", "5m"])
    return BinanceFeed(config)


def _make_candle(
    close: float,
    interval: str = "1m",
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
) -> Candle:
    return Candle(
        timestamp=datetime.now(timezone.utc),
        open=open_ or close * 0.999,
        high=high or close * 1.001,
        low=low or close * 0.998,
        close=close,
        volume=100.0,
        interval=interval,
    )


class TestCandleBuffer:
    def test_add_candle(self, feed: BinanceFeed) -> None:
        candle = _make_candle(85000.0)
        feed.add_candle(candle)
        assert feed.candle_count("1m") == 1

    def test_get_recent_candles(self, feed: BinanceFeed) -> None:
        for i in range(10):
            feed.add_candle(_make_candle(85000.0 + i * 10))

        candles = feed.get_recent_candles("1m", 5)
        assert len(candles) == 5
        # Should be the last 5 (most recent)
        assert candles[-1].close == 85090.0

    def test_get_recent_more_than_available(self, feed: BinanceFeed) -> None:
        feed.add_candle(_make_candle(85000.0))
        candles = feed.get_recent_candles("1m", 100)
        assert len(candles) == 1

    def test_different_intervals(self, feed: BinanceFeed) -> None:
        feed.add_candle(_make_candle(85000.0, interval="1m"))
        feed.add_candle(_make_candle(85100.0, interval="5m"))
        assert feed.candle_count("1m") == 1
        assert feed.candle_count("5m") == 1

    def test_buffer_max_size(self) -> None:
        config = BinanceFeedConfig(buffer_size=5)
        feed = BinanceFeed(config)
        for i in range(10):
            feed.add_candle(_make_candle(85000.0 + i))
        assert feed.candle_count("1m") == 5


class TestClosePrices:
    def test_get_close_prices(self, feed: BinanceFeed) -> None:
        prices = [85000.0, 85050.0, 85100.0]
        for p in prices:
            feed.add_candle(_make_candle(p))

        result = feed.get_close_prices("1m", 3)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, prices)


class TestRealisedVolatility:
    def test_vol_with_constant_prices(self, feed: BinanceFeed) -> None:
        for _ in range(10):
            feed.add_candle(_make_candle(85000.0))
        vol = feed.get_realised_vol("1m", 5)
        assert vol == pytest.approx(0.0)

    def test_vol_with_varying_prices(self, feed: BinanceFeed) -> None:
        prices = [85000, 85100, 84900, 85200, 84800, 85300]
        for p in prices:
            feed.add_candle(_make_candle(float(p)))
        vol = feed.get_realised_vol("1m", 5)
        assert vol > 0

    def test_vol_insufficient_data(self, feed: BinanceFeed) -> None:
        feed.add_candle(_make_candle(85000.0))
        vol = feed.get_realised_vol("1m", 5)
        assert vol == 0.0

    def test_vol_empty_buffer(self, feed: BinanceFeed) -> None:
        vol = feed.get_realised_vol("1m", 5)
        assert vol == 0.0

    def test_vol_two_candles(self, feed: BinanceFeed) -> None:
        feed.add_candle(_make_candle(85000.0))
        feed.add_candle(_make_candle(85100.0))
        vol = feed.get_realised_vol("1m", 1)
        # With lookback=1, we get 2 candles -> 1 log return -> std of 1 element = 0
        assert vol == 0.0


class TestGarmanKlassVol:
    def test_gk_vol_basic(self, feed: BinanceFeed) -> None:
        for _ in range(10):
            feed.add_candle(
                Candle(
                    timestamp=datetime.now(timezone.utc),
                    open=85000.0,
                    high=85200.0,
                    low=84800.0,
                    close=85100.0,
                    volume=100.0,
                    interval="1m",
                )
            )
        gk = feed.get_garman_klass_vol("1m", 10)
        assert gk > 0

    def test_gk_vol_insufficient_data(self, feed: BinanceFeed) -> None:
        feed.add_candle(_make_candle(85000.0))
        gk = feed.get_garman_klass_vol("1m", 5)
        assert gk >= 0


class TestCallbacks:
    def test_callback_invoked(self, feed: BinanceFeed) -> None:
        received = []
        feed.on_candle(lambda c: received.append(c))
        feed.add_candle(_make_candle(85000.0))
        assert len(received) == 1


class TestWebSocketParsing:
    def test_handle_kline_message(self, feed: BinanceFeed) -> None:
        msg = json.dumps({
            "k": {
                "t": 1710460800000,
                "o": "85000.00",
                "h": "85200.00",
                "l": "84900.00",
                "c": "85100.00",
                "v": "150.5",
                "i": "1m",
                "x": True,  # Closed candle
            }
        })
        feed._handle_message(msg)
        assert feed.candle_count("1m") == 1
        candles = feed.get_recent_candles("1m", 1)
        assert candles[0].close == 85100.0

    def test_handle_open_candle_not_buffered(self, feed: BinanceFeed) -> None:
        msg = json.dumps({
            "k": {
                "t": 1710460800000,
                "o": "85000.00",
                "h": "85100.00",
                "l": "84950.00",
                "c": "85050.00",
                "v": "50.0",
                "i": "1m",
                "x": False,  # Not closed yet
            }
        })
        feed._handle_message(msg)
        assert feed.candle_count("1m") == 0

    def test_handle_invalid_message(self, feed: BinanceFeed) -> None:
        feed._handle_message("not json")
        assert feed.candle_count("1m") == 0
