"""Binance WebSocket data feed for real-time crypto candle data."""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List


@dataclass
class Candle:
    """OHLCV candle from Binance."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str


@dataclass
class BinanceFeedConfig:
    """Configuration for BinanceFeed."""
    symbol: str = "BTCUSDT"
    intervals: List[str] = field(default_factory=lambda: ["1m"])
    buffer_size: int = 500


class BinanceFeed:
    """
    Binance crypto data feed with per-interval candle buffers.

    Supports:
    - Manual candle injection (add_candle) for backtesting
    - WebSocket message parsing (_handle_message) for live trading
    - Realised volatility and Garman-Klass volatility estimation
    """

    def __init__(self, config: BinanceFeedConfig | None = None) -> None:
        self.config = config or BinanceFeedConfig()
        # Per-interval rolling buffer
        self._buffers: Dict[str, Deque[Candle]] = {
            interval: deque(maxlen=self.config.buffer_size)
            for interval in self.config.intervals
        }
        # Fallback buffer for intervals not in config
        self._default_interval = self.config.intervals[0] if self.config.intervals else "1m"
        self._callbacks: List[Callable[[Candle], None]] = []

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def add_candle(self, candle: Candle) -> None:
        """Add a candle to the appropriate interval buffer."""
        interval = candle.interval
        if interval not in self._buffers:
            self._buffers[interval] = deque(maxlen=self.config.buffer_size)
        self._buffers[interval].append(candle)
        for cb in self._callbacks:
            cb(candle)

    def candle_count(self, interval: str) -> int:
        """Return the number of candles in the given interval buffer."""
        return len(self._buffers.get(interval, []))

    def get_recent_candles(self, interval: str, n: int) -> List[Candle]:
        """Return the n most recent candles for the given interval."""
        buf = list(self._buffers.get(interval, []))
        return buf[-n:]

    def get_close_prices(self, interval: str, n: int) -> List[float]:
        """Return the n most recent close prices."""
        return [c.close for c in self.get_recent_candles(interval, n)]

    # ------------------------------------------------------------------
    # Volatility estimators
    # ------------------------------------------------------------------

    def get_realised_vol(self, interval: str, lookback: int) -> float:
        """
        Compute realised close-to-close volatility as annualised std of log returns.
        Returns 0.0 if there is insufficient data.
        """
        prices = self.get_close_prices(interval, lookback + 1)
        if len(prices) < 2:
            return 0.0
        returns = [
            math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
        ]
        if len(returns) < 2:
            return 0.0
        n = len(returns)
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        return math.sqrt(variance)

    def get_garman_klass_vol(self, interval: str, lookback: int) -> float:
        """
        Compute Garman-Klass volatility estimator using OHLC data.
        More efficient than close-to-close volatility.
        """
        candles = self.get_recent_candles(interval, lookback)
        if not candles:
            return 0.0

        gk_sum = 0.0
        for c in candles:
            if c.high <= 0 or c.low <= 0 or c.open <= 0 or c.close <= 0:
                continue
            try:
                hl = math.log(c.high / c.low) ** 2
                co = math.log(c.close / c.open) ** 2
                gk_sum += 0.5 * hl - (2 * math.log(2) - 1) * co
            except (ValueError, ZeroDivisionError):
                continue

        if len(candles) == 0:
            return 0.0
        return math.sqrt(gk_sum / len(candles))

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_candle(self, callback: Callable[[Candle], None]) -> None:
        """Register a callback called whenever a new candle is added."""
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # WebSocket message parsing
    # ------------------------------------------------------------------

    def _handle_message(self, raw: str) -> None:
        """
        Parse a Binance kline WebSocket JSON message.
        Only buffers closed (x=True) candles.
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        kline = data.get("k")
        if not kline:
            return

        if not kline.get("x", False):
            return  # Candle not yet closed

        try:
            candle = Candle(
                timestamp=datetime.fromtimestamp(kline["t"] / 1000, tz=timezone.utc),
                open=float(kline["o"]),
                high=float(kline["h"]),
                low=float(kline["l"]),
                close=float(kline["c"]),
                volume=float(kline["v"]),
                interval=kline["i"],
            )
            self.add_candle(candle)
        except (KeyError, ValueError):
            return
