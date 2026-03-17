"""Volatility Compression Straddle strategy for Polymarket.

Exploits BTC realised volatility clustering on Polymarket's 5/15-minute
crypto markets. When realised vol compresses to abnormally low levels,
buys both UP and DOWN legs (straddle) expecting a vol expansion and
directional breakout.

WARNING: This strategy faces taker fees (up to 1.56%) and a 250ms speed
bump on Polymarket crypto short-duration markets. Edge after fees may
be marginal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bifs_quant_engine.data.binance_feed import BinanceFeed
from bifs_quant_engine.strategies.strategy_protocol import (
    BaseStrategy,
    Signal,
    SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class VolStraddleConfig:
    """Configuration for the Volatility Compression Straddle.

    Attributes:
        vol_lookback_minutes: Lookback window for realised vol (in 1m candles).
        vol_threshold_ratio: Enter when current vol < ratio * historical avg vol.
        historical_vol_lookback: Lookback for historical avg vol (in 1m candles).
        max_position_usd: Maximum USDC per straddle (split across both legs).
        max_straddle_cost: Maximum combined cost of YES + NO legs.
        holding_period_minutes: Hold duration before forced exit.
        take_profit_pct: Take profit on combined position.
        stop_loss_pct: Stop loss on combined position.
        taker_fee_bps: Expected taker fee in basis points.
        max_straddles_per_day: Maximum straddles per trading day.
    """

    vol_lookback_minutes: int = 5
    vol_threshold_ratio: float = 0.5
    historical_vol_lookback: int = 1440  # 24 hours of 1m candles
    max_position_usd: Decimal = Decimal("300")
    max_straddle_cost: Decimal = Decimal("0.65")
    holding_period_minutes: int = 60
    take_profit_pct: float = 0.15
    stop_loss_pct: float = 0.10
    taker_fee_bps: int = 156
    max_straddles_per_day: int = 3


@dataclass
class StraddlePosition:
    """Tracks an active straddle (both legs).

    Attributes:
        yes_token_id: Token ID for the UP/YES leg.
        no_token_id: Token ID for the DOWN/NO leg.
        yes_price: Entry price for YES leg.
        no_price: Entry price for NO leg.
        shares_per_leg: Number of shares per leg.
        entry_time: When the straddle was opened.
        entry_vol: Realised vol at entry.
    """

    yes_token_id: str
    no_token_id: str
    yes_price: Decimal
    no_price: Decimal
    shares_per_leg: int
    entry_time: datetime
    entry_vol: float

    @property
    def total_cost(self) -> Decimal:
        """Total cost of both legs."""
        return (self.yes_price + self.no_price) * self.shares_per_leg

    @property
    def breakeven_payout(self) -> Decimal:
        """Minimum payout needed to break even."""
        return self.total_cost


class VolStraddleStrategy(BaseStrategy):
    """Volatility Compression Straddle on Polymarket BTC markets.

    Monitors Binance BTC/USDT 1-minute candles for vol compression
    events, then buys both YES and NO legs on the current Polymarket
    15-minute BTC up/down market.

    Example:
        config = VolStraddleConfig()
        feed = BinanceFeed(BinanceFeedConfig())
        strategy = VolStraddleStrategy(config, feed)

        signals = strategy.generate_signals(pd.DataFrame(), {})
    """

    def __init__(
        self,
        config: VolStraddleConfig,
        binance_feed: BinanceFeed,
    ) -> None:
        """Initialize strategy.

        Args:
            config: Strategy configuration.
            binance_feed: Binance data feed for BTC candles.
        """
        super().__init__(name="vol_straddle")
        self._config = config
        self._feed = binance_feed

        # State
        self._active_straddle: Optional[StraddlePosition] = None
        self._straddles_today: int = 0
        self._last_reset_date: Optional[datetime] = None

    @property
    def in_straddle(self) -> bool:
        """Whether a straddle position is currently active."""
        return self._active_straddle is not None

    @property
    def straddles_today(self) -> int:
        """Number of straddles executed today."""
        return self._straddles_today

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: Dict[str, Decimal],
    ) -> List[Signal]:
        """Generate straddle signals based on vol compression.

        Args:
            market_data: Ignored (uses internal Binance feed).
            current_positions: Current positions by token_id.

        Returns:
            List of signals (0 or 2 for straddle entry, 0 or 2 for exit).
        """
        signals: List[Signal] = []

        # Daily reset
        self._maybe_reset_daily()

        # Check exit conditions first
        if self._active_straddle is not None:
            exit_signals = self._check_exit_conditions()
            signals.extend(exit_signals)
            if exit_signals:
                return signals  # Exit takes priority

        # Check entry conditions
        if self._active_straddle is None:
            entry_signals = self._check_entry_conditions()
            signals.extend(entry_signals)

        return signals

    def detect_vol_compression(self) -> Tuple[bool, float, float]:
        """Detect whether BTC volatility is compressed.

        Returns:
            Tuple of (is_compressed, current_vol, historical_avg_vol).
        """
        current_vol = self._feed.get_realised_vol(
            "1m", self._config.vol_lookback_minutes
        )
        historical_vol = self._feed.get_realised_vol(
            "1m", self._config.historical_vol_lookback
        )

        if historical_vol <= 0:
            return False, current_vol, historical_vol

        ratio = current_vol / historical_vol
        compressed = ratio < self._config.vol_threshold_ratio

        if compressed:
            logger.info(
                "Vol compression detected: current=%.6f, historical=%.6f, ratio=%.3f",
                current_vol,
                historical_vol,
                ratio,
            )

        return compressed, current_vol, historical_vol

    def compute_straddle_ev(
        self,
        yes_price: Decimal,
        no_price: Decimal,
        win_probability: float,
    ) -> Decimal:
        """Compute expected value of a straddle.

        Args:
            yes_price: YES leg price.
            no_price: NO leg price.
            win_probability: Probability that BTC moves (either direction).

        Returns:
            Expected value per share.
        """
        total_cost = yes_price + no_price
        payout = Decimal("1.00")  # One side always pays $1

        # Fee impact
        fee_rate = Decimal(self._config.taker_fee_bps) / Decimal("10000")
        fee_per_leg = (yes_price + no_price) / 2 * fee_rate
        total_fee = fee_per_leg * 2

        gross_profit = payout - total_cost
        net_profit = gross_profit - total_fee

        ev = Decimal(str(win_probability)) * net_profit - (
            Decimal(str(1 - win_probability)) * total_cost
        )

        return ev

    def record_straddle_entry(
        self,
        yes_token_id: str,
        no_token_id: str,
        yes_price: Decimal,
        no_price: Decimal,
        shares: int,
        vol: float,
    ) -> None:
        """Record a straddle entry.

        Args:
            yes_token_id: YES token ID.
            no_token_id: NO token ID.
            yes_price: Entry price for YES.
            no_price: Entry price for NO.
            shares: Shares per leg.
            vol: Realised vol at entry.
        """
        self._active_straddle = StraddlePosition(
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            yes_price=yes_price,
            no_price=no_price,
            shares_per_leg=shares,
            entry_time=datetime.now(timezone.utc),
            entry_vol=vol,
        )
        self._straddles_today += 1

    def record_straddle_exit(self) -> None:
        """Record a straddle exit."""
        self._active_straddle = None

    def _check_entry_conditions(self) -> List[Signal]:
        """Check if conditions are met for a straddle entry.

        Returns:
            List of 2 signals (YES + NO) if entry conditions met, else [].
        """
        # Check daily limit
        if self._straddles_today >= self._config.max_straddles_per_day:
            return []

        # Check sufficient data
        if self._feed.candle_count("1m") < self._config.historical_vol_lookback:
            return []

        # Detect vol compression
        compressed, current_vol, hist_vol = self.detect_vol_compression()
        if not compressed:
            return []

        # We'd need actual Polymarket market data here to get token IDs
        # and prices. For now, emit signals with metadata indicating
        # straddle intent.
        signals = [
            Signal(
                strategy_id=self._name,
                symbol="__btc_yes_placeholder__",
                signal_type=SignalType.LONG,
                weight=0.5,
                confidence=min(1.0, hist_vol / max(current_vol, 1e-10)),
                metadata={
                    "leg": "yes",
                    "straddle": True,
                    "current_vol": current_vol,
                    "historical_vol": hist_vol,
                    "vol_ratio": current_vol / max(hist_vol, 1e-10),
                    "max_shares": int(
                        float(self._config.max_position_usd / 2)
                        / 0.30  # Approx price
                    ),
                },
            ),
            Signal(
                strategy_id=self._name,
                symbol="__btc_no_placeholder__",
                signal_type=SignalType.LONG,
                weight=0.5,
                confidence=min(1.0, hist_vol / max(current_vol, 1e-10)),
                metadata={
                    "leg": "no",
                    "straddle": True,
                    "current_vol": current_vol,
                    "historical_vol": hist_vol,
                    "vol_ratio": current_vol / max(hist_vol, 1e-10),
                    "max_shares": int(
                        float(self._config.max_position_usd / 2)
                        / 0.30
                    ),
                },
            ),
        ]

        logger.info(
            "Straddle entry signal: vol=%.6f (%.1f%% of historical)",
            current_vol,
            (current_vol / max(hist_vol, 1e-10)) * 100,
        )

        return signals

    def _check_exit_conditions(self) -> List[Signal]:
        """Check exit conditions for the active straddle.

        Returns:
            List of 2 CLOSE signals if exit triggered, else [].
        """
        if self._active_straddle is None:
            return []

        straddle = self._active_straddle
        now = datetime.now(timezone.utc)

        # Check holding period
        minutes_held = (now - straddle.entry_time).total_seconds() / 60
        should_exit = False
        reason = ""

        if minutes_held >= self._config.holding_period_minutes:
            should_exit = True
            reason = f"holding_period ({minutes_held:.0f}m)"

        # Check vol expansion (take profit signal)
        current_vol = self._feed.get_realised_vol(
            "1m", self._config.vol_lookback_minutes
        )
        if straddle.entry_vol > 0 and current_vol > straddle.entry_vol * 3:
            should_exit = True
            reason = f"vol_expansion (entry={straddle.entry_vol:.6f}, now={current_vol:.6f})"

        if not should_exit:
            return []

        signals = [
            Signal(
                strategy_id=self._name,
                symbol=straddle.yes_token_id,
                signal_type=SignalType.CLOSE,
                weight=0.0,
                confidence=1.0,
                metadata={"exit_reason": reason, "leg": "yes"},
            ),
            Signal(
                strategy_id=self._name,
                symbol=straddle.no_token_id,
                signal_type=SignalType.CLOSE,
                weight=0.0,
                confidence=1.0,
                metadata={"exit_reason": reason, "leg": "no"},
            ),
        ]

        logger.info("Straddle exit: %s", reason)
        return signals

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if the date has changed."""
        now = datetime.now(timezone.utc)
        if self._last_reset_date is None or self._last_reset_date.date() != now.date():
            self._straddles_today = 0
            self._last_reset_date = now
