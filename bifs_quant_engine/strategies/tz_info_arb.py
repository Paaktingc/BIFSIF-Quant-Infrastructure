"""Timezone Information Arbitrage strategy for Polymarket.

Exploits information latency on event-driven Polymarket contracts during
US off-hours. When official government actions (central bank decisions,
parliamentary votes, regulatory approvals) are published in non-US
timezones, Polymarket prices may remain stale because the majority of
active traders are asleep.

The strategy monitors RSS feeds and APIs, extracts events via NLP,
matches them to active Polymarket contracts, and trades mispriced
contracts before the US market wakes up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import pandas as pd

from bifs_quant_engine.data.news_monitor import FeedSource, NewsItem, NewsMonitor
from bifs_quant_engine.data.polymarket_api import PolymarketDataAPI
from bifs_quant_engine.nlp.contract_matcher import ContractMatch, ContractMatcher
from bifs_quant_engine.nlp.entity_extractor import EntityExtractor, ExtractionResult
from bifs_quant_engine.strategies.strategy_protocol import (
    BaseStrategy,
    Signal,
    SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class TZInfoArbConfig:
    """Configuration for the Timezone Information Arbitrage strategy.

    Attributes:
        max_position_per_contract: Maximum USDC per single contract.
        min_edge_threshold: Minimum price edge to trade (e.g., 0.05 = 5 cents).
        active_hours_utc: Trading windows as (start_hour, end_hour) pairs.
        max_simultaneous_positions: Maximum open positions.
        sentiment_threshold: Minimum sentiment confidence to act.
        similarity_threshold: Minimum cosine similarity for contract match.
        exit_edge_threshold: Close when edge compresses below this.
        max_holding_hours: Maximum hours to hold before exiting.
        require_two_source: Require corroboration from 2nd source.
        contract_refresh_minutes: How often to refresh contract embeddings.
    """

    max_position_per_contract: Decimal = Decimal("500")
    min_edge_threshold: float = 0.05
    active_hours_utc: List[Tuple[int, int]] = field(
        default_factory=lambda: [(22, 6)]
    )
    max_simultaneous_positions: int = 5
    sentiment_threshold: float = 0.7
    similarity_threshold: float = 0.65
    exit_edge_threshold: float = 0.02
    max_holding_hours: int = 12
    require_two_source: bool = True
    contract_refresh_minutes: int = 30


@dataclass
class OpenPosition:
    """Tracks an open position for exit monitoring.

    Attributes:
        token_id: Polymarket token ID.
        condition_id: Market condition ID.
        entry_price: Price paid for shares.
        quantity: Number of shares held.
        entry_time: When position was opened.
        entry_edge: Edge at time of entry.
        direction: "yes" or "no".
    """

    token_id: str
    condition_id: str
    entry_price: Decimal
    quantity: int
    entry_time: datetime
    entry_edge: Decimal
    direction: str


class TZInfoArbStrategy(BaseStrategy):
    """Timezone Information Arbitrage on Polymarket event contracts.

    Monitors global news feeds during US off-hours, matches events to
    Polymarket contracts via NLP, and trades mispriced contracts.

    Example:
        config = TZInfoArbConfig(max_position_per_contract=Decimal("500"))
        api = PolymarketDataAPI()
        strategy = TZInfoArbStrategy(config, api, feed_sources=[...])

        signals = strategy.generate_signals(
            market_data=pd.DataFrame(),
            current_positions={},
        )
    """

    def __init__(
        self,
        config: TZInfoArbConfig,
        polymarket_api: PolymarketDataAPI,
        feed_sources: Optional[List[FeedSource]] = None,
    ) -> None:
        """Initialize strategy.

        Args:
            config: Strategy configuration.
            polymarket_api: Polymarket data API client.
            feed_sources: News feed sources to monitor.
        """
        super().__init__(name="tz_info_arb")
        self._config = config
        self._api = polymarket_api

        # Components
        self._news_monitor = NewsMonitor(feed_sources or [])
        self._extractor = EntityExtractor(lazy_load=True)
        self._matcher = ContractMatcher(
            polymarket_api,
            similarity_threshold=config.similarity_threshold,
        )

        # State
        self._open_positions: Dict[str, OpenPosition] = {}
        self._last_refresh: Optional[datetime] = None
        self._pending_signals: List[ExtractionResult] = []

    @property
    def universe(self) -> List[str]:
        """Currently tracked token IDs."""
        return list(self._open_positions.keys())

    @property
    def open_position_count(self) -> int:
        """Number of open positions."""
        return len(self._open_positions)

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: Dict[str, Decimal],
    ) -> List[Signal]:
        """Generate trading signals from news feeds.

        This strategy is event-driven and ignores the market_data parameter.
        Instead, it polls news feeds, extracts events, and matches them to
        Polymarket contracts.

        Args:
            market_data: Ignored (strategy uses internal news feeds).
            current_positions: Current position quantities by token_id.

        Returns:
            List of trading signals.
        """
        signals: List[Signal] = []

        # Check active hours
        if not self._is_active_hours():
            return signals

        # Refresh contract cache if stale
        self._maybe_refresh_contracts()

        # Poll for new news
        new_items = self._news_monitor.poll_all()

        # Process each new item
        for item in new_items:
            entry_signals = self._process_news_item(item, current_positions)
            signals.extend(entry_signals)

        # Check exit conditions for open positions
        exit_signals = self._check_exits(current_positions)
        signals.extend(exit_signals)

        return signals

    def record_entry(
        self,
        token_id: str,
        condition_id: str,
        price: Decimal,
        quantity: int,
        direction: str,
        edge: Decimal,
    ) -> None:
        """Record a new position entry.

        Args:
            token_id: Polymarket token ID.
            condition_id: Market condition ID.
            price: Entry price.
            quantity: Number of shares.
            direction: "yes" or "no".
            edge: Edge at entry.
        """
        self._open_positions[token_id] = OpenPosition(
            token_id=token_id,
            condition_id=condition_id,
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now(timezone.utc),
            entry_edge=edge,
            direction=direction,
        )

    def record_exit(self, token_id: str) -> None:
        """Record a position exit.

        Args:
            token_id: Token ID to remove.
        """
        self._open_positions.pop(token_id, None)

    def _process_news_item(
        self,
        item: NewsItem,
        current_positions: Dict[str, Decimal],
    ) -> List[Signal]:
        """Process a single news item through the NLP pipeline.

        Args:
            item: News item to process.
            current_positions: Current positions.

        Returns:
            List of entry signals.
        """
        signals: List[Signal] = []

        # Extract entities and sentiment
        extraction = self._extractor.extract(item)

        # Check sentiment confidence
        if abs(extraction.sentiment_score) < self._config.sentiment_threshold:
            logger.debug(
                "Low confidence (%.2f) for: %s",
                extraction.sentiment_score,
                item.title[:60],
            )
            return signals

        # Check if capacity allows new positions
        if len(self._open_positions) >= self._config.max_simultaneous_positions:
            logger.debug("Max positions reached (%d)", self._config.max_simultaneous_positions)
            return signals

        # Find matching contracts
        matches = self._matcher.find_matches(extraction, top_k=3)

        for match in matches:
            signal = self._evaluate_match(match, extraction, current_positions)
            if signal is not None:
                signals.append(signal)

        return signals

    def _evaluate_match(
        self,
        match: ContractMatch,
        extraction: ExtractionResult,
        current_positions: Dict[str, Decimal],
    ) -> Optional[Signal]:
        """Evaluate a contract match and generate a signal if profitable.

        Args:
            match: The contract match.
            extraction: NLP extraction result.
            current_positions: Current positions.

        Returns:
            Signal or None if not profitable.
        """
        # Check edge threshold
        if abs(float(match.edge)) < self._config.min_edge_threshold:
            return None

        # Determine which token to buy
        if match.implied_direction == "yes" and match.market.tokens:
            token = match.market.tokens[0]  # YES token
        elif match.implied_direction == "no" and len(match.market.tokens) > 1:
            token = match.market.tokens[1]  # NO token
        else:
            return None

        # Skip if already positioned
        if token.token_id in current_positions:
            return None
        if token.token_id in self._open_positions:
            return None

        # Calculate position size
        price = token.price
        if price <= 0 or price >= Decimal("1"):
            return None

        max_shares = int(self._config.max_position_per_contract / price)
        if max_shares <= 0:
            return None

        signal = Signal(
            strategy_id=self._name,
            symbol=token.token_id,
            signal_type=SignalType.LONG,
            weight=float(match.edge),
            confidence=match.similarity_score * abs(extraction.sentiment_score),
            metadata={
                "condition_id": match.market.condition_id,
                "question": match.market.question,
                "direction": match.implied_direction,
                "edge": str(match.edge),
                "similarity": match.similarity_score,
                "sentiment": extraction.sentiment,
                "sentiment_score": extraction.sentiment_score,
                "event_type": extraction.event_type,
                "entry_price": str(price),
                "max_shares": max_shares,
                "source": extraction.news_item.source,
            },
        )

        logger.info(
            "Signal: BUY %s @ %s (edge=%.3f, sim=%.3f) — %s",
            token.token_id[:12],
            price,
            float(match.edge),
            match.similarity_score,
            match.market.question[:50],
        )

        return signal

    def _check_exits(
        self,
        current_positions: Dict[str, Decimal],
    ) -> List[Signal]:
        """Check exit conditions for open positions.

        Args:
            current_positions: Current positions.

        Returns:
            List of exit signals.
        """
        signals: List[Signal] = []
        now = datetime.now(timezone.utc)

        for token_id, pos in list(self._open_positions.items()):
            should_exit = False
            reason = ""

            # Check max holding time
            hours_held = (now - pos.entry_time).total_seconds() / 3600
            if hours_held >= self._config.max_holding_hours:
                should_exit = True
                reason = f"max_holding_hours ({hours_held:.1f}h)"

            if should_exit:
                signal = Signal(
                    strategy_id=self._name,
                    symbol=token_id,
                    signal_type=SignalType.CLOSE,
                    weight=0.0,
                    confidence=1.0,
                    metadata={
                        "exit_reason": reason,
                        "condition_id": pos.condition_id,
                        "entry_price": str(pos.entry_price),
                        "hours_held": hours_held,
                    },
                )
                signals.append(signal)
                logger.info("Exit signal: %s — %s", token_id[:12], reason)

        return signals

    def _is_active_hours(self) -> bool:
        """Check if current time is within active trading hours.

        Returns:
            True if within active hours.
        """
        now = datetime.now(timezone.utc)
        hour = now.hour

        for start, end in self._config.active_hours_utc:
            if start > end:
                # Wraps midnight (e.g., 22:00 to 06:00)
                if hour >= start or hour < end:
                    return True
            else:
                if start <= hour < end:
                    return True

        return False

    def _maybe_refresh_contracts(self) -> None:
        """Refresh contract embeddings if cache is stale."""
        now = datetime.now(timezone.utc)

        if self._last_refresh is not None:
            elapsed = (now - self._last_refresh).total_seconds() / 60
            if elapsed < self._config.contract_refresh_minutes:
                return

        try:
            count = self._matcher.refresh_contract_embeddings(
                categories=["Politics", "Economics", "Crypto"],
                limit=200,
            )
            self._last_refresh = now
            logger.info("Refreshed %d contract embeddings", count)
        except Exception as e:
            logger.error("Failed to refresh contracts: %s", e)
