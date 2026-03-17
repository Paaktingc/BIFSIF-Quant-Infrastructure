"""Async trading session for Polymarket strategies.

Coordinates concurrent data feeds (RSS polling, Binance WebSocket),
strategy signal generation, and order execution via an asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from bifs_quant_engine.core.enums import OrderSide, OrderType
from bifs_quant_engine.core.models import Order
from bifs_quant_engine.data.binance_feed import BinanceFeed
from bifs_quant_engine.data.news_monitor import NewsMonitor
from bifs_quant_engine.strategies.strategy_protocol import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Trading session lifecycle states."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class PolymarketSessionConfig:
    """Configuration for the Polymarket trading session.

    Attributes:
        signal_check_interval: Seconds between strategy signal checks.
        position_check_interval: Seconds between position reconciliation.
        max_daily_loss: Maximum daily loss in USDC before halt.
        enable_binance_feed: Whether to connect Binance WebSocket.
    """

    signal_check_interval: int = 30
    position_check_interval: int = 60
    max_daily_loss: Decimal = Decimal("200")
    enable_binance_feed: bool = True


class PolymarketSession:
    """Async trading session orchestrating Polymarket strategies.

    Unlike the equity TradingSession which uses threading, this session
    uses asyncio to coordinate:
    - RSS feed polling (NewsMonitor) for Strategy 1
    - Binance WebSocket streaming for Strategy 2
    - Periodic signal generation and order execution
    - Position monitoring and risk checks

    Example:
        session = PolymarketSession(
            broker=polymarket_broker,
            strategies=[tz_info_arb, vol_straddle],
            config=PolymarketSessionConfig(),
        )
        asyncio.run(session.run())
    """

    def __init__(
        self,
        broker: object,  # BrokerAdapter (PolymarketBroker or Paper)
        strategies: List[BaseStrategy],
        config: Optional[PolymarketSessionConfig] = None,
        binance_feed: Optional[BinanceFeed] = None,
    ) -> None:
        """Initialize session.

        Args:
            broker: Polymarket broker adapter.
            strategies: List of strategies to run.
            config: Session configuration.
            binance_feed: Optional Binance feed for vol straddle.
        """
        self._broker = broker
        self._strategies = {s.name: s for s in strategies}
        self._config = config or PolymarketSessionConfig()
        self._binance_feed = binance_feed

        self._state = SessionState.IDLE
        self._stop_event = asyncio.Event()
        self._daily_pnl = Decimal("0")
        self._signals_generated = 0
        self._orders_submitted = 0

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state

    @property
    def strategy_names(self) -> List[str]:
        """Names of registered strategies."""
        return list(self._strategies.keys())

    @property
    def signals_generated(self) -> int:
        """Total signals generated this session."""
        return self._signals_generated

    @property
    def orders_submitted(self) -> int:
        """Total orders submitted this session."""
        return self._orders_submitted

    async def run(self) -> None:
        """Start the trading session.

        Launches concurrent tasks for feed monitoring, signal generation,
        and position management. Runs until stop() is called or a fatal
        error occurs.
        """
        self._state = SessionState.STARTING
        logger.info("Starting Polymarket trading session")

        # Connect broker
        try:
            self._broker.connect()
        except Exception as e:
            logger.error("Failed to connect broker: %s", e)
            self._state = SessionState.STOPPED
            return

        self._state = SessionState.RUNNING
        logger.info(
            "Session running with strategies: %s",
            ", ".join(self.strategy_names),
        )

        tasks = [
            asyncio.create_task(self._signal_loop()),
            asyncio.create_task(self._position_monitor_loop()),
        ]

        if self._config.enable_binance_feed and self._binance_feed:
            tasks.append(asyncio.create_task(self._binance_feed.connect()))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Session tasks cancelled")
        except Exception as e:
            logger.error("Session error: %s", e)
        finally:
            self._state = SessionState.STOPPING
            await self._shutdown(tasks)

    async def stop(self) -> None:
        """Signal the session to stop gracefully."""
        logger.info("Stop requested")
        self._stop_event.set()

    async def _signal_loop(self) -> None:
        """Periodically generate signals from all strategies."""
        import pandas as pd

        while not self._stop_event.is_set():
            try:
                current_positions = self._get_position_quantities()

                for name, strategy in self._strategies.items():
                    signals = strategy.generate_signals(
                        pd.DataFrame(), current_positions
                    )

                    self._signals_generated += len(signals)

                    for signal in signals:
                        await self._process_signal(signal)

            except Exception as e:
                logger.error("Signal loop error: %s", e)

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._config.signal_check_interval,
                )
                break
            except asyncio.TimeoutError:
                continue

    async def _position_monitor_loop(self) -> None:
        """Periodically check positions and P&L."""
        while not self._stop_event.is_set():
            try:
                positions = self._broker.get_positions()
                cash = self._broker.get_cash_balance()

                logger.debug(
                    "Positions: %d active, cash: $%.2f",
                    len(positions),
                    cash,
                )

                # Check daily loss limit
                if self._daily_pnl < -self._config.max_daily_loss:
                    logger.warning(
                        "Daily loss limit hit ($%.2f). Halting.",
                        self._daily_pnl,
                    )
                    await self.stop()

            except Exception as e:
                logger.error("Position monitor error: %s", e)

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._config.position_check_interval,
                )
                break
            except asyncio.TimeoutError:
                continue

    async def _process_signal(self, signal: Signal) -> None:
        """Convert a signal to an order and submit.

        Args:
            signal: Trading signal from a strategy.
        """
        if signal.signal_type == SignalType.HOLD:
            return

        side = OrderSide.BUY if signal.signal_type == SignalType.LONG else OrderSide.SELL

        # Get entry price from metadata or quote
        entry_price = None
        if "entry_price" in signal.metadata:
            entry_price = Decimal(str(signal.metadata["entry_price"]))

        max_shares = signal.metadata.get("max_shares", 100)

        order = Order(
            symbol=signal.symbol,
            side=side,
            quantity=int(max_shares),
            order_type=OrderType.LIMIT,
            limit_price=entry_price,
            strategy_id=signal.strategy_id,
        )

        try:
            result = await asyncio.to_thread(self._broker.submit_order, order)
            self._orders_submitted += 1
            logger.info(
                "Order %s: %s %s %d @ %s (status=%s)",
                result.order_id,
                side.value,
                signal.symbol[:12],
                order.quantity,
                entry_price,
                result.status.name,
            )
        except Exception as e:
            logger.error("Order submission failed: %s", e)

    def _get_position_quantities(self) -> Dict[str, Decimal]:
        """Get current position quantities as a dict.

        Returns:
            Dict mapping symbol to quantity.
        """
        try:
            positions = self._broker.get_positions()
            return {
                sym: Decimal(str(pos.quantity))
                for sym, pos in positions.items()
            }
        except Exception:
            return {}

    async def _shutdown(self, tasks: List[asyncio.Task]) -> None:
        """Gracefully shut down all tasks.

        Args:
            tasks: List of running tasks to cancel.
        """
        logger.info("Shutting down session")

        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        try:
            self._broker.disconnect()
        except Exception as e:
            logger.warning("Broker disconnect error: %s", e)

        if self._binance_feed:
            try:
                await self._binance_feed.disconnect()
            except Exception as e:
                logger.warning("Binance disconnect error: %s", e)

        self._state = SessionState.STOPPED
        logger.info(
            "Session stopped. Signals: %d, Orders: %d",
            self._signals_generated,
            self._orders_submitted,
        )
