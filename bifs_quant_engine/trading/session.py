"""Trading session for live/paper trading.

Orchestrates the complete trading flow:
- Market data subscription
- Strategy signal generation
- Risk checks
- Order execution
- Position and P&L updates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID
import logging
import threading
import time as time_module

import pandas as pd

from bifs_quant_engine.core.enums import OrderSide, OrderStatus
from bifs_quant_engine.core.models import Order, Fill, Position
from bifs_quant_engine.core.protocols import BrokerAdapter
from bifs_quant_engine.execution.execution_engine import ExecutionEngineImpl
from bifs_quant_engine.strategies.strategy_protocol import BaseStrategy, Signal, SignalType
from bifs_quant_engine.strategies.orchestrator import StrategyOrchestrator, AllocationMethod
from bifs_quant_engine.monitoring.metrics import MetricsCollector
from bifs_quant_engine.monitoring.alerts import AlertManager
from bifs_quant_engine.compliance.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Trading session state."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class SessionConfig:
    """Trading session configuration.
    
    Attributes:
        initial_capital: Starting capital
        max_position_pct: Maximum position as % of capital
        max_daily_loss: Maximum daily loss to trigger halt
        run_interval_seconds: Interval between strategy runs
        market_open: Market open time (UTC)
        market_close: Market close time (UTC)
    """
    initial_capital: Decimal = Decimal("100000")
    max_position_pct: float = 0.10
    max_daily_loss: Decimal = Decimal("-5000")
    run_interval_seconds: int = 60
    market_open: time = time(14, 30)  # 9:30 AM ET in UTC
    market_close: time = time(21, 0)  # 4:00 PM ET in UTC


class TradingSession:
    """Orchestrates live/paper trading session.
    
    Coordinates:
    - Broker connection
    - Strategy execution via orchestrator
    - Order execution via engine
    - Monitoring and alerting
    - Audit logging
    
    Example:
        broker = PaperBroker()
        session = TradingSession(broker)
        
        strategy = MomentumStrategy()
        session.add_strategy(strategy)
        
        session.start()
        # ... trading happens ...
        session.stop()
    """
    
    def __init__(
        self,
        broker: BrokerAdapter,
        config: Optional[SessionConfig] = None,
    ) -> None:
        """Initialize trading session.
        
        Args:
            broker: Broker adapter for order execution
            config: Session configuration
        """
        self._broker = broker
        self._config = config or SessionConfig()
        
        # Core components
        self._execution_engine = ExecutionEngineImpl(broker)
        self._orchestrator = StrategyOrchestrator(
            total_capital=self._config.initial_capital,
            allocation_method=AllocationMethod.EQUAL,
        )
        self._metrics = MetricsCollector(initial_nav=self._config.initial_capital)
        self._alerts = AlertManager()
        self._audit = AuditLogger()
        
        # Session state
        self._state = SessionState.IDLE
        self._start_time: Optional[datetime] = None
        self._last_run: Optional[datetime] = None
        
        # Market data cache
        self._price_history: Optional[pd.DataFrame] = None
        self._current_quotes: Dict[str, Dict[str, Decimal]] = {}
        
        # Threading
        self._run_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Callbacks
        self._on_signal_callbacks: List[Callable[[Signal], None]] = []
        self._on_fill_callbacks: List[Callable[[Fill], None]] = []
        self._on_error_callbacks: List[Callable[[Exception], None]] = []
    
    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Returns True if session is actively trading."""
        return self._state == SessionState.RUNNING
    
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a strategy to the session.
        
        Args:
            strategy: Strategy to add
        """
        self._orchestrator.register_strategy(strategy)
        logger.info(f"Added strategy: {strategy.name}")
    
    def remove_strategy(self, name: str) -> bool:
        """Remove a strategy from the session.
        
        Args:
            name: Strategy name
            
        Returns:
            True if removed
        """
        return self._orchestrator.deregister_strategy(name)
    
    def update_quotes(self, quotes: Dict[str, Dict[str, float]]) -> None:
        """Update current market quotes.
        
        Args:
            quotes: Mapping of symbol to quote dict
        """
        self._current_quotes = {
            sym: {k: Decimal(str(v)) for k, v in q.items()}
            for sym, q in quotes.items()
        }
        
        # Also update broker quotes if paper trading
        if hasattr(self._broker, 'update_quotes'):
            self._broker.update_quotes(quotes)
    
    def set_price_history(self, history: pd.DataFrame) -> None:
        """Set historical price data for strategy calculations.
        
        Args:
            history: DataFrame with symbol columns and datetime index
        """
        self._price_history = history
    
    def start(self) -> None:
        """Start the trading session.
        
        Begins the main trading loop in a background thread.
        """
        if self._state not in (SessionState.IDLE, SessionState.STOPPED):
            raise RuntimeError(f"Cannot start from state {self._state}")
        
        self._state = SessionState.STARTING
        self._start_time = datetime.now(timezone.utc)
        self._stop_event.clear()
        
        # Connect broker
        self._broker.connect()
        
        # Start main loop
        self._run_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._run_thread.start()
        
        self._state = SessionState.RUNNING
        logger.info("Trading session started")
    
    def stop(self) -> None:
        """Stop the trading session gracefully.
        
        Cancels open orders and disconnects.
        """
        if self._state not in (SessionState.RUNNING, SessionState.PAUSED):
            return
        
        self._state = SessionState.STOPPING
        self._stop_event.set()
        
        # Wait for run thread
        if self._run_thread:
            self._run_thread.join(timeout=10)
        
        # Cancel all open orders
        cancelled = self._execution_engine.cancel_all_orders()
        logger.info(f"Cancelled {cancelled} open orders")
        
        # Disconnect broker
        self._broker.disconnect()
        
        self._state = SessionState.STOPPED
        logger.info("Trading session stopped")
    
    def pause(self) -> None:
        """Pause trading (stop generating new orders)."""
        if self._state == SessionState.RUNNING:
            self._state = SessionState.PAUSED
            logger.info("Trading session paused")
    
    def resume(self) -> None:
        """Resume trading after pause."""
        if self._state == SessionState.PAUSED:
            self._state = SessionState.RUNNING
            logger.info("Trading session resumed")
    
    def run_once(self) -> Dict[str, Any]:
        """Execute a single trading cycle.
        
        Returns:
            Cycle report with signals, orders, fills
        """
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": [],
            "orders": [],
            "fills": [],
            "errors": [],
        }
        
        try:
            # Check market hours
            if not self._is_market_open():
                report["status"] = "market_closed"
                return report
            
            # Generate signals
            signals = self._generate_signals()
            report["signals"] = [
                {"strategy": s.strategy_id, "symbol": s.symbol, "type": s.signal_type.value}
                for s in signals
            ]
            
            # Convert to orders
            orders = self._signals_to_orders(signals)
            
            # Execute orders
            results = self._execution_engine.submit_orders(orders)
            report["orders"] = [
                {"symbol": o.symbol, "side": o.side.value, "quantity": o.quantity, "status": o.status.value}
                for o in results
            ]
            
            # Record fills
            for order in results:
                if order.status == OrderStatus.FILLED:
                    self._on_order_filled(order)
                    report["fills"].append({
                        "symbol": order.symbol,
                        "quantity": order.filled_quantity,
                        "price": str(order.avg_fill_price),
                    })
            
            # Update metrics
            self._update_metrics()
            
            # Check alerts
            self._check_alerts()
            
            report["status"] = "success"
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            report["errors"].append(str(e))
            report["status"] = "error"
            
            for callback in self._on_error_callbacks:
                callback(e)
        
        self._last_run = datetime.now(timezone.utc)
        return report
    
    def get_status(self) -> Dict[str, Any]:
        """Get current session status.
        
        Returns:
            Status dict with state, metrics, positions
        """
        portfolio = self._metrics.get_portfolio_metrics()
        
        return {
            "state": self._state.value,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "nav": str(portfolio.nav) if portfolio else None,
            "pnl_today": str(portfolio.pnl_today) if portfolio else None,
            "position_count": portfolio.position_count if portfolio else 0,
            "strategies": list(self._orchestrator._strategies.keys()),
        }
    
    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        """Register callback for signal events."""
        self._on_signal_callbacks.append(callback)
    
    def on_fill(self, callback: Callable[[Fill], None]) -> None:
        """Register callback for fill events."""
        self._on_fill_callbacks.append(callback)
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for error events."""
        self._on_error_callbacks.append(callback)
    
    def _run_loop(self) -> None:
        """Main trading loop."""
        while not self._stop_event.is_set():
            if self._state == SessionState.RUNNING:
                self.run_once()
            
            # Wait for next interval
            self._stop_event.wait(self._config.run_interval_seconds)
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(timezone.utc).time()
        return self._config.market_open <= now <= self._config.market_close
    
    def _generate_signals(self) -> List[Signal]:
        """Generate signals from all strategies."""
        if self._price_history is None:
            return []
        
        signals = self._orchestrator.generate_signals(
            datetime.now(timezone.utc),
            self._price_history,
        )
        
        for signal in signals:
            self._audit.log_strategy_signal(
                strategy_id=signal.strategy_id,
                signal_type=signal.signal_type.value,
                symbol=signal.symbol,
                weight=signal.weight,
            )
            
            for callback in self._on_signal_callbacks:
                callback(signal)
        
        return signals
    
    def _signals_to_orders(self, signals: List[Signal]) -> List[Order]:
        """Convert signals to executable orders."""
        orders = []
        
        capital = self._config.initial_capital
        
        for signal in signals:
            # Get current price
            quote = self._current_quotes.get(signal.symbol)
            if not quote:
                continue
            
            price = quote.get("last", quote.get("ask", Decimal("0")))
            if price <= 0:
                continue
            
            # Calculate order size
            position_value = abs(signal.weight) * capital
            quantity = int(position_value / price)
            
            if quantity <= 0:
                continue
            
            # Determine side
            if signal.signal_type == SignalType.LONG:
                side = OrderSide.BUY
            elif signal.signal_type == SignalType.SHORT:
                side = OrderSide.SHORT
            else:
                continue
            
            order = Order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                strategy_id=signal.strategy_id,
            )
            
            self._audit.log_order_submitted(order, strategy_id=signal.strategy_id)
            orders.append(order)
        
        return orders
    
    def _on_order_filled(self, order: Order) -> None:
        """Handle order fill."""
        if order.avg_fill_price:
            self._audit.log_order_filled(order, order.avg_fill_price)
        
        # Record fill in orchestrator
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            price=order.avg_fill_price or Decimal("0"),
        )
        self._orchestrator.on_fill(fill)
        
        for callback in self._on_fill_callbacks:
            callback(fill)
    
    def _update_metrics(self) -> None:
        """Update portfolio metrics."""
        positions = self._broker.get_positions()
        cash = self._broker.get_cash_balance()
        
        # Calculate NAV
        nav = cash
        for pos in positions.values():
            quote = self._current_quotes.get(pos.symbol)
            if quote:
                price = quote.get("last", pos.avg_cost)
                nav += pos.quantity * price
        
        self._metrics.update_portfolio(nav=nav, cash=cash, positions=positions)
    
    def _check_alerts(self) -> None:
        """Check for alert conditions."""
        portfolio = self._metrics.get_portfolio_metrics()
        if not portfolio:
            return
        
        # Check daily loss
        if portfolio.pnl_today < self._config.max_daily_loss:
            logger.warning(f"Max daily loss breached: {portfolio.pnl_today}")
            self.pause()
