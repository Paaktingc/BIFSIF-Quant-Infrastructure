"""End-to-end integration tests.

Tests the complete flow from strategy signals through execution to fills.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd

from bifs_quant_engine.core.enums import OrderSide, OrderStatus
from bifs_quant_engine.core.models import Order, Fill
from bifs_quant_engine.brokers.paper_broker import PaperBroker, PaperBrokerConfig
from bifs_quant_engine.execution.execution_engine import ExecutionEngineImpl, ExecutionConfig
from bifs_quant_engine.strategies.strategy_protocol import BaseStrategy, Signal, SignalType
from bifs_quant_engine.strategies.orchestrator import StrategyOrchestrator, AllocationMethod
from bifs_quant_engine.backtest.realistic_backtest import RealisticBacktest, BacktestConfig
from bifs_quant_engine.monitoring.metrics import MetricsCollector
from bifs_quant_engine.monitoring.alerts import AlertManager, AlertConfig, AlertType, AlertLevel
from bifs_quant_engine.compliance.audit_logger import AuditLogger


class SimpleMomentumStrategy(BaseStrategy):
    """Simple momentum strategy for testing."""
    
    def __init__(self, lookback: int = 5):
        super().__init__("momentum", ["AAPL", "GOOG", "MSFT"])
        self.lookback = lookback
    
    def generate_target_weights(
        self, 
        date: datetime, 
        price_history: pd.DataFrame,
    ) -> dict:
        """Generate weights based on momentum."""
        weights = {}
        
        if len(price_history) < self.lookback:
            return weights
        
        for symbol in self._universe:
            if symbol not in price_history.columns:
                continue
            
            returns = price_history[symbol].pct_change(self.lookback).iloc[-1]
            if pd.notna(returns):
                # Long winners, short losers
                weights[symbol] = float(np.sign(returns) * 0.1)
        
        return weights
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: dict,
    ) -> list:
        """Generate trading signals."""
        weights = self.generate_target_weights(datetime.now(), market_data)
        signals = []
        
        for symbol, weight in weights.items():
            if weight > 0:
                signals.append(Signal(
                    strategy_id=self.name,
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    weight=weight,
                ))
            elif weight < 0:
                signals.append(Signal(
                    strategy_id=self.name,
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    weight=weight,
                ))
        
        return signals


class TestEndToEndBacktest:
    """Integration tests for backtesting flow."""
    
    @pytest.fixture
    def price_data(self):
        """Generate test price data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        
        data = {}
        for symbol in ["AAPL", "GOOG", "MSFT"]:
            base = 100 + np.random.randint(0, 100)
            returns = np.random.randn(100) * 0.02
            prices = base * np.cumprod(1 + returns)
            data[symbol] = prices
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def strategy(self):
        """Create test strategy."""
        return SimpleMomentumStrategy(lookback=5)
    
    def test_backtest_end_to_end(self, strategy, price_data):
        """Test complete backtest flow."""
        config = BacktestConfig(
            initial_cash=Decimal("100000"),
            slippage_bps=5,
            commission_per_share=Decimal("0.01"),
        )
        
        # Create backtest with strategy, universe, and provide price data loader
        def data_loader(universe, start, end):
            return price_data
        
        backtest = RealisticBacktest(
            strategy=strategy,
            universe=["AAPL", "GOOG", "MSFT"],
            start="2020-01-01",
            end="2020-05-22",
            config=config,
            data_loader=data_loader,
        )
        result = backtest.run()
        
        # Verify result structure
        assert result is not None
        assert len(result.equity_curve) > 0
        assert float(result.equity_curve.iloc[0]) == float(config.initial_cash)
        
        # Verify analytics
        analytics = result.analytics
        assert analytics.total_return is not None
    
    def test_backtest_with_trades(self, strategy, price_data):
        """Test that backtest executes trades."""
        config = BacktestConfig(
            initial_cash=Decimal("100000"),
            slippage_bps=0,
            commission_per_share=Decimal("0"),
        )
        
        def data_loader(universe, start, end):
            return price_data
        
        backtest = RealisticBacktest(
            strategy=strategy,
            universe=["AAPL", "GOOG", "MSFT"],
            start="2020-01-01",
            config=config,
            data_loader=data_loader,
        )
        result = backtest.run()
        
        # Should have some trades
        assert len(result.trades) > 0
        
        # Verify trade structure
        trade = result.trades[0]
        assert "symbol" in trade
        assert "quantity" in trade
        assert "price" in trade


class TestEndToEndPaperTrading:
    """Integration tests for paper trading flow."""
    
    @pytest.fixture
    def broker(self):
        """Create paper broker with quotes."""
        config = PaperBrokerConfig(
            initial_cash=Decimal("100000"),
            latency_ms=0,
            slippage_bps=0,
        )
        broker = PaperBroker(config)
        broker.connect()
        broker.update_quotes({
            "AAPL": {"bid": 150.0, "ask": 150.0, "last": 150.0},
            "GOOG": {"bid": 100.0, "ask": 100.0, "last": 100.0},
            "MSFT": {"bid": 300.0, "ask": 300.0, "last": 300.0},
        })
        return broker
    
    @pytest.fixture
    def execution_engine(self, broker):
        """Create execution engine."""
        return ExecutionEngineImpl(broker)
    
    @pytest.fixture
    def orchestrator(self):
        """Create strategy orchestrator."""
        return StrategyOrchestrator(
            total_capital=Decimal("100000"),
            allocation_method=AllocationMethod.EQUAL,
        )
    
    def test_signal_to_execution_flow(self, broker, execution_engine, orchestrator):
        """Test complete signal → order → fill flow."""
        # Register strategy
        strategy = SimpleMomentumStrategy()
        orchestrator.register_strategy(strategy)
        
        # Create price data that will generate signals
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        price_data = pd.DataFrame({
            "AAPL": [150 + i for i in range(20)],  # Rising
            "GOOG": [100 - i * 0.5 for i in range(20)],  # Falling
            "MSFT": [300] * 20,  # Flat
        }, index=dates)
        
        # Generate signals directly from strategy
        signals = strategy.generate_signals(price_data, {})
        
        # Convert signals to orders
        orders = []
        for signal in signals:
            if signal.signal_type == SignalType.LONG:
                side = OrderSide.BUY
            else:
                side = OrderSide.SHORT
            
            orders.append(Order(
                symbol=signal.symbol,
                side=side,
                quantity=10,
                strategy_id=signal.strategy_id,
            ))
        
        # Execute orders
        results = execution_engine.submit_orders(orders)
        
        # Verify execution
        assert len(results) > 0
        filled = [r for r in results if r.status == OrderStatus.FILLED]
        assert len(filled) > 0
        
        # Verify positions updated
        positions = broker.get_positions()
        assert len(positions) > 0
    
    def test_position_reconciliation(self, broker, execution_engine):
        """Test position sync between engine and broker."""
        # Execute some orders
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        execution_engine.submit_orders([order])
        
        # Reconcile
        report = execution_engine.reconcile()
        
        # Positions should match
        broker_positions = broker.get_positions()
        engine_positions = execution_engine.sync_positions()
        
        for symbol, broker_pos in broker_positions.items():
            assert symbol in engine_positions
            assert engine_positions[symbol].quantity == broker_pos.quantity


class TestEndToEndMonitoring:
    """Integration tests for monitoring during trading."""
    
    @pytest.fixture
    def broker(self):
        """Create paper broker."""
        config = PaperBrokerConfig(
            initial_cash=Decimal("100000"),
            latency_ms=0,
        )
        broker = PaperBroker(config)
        broker.connect()
        broker.update_quotes({
            "AAPL": {"bid": 150.0, "ask": 150.0, "last": 150.0},
        })
        return broker
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector."""
        return MetricsCollector(initial_nav=Decimal("100000"))
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager with test rules."""
        manager = AlertManager()
        manager.add_rule(AlertConfig(
            name="test_drawdown",
            alert_type=AlertType.DRAWDOWN,
            level=AlertLevel.WARNING,
            threshold=-0.01,  # 1% drawdown trigger
            comparison="below",
        ))
        return manager
    
    def test_metrics_during_trading(self, broker, metrics_collector):
        """Test metrics update during trading."""
        # Initial state
        nav = broker.get_cash_balance()
        metrics = metrics_collector.update_portfolio(
            nav=nav,
            cash=nav,
            positions=broker.get_positions(),
        )
        
        assert metrics.nav == Decimal("100000")
        assert metrics.position_count == 0
        
        # After trade
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(order)
        
        positions = broker.get_positions()
        cash = broker.get_cash_balance()
        nav = cash + sum(
            p.quantity * p.avg_cost 
            for p in positions.values()
        )
        
        metrics = metrics_collector.update_portfolio(
            nav=nav,
            cash=cash,
            positions=positions,
        )
        
        assert metrics.position_count == 1
        assert metrics.long_value > 0
    
    def test_alerts_on_threshold_breach(self, alert_manager):
        """Test alert triggering on metric breach."""
        # Simulate drawdown
        alerts = alert_manager.check_metric(AlertType.DRAWDOWN, -0.02)
        
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING
        assert "test_drawdown" in alerts[0].message


class TestEndToEndCompliance:
    """Integration tests for compliance and audit trail."""
    
    @pytest.fixture
    def broker(self):
        """Create paper broker."""
        config = PaperBrokerConfig(
            initial_cash=Decimal("100000"),
            latency_ms=0,
        )
        broker = PaperBroker(config)
        broker.connect()
        broker.update_quotes({
            "AAPL": {"bid": 150.0, "ask": 150.0, "last": 150.0},
        })
        return broker
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger."""
        return AuditLogger()
    
    def test_audit_trail_during_trading(self, broker, audit_logger):
        """Test audit logging during order lifecycle."""
        # Log order submission
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        audit_logger.log_order_submitted(order, strategy_id="test_strategy")
        
        # Execute order
        result = broker.submit_order(order)
        
        # Log fill
        if result.status == OrderStatus.FILLED:
            audit_logger.log_order_filled(
                order=result,
                fill_price=result.avg_fill_price,
            )
        
        # Verify audit trail
        entries = audit_logger.get_entries()
        assert len(entries) >= 3  # System start + submit + fill
        
        # Verify integrity
        assert audit_logger.verify_integrity()
    
    def test_audit_risk_decisions(self, audit_logger):
        """Test audit logging for risk decisions."""
        from uuid import uuid4
        
        order_id = uuid4()
        
        # Log risk approval
        audit_logger.log_risk_decision(
            order_id=order_id,
            decision="approved",
            reason="Within position limits",
            risk_metrics={"exposure": 0.05},
        )
        
        # Query by order
        entries = audit_logger.get_entries(order_id=order_id)
        assert len(entries) == 1
        assert entries[0].data["decision"] == "approved"
