"""Unit tests for multi-strategy support."""

import pytest
from datetime import datetime
from decimal import Decimal

import pandas as pd

from bifs_quant_engine.core.enums import OrderSide
from bifs_quant_engine.core.models import Fill, Position
from bifs_quant_engine.strategies.strategy_protocol import (
    BaseStrategy, 
    Signal, 
    SignalType,
)
from bifs_quant_engine.strategies.orchestrator import (
    StrategyOrchestrator,
    AllocationMethod,
    StrategyAllocation,
)


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, name: str, weights: dict = None):
        super().__init__(name, list(weights.keys()) if weights else [])
        self._weights = weights or {}
    
    def generate_target_weights(self, date, price_history):
        return self._weights
    
    def generate_signals(self, market_data, current_positions):
        return [
            Signal(
                strategy_id=self.name,
                symbol=sym,
                signal_type=SignalType.LONG if w > 0 else SignalType.SHORT,
                weight=w,
            )
            for sym, w in self._weights.items()
        ]


class TestStrategyProtocol:
    """Tests for StrategyProtocol and BaseStrategy."""
    
    def test_base_strategy_name(self):
        """Should have name property."""
        strategy = BaseStrategy("test_strategy")
        assert strategy.name == "test_strategy"
    
    def test_base_strategy_universe(self):
        """Should track universe."""
        strategy = BaseStrategy("test", universe=["AAPL", "GOOG"])
        assert "AAPL" in strategy.universe
        assert "GOOG" in strategy.universe
    
    def test_base_strategy_on_fill(self):
        """Should track fills."""
        strategy = BaseStrategy("test")
        
        fill = Fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal("150"),
        )
        strategy.on_fill(fill)
        
        fills = strategy.get_fills()
        assert len(fills) == 1
        assert fills[0].symbol == "AAPL"
    
    def test_base_strategy_reset(self):
        """Should reset state."""
        strategy = BaseStrategy("test")
        
        fill = Fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal("150"),
        )
        strategy.on_fill(fill)
        strategy.reset()
        
        assert len(strategy.get_fills()) == 0


class TestSignal:
    """Tests for Signal dataclass."""
    
    def test_signal_creation(self):
        """Should create signal with defaults."""
        signal = Signal(
            strategy_id="momentum",
            symbol="AAPL",
            signal_type=SignalType.LONG,
        )
        
        assert signal.strategy_id == "momentum"
        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.LONG
        assert signal.confidence == 1.0
    
    def test_signal_with_weight(self):
        """Should accept weight parameter."""
        signal = Signal(
            strategy_id="momentum",
            symbol="AAPL",
            signal_type=SignalType.LONG,
            weight=0.5,
        )
        
        assert signal.weight == 0.5


class TestStrategyOrchestrator:
    """Tests for StrategyOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        return StrategyOrchestrator(
            total_capital=Decimal("1000000"),
            allocation_method=AllocationMethod.EQUAL,
        )
    
    def test_register_strategy(self, orchestrator):
        """Should register strategy."""
        strategy = MockStrategy("test", {"AAPL": 0.5})
        orchestrator.register_strategy(strategy)
        
        assert "test" in orchestrator.get_active_strategies()
    
    def test_deregister_strategy(self, orchestrator):
        """Should deregister strategy."""
        strategy = MockStrategy("test", {"AAPL": 0.5})
        orchestrator.register_strategy(strategy)
        
        result = orchestrator.deregister_strategy("test")
        
        assert result is True
        assert "test" not in orchestrator.get_active_strategies()
    
    def test_deregister_unknown_strategy(self, orchestrator):
        """Should return False for unknown strategy."""
        result = orchestrator.deregister_strategy("unknown")
        assert result is False
    
    def test_set_strategy_active(self, orchestrator):
        """Should enable/disable strategy."""
        strategy = MockStrategy("test", {"AAPL": 0.5})
        orchestrator.register_strategy(strategy)
        
        orchestrator.set_strategy_active("test", False)
        assert "test" not in orchestrator.get_active_strategies()
        
        orchestrator.set_strategy_active("test", True)
        assert "test" in orchestrator.get_active_strategies()


class TestStrategyOrchestratorAllocation:
    """Tests for capital allocation."""
    
    def test_equal_allocation_single(self):
        """Single strategy should get 100%."""
        orchestrator = StrategyOrchestrator(
            allocation_method=AllocationMethod.EQUAL
        )
        
        strategy = MockStrategy("test", {"AAPL": 0.5})
        orchestrator.register_strategy(strategy)
        
        assert orchestrator.get_total_weight() == Decimal("1")
    
    def test_equal_allocation_multiple(self):
        """Multiple strategies should split equally."""
        orchestrator = StrategyOrchestrator(
            allocation_method=AllocationMethod.EQUAL
        )
        
        orchestrator.register_strategy(MockStrategy("s1", {"AAPL": 0.5}))
        orchestrator.register_strategy(MockStrategy("s2", {"GOOG": 0.5}))
        
        # Each should have 0.5 weight
        total = orchestrator.get_total_weight()
        assert total == Decimal("1")
    
    def test_custom_allocation(self):
        """Should accept custom weights."""
        orchestrator = StrategyOrchestrator(
            allocation_method=AllocationMethod.CUSTOM
        )
        
        orchestrator.register_strategy(
            MockStrategy("s1", {}), 
            weight=Decimal("0.7"),
        )
        orchestrator.register_strategy(
            MockStrategy("s2", {}), 
            weight=Decimal("0.3"),
        )
        
        assert orchestrator.get_total_weight() == Decimal("1.0")


class TestStrategyOrchestratorAggregation:
    """Tests for signal/weight aggregation."""
    
    @pytest.fixture
    def orchestrator_with_strategies(self):
        orchestrator = StrategyOrchestrator(
            allocation_method=AllocationMethod.EQUAL
        )
        
        # Two strategies with overlapping positions
        orchestrator.register_strategy(
            MockStrategy("s1", {"AAPL": 0.5, "GOOG": 0.3})
        )
        orchestrator.register_strategy(
            MockStrategy("s2", {"AAPL": 0.2, "MSFT": 0.6})
        )
        
        return orchestrator
    
    def test_aggregate_weights(self, orchestrator_with_strategies):
        """Should aggregate weights from multiple strategies."""
        now = datetime.now()
        prices = pd.DataFrame({"AAPL": [100], "GOOG": [50], "MSFT": [200]})
        
        weights = orchestrator_with_strategies.aggregate_target_weights(
            now, 
            prices,
        )
        
        # AAPL should have combined weight from both strategies
        assert "AAPL" in weights
        assert "GOOG" in weights
        assert "MSFT" in weights
    
    def test_generate_signals(self, orchestrator_with_strategies):
        """Should generate signals from all strategies."""
        prices = pd.DataFrame({"AAPL": [100], "GOOG": [50], "MSFT": [200]})
        
        signals = orchestrator_with_strategies.generate_signals(
            prices,
            {},  # No current positions
        )
        
        # Should have signals from both strategies
        assert len(signals) > 0
        
        # Each signal should have strategy_id
        strategy_ids = {s.strategy_id for s in signals}
        assert "s1" in strategy_ids
        assert "s2" in strategy_ids


class TestStrategyOrchestratorFills:
    """Tests for fill handling."""
    
    def test_on_fill_updates_metrics(self):
        """Should track fills per strategy."""
        orchestrator = StrategyOrchestrator()
        strategy = MockStrategy("test", {"AAPL": 0.5})
        orchestrator.register_strategy(strategy)
        
        fill = Fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal("150"),
        )
        orchestrator.on_fill(fill, "test")
        
        metrics = orchestrator.get_strategy_metrics("test")
        assert metrics.fills_count == 1
    
    def test_on_fill_updates_positions(self):
        """Should update strategy positions."""
        orchestrator = StrategyOrchestrator()
        strategy = MockStrategy("test", {"AAPL": 0.5})
        orchestrator.register_strategy(strategy)
        
        fill = Fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal("150"),
        )
        orchestrator.on_fill(fill, "test")
        
        metrics = orchestrator.get_strategy_metrics("test")
        assert "AAPL" in metrics.positions
        assert metrics.positions["AAPL"].quantity == 100


class TestStrategyOrchestratorExposureLimit:
    """Tests for exposure limits."""
    
    def test_exposure_limit_applied(self):
        """Should limit gross exposure."""
        orchestrator = StrategyOrchestrator(
            max_gross_exposure=Decimal("1.0"),  # No leverage
        )
        
        # Strategy wants 150% long
        strategy = MockStrategy("aggressive", {
            "AAPL": 0.5, 
            "GOOG": 0.5, 
            "MSFT": 0.5,
        })
        orchestrator.register_strategy(strategy)
        
        weights = orchestrator.aggregate_target_weights(
            datetime.now(),
            pd.DataFrame(),
        )
        
        # Gross exposure should be scaled down to 1.0
        gross = sum(abs(w) for w in weights.values())
        assert gross <= 1.0
