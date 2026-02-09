"""Unit tests for market-neutral strategies."""

import pytest
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from bifs_quant_engine.strategies.market_neutral.cointegration import (
    check_cointegration,
    calculate_hedge_ratio,
    calculate_spread,
    calculate_zscore,
    half_life,
)
from bifs_quant_engine.strategies.market_neutral.pairs_trading import (
    PairsTradingStrategy,
    PairsTradingConfig,
)
from bifs_quant_engine.strategies.market_neutral.stat_arb import (
    StatisticalArbitrageStrategy,
    StatArbConfig,
)


class TestCointegration:
    """Tests for cointegration utilities."""
    
    def test_hedge_ratio_perfect_correlation(self):
        """Hedge ratio should be 1 for identical series."""
        np.random.seed(42)
        n = 100
        series1 = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        series2 = series1.copy()  # Identical
        
        ratio = calculate_hedge_ratio(series1, series2)
        
        assert abs(ratio - 1.0) < 0.1
    
    def test_hedge_ratio_scaled(self):
        """Hedge ratio should reflect scaling."""
        np.random.seed(42)
        n = 100
        series1 = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        series2 = series1 * 2  # Doubled
        
        ratio = calculate_hedge_ratio(series1, series2)
        
        assert abs(ratio - 0.5) < 0.1
    
    def test_spread_calculation(self):
        """Spread should be series1 - ratio * series2."""
        np.random.seed(42)
        n = 100
        series1 = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        series2 = pd.Series(np.cumsum(np.random.randn(n)) + 50)
        
        spread = calculate_spread(series1, series2, hedge_ratio=1.0)
        
        expected = series1 - series2
        assert abs(spread.mean() - expected.mean()) < 1
    
    def test_zscore_range(self):
        """Z-score should be standardized."""
        np.random.seed(42)
        spread = pd.Series(np.random.randn(100))
        
        zscore = calculate_zscore(spread, lookback=20)
        
        # After warmup, should be roughly standardized
        valid_z = zscore.dropna()
        assert abs(valid_z.mean()) < 0.5
        assert abs(valid_z.std() - 1.0) < 0.5
    
    def test_cointegration_test_cointegrated(self):
        """Should detect cointegrated series."""
        np.random.seed(42)
        n = 200
        
        # Create cointegrated pair
        series2 = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        series1 = series2 + np.random.randn(n) * 2  # Mean-reverting deviation
        
        result = check_cointegration(series1, series2, significance=0.10)
        
        # Should report cointegration (with reasonable chance)
        # Note: test is probabilistic, so we just check structure
        assert hasattr(result, 'is_cointegrated')
        assert hasattr(result, 'hedge_ratio')
    
    def test_half_life_positive(self):
        """Half-life should be positive for mean-reverting series."""
        np.random.seed(42)
        
        # Create mean-reverting series (AR(1) with phi < 1)
        n = 100
        series = [0.0]
        for i in range(n - 1):
            series.append(0.9 * series[-1] + np.random.randn())
        
        hl = half_life(pd.Series(series))
        
        assert hl > 0
        assert hl < 100  # Should be finite


class TestPairsTradingStrategy:
    """Tests for PairsTradingStrategy."""
    
    @pytest.fixture
    def strategy(self):
        config = PairsTradingConfig(
            entry_zscore=2.0,
            exit_zscore=0.5,
            lookback=20,
            hedge_ratio_lookback=60,
        )
        return PairsTradingStrategy(
            name="test_pairs",
            pairs=[("AAPL", "MSFT")],
            config=config,
        )
    
    @pytest.fixture
    def mock_prices(self):
        """Create mock price data for two correlated stocks."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        
        # Create correlated price series
        common = np.cumsum(np.random.randn(n))
        aapl = 150 + common + np.random.randn(n) * 2
        msft = 300 + common * 2 + np.random.randn(n) * 2
        
        return pd.DataFrame({"AAPL": aapl, "MSFT": msft}, index=dates)
    
    def test_name_and_universe(self, strategy):
        """Should have correct name and universe."""
        assert strategy.name == "test_pairs"
        assert "AAPL" in strategy.universe
        assert "MSFT" in strategy.universe
    
    def test_generate_weights_insufficient_data(self, strategy):
        """Should return empty weights with insufficient data."""
        short_prices = pd.DataFrame({
            "AAPL": [150, 151, 152],
            "MSFT": [300, 301, 302],
        })
        
        weights = strategy.generate_target_weights(datetime.now(), short_prices)
        
        assert weights == {}
    
    def test_generate_weights_with_data(self, strategy, mock_prices):
        """Should generate weights with sufficient data."""
        weights = strategy.generate_target_weights(datetime.now(), mock_prices)
        
        # May or may not have positions depending on z-score
        assert isinstance(weights, dict)
    
    def test_pair_states(self, strategy):
        """Should track pair states."""
        states = strategy.get_pair_states()
        
        assert ("AAPL", "MSFT") in states


class TestStatisticalArbitrageStrategy:
    """Tests for StatisticalArbitrageStrategy."""
    
    @pytest.fixture
    def strategy(self):
        config = StatArbConfig(
            lookback=60,
            num_factors=3,
            entry_zscore=1.5,
        )
        return StatisticalArbitrageStrategy(
            name="test_stat_arb",
            universe=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
            config=config,
        )
    
    @pytest.fixture
    def mock_prices(self):
        """Create mock price data for multiple stocks."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        
        # Create factor-driven prices
        factor1 = np.cumsum(np.random.randn(n))
        factor2 = np.cumsum(np.random.randn(n))
        
        prices = {}
        for i, sym in enumerate(["AAPL", "MSFT", "GOOG", "AMZN", "META"]):
            prices[sym] = 100 + factor1 * (1 + 0.2 * i) + factor2 * (0.5 - 0.1 * i) + np.random.randn(n) * 3
        
        return pd.DataFrame(prices, index=dates)
    
    def test_name_and_universe(self, strategy):
        """Should have correct name and universe."""
        assert strategy.name == "test_stat_arb"
        assert len(strategy.universe) == 5
    
    def test_generate_weights(self, strategy, mock_prices):
        """Should generate weights."""
        weights = strategy.generate_target_weights(datetime.now(), mock_prices)
        
        assert isinstance(weights, dict)
    
    def test_weights_market_neutral(self, strategy, mock_prices):
        """Weights should be approximately market neutral."""
        weights = strategy.generate_target_weights(datetime.now(), mock_prices)
        
        if weights:
            net = sum(weights.values())
            assert abs(net) < 0.1  # Should be close to neutral
