"""Unit tests for realistic backtest engine."""

import pytest
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from bifs_quant_engine.backtest.realistic_backtest import (
    RealisticBacktest,
    BacktestConfig,
    BacktestResult,
)
from bifs_quant_engine.backtest.performance_analytics import PerformanceAnalytics


class MockStrategy:
    """Mock strategy for testing."""
    
    def __init__(self, weights: dict = None):
        self.weights = weights or {"AAPL": 0.5, "GOOG": 0.5}
        self.name = "mock"
    
    def generate_target_weights(self, date, price_history):
        return self.weights


def create_mock_prices(start="2020-01-01", periods=100):
    """Create mock price data."""
    dates = pd.date_range(start=start, periods=periods, freq="B")
    np.random.seed(42)
    
    # Generate realistic price paths
    aapl_returns = np.random.normal(0.001, 0.02, periods)
    goog_returns = np.random.normal(0.001, 0.02, periods)
    
    aapl_prices = 100 * np.cumprod(1 + aapl_returns)
    goog_prices = 50 * np.cumprod(1 + goog_returns)
    
    return pd.DataFrame({
        "AAPL": aapl_prices,
        "GOOG": goog_prices,
    }, index=dates)


class TestBacktestConfig:
    """Tests for BacktestConfig."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        config = BacktestConfig()
        
        assert config.initial_cash == Decimal("100000")
        assert config.slippage_bps == 5
        assert config.commission_per_share == Decimal("0.005")
    
    def test_custom_values(self):
        """Should accept custom values."""
        config = BacktestConfig(
            initial_cash=Decimal("50000"),
            slippage_bps=10,
        )
        
        assert config.initial_cash == Decimal("50000")
        assert config.slippage_bps == 10


class TestRealisticBacktest:
    """Tests for RealisticBacktest."""
    
    @pytest.fixture
    def mock_prices(self):
        return create_mock_prices()
    
    @pytest.fixture
    def backtest(self, mock_prices):
        strategy = MockStrategy()
        config = BacktestConfig(
            initial_cash=Decimal("100000"),
            slippage_bps=0,
            commission_per_share=Decimal("0"),
            min_commission=Decimal("0"),
        )
        
        def loader(universe, start, end):
            return mock_prices
        
        return RealisticBacktest(
            strategy=strategy,
            universe=["AAPL", "GOOG"],
            config=config,
            rebalance_frequency=20,
            data_loader=loader,
        )
    
    def test_run_returns_result(self, backtest):
        """Should return BacktestResult."""
        result = backtest.run()
        
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert len(result.returns) > 0
    
    def test_equity_curve_starts_at_initial_cash(self, backtest):
        """Equity should start at initial cash."""
        result = backtest.run()
        
        # First value should be close to initial cash
        assert abs(result.equity_curve.iloc[0] - 100000) < 1000
    
    def test_trades_recorded(self, backtest):
        """Should record trades."""
        result = backtest.run()
        
        # Should have trades from rebalancing
        assert len(result.trades) > 0
        
        # Each trade should have required fields
        trade = result.trades[0]
        assert "symbol" in trade
        assert "side" in trade
        assert "quantity" in trade
        assert "price" in trade
    
    def test_slippage_applied(self, mock_prices):
        """Slippage should increase costs."""
        strategy = MockStrategy({"AAPL": 1.0})
        
        # No slippage
        config_no_slip = BacktestConfig(
            slippage_bps=0,
            commission_per_share=Decimal("0"),
            min_commission=Decimal("0"),
        )
        bt_no_slip = RealisticBacktest(
            strategy=strategy,
            universe=["AAPL"],
            config=config_no_slip,
            data_loader=lambda u, s, e: mock_prices,
        )
        
        # With slippage
        config_slip = BacktestConfig(
            slippage_bps=100,  # 1%
            commission_per_share=Decimal("0"),
            min_commission=Decimal("0"),
        )
        bt_slip = RealisticBacktest(
            strategy=strategy,
            universe=["AAPL"],
            config=config_slip,
            data_loader=lambda u, s, e: mock_prices,
        )
        
        result_no_slip = bt_no_slip.run()
        result_slip = bt_slip.run()
        
        # Slippage should reduce returns
        assert result_slip.equity_curve.iloc[-1] < result_no_slip.equity_curve.iloc[-1]
    
    def test_commissions_applied(self, mock_prices):
        """Commissions should reduce returns."""
        strategy = MockStrategy({"AAPL": 1.0})
        
        # No commissions
        config_no_comm = BacktestConfig(
            slippage_bps=0,
            commission_per_share=Decimal("0"),
            min_commission=Decimal("0"),
        )
        bt_no_comm = RealisticBacktest(
            strategy=strategy,
            universe=["AAPL"],
            config=config_no_comm,
            data_loader=lambda u, s, e: mock_prices,
        )
        
        # With commissions
        config_comm = BacktestConfig(
            slippage_bps=0,
            commission_per_share=Decimal("0.01"),
            min_commission=Decimal("5.00"),
        )
        bt_comm = RealisticBacktest(
            strategy=strategy,
            universe=["AAPL"],
            config=config_comm,
            data_loader=lambda u, s, e: mock_prices,
        )
        
        result_no_comm = bt_no_comm.run()
        result_comm = bt_comm.run()
        
        # Commissions should reduce returns
        assert result_comm.equity_curve.iloc[-1] < result_no_comm.equity_curve.iloc[-1]


class TestPerformanceAnalytics:
    """Tests for PerformanceAnalytics."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample equity curve and returns."""
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        np.random.seed(42)
        
        # Generate returns with positive drift
        returns = pd.Series(
            np.random.normal(0.001, 0.01, 252),
            index=dates,
        )
        
        # Build equity curve
        equity = pd.Series(
            100000 * np.cumprod(1 + returns),
            index=dates,
        )
        
        return returns, equity
    
    def test_total_return(self, sample_data):
        """Should calculate total return."""
        returns, equity = sample_data
        analytics = PerformanceAnalytics(returns, equity)
        
        expected = (equity.iloc[-1] / equity.iloc[0]) - 1
        assert abs(analytics.total_return - expected) < 0.0001
    
    def test_sharpe_ratio(self, sample_data):
        """Should calculate Sharpe ratio."""
        returns, equity = sample_data
        analytics = PerformanceAnalytics(returns, equity)
        
        # Sharpe should be reasonable for positive-drift series
        assert analytics.sharpe_ratio > 0
    
    def test_max_drawdown(self, sample_data):
        """Should calculate max drawdown."""
        returns, equity = sample_data
        analytics = PerformanceAnalytics(returns, equity)
        
        # Max drawdown should be negative
        assert analytics.max_drawdown <= 0
    
    def test_win_rate(self, sample_data):
        """Should calculate win rate."""
        returns, equity = sample_data
        analytics = PerformanceAnalytics(returns, equity)
        
        # Win rate should be between 0 and 1
        assert 0 <= analytics.win_rate <= 1
    
    def test_summary(self, sample_data):
        """Should generate summary string."""
        returns, equity = sample_data
        analytics = PerformanceAnalytics(returns, equity)
        
        summary = analytics.summary()
        
        assert "PERFORMANCE SUMMARY" in summary
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary
    
    def test_to_dict(self, sample_data):
        """Should export metrics as dict."""
        returns, equity = sample_data
        analytics = PerformanceAnalytics(returns, equity)
        
        metrics = analytics.to_dict()
        
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics


class TestBacktestResultAnalytics:
    """Tests for analytics integration in BacktestResult."""
    
    def test_analytics_lazy_load(self):
        """Analytics should be computed lazily."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        equity = pd.Series([100000 + i * 100 for i in range(10)], index=dates)
        returns = equity.pct_change().fillna(0)
        
        result = BacktestResult(
            equity_curve=equity,
            returns=returns,
            trades=[],
            snapshots=[],
            config=BacktestConfig(),
        )
        
        # First access should compute
        analytics = result.analytics
        assert isinstance(analytics, PerformanceAnalytics)
        
        # Second access should return same instance
        assert result.analytics is analytics
