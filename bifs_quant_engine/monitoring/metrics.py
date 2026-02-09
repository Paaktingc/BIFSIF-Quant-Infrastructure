"""Metrics collection for real-time monitoring.

Collects and tracks:
- Portfolio metrics (NAV, P&L, positions)
- Risk metrics (exposure, drawdown, VaR)
- Strategy metrics (fills, win rate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics.
    
    Attributes:
        timestamp: When metrics were captured
        nav: Net asset value
        cash: Cash balance
        pnl_today: Today's P&L
        pnl_total: Total P&L since inception
        long_value: Long exposure value
        short_value: Short exposure value
        gross_exposure: Long + |Short|
        net_exposure: Long - |Short|
        position_count: Number of positions
    """
    timestamp: datetime
    nav: Decimal
    cash: Decimal
    pnl_today: Decimal = Decimal("0")
    pnl_total: Decimal = Decimal("0")
    long_value: Decimal = Decimal("0")
    short_value: Decimal = Decimal("0")
    gross_exposure: Decimal = Decimal("0")
    net_exposure: Decimal = Decimal("0")
    position_count: int = 0


@dataclass
class RiskMetrics:
    """Risk-level metrics.
    
    Attributes:
        timestamp: When metrics were captured
        var_95: Value at Risk (95% confidence)
        var_99: Value at Risk (99% confidence)
        max_drawdown: Maximum drawdown
        current_drawdown: Current drawdown
        beta: Portfolio beta to benchmark
        sharpe_rolling: Rolling Sharpe ratio (20-day)
        volatility: Annualized volatility
    """
    timestamp: datetime
    var_95: Decimal = Decimal("0")
    var_99: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    current_drawdown: Decimal = Decimal("0")
    beta: Decimal = Decimal("0")
    sharpe_rolling: Decimal = Decimal("0")
    volatility: Decimal = Decimal("0")


@dataclass
class StrategyMetricsSnapshot:
    """Metrics for a single strategy."""
    strategy_name: str
    pnl_today: Decimal = Decimal("0")
    pnl_total: Decimal = Decimal("0")
    position_count: int = 0
    fills_today: int = 0
    win_rate: float = 0.0


class MetricsCollector:
    """Collects and tracks real-time metrics.
    
    Features:
    - Portfolio and risk metrics collection
    - Historical metric storage
    - Rolling calculations (Sharpe, volatility)
    
    Example:
        collector = MetricsCollector()
        collector.update_portfolio(nav=Decimal("100000"), cash=Decimal("10000"))
        metrics = collector.get_current_metrics()
    """
    
    def __init__(
        self,
        initial_nav: Decimal = Decimal("0"),
        lookback_days: int = 252,
    ) -> None:
        """Initialize collector.
        
        Args:
            initial_nav: Starting NAV for P&L calculation
            lookback_days: Days of history to keep
        """
        self._initial_nav = initial_nav
        self._lookback_days = lookback_days
        
        # Current state
        self._current_portfolio: Optional[PortfolioMetrics] = None
        self._current_risk: Optional[RiskMetrics] = None
        self._strategy_metrics: Dict[str, StrategyMetricsSnapshot] = {}
        
        # Historical data
        self._nav_history: List[Tuple[datetime, Decimal]] = []
        self._daily_returns: List[float] = []
        self._high_water_mark: Decimal = initial_nav
        self._day_start_nav: Decimal = initial_nav
    
    def update_portfolio(
        self,
        nav: Decimal,
        cash: Decimal,
        positions: Optional[Dict[str, "Position"]] = None,
    ) -> PortfolioMetrics:
        """Update portfolio metrics.
        
        Args:
            nav: Current net asset value
            cash: Current cash balance
            positions: Current positions
            
        Returns:
            Updated metrics
        """
        now = datetime.now(timezone.utc)
        
        # Calculate exposures from positions
        long_value = Decimal("0")
        short_value = Decimal("0")
        position_count = 0
        
        if positions:
            for pos in positions.values():
                if pos.quantity > 0:
                    long_value += pos.quantity * pos.avg_cost
                    position_count += 1
                elif pos.quantity < 0:
                    short_value += abs(pos.quantity) * pos.avg_cost
                    position_count += 1
        
        # Calculate P&L
        pnl_today = nav - self._day_start_nav
        pnl_total = nav - self._initial_nav
        
        # Update high water mark
        if nav > self._high_water_mark:
            self._high_water_mark = nav
        
        # Store NAV for history
        self._nav_history.append((now, nav))
        self._trim_history()
        
        # Create metrics
        metrics = PortfolioMetrics(
            timestamp=now,
            nav=nav,
            cash=cash,
            pnl_today=pnl_today,
            pnl_total=pnl_total,
            long_value=long_value,
            short_value=short_value,
            gross_exposure=long_value + short_value,
            net_exposure=long_value - short_value,
            position_count=position_count,
        )
        
        self._current_portfolio = metrics
        return metrics
    
    def update_risk(self, returns: Optional[pd.Series] = None) -> RiskMetrics:
        """Update risk metrics.
        
        Args:
            returns: Optional return series for calculations
            
        Returns:
            Updated risk metrics
        """
        now = datetime.now(timezone.utc)
        
        var_95 = Decimal("0")
        var_99 = Decimal("0")
        volatility = Decimal("0")
        sharpe = Decimal("0")
        
        if returns is not None and len(returns) > 10:
            # Calculate VaR
            returns_clean = returns.dropna()
            var_95 = Decimal(str(returns_clean.quantile(0.05)))
            var_99 = Decimal(str(returns_clean.quantile(0.01)))
            
            # Annualized volatility
            daily_vol = float(returns_clean.std())
            annual_vol = daily_vol * (252 ** 0.5)
            volatility = Decimal(str(annual_vol))
            
            # Sharpe ratio (assuming 2% risk-free rate)
            annual_return = float(returns_clean.mean()) * 252
            rf = 0.02
            if annual_vol > 0:
                sharpe = Decimal(str((annual_return - rf) / annual_vol))
        
        # Calculate drawdown
        current_nav = self._current_portfolio.nav if self._current_portfolio else Decimal("0")
        current_dd = Decimal("0")
        if self._high_water_mark > 0:
            current_dd = (current_nav - self._high_water_mark) / self._high_water_mark
        
        metrics = RiskMetrics(
            timestamp=now,
            var_95=var_95,
            var_99=var_99,
            max_drawdown=min(current_dd, Decimal("0")),
            current_drawdown=current_dd if current_dd < 0 else Decimal("0"),
            volatility=volatility,
            sharpe_rolling=sharpe,
        )
        
        self._current_risk = metrics
        return metrics
    
    def update_strategy(
        self,
        strategy_name: str,
        pnl_today: Decimal = Decimal("0"),
        pnl_total: Decimal = Decimal("0"),
        position_count: int = 0,
        fills_today: int = 0,
        win_rate: float = 0.0,
    ) -> None:
        """Update strategy metrics.
        
        Args:
            strategy_name: Strategy identifier
            pnl_today: Today's P&L
            pnl_total: Total P&L
            position_count: Number of positions
            fills_today: Fills executed today
            win_rate: Win rate
        """
        self._strategy_metrics[strategy_name] = StrategyMetricsSnapshot(
            strategy_name=strategy_name,
            pnl_today=pnl_today,
            pnl_total=pnl_total,
            position_count=position_count,
            fills_today=fills_today,
            win_rate=win_rate,
        )
    
    def get_portfolio_metrics(self) -> Optional[PortfolioMetrics]:
        """Get current portfolio metrics."""
        return self._current_portfolio
    
    def get_risk_metrics(self) -> Optional[RiskMetrics]:
        """Get current risk metrics."""
        return self._current_risk
    
    def get_strategy_metrics(self) -> Dict[str, StrategyMetricsSnapshot]:
        """Get all strategy metrics."""
        return dict(self._strategy_metrics)
    
    def start_new_day(self) -> None:
        """Mark start of new trading day."""
        if self._current_portfolio:
            self._day_start_nav = self._current_portfolio.nav
        
        # Reset strategy daily metrics
        for metrics in self._strategy_metrics.values():
            metrics.pnl_today = Decimal("0")
            metrics.fills_today = 0
    
    def _trim_history(self) -> None:
        """Remove old history beyond lookback."""
        if len(self._nav_history) > self._lookback_days * 10:
            self._nav_history = self._nav_history[-self._lookback_days * 10:]


from typing import Tuple
