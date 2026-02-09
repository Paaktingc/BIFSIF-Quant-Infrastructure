"""Performance analytics for backtest results.

Provides comprehensive performance metrics including:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Win/loss statistics
- Return attribution
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DrawdownInfo:
    """Information about a drawdown period.
    
    Attributes:
        start_date: When drawdown began
        end_date: When drawdown ended (or ongoing)
        recovery_date: When recovered to previous high (or None)
        max_drawdown: Maximum drawdown percentage
        duration_days: Days from start to recovery (or current)
    """
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    recovery_date: Optional[pd.Timestamp]
    max_drawdown: float
    duration_days: int


class PerformanceAnalytics:
    """Comprehensive performance analytics.
    
    Calculates key metrics from returns and equity curve.
    
    Example:
        analytics = PerformanceAnalytics(returns, equity_curve)
        print(analytics.sharpe_ratio)
        print(analytics.max_drawdown)
        print(analytics.summary())
    """
    
    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
    
    def __init__(
        self, 
        returns: pd.Series,
        equity_curve: pd.Series,
        risk_free_rate: float = 0.02,
    ) -> None:
        """Initialize analytics.
        
        Args:
            returns: Daily returns series
            equity_curve: Portfolio value series
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self._returns = returns.dropna()
        self._equity = equity_curve
        self._rf_rate = risk_free_rate
        
        # Pre-compute common values
        self._trading_days = len(self._returns)
        self._daily_rf = (1 + self._rf_rate) ** (1/self.TRADING_DAYS_PER_YEAR) - 1
        
    # ─────────────────────────────────────────────────────────────────────────
    # RETURN METRICS
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def total_return(self) -> float:
        """Total cumulative return."""
        if len(self._equity) < 2:
            return 0.0
        return float((self._equity.iloc[-1] / self._equity.iloc[0]) - 1)
    
    @property
    def annual_return(self) -> float:
        """Annualized return (CAGR)."""
        if self._trading_days < 1:
            return 0.0
        years = self._trading_days / self.TRADING_DAYS_PER_YEAR
        if years <= 0:
            return 0.0
        return float((1 + self.total_return) ** (1/years) - 1)
    
    @property
    def daily_return_mean(self) -> float:
        """Average daily return."""
        return float(self._returns.mean())
    
    @property
    def daily_return_std(self) -> float:
        """Daily return standard deviation."""
        return float(self._returns.std())
    
    @property
    def annual_volatility(self) -> float:
        """Annualized volatility."""
        return self.daily_return_std * np.sqrt(self.TRADING_DAYS_PER_YEAR)
    
    # ─────────────────────────────────────────────────────────────────────────
    # RISK-ADJUSTED METRICS
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio (annualized)."""
        if self.annual_volatility == 0:
            return 0.0
        excess_return = self.annual_return - self._rf_rate
        return excess_return / self.annual_volatility
    
    @property
    def sortino_ratio(self) -> float:
        """Sortino ratio (annualized).
        
        Uses downside deviation instead of total volatility.
        """
        downside_returns = self._returns[self._returns < self._daily_rf]
        if len(downside_returns) == 0:
            return float('inf') if self.annual_return > self._rf_rate else 0.0
        
        downside_std = float(downside_returns.std())
        annual_downside = downside_std * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        if annual_downside == 0:
            return 0.0
        
        excess_return = self.annual_return - self._rf_rate
        return excess_return / annual_downside
    
    @property
    def calmar_ratio(self) -> float:
        """Calmar ratio (annual return / max drawdown)."""
        mdd = abs(self.max_drawdown)
        if mdd == 0:
            return float('inf') if self.annual_return > 0 else 0.0
        return self.annual_return / mdd
    
    # ─────────────────────────────────────────────────────────────────────────
    # DRAWDOWN METRICS
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown (as negative percentage)."""
        dd_series = self._calculate_drawdown_series()
        if len(dd_series) == 0:
            return 0.0
        return float(dd_series.min())
    
    @property
    def max_drawdown_duration(self) -> int:
        """Duration of longest drawdown in days."""
        drawdowns = self._find_drawdown_periods()
        if not drawdowns:
            return 0
        return max(d.duration_days for d in drawdowns)
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        dd_series = self._calculate_drawdown_series()
        if len(dd_series) == 0:
            return 0.0
        return float(dd_series.iloc[-1])
    
    def _calculate_drawdown_series(self) -> pd.Series:
        """Calculate drawdown at each point."""
        rolling_max = self._equity.expanding().max()
        drawdowns = (self._equity - rolling_max) / rolling_max
        return drawdowns
    
    def _find_drawdown_periods(self) -> list[DrawdownInfo]:
        """Find all drawdown periods."""
        dd_series = self._calculate_drawdown_series()
        periods = []
        
        in_drawdown = False
        start_idx = None
        max_dd = 0.0
        
        for i, (date, dd) in enumerate(dd_series.items()):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
                max_dd = dd
            elif dd < 0 and in_drawdown:
                # Continue drawdown
                max_dd = min(max_dd, dd)
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                periods.append(DrawdownInfo(
                    start_date=dd_series.index[start_idx],
                    end_date=dd_series.index[i-1],
                    recovery_date=date,
                    max_drawdown=max_dd,
                    duration_days=i - start_idx,
                ))
        
        # Handle ongoing drawdown
        if in_drawdown:
            periods.append(DrawdownInfo(
                start_date=dd_series.index[start_idx],
                end_date=dd_series.index[-1],
                recovery_date=None,
                max_drawdown=max_dd,
                duration_days=len(dd_series) - start_idx,
            ))
        
        return periods
    
    # ─────────────────────────────────────────────────────────────────────────
    # WIN/LOSS METRICS
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def win_rate(self) -> float:
        """Percentage of positive return days."""
        if len(self._returns) == 0:
            return 0.0
        wins = (self._returns > 0).sum()
        return float(wins / len(self._returns))
    
    @property
    def profit_factor(self) -> float:
        """Ratio of gross profit to gross loss."""
        gains = self._returns[self._returns > 0].sum()
        losses = abs(self._returns[self._returns < 0].sum())
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        return float(gains / losses)
    
    @property
    def avg_win(self) -> float:
        """Average winning day return."""
        wins = self._returns[self._returns > 0]
        return float(wins.mean()) if len(wins) > 0 else 0.0
    
    @property
    def avg_loss(self) -> float:
        """Average losing day return."""
        losses = self._returns[self._returns < 0]
        return float(losses.mean()) if len(losses) > 0 else 0.0
    
    @property
    def best_day(self) -> float:
        """Best single day return."""
        return float(self._returns.max()) if len(self._returns) > 0 else 0.0
    
    @property
    def worst_day(self) -> float:
        """Worst single day return."""
        return float(self._returns.min()) if len(self._returns) > 0 else 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # MONTHLY/YEARLY RETURNS
    # ─────────────────────────────────────────────────────────────────────────
    
    def monthly_returns(self) -> pd.Series:
        """Calculate monthly returns."""
        return self._equity.resample('M').last().pct_change().dropna()
    
    def yearly_returns(self) -> pd.Series:
        """Calculate yearly returns."""
        return self._equity.resample('Y').last().pct_change().dropna()
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    
    def summary(self) -> str:
        """Generate formatted summary of all metrics."""
        lines = [
            "=" * 50,
            "PERFORMANCE SUMMARY",
            "=" * 50,
            "",
            "RETURNS",
            f"  Total Return:      {self.total_return:>10.2%}",
            f"  Annual Return:     {self.annual_return:>10.2%}",
            f"  Annual Volatility: {self.annual_volatility:>10.2%}",
            "",
            "RISK-ADJUSTED",
            f"  Sharpe Ratio:      {self.sharpe_ratio:>10.2f}",
            f"  Sortino Ratio:     {self.sortino_ratio:>10.2f}",
            f"  Calmar Ratio:      {self.calmar_ratio:>10.2f}",
            "",
            "DRAWDOWNS",
            f"  Max Drawdown:      {self.max_drawdown:>10.2%}",
            f"  Max DD Duration:   {self.max_drawdown_duration:>10d} days",
            f"  Current Drawdown:  {self.current_drawdown:>10.2%}",
            "",
            "WIN/LOSS",
            f"  Win Rate:          {self.win_rate:>10.2%}",
            f"  Profit Factor:     {self.profit_factor:>10.2f}",
            f"  Avg Win:           {self.avg_win:>10.4%}",
            f"  Avg Loss:          {self.avg_loss:>10.4%}",
            f"  Best Day:          {self.best_day:>10.2%}",
            f"  Worst Day:         {self.worst_day:>10.2%}",
            "",
            "=" * 50,
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Export all metrics as dictionary."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "current_drawdown": self.current_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "best_day": self.best_day,
            "worst_day": self.worst_day,
            "trading_days": self._trading_days,
        }
