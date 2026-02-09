"""Dashboard output for terminal display.

Provides formatted text output for monitoring:
- Portfolio status
- Position summary
- Alert summary
- Strategy performance
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
import sys

from bifs_quant_engine.monitoring.metrics import (
    PortfolioMetrics,
    RiskMetrics,
    StrategyMetricsSnapshot,
)
from bifs_quant_engine.monitoring.alerts import Alert, AlertLevel


class DashboardOutput:
    """Text-based dashboard display.
    
    Outputs formatted monitoring data to terminal.
    
    Example:
        dashboard = DashboardOutput()
        dashboard.render(
            portfolio=portfolio_metrics,
            risk=risk_metrics,
            strategies=strategy_metrics,
            alerts=active_alerts,
        )
    """
    
    def __init__(
        self, 
        width: int = 80,
        use_color: bool = True,
    ) -> None:
        """Initialize dashboard.
        
        Args:
            width: Terminal width
            use_color: Enable ANSI colors
        """
        self._width = width
        self._use_color = use_color and sys.stdout.isatty()
    
    def render(
        self,
        portfolio: Optional[PortfolioMetrics] = None,
        risk: Optional[RiskMetrics] = None,
        strategies: Optional[Dict[str, StrategyMetricsSnapshot]] = None,
        alerts: Optional[List[Alert]] = None,
    ) -> str:
        """Render complete dashboard.
        
        Args:
            portfolio: Portfolio metrics
            risk: Risk metrics
            strategies: Strategy metrics by name
            alerts: Active alerts
            
        Returns:
            Formatted dashboard string
        """
        lines = []
        
        # Header
        lines.append(self._header("TRADING DASHBOARD"))
        lines.append(f"  Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Portfolio section
        if portfolio:
            lines.extend(self._render_portfolio(portfolio))
            lines.append("")
        
        # Risk section
        if risk:
            lines.extend(self._render_risk(risk))
            lines.append("")
        
        # Strategies section
        if strategies:
            lines.extend(self._render_strategies(strategies))
            lines.append("")
        
        # Alerts section
        if alerts:
            lines.extend(self._render_alerts(alerts))
            lines.append("")
        
        lines.append(self._separator("="))
        
        return "\n".join(lines)
    
    def print(
        self,
        portfolio: Optional[PortfolioMetrics] = None,
        risk: Optional[RiskMetrics] = None,
        strategies: Optional[Dict[str, StrategyMetricsSnapshot]] = None,
        alerts: Optional[List[Alert]] = None,
    ) -> None:
        """Render and print dashboard to stdout."""
        output = self.render(portfolio, risk, strategies, alerts)
        print(output)
    
    def _render_portfolio(self, metrics: PortfolioMetrics) -> List[str]:
        """Render portfolio section."""
        lines = [self._header("PORTFOLIO")]
        
        nav_color = self._color("green") if metrics.pnl_today >= 0 else self._color("red")
        
        lines.append(f"  {'NAV:':<20} {self._fmt_money(metrics.nav)}")
        lines.append(f"  {'Cash:':<20} {self._fmt_money(metrics.cash)}")
        lines.append(
            f"  {'P&L Today:':<20} {nav_color}{self._fmt_money(metrics.pnl_today, sign=True)}{self._color('reset')}"
        )
        lines.append(
            f"  {'P&L Total:':<20} {self._fmt_money(metrics.pnl_total, sign=True)}"
        )
        lines.append(f"  {'Positions:':<20} {metrics.position_count}")
        lines.append("")
        lines.append(f"  {'Long Exposure:':<20} {self._fmt_money(metrics.long_value)}")
        lines.append(f"  {'Short Exposure:':<20} {self._fmt_money(metrics.short_value)}")
        lines.append(f"  {'Gross Exposure:':<20} {self._fmt_money(metrics.gross_exposure)}")
        lines.append(f"  {'Net Exposure:':<20} {self._fmt_money(metrics.net_exposure)}")
        
        return lines
    
    def _render_risk(self, metrics: RiskMetrics) -> List[str]:
        """Render risk section."""
        lines = [self._header("RISK METRICS")]
        
        dd_color = self._color("red") if metrics.current_drawdown < Decimal("-0.05") else self._color("yellow")
        
        lines.append(f"  {'VaR 95%:':<20} {self._fmt_pct(metrics.var_95)}")
        lines.append(f"  {'VaR 99%:':<20} {self._fmt_pct(metrics.var_99)}")
        lines.append(
            f"  {'Current Drawdown:':<20} {dd_color}{self._fmt_pct(metrics.current_drawdown)}{self._color('reset')}"
        )
        lines.append(f"  {'Max Drawdown:':<20} {self._fmt_pct(metrics.max_drawdown)}")
        lines.append(f"  {'Volatility (Ann):':<20} {self._fmt_pct(metrics.volatility)}")
        lines.append(f"  {'Sharpe (20d):':<20} {float(metrics.sharpe_rolling):.2f}")
        
        return lines
    
    def _render_strategies(
        self, 
        strategies: Dict[str, StrategyMetricsSnapshot],
    ) -> List[str]:
        """Render strategies section."""
        lines = [self._header("STRATEGIES")]
        
        # Table header
        header = f"  {'Strategy':<20} {'P&L Today':>12} {'P&L Total':>12} {'Positions':>10} {'Fills':>8}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        
        for name, metrics in strategies.items():
            pnl_color = self._color("green") if metrics.pnl_today >= 0 else self._color("red")
            row = (
                f"  {name:<20} "
                f"{pnl_color}{self._fmt_money(metrics.pnl_today, compact=True):>12}{self._color('reset')} "
                f"{self._fmt_money(metrics.pnl_total, compact=True):>12} "
                f"{metrics.position_count:>10} "
                f"{metrics.fills_today:>8}"
            )
            lines.append(row)
        
        return lines
    
    def _render_alerts(self, alerts: List[Alert]) -> List[str]:
        """Render alerts section."""
        lines = [self._header("ACTIVE ALERTS")]
        
        if not alerts:
            lines.append("  No active alerts")
            return lines
        
        for alert in alerts[:10]:  # Limit to 10
            level_color = {
                AlertLevel.INFO: self._color("blue"),
                AlertLevel.WARNING: self._color("yellow"),
                AlertLevel.CRITICAL: self._color("red"),
            }.get(alert.level, "")
            
            time_str = alert.timestamp.strftime("%H:%M:%S")
            lines.append(
                f"  {time_str} [{level_color}{alert.level.value:^8}{self._color('reset')}] {alert.message}"
            )
        
        if len(alerts) > 10:
            lines.append(f"  ... and {len(alerts) - 10} more alerts")
        
        return lines
    
    def _header(self, title: str) -> str:
        """Create section header."""
        return self._separator("=") + f"\n  {self._color('bold')}{title}{self._color('reset')}\n" + self._separator("-")
    
    def _separator(self, char: str = "-") -> str:
        """Create line separator."""
        return char * self._width
    
    def _fmt_money(
        self, 
        value: Decimal, 
        sign: bool = False,
        compact: bool = False,
    ) -> str:
        """Format monetary value."""
        val = float(value)
        
        if compact:
            if abs(val) >= 1_000_000:
                return f"${val/1_000_000:+.2f}M" if sign else f"${val/1_000_000:.2f}M"
            elif abs(val) >= 1_000:
                return f"${val/1_000:+.1f}K" if sign else f"${val/1_000:.1f}K"
        
        if sign:
            return f"${val:+,.2f}"
        return f"${val:,.2f}"
    
    def _fmt_pct(self, value: Decimal) -> str:
        """Format percentage value."""
        return f"{float(value) * 100:.2f}%"
    
    def _color(self, name: str) -> str:
        """Get ANSI color code."""
        if not self._use_color:
            return ""
        
        colors = {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
        }
        return colors.get(name, "")
