"""Flask-based web dashboard for trading monitoring.

Provides:
- REST API for portfolio, strategies, alerts
- WebSocket for real-time metrics
- HTML dashboard UI
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from functools import wraps
from typing import Any, Dict, List, Optional
import json
import threading

try:
    from flask import Flask, jsonify, render_template_string, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from bifs_quant_engine.monitoring.metrics import MetricsCollector, PortfolioMetrics
from bifs_quant_engine.monitoring.alerts import AlertManager, Alert


def decimal_serializer(obj):
    """JSON serializer for Decimal objects."""
    if isinstance(obj, Decimal):
        return float(obj)
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BIFS Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-card: #16213e;
            --text-primary: #e8e8e8;
            --text-secondary: #a0a0a0;
            --accent-blue: #4cc9f0;
            --accent-green: #06d6a0;
            --accent-red: #ef476f;
            --accent-yellow: #ffd166;
            --border-color: #2d2d44;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 0;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .logo {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        
        .card-label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        
        .card-value {
            font-size: 28px;
            font-weight: 600;
        }
        
        .positive { color: var(--accent-green); }
        .negative { color: var(--accent-red); }
        
        .card-value.small {
            font-size: 18px;
        }
        
        .section {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid var(--border-color);
        }
        
        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }
        
        .chart-container {
            height: 300px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            color: var(--text-secondary);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }
        
        tbody tr:hover {
            background: rgba(255,255,255,0.02);
        }
        
        .alert-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        
        .alert-warning {
            background: rgba(255, 209, 102, 0.1);
            border-left: 3px solid var(--accent-yellow);
        }
        
        .alert-critical {
            background: rgba(239, 71, 111, 0.1);
            border-left: 3px solid var(--accent-red);
        }
        
        .btn-refresh {
            background: var(--accent-blue);
            color: var(--bg-primary);
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: opacity 0.2s;
        }
        
        .btn-refresh:hover {
            opacity: 0.8;
        }
        
        @media (max-width: 1200px) {
            .grid { grid-template-columns: repeat(2, 1fr); }
        }
        
        @media (max-width: 600px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">BIFS Trading</div>
            <div class="status">
                <span class="status-dot"></span>
                <span id="session-status">Connected</span>
                <button class="btn-refresh" onclick="refreshData()">Refresh</button>
            </div>
        </header>
        
        <div class="grid">
            <div class="card">
                <div class="card-label">Net Asset Value</div>
                <div class="card-value" id="nav">$--</div>
            </div>
            <div class="card">
                <div class="card-label">Today's P&L</div>
                <div class="card-value" id="pnl">$--</div>
            </div>
            <div class="card">
                <div class="card-label">Positions</div>
                <div class="card-value small" id="positions">--</div>
            </div>
            <div class="card">
                <div class="card-label">Exposure</div>
                <div class="card-value small" id="exposure">--%</div>
            </div>
        </div>
        
        <div class="section">
            <h3 class="section-title">Equity Curve</h3>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 24px;">
            <div class="section">
                <h3 class="section-title">Open Positions</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Quantity</th>
                            <th>Avg Cost</th>
                            <th>Market Value</th>
                            <th>Unrealized P&L</th>
                        </tr>
                    </thead>
                    <tbody id="positions-table">
                        <tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">No positions</td></tr>
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h3 class="section-title">Recent Alerts</h3>
                <div id="alerts-container">
                    <div style="text-align: center; color: var(--text-secondary); padding: 20px;">No alerts</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let equityChart;
        
        function initChart() {
            const ctx = document.getElementById('equityChart').getContext('2d');
            equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [],
                        borderColor: '#4cc9f0',
                        backgroundColor: 'rgba(76, 201, 240, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#a0a0a0' }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { 
                                color: '#a0a0a0',
                                callback: (value) => '$' + value.toLocaleString()
                            }
                        }
                    }
                }
            });
        }
        
        function formatMoney(value) {
            const num = parseFloat(value);
            return '$' + num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }
        
        function updateDashboard(data) {
            // Update cards
            if (data.portfolio) {
                document.getElementById('nav').textContent = formatMoney(data.portfolio.nav || 0);
                
                const pnl = parseFloat(data.portfolio.pnl_today || 0);
                const pnlEl = document.getElementById('pnl');
                pnlEl.textContent = formatMoney(pnl);
                pnlEl.className = 'card-value ' + (pnl >= 0 ? 'positive' : 'negative');
                
                document.getElementById('positions').textContent = data.portfolio.position_count || 0;
                document.getElementById('exposure').textContent = ((data.portfolio.gross_exposure || 0) * 100).toFixed(1) + '%';
            }
            
            // Update positions table
            if (data.positions && data.positions.length > 0) {
                const tbody = document.getElementById('positions-table');
                tbody.innerHTML = data.positions.map(pos => `
                    <tr>
                        <td>${pos.symbol}</td>
                        <td>${pos.quantity}</td>
                        <td>${formatMoney(pos.avg_cost)}</td>
                        <td>${formatMoney(pos.market_value || pos.quantity * pos.avg_cost)}</td>
                        <td class="${parseFloat(pos.unrealized_pnl || 0) >= 0 ? 'positive' : 'negative'}">${formatMoney(pos.unrealized_pnl || 0)}</td>
                    </tr>
                `).join('');
            }
            
            // Update alerts
            if (data.alerts && data.alerts.length > 0) {
                const container = document.getElementById('alerts-container');
                container.innerHTML = data.alerts.map(alert => `
                    <div class="alert-item alert-${alert.level}">
                        <span>${alert.message}</span>
                    </div>
                `).join('');
            }
            
            // Update chart
            if (data.equity_history && equityChart) {
                equityChart.data.labels = data.equity_history.map(p => p.date);
                equityChart.data.datasets[0].data = data.equity_history.map(p => p.value);
                equityChart.update();
            }
        }
        
        async function refreshData() {
            try {
                const response = await fetch('/api/dashboard');
                const data = await response.json();
                updateDashboard(data);
            } catch (err) {
                console.error('Failed to fetch data:', err);
            }
        }
        
        // Initialize
        initChart();
        refreshData();
        
        // Auto-refresh every 5 seconds
        setInterval(refreshData, 5000);
    </script>
</body>
</html>
"""


class DashboardApp:
    """Flask-based trading dashboard application.
    
    Provides REST API and web UI for monitoring trading activity.
    
    Example:
        from bifs_quant_engine.monitoring.metrics import MetricsCollector
        
        metrics = MetricsCollector(initial_nav=100000)
        app = DashboardApp(metrics_collector=metrics)
        app.run(port=5000)
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        """Initialize dashboard.
        
        Args:
            metrics_collector: Metrics source
            alert_manager: Alert source
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask not installed. Run: pip install flask flask-cors")
        
        self._metrics = metrics_collector or MetricsCollector(initial_nav=Decimal("100000"))
        self._alerts = alert_manager or AlertManager()
        
        # Position and equity data
        self._positions: List[Dict[str, Any]] = []
        self._equity_history: List[Dict[str, Any]] = []
        
        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        
        self._register_routes()
    
    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Update position data for display."""
        self._positions = positions
    
    def add_equity_point(self, date: str, value: float) -> None:
        """Add point to equity history."""
        self._equity_history.append({"date": date, "value": value})
        # Keep last 100 points
        if len(self._equity_history) > 100:
            self._equity_history = self._equity_history[-100:]
    
    def _register_routes(self) -> None:
        """Register Flask routes."""
        
        @self.app.route("/")
        def index():
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route("/api/dashboard")
        def dashboard_data():
            portfolio = self._metrics.get_portfolio_metrics()
            recent_alerts = self._alerts.get_recent_alerts(limit=5)
            
            return jsonify({
                "portfolio": {
                    "nav": float(portfolio.nav) if portfolio else 0,
                    "pnl_today": float(portfolio.pnl_today) if portfolio else 0,
                    "position_count": portfolio.position_count if portfolio else 0,
                    "gross_exposure": float(portfolio.gross_exposure) if portfolio else 0,
                    "long_value": float(portfolio.long_value) if portfolio else 0,
                    "short_value": float(portfolio.short_value) if portfolio else 0,
                },
                "positions": self._positions,
                "alerts": [
                    {"level": a.level.value, "message": a.message}
                    for a in recent_alerts
                ],
                "equity_history": self._equity_history,
            })
        
        @self.app.route("/api/portfolio")
        def get_portfolio():
            portfolio = self._metrics.get_portfolio_metrics()
            if not portfolio:
                return jsonify({"error": "No portfolio data"}), 404
            
            return jsonify({
                "nav": float(portfolio.nav),
                "cash": float(portfolio.cash),
                "pnl_today": float(portfolio.pnl_today),
                "position_count": portfolio.position_count,
                "long_value": float(portfolio.long_value),
                "short_value": float(portfolio.short_value),
                "gross_exposure": float(portfolio.gross_exposure),
            })
        
        @self.app.route("/api/positions")
        def get_positions():
            return jsonify(self._positions)
        
        @self.app.route("/api/alerts")
        def get_alerts():
            alerts = self._alerts.get_recent_alerts(limit=20)
            return jsonify([
                {
                    "id": str(a.alert_id),
                    "level": a.level.value,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in alerts
            ])
        
        @self.app.route("/api/strategies")
        def get_strategies():
            # Return strategy metrics
            strategy_metrics = self._metrics.get_strategy_metrics()
            return jsonify([
                {
                    "name": sm.strategy_id,
                    "allocation": float(sm.allocation),
                    "pnl": float(sm.pnl),
                    "positions": sm.position_count,
                }
                for sm in strategy_metrics
            ])
        
        @self.app.route("/api/health")
        def health_check():
            return jsonify({"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})
    
    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
        """Start the dashboard server.
        
        Args:
            host: Bind address
            port: Listen port
            debug: Enable debug mode
        """
        self.app.run(host=host, port=port, debug=debug)
    
    def run_background(self, host: str = "0.0.0.0", port: int = 5000) -> threading.Thread:
        """Start dashboard in background thread.
        
        Args:
            host: Bind address
            port: Listen port
            
        Returns:
            Thread running the server
        """
        thread = threading.Thread(
            target=self.app.run,
            kwargs={"host": host, "port": port, "debug": False},
            daemon=True,
        )
        thread.start()
        return thread


def create_app(
    metrics_collector: Optional[MetricsCollector] = None,
    alert_manager: Optional[AlertManager] = None,
) -> Flask:
    """Factory function to create Flask app.
    
    Useful for WSGI servers like gunicorn.
    
    Args:
        metrics_collector: Metrics source
        alert_manager: Alert source
        
    Returns:
        Configured Flask app
    """
    dashboard = DashboardApp(metrics_collector, alert_manager)
    return dashboard.app
