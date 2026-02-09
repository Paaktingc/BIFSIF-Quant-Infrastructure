# BIFS Quant Engine - Operations Manual

A complete guide to installing, configuring, and operating the BIFS Quant Engine trading system.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
4. [Running Backtests](#4-running-backtests)
5. [Paper Trading](#5-paper-trading)
6. [Live Trading](#6-live-trading)
7. [Web Dashboard](#7-web-dashboard)
8. [Monitoring & Alerts](#8-monitoring--alerts)
9. [Compliance & Audit](#9-compliance--audit)
10. [Creating Custom Strategies](#10-creating-custom-strategies)
11. [Troubleshooting](#11-troubleshooting)
12. [Command Reference](#12-command-reference)

---

## 1. System Requirements

### Hardware
- CPU: 4+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB+ for historical data

### Software
- Python 3.10 or higher
- pip package manager
- Git (for version control)

### Network
- Internet connection for live market data
- Port 5000 available for dashboard (configurable)

---

## 2. Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd BIFSIF-Quant-Infrastructure
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Run tests to verify everything works
pytest tests/ -v

# Expected output: 168+ tests passed
```

### Step 5: Set PYTHONPATH

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export PYTHONPATH="/path/to/BIFSIF-Quant-Infrastructure:$PYTHONPATH"
```

---

## 3. Configuration

### 3.1 Broker Configuration

Edit `bifs_quant_engine/config/broker_config.yaml`:

```yaml
# Trading mode: 'paper' or 'live'
mode: paper

# Paper trading settings
paper:
  initial_cash: 100000
  latency_ms: 50
  slippage_bps: 5
  commission_per_share: 0.01
  min_commission: 1.00

# Live trading (Alpaca)
live:
  api_key: ${ALPACA_API_KEY}
  api_secret: ${ALPACA_API_SECRET}
  base_url: https://paper-api.alpaca.markets  # Use paper-api for testing
```

### 3.2 Environment Variables (for Live Trading)

```bash
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_API_SECRET="your_api_secret_here"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

### 3.3 Risk Limits

Configure in your strategy or session:

```python
from bifs_quant_engine.trading.session import SessionConfig

config = SessionConfig(
    initial_capital=100000,
    max_position_pct=0.10,      # 10% max per position
    max_daily_loss=-5000,       # Stop trading at $5k loss
    run_interval_seconds=60,    # Run strategy every minute
)
```

---

## 4. Running Backtests

### 4.1 Basic Backtest

```python
from bifs_quant_engine.backtest import RealisticBacktest, BacktestConfig
from my_strategy import MyStrategy

# Configure backtest
config = BacktestConfig(
    initial_cash=100000,
    slippage_bps=5,
    commission_per_share=0.01,
    market_impact_bps=10,
)

# Create and run backtest
backtest = RealisticBacktest(
    strategy=MyStrategy(),
    universe=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
    start="2020-01-01",
    end="2023-12-31",
    config=config,
)

result = backtest.run()

# View results
print(f"Total Return: {result.analytics.total_return:.2%}")
print(f"Sharpe Ratio: {result.analytics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.analytics.max_drawdown:.2%}")
print(f"Total Trades: {len(result.trades)}")
```

### 4.2 Analyzing Results

```python
# Get detailed analytics
analytics = result.analytics

# Key metrics
print(f"CAGR: {analytics.cagr:.2%}")
print(f"Volatility: {analytics.volatility:.2%}")
print(f"Sortino Ratio: {analytics.sortino_ratio:.2f}")
print(f"Calmar Ratio: {analytics.calmar_ratio:.2f}")
print(f"Win Rate: {analytics.win_rate:.2%}")

# Export equity curve
result.equity_curve.to_csv("equity_curve.csv")

# Export trades
import pandas as pd
pd.DataFrame(result.trades).to_csv("trades.csv")
```

---

## 5. Paper Trading

### 5.1 Quick Start - Run Demo

```bash
PYTHONPATH="." python examples/paper_trading_demo.py
```

### 5.2 Custom Paper Trading Session

```python
from decimal import Decimal
from bifs_quant_engine.brokers.paper_broker import PaperBroker, PaperBrokerConfig
from bifs_quant_engine.trading.session import TradingSession, SessionConfig
from my_strategy import MyStrategy

# Step 1: Create broker
broker = PaperBroker(PaperBrokerConfig(
    initial_cash=Decimal("100000"),
    latency_ms=50,
    slippage_bps=5,
))

# Step 2: Create session
session = TradingSession(broker, SessionConfig(
    initial_capital=Decimal("100000"),
    run_interval_seconds=60,
))

# Step 3: Add strategies
session.add_strategy(MyStrategy())

# Step 4: Set market data
import pandas as pd
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
session.set_price_history(prices)

# Set current quotes
session.update_quotes({
    "AAPL": {"bid": 150.0, "ask": 150.1, "last": 150.05},
    "MSFT": {"bid": 300.0, "ask": 300.2, "last": 300.10},
})

# Step 5: Start trading
session.start()

# Monitor status
print(session.get_status())

# Step 6: Stop when done
session.stop()
```

### 5.3 Running Single Cycle (Manual Mode)

```python
# Run one trading iteration manually
report = session.run_once()

print(f"Signals: {report['signals']}")
print(f"Orders: {report['orders']}")
print(f"Fills: {report['fills']}")
```

---

## 6. Live Trading

> ⚠️ **WARNING**: Live trading involves real money. Test thoroughly in paper mode first!

### 6.1 Setup Alpaca Account

1. Create account at https://alpaca.markets
2. Generate API keys (paper trading first!)
3. Set environment variables

### 6.2 Live Trading Code

```python
from bifs_quant_engine.brokers.alpaca_broker import AlpacaBroker
from bifs_quant_engine.trading.session import TradingSession, SessionConfig

# Create live broker (uses env vars)
broker = AlpacaBroker()

# Create session with conservative limits
session = TradingSession(broker, SessionConfig(
    initial_capital=50000,
    max_position_pct=0.05,  # 5% max per position
    max_daily_loss=-1000,   # $1k daily loss limit
))

session.add_strategy(MyStrategy())
session.start()
```

### 6.3 Safety Checklist

Before going live:
- [ ] Paper trading for at least 2 weeks
- [ ] Verified all risk limits work
- [ ] Tested order cancellation
- [ ] Reviewed audit logs
- [ ] Set up monitoring alerts
- [ ] Have emergency stop procedure ready

---

## 7. Web Dashboard

### 7.1 Start Dashboard

```python
from bifs_quant_engine.dashboard import DashboardApp
from bifs_quant_engine.monitoring.metrics import MetricsCollector

# Create with metrics
metrics = MetricsCollector(initial_nav=100000)
app = DashboardApp(metrics_collector=metrics)

# Start server
app.run(host="0.0.0.0", port=5000)
```

Then open: http://localhost:5000

### 7.2 Dashboard Features

- **Portfolio Overview**: NAV, P&L, exposure
- **Positions Table**: Current holdings with unrealized P&L
- **Equity Chart**: Historical portfolio value
- **Alerts Panel**: Recent alerts and warnings
- **Auto-refresh**: Updates every 5 seconds

### 7.3 Running in Background

```python
# Start dashboard in background thread
thread = app.run_background(port=5000)

# Your trading continues...
session.start()
```

---

## 8. Monitoring & Alerts

### 8.1 Setup Metrics Collection

```python
from bifs_quant_engine.monitoring.metrics import MetricsCollector
from bifs_quant_engine.monitoring.alerts import AlertManager, AlertConfig, AlertType, AlertLevel

# Create collectors
metrics = MetricsCollector(initial_nav=100000)
alerts = AlertManager()

# Add alert rules
alerts.add_rule(AlertConfig(
    name="daily_loss_warning",
    alert_type=AlertType.PNL,
    level=AlertLevel.WARNING,
    threshold=-2000,
    comparison="below",
))

alerts.add_rule(AlertConfig(
    name="drawdown_critical",
    alert_type=AlertType.DRAWDOWN,
    level=AlertLevel.CRITICAL,
    threshold=-0.05,  # 5% drawdown
    comparison="below",
))
```

### 8.2 Check Alerts

```python
# Manual check
portfolio = metrics.get_portfolio_metrics()
triggered = alerts.check_metric(AlertType.PNL, portfolio.pnl_today)

for alert in triggered:
    print(f"[{alert.level}] {alert.message}")
```

---

## 9. Compliance & Audit

### 9.1 Enable Audit Logging

```python
from bifs_quant_engine.compliance.audit_logger import AuditLogger

audit = AuditLogger(persist_path="/path/to/audit_logs")

# Logging happens automatically when integrated with session
# Or log manually:
audit.log_order_submitted(order, strategy_id="my_strategy")
audit.log_order_filled(order, fill_price)
audit.log_risk_decision(order_id, "approved", "Within limits")
```

### 9.2 Generate Reports

```python
from bifs_quant_engine.compliance.compliance_reporter import ComplianceReporter

reporter = ComplianceReporter(audit_logger=audit)

# Generate trade log
reporter.generate_trade_log(
    start_date="2024-01-01",
    end_date="2024-01-31",
    output_path="trade_log.csv",
    format="csv",
)

# Generate daily summary
reporter.generate_daily_summary(
    date="2024-01-15",
    output_path="daily_summary.html",
    format="html",
)
```

### 9.3 Verify Audit Integrity

```python
# Check audit trail hasn't been tampered with
is_valid = audit.verify_integrity()
print(f"Audit integrity: {'VALID' if is_valid else 'CORRUPTED'}")
```

---

## 10. Creating Custom Strategies

### 10.1 Strategy Template

```python
from datetime import datetime
import pandas as pd
from bifs_quant_engine.strategies.strategy_protocol import BaseStrategy, Signal, SignalType

class MyCustomStrategy(BaseStrategy):
    """Template for custom strategy."""
    
    def __init__(self, lookback: int = 20):
        super().__init__(
            name="my_strategy",
            universe=["AAPL", "MSFT", "GOOG"],
        )
        self.lookback = lookback
    
    def generate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> dict:
        """Generate target portfolio weights.
        
        Args:
            date: Current date
            price_history: Historical prices DataFrame
            
        Returns:
            Dict mapping symbol to weight (-1 to 1)
        """
        if len(price_history) < self.lookback:
            return {}
        
        weights = {}
        for symbol in self._universe:
            if symbol not in price_history.columns:
                continue
            
            # Your logic here
            # Example: momentum
            returns = price_history[symbol].pct_change(self.lookback).iloc[-1]
            weights[symbol] = 0.1 if returns > 0 else -0.1
        
        return weights
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: dict,
    ) -> list:
        """Generate trading signals for live trading."""
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
                    weight=abs(weight),
                ))
        
        return signals
```

### 10.2 Market-Neutral Strategy (Pairs Trading)

```python
from bifs_quant_engine.strategies.market_neutral import PairsTradingStrategy

strategy = PairsTradingStrategy(
    pairs=[("AAPL", "MSFT"), ("GOOG", "META")],
    lookback=60,
    entry_zscore=2.0,
    exit_zscore=0.5,
    stop_zscore=4.0,
)
```

---

## 11. Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: bifs_quant_engine` | Set PYTHONPATH: `export PYTHONPATH="/path/to/project"` |
| `No quote available for SYMBOL` | Call `broker.update_quotes()` before trading |
| Orders not filling | Check broker.is_connected, verify quotes are set |
| Dashboard not loading | Ensure Flask is installed: `pip install flask flask-cors` |
| Alpaca API error | Verify env vars and API key permissions |

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Verify System Health

```bash
# Run all tests
pytest tests/ -v

# Check specific component
pytest tests/unit/test_paper_broker.py -v
```

---

## 12. Command Reference

### Testing
```bash
pytest tests/ -v                    # All tests
pytest tests/unit/ -v               # Unit tests only
pytest tests/integration/ -v        # Integration tests only
pytest tests/ -k "backtest" -v      # Tests matching pattern
```

### Running Examples
```bash
PYTHONPATH="." python examples/paper_trading_demo.py
```

### Starting Dashboard
```bash
PYTHONPATH="." python -c "from bifs_quant_engine.dashboard import DashboardApp; DashboardApp().run(port=5000)"
```

### Python REPL Quick Start
```python
# In Python shell
from bifs_quant_engine.brokers.paper_broker import PaperBroker
from bifs_quant_engine.trading.session import TradingSession

broker = PaperBroker()
session = TradingSession(broker)
```

---

## Appendix: Project Structure

```
BIFSIF-Quant-Infrastructure/
├── bifs_quant_engine/
│   ├── brokers/           # Broker adapters
│   │   ├── paper_broker.py
│   │   └── alpaca_broker.py
│   ├── execution/         # Order execution
│   │   └── execution_engine.py
│   ├── backtest/          # Backtesting
│   │   ├── realistic_backtest.py
│   │   └── performance_analytics.py
│   ├── strategies/        # Strategy framework
│   │   ├── strategy_protocol.py
│   │   ├── orchestrator.py
│   │   └── market_neutral/
│   ├── monitoring/        # Metrics & alerts
│   │   ├── metrics.py
│   │   ├── alerts.py
│   │   └── dashboard.py
│   ├── compliance/        # Audit & reporting
│   │   ├── audit_logger.py
│   │   └── compliance_reporter.py
│   ├── trading/           # Live trading session
│   │   └── session.py
│   └── dashboard/         # Web UI
│       └── app.py
├── examples/
│   └── paper_trading_demo.py
├── docs/
│   ├── getting_started.md
│   └── operations_manual.md
├── tests/
│   ├── unit/
│   └── integration/
└── README.md
```

---

*Last Updated: February 2026*
