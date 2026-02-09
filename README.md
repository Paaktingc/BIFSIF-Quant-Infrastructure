# BIFS Quant Engine

Production-grade quantitative trading infrastructure for hedge fund operations.

## Features

- **Multi-Strategy Support** - Run multiple strategies with capital allocation and P&L tracking
- **Market-Neutral Strategies** - Pairs trading and statistical arbitrage with cointegration testing
- **Realistic Backtesting** - Transaction costs, slippage, market impact modeling
- **Paper & Live Trading** - PaperBroker for simulation, AlpacaBroker for live execution
- **Risk Management** - Position limits, drawdown controls, circuit breakers
- **Monitoring & Alerting** - Real-time metrics, configurable alerts, web dashboard
- **Compliance & Audit** - Hash-chained audit log, trade reporting

## Quick Start

```python
from bifs_quant_engine.brokers.paper_broker import PaperBroker, PaperBrokerConfig
from bifs_quant_engine.trading.session import TradingSession, SessionConfig

# Create paper broker
broker = PaperBroker(PaperBrokerConfig(initial_cash=100000))

# Create trading session
session = TradingSession(broker, SessionConfig())

# Add your strategy
from my_strategies import MomentumStrategy
session.add_strategy(MomentumStrategy())

# Start trading
session.start()
```

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- pandas
- numpy
- alpaca-trade-api (for live trading)
- flask, flask-cors (for web dashboard)

## Project Structure

```
bifs_quant_engine/
├── brokers/           # Broker adapters (Paper, Alpaca)
├── execution/         # Order execution engine
├── backtest/          # Backtesting framework
├── strategies/        # Strategy protocols and implementations
│   └── market_neutral/  # Pairs trading, stat arb
├── monitoring/        # Metrics, alerts, dashboard output
├── compliance/        # Audit logging, reporting
├── trading/           # Live trading session management
└── dashboard/         # Web-based monitoring UI
```

## Examples

### Run a Backtest

```python
from bifs_quant_engine.backtest import RealisticBacktest, BacktestConfig
from my_strategies import MomentumStrategy

backtest = RealisticBacktest(
    strategy=MomentumStrategy(),
    universe=["SPY", "QQQ", "IWM"],
    start="2020-01-01",
    end="2023-12-31",
    config=BacktestConfig(
        initial_cash=100000,
        slippage_bps=5,
        commission_per_share=0.01,
    ),
)

result = backtest.run()
print(result.analytics.summary())
```

### Paper Trading

```bash
python examples/paper_trading_demo.py
```

### Start Dashboard

```python
from bifs_quant_engine.dashboard import DashboardApp
from bifs_quant_engine.monitoring.metrics import MetricsCollector

metrics = MetricsCollector(initial_nav=100000)
app = DashboardApp(metrics_collector=metrics)
app.run(port=5000)
```

Then open http://localhost:5000 in your browser.

## Documentation

- [Getting Started Guide](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Strategy Development](docs/strategies.md)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## License

Proprietary - BIFS Investment Fund
