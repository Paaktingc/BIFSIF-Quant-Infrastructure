# Getting Started

This guide walks you through setting up and using the BIFS Quant Engine.

## Prerequisites

- Python 3.10+
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BIFSIF-Quant-Infrastructure
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For live trading with Alpaca, set environment variables:
```bash
export ALPACA_API_KEY=your_api_key
export ALPACA_API_SECRET=your_api_secret
export ALPACA_BASE_URL=https://paper-api.alpaca.markets  # for paper trading
```

## Your First Backtest

Create a simple momentum strategy:

```python
# my_strategy.py
from datetime import datetime
import pandas as pd
from bifs_quant_engine.strategies.strategy_protocol import BaseStrategy

class SimpleStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("simple_momentum", ["AAPL", "MSFT", "GOOG"])
    
    def generate_target_weights(self, date: datetime, prices: pd.DataFrame) -> dict:
        if len(prices) < 20:
            return {}
        
        # Equal weight top performers
        returns = prices.pct_change(20).iloc[-1]
        top = returns.nlargest(2).index.tolist()
        
        return {symbol: 0.5 for symbol in top}
```

Run the backtest:

```python
from bifs_quant_engine.backtest import RealisticBacktest, BacktestConfig
from my_strategy import SimpleStrategy

backtest = RealisticBacktest(
    strategy=SimpleStrategy(),
    universe=["AAPL", "MSFT", "GOOG"],
    start="2022-01-01",
    config=BacktestConfig(initial_cash=100000),
)

result = backtest.run()

# View results
print(f"Total Return: {result.analytics.total_return:.2%}")
print(f"Sharpe Ratio: {result.analytics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.analytics.max_drawdown:.2%}")
```

## Paper Trading

Test your strategy in real-time with simulated execution:

```python
from bifs_quant_engine.brokers.paper_broker import PaperBroker, PaperBrokerConfig
from bifs_quant_engine.trading.session import TradingSession

# Setup
broker = PaperBroker(PaperBrokerConfig(initial_cash=100000))
session = TradingSession(broker)

# Add strategy
session.add_strategy(SimpleStrategy())

# Set market data (you'd typically fetch real data)
session.set_price_history(prices)
session.update_quotes(current_quotes)

# Start trading
session.start()

# Monitor status
print(session.get_status())

# Stop when done
session.stop()
```

## Live Trading

Switch to live trading by changing the broker:

```python
from bifs_quant_engine.brokers.alpaca_broker import AlpacaBroker

broker = AlpacaBroker()  # Uses environment variables for credentials
session = TradingSession(broker)

# Same strategy works with live broker
session.add_strategy(SimpleStrategy())
session.start()
```

## Monitoring Dashboard

Launch the web dashboard:

```python
from bifs_quant_engine.dashboard import DashboardApp
from bifs_quant_engine.monitoring.metrics import MetricsCollector

metrics = MetricsCollector(initial_nav=100000)
app = DashboardApp(metrics_collector=metrics)
app.run(port=5000)
```

Open http://localhost:5000 to view:
- Portfolio NAV and P&L
- Open positions
- Strategy performance
- Active alerts

## Next Steps

- [Developing Custom Strategies](strategies.md)
- [API Reference](api_reference.md)
- [Risk Management Configuration](risk.md)
