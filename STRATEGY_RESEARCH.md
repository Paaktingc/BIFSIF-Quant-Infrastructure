# Strategy Research Guide

This guide outlines the workflow for researching and developing quantitative strategies using the BIFSIF Quant Infrastructure.

## Research Workflow

1. **Idea Generation**: Define the hypothesis (e.g., "Stocks above their 200-day moving average outperform").
2. **Data Analysis**: Explore price data to validate the hypothesis.
3. **Strategy Implementation**: Code the logic in a new Strategy class.
4. **Backtesting**: Run the strategy against historical data.
5. **Evaluation**: Analyze Sharpe ratio, Drawdown, and Stability.

## implementation Steps

### 1. Create a New Strategy Class

Create a new file in `bifs_quant_engine/strategies/` (e.g., `my_strategy.py`). Inherit from `Strategy`.

```python
from bifs_quant_engine.strategies.base import Strategy
from datetime import datetime
import pandas as pd
from typing import Dict

class MyNewStrategy(Strategy):
    def generate_target_weights(self, date: datetime, price_history: pd.DataFrame) -> Dict[str, float]:
        # Your logic here
        return {"SPY": 1.0}
```

### 2. Run Backtest

Use the research template script to run your backtest.

```bash
python research/template_strategy_research.py
```

### 3. Analyze Results

Check the console output for metrics:

- **Total Return**: Overall performance.
- **Sharpe Ratio**: Risk-adjusted return.
- **Max Drawdown**: Worst peak-to-trough decline.

## Key Components

- **`bifs_quant_engine.strategies.base.Strategy`**: The base class you must inherit from.
- **`bifs_quant_engine.core.backtest.Backtest`**: The engine that simulates trading.
- **`bifs_quant_engine.core.data_api.MarketDataAPI`**: Retrieves historical data.

## Tips

- **Look-ahead Bias**: Ensure you don't use future data (e.g., today's close price to decide today's trade if you are trading at the open). The current backtester assumes trading at the Close, so using Close is acceptable *if* the decision is made just before close.
- **Overfitting**: Avoid tweaking parameters (e.g., moving average window) to perfectly fit the past. Use simple, robust logic.
