"""
Research Script for Mean Reversion Strategy
===========================================

Backtests the MeanReversionStrategy on historical data.
"""

import sys
from pathlib import Path

# Ensure the package is visible
sys.path.append(str(Path.cwd()))

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from bifs_quant_engine.strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from bifs_quant_engine.core.backtest import Backtest
from bifs_quant_engine.core.data_api import MarketDataAPI

def main():
    # 1. Setup
    symbol = "SPY"
    universe = [symbol]
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    print(f"Researching Mean Reversion on {symbol}...")
    
    # 2. Configure Strategy
    config = MeanReversionConfig(
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        bb_period=20,
        bb_std=2.0
    )
    
    strategy = MeanReversionStrategy(
        name="mean_reversion_spy",
        universe=universe,
        config=config
    )
    
    # 3. Validation Backtest
    # We use the realistic Backtest engine, not just a vectorbt one, to ensure it works in our system.
    
    backtest = Backtest(
        strategy=strategy,
        universe=universe,
        start=start_date,
        end=end_date,
        initial_cash=100000.0
    )
    
    result = backtest.run()
    
    # 4. Results
    equity_curve = result.equity_curve
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0
    
    print("Backtest Complete.")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    # trades is a list of dictionaries, not objects, so we just count them
    print(f"Trades: {len(result.trades)}")
    
    # 5. Plot
    eq = result.equity_curve
    plt.figure(figsize=(10, 6))
    plt.plot(eq.index, eq.values, label="Strategy Equity")
    plt.title(f"Mean Reversion Strategy - {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True)
    
    output_path = "mean_reversion_research.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
