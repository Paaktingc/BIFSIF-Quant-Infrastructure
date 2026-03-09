"""
Template script for strategy research.
Use this to test new strategy ideas against historical data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, List

from bifs_quant_engine.core.backtest import Backtest, BacktestResult
from bifs_quant_engine.strategies.base import Strategy

# --- Define specific strategy here or import it ---

# --- Define specific strategy here or import it ---
from bifs_quant_engine.strategies.mean_reversion import MeanReversionStrategy

def analyze_results(result: BacktestResult):
    """Print stats and plot."""
    eq = result.equity_curve
    
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    daily = eq.pct_change().dropna()
    ann_ret = daily.mean() * 252
    ann_vol = daily.std() * (252 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    max_drawdown = (eq / eq.cummax() - 1).min()

    print("-" * 40)
    print(f"Backtest Results")
    print("-" * 40)
    print(f"Total Return:      {total_ret:.2%}")
    print(f"Annualized Return: {ann_ret:.2%}")
    print(f"Annualized Vol:    {ann_vol:.2%}")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Max Drawdown:      {max_drawdown:.2%}")
    print(f"Trades Executed:   {len(result.trades)}")
    print("-" * 40)

    # Plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(eq.index, eq.values, label="Strategy Equity")
        plt.title("Strategy Research Backtest")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True)
        plt.savefig("research_plot.png")
        print("Plot saved to 'research_plot.png'")
    except Exception as e:
        print(f"Could not save plot: {e}")


def main():
    # 1. Configuration
    universe = ["SPY", "QQQ", "TLT", "EEM"]
    
    # 2. Instantiate Strategy
    strat = MeanReversionStrategy(
        name="MeanReversion_v1", 
        universe=universe
    )

    # 3. Setup Backtest
    bt = Backtest(
        strategy=strat,
        universe=universe,
        start="2018-01-01",
        end=None, 
        initial_cash=100_000.0,
        rebalance_every_n_days=1, 
    )

    # 4. Run
    print("Running backtest for MeanReversionStrategy...")
    result = bt.run()

    # 5. Analyze
    analyze_results(result)


if __name__ == "__main__":
    main()
