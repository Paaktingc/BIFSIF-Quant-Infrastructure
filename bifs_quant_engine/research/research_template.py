"""
Research Template Strategy
==========================

This script serves as a template for researching new trading strategies.
It demonstrates:
1. Loading data using the project's MarketDataAPI
2. Calculating technical indicators
3. Running a vectorized backtest (Pandas-based) to validate the hypothesis
4. Visualizing the results

Usage:
    Run this script from the project root:
    $ python bifs_quant_engine/research/research_template.py
"""

import sys
from pathlib import Path

# Ensure the package is visible (if running as a script from root)
sys.path.append(str(Path.cwd()))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the project's data loader
from bifs_quant_engine.core.data_api import MarketDataAPI


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe.
    Modify this function to test different signals.
    """
    # Example: Simple Moving Average Crossover using Close price
    # Note: MarketDataAPI.get_prices_for_universe returns columns as symbols,
    # but get_history returns OHLCV. We'll use get_history for a single asset here.
    
    # Calculate 50-day and 200-day Simple Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate RSI (14-day) - A common momentum indicator
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on indicators.
    Returns the dataframe with a 'Signal' column:
    1 = Long, -1 = Short, 0 = Neutral
    """
    df['Signal'] = 0
    
    # Example Strategy: Golden Cross
    # Long when SMA_50 > SMA_200
    df.loc[df['SMA_50'] > df['SMA_200'], 'Signal'] = 1
    
    # Filter: Only take long positions if RSI < 70 (not overbought)
    # df.loc[df['RSI'] > 70, 'Signal'] = 0
    
    return df


def run_vectorized_backtest(df: pd.DataFrame) -> pd.Series:
    """
    Simulate strategy performance using vectorized operations.
    Returns the cumulative strategy returns.
    """
    # Calculate daily returns of the asset
    df['Market_Returns'] = df['Close'].pct_change()
    
    # Shift signal by 1 day to avoid look-ahead bias
    # We trade at the Open/Close of the NEXT day based on TODAY's signal
    df['Strategy_Returns'] = df['Market_Returns'] * df['Signal'].shift(1)
    
    # Calculate cumulative returns
    df['Cumulative_Market_Returns'] = (1 + df['Market_Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    
    return df


def analyze_performance(df: pd.DataFrame):
    """
    Print performance metrics.
    """
    strategy_total_return = df['Cumulative_Strategy_Returns'].iloc[-1] - 1
    market_total_return = df['Cumulative_Market_Returns'].iloc[-1] - 1
    
    # Sharpe Ratio (assuming 252 trading days, risk-free rate = 0 for simplicity)
    strategy_sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252)
    
    print("-" * 40)
    print(f"Performance Summary:")
    print("-" * 40)
    print(f"Total Strategy Return: {strategy_total_return:.2%}")
    print(f"Total Market Return:   {market_total_return:.2%}")
    print(f"Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
    print("-" * 40)


def plot_results(df: pd.DataFrame, symbol: str):
    """
    Visualize the strategy performance vs the benchmark.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative_Market_Returns'], label=f'{symbol} (Buy & Hold)', alpha=0.6)
    plt.plot(df.index, df['Cumulative_Strategy_Returns'], label='Strategy', linewidth=2)
    
    plt.title(f'Strategy Research: {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot instead of showing it (better for headless environments)
    output_path = f"research_results_{symbol}.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def main():
    # 1. Define Universe & Parameters
    symbol = "SPY"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    print(f"Starting research for {symbol} from {start_date} to {end_date}...")
    
    # 2. Load Data
    # Point to the correct data directory where SPY.csv lives
    data_dir = Path(__file__).parent.parent / "data"
    data_api = MarketDataAPI(data_dir=data_dir)
    
    try:
        df = data_api.get_history(symbol, start=start_date, end=end_date)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback for the template if download fails and no data exists
        print("Creating dummy data for demonstration...")
        dates = pd.date_range(start=start_date, end=end_date)
        df = pd.DataFrame(index=dates)
        df['Close'] = 100 + np.random.randn(len(dates)).cumsum()
        df['Open'] = df['Close']
        df['High'] = df['Close']
        df['Low'] = df['Close']
        df['Volume'] = 1000000

    print(f"Loaded {len(df)} rows of data.")
    
    # 3. Calculate Indicators
    df = calculate_indicators(df)
    
    # 4. Generate Signals
    df = generate_signals(df)
    
    # 5. Run Backtest
    df = run_vectorized_backtest(df)
    
    # 6. Analyze & Visualize
    analyze_performance(df)
    plot_results(df, symbol)


if __name__ == "__main__":
    main()
