"""
Verification Script for Market Neutral Strategies
=================================================

Tests the PairsTradingStrategy and StatisticalArbitrageStrategy using
synthetic cointegrated data to ensure logic correctness.
"""

import sys
from pathlib import Path

# Ensure the package is visible
sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from bifs_quant_engine.strategies.market_neutral.pairs_trading import PairsTradingStrategy, PairsTradingConfig
from bifs_quant_engine.strategies.market_neutral.stat_arb import StatisticalArbitrageStrategy, StatArbConfig

def generate_correlated_data(n_days=500):
    """Generate two cointegrated price series."""
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=x) for x in range(n_days)]
    dates = sorted(dates)
    
    # Common factor (random walk)
    common = np.cumsum(np.random.normal(0, 1, n_days))
    
    # Noise
    noise1 = np.random.normal(0, 0.5, n_days)
    noise2 = np.random.normal(0, 0.5, n_days)
    
    # Series 1 & 2
    s1 = 100 + common + noise1
    s2 = 100 + common + noise2 # 1:1 relationship with s1
    
    # Make them diverge temporarily to trigger signals
    # Divergence at index 300-350
    s2[300:350] += 5.0 
    
    df = pd.DataFrame(index=dates)
    df['A'] = s1
    df['B'] = s2
    return df

def test_pairs_trading():
    print("\nTesting PairsTradingStrategy...")
    
    df = generate_correlated_data()
    
    config = PairsTradingConfig(
        entry_zscore=1.5,
        exit_zscore=0.0,
        lookback=20,
        hedge_ratio_lookback=60
    )
    
    strategy = PairsTradingStrategy(
        name="test_pair",
        pairs=[("A", "B")],
        config=config
    )
    
    # Run through data
    # We need a warm-up period
    warmup = 100
    
    signals_generated = 0
    target_weights_history = []
    
    for i in range(warmup, len(df)):
        current_date = df.index[i]
        history = df.iloc[:i+1]
        
        weights = strategy.generate_target_weights(current_date, history)
        if weights:
            signals_generated += 1
            target_weights_history.append(weights)
            # print(f"Date: {current_date.date()}, Weights: {weights}")
            
    print(f"Signals generated: {signals_generated}")
    if signals_generated > 0:
        print("SUCCESS: Pairs trading strategy generated signals.")
        return True
    else:
        print("FAILURE: Pairs trading strategy failed to generate signals.")
        return False

def test_stat_arb():
    print("\nTesting StatisticalArbitrageStrategy...")
    
    # Generate 5 assets with some common factors
    np.random.seed(100)
    n_days = 500
    dates = [datetime.now() - timedelta(days=x) for x in range(n_days)]
    dates = sorted(dates)
    
    # Factors (Random Walks)
    f1 = np.cumsum(np.random.normal(0, 1, n_days))
    f2 = np.cumsum(np.random.normal(0, 1, n_days))
    
    # Generate AR(1) residuals which are mean-reverting
    # e_t = phi * e_{t-1} + noise
    # phi = 0.8 implies half-life of ~3 days
    def generate_ar1(n, phi=0.8, sigma=0.5):
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = phi * e[t-1] + np.random.normal(0, sigma)
        return e
    
    e1 = generate_ar1(n_days)
    e2 = generate_ar1(n_days)
    e3 = generate_ar1(n_days)
    
    df = pd.DataFrame(index=dates)
    
    # Create assets from factors + mean-reverting residuals
    df['S1'] = 100 + 1.0 * f1 + 0.5 * f2 + e1
    df['S2'] = 100 + 1.2 * f1 + 0.3 * f2 + e2
    df['S3'] = 100 + 0.8 * f1 + 0.8 * f2 + e3
    
    # Create a divergence
    # df.loc[df.index[-50:], 'S1'] += 3.0 # Shock S1 - Use .loc to avoid ChainedAssignment
    
    # Correction: To force a mean reversion opportunity, we simply deviate S1 from its factor model
    # The original code did += 3.0 which is a permanent step change, not necessarily mean reverting quickly.
    # Let's add a temporary bump that reverts
    
    shock = np.linspace(0, 3.0, 10)
    shock = np.concatenate([shock, np.linspace(3.0, 0, 10)])
    # Add shock to middle of the test period to ensure we see it
    start_idx = len(df) - 30
    end_idx = start_idx + len(shock)
    
    # Use .loc with proper indexer to avoid read-only array issues
    df.loc[df.index[start_idx:end_idx], 'S1'] += shock
    
    config = StatArbConfig(
        lookback=60,
        num_factors=2,
        entry_zscore=1.5,
        rebalance_threshold=0.0,
        min_half_life=0.1,  # Relax constraint for synthetic test
        max_half_life=100.0 # Relax constraint
    )
    
    strategy = StatisticalArbitrageStrategy(
        name="test_stat_arb",
        universe=['S1', 'S2', 'S3'],
        config=config
    )
    
    signals_count = 0
    
    # Test last 20 days
    print("  Debug Loop:")
    for i in range(len(df)-20, len(df)):
        current_date = df.index[i]
        history = df.iloc[:i+1]
        
        weights = strategy.generate_target_weights(current_date, history)
        if weights:
            signals_count += 1
            print(f"    Date: {current_date.date()}, Weights: {weights}")
        # else:
            # Uncomment to debug failures
            # print(f"    Date: {current_date.date()} - No signal")
            
    print(f"Signals generated: {signals_count}")
    if signals_count > 0:
        print("SUCCESS: Stat Arb strategy generated signals.")
        return True
    else:
        print("FAILURE: Stat Arb strategy failed to generate signals.")
        return False

def main():
    pairs_ok = test_pairs_trading()
    stat_ok = test_stat_arb()
    
    if pairs_ok and stat_ok:
        print("\nAll Market Neutral verification tests PASSED.")
    else:
        print("\nSome verification tests FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
