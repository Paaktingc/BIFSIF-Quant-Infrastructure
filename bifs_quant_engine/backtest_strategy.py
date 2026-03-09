from __future__ import annotations

import matplotlib.pyplot as plt

from bifs_quant_engine.core.backtest import Backtest
from bifs_quant_engine.strategies.etf_momentum import ETFMomentumStrategy


def main():
    universe = ["SPY", "QQQ", "EEM", "TLT", "GLD"]

    strat = ETFMomentumStrategy(
        universe=universe,
        lookback_days=126,
        top_n=3,
    )

    bt = Backtest(
        strategy=strat,
        universe=universe,
        start="2015-01-01",
        end=None,
        initial_cash=100_000.0,
        rebalance_every_n_days=20,
    )

    result = bt.run()
    eq = result.equity_curve

    # basic stats
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    daily = eq.pct_change().dropna()
    ann_ret = daily.mean() * 252
    ann_vol = daily.std() * (252 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    print(f"Total return: {total_ret:.2%}")
    print(f"Annualised return: {ann_ret:.2%}")
    print(f"Annualised vol: {ann_vol:.2%}")
    print(f"Sharpe (no rf): {sharpe:.2f}")
    print(f"Number of trades: {len(result.trades)}")

    # plot
    plt.plot(eq.index, eq.values)
    plt.title("ETF Momentum Backtest – Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.savefig('backtest_result.png')
    print("Plot saved to backtest_result.png")


if __name__ == "__main__":
    main()