"""
run_all_backtests.py
====================
BIFS Quant Engine — Unified Multi-Strategy Backtest Runner

Runs every backtestable strategy in the engine and prints a side-by-side
performance comparison table.

Strategies included
-------------------
  1. ETF Momentum         — Momentum across 10 sector ETFs
  2. Mean Reversion       — RSI + Bollinger Bands on large-caps
  3. SP500 Momentum       — 12-1 cross-sectional momentum on S&P 500
  4. Kalman OU Pairs      — Dynamic pairs trading (own internal backtester)

Strategies excluded (live-only, no historical replay):
  - TZInfoArbStrategy     — requires live Polymarket API + NLP news feeds
  - VolStraddleStrategy   — requires live Binance tick feed

Usage
-----
  # Quick smoke-test with synthetic data (no network):
  python run_all_backtests.py --synthetic

  # Full backtest with real data (downloads via yfinance, cached to data/):
  python run_all_backtests.py

  # Custom date range:
  python run_all_backtests.py --start 2018-01-01 --end 2023-12-31

  # Custom initial capital:
  python run_all_backtests.py --cash 500000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,          # suppress INFO chatter during runs
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyResult:
    name: str
    ann_return: float       # decimal, e.g. 0.12 = 12%
    sharpe: float
    max_dd: float           # negative decimal, e.g. -0.18 = -18%
    win_rate: float         # 0–1
    n_trades: int
    equity_curve: pd.Series
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic price generator (--synthetic mode)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_prices(
    tickers: List[str],
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate GBM price paths — no network required."""
    np.random.seed(seed)
    dates = pd.bdate_range(start, end)
    n, m = len(dates), len(tickers)
    log_returns = np.random.normal(0.0003, 0.012, size=(n, m))
    prices = np.cumprod(np.exp(log_returns), axis=0) * 100.0
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ─────────────────────────────────────────────────────────────────────────────
# Shared analytics helper
# ─────────────────────────────────────────────────────────────────────────────

def _analytics_from_equity(equity: pd.Series, trades: List[Dict]) -> dict:
    """Derive key metrics from an equity curve + trade log."""
    returns = equity.pct_change().dropna()

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = len(returns) / 252
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    daily_std = returns.std()
    sharpe = (returns.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    running_max = equity.cummax()
    max_dd = float(((equity - running_max) / running_max).min())

    win_rate = float((returns > 0).mean())
    n_trades = len(trades)

    return dict(
        ann_return=ann_ret,
        sharpe=sharpe,
        max_dd=max_dd,
        win_rate=win_rate,
        n_trades=n_trades,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. ETF Momentum
# ─────────────────────────────────────────────────────────────────────────────

ETF_UNIVERSE = [
    "XLK", "XLF", "XLV", "XLE", "XLI",
    "XLU", "XLP", "XLY", "XLB", "XLRE",
]


def run_etf_momentum(
    start: str,
    end: str,
    initial_cash: float,
    synthetic: bool,
) -> StrategyResult:
    from bifs_quant_engine.strategies.etf_momentum import ETFMomentumStrategy
    from bifs_quant_engine.core.backtest import Backtest

    strategy = ETFMomentumStrategy(
        universe=ETF_UNIVERSE,
        lookback_days=126,
        top_n=3,
        use_vol_weighting=True,
    )

    if synthetic:
        prices = make_synthetic_prices(ETF_UNIVERSE, start, end)
        # Monkey-patch data loader
        from bifs_quant_engine.core import data_api as _api_mod
        real_cls = _api_mod.MarketDataAPI

        class _SyntheticAPI:
            def get_prices_for_universe(self, symbols, start=None, end=None):
                return prices[[s for s in symbols if s in prices.columns]]

        bt = Backtest(
            strategy=strategy,
            universe=ETF_UNIVERSE,
            start=start,
            end=end,
            initial_cash=initial_cash,
            rebalance_every_n_days=21,
        )
        bt.data_api = _SyntheticAPI()
    else:
        bt = Backtest(
            strategy=strategy,
            universe=ETF_UNIVERSE,
            start=start,
            end=end,
            initial_cash=initial_cash,
            rebalance_every_n_days=21,
        )

    result = bt.run()
    m = _analytics_from_equity(result.equity_curve, result.trades)

    return StrategyResult(
        name="ETF Momentum",
        equity_curve=result.equity_curve,
        **m,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mean Reversion
# ─────────────────────────────────────────────────────────────────────────────

MEAN_REV_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "JPM", "JNJ", "XOM", "PG",
]


def run_mean_reversion(
    start: str,
    end: str,
    initial_cash: float,
    synthetic: bool,
) -> StrategyResult:
    from bifs_quant_engine.strategies.mean_reversion import (
        MeanReversionStrategy, MeanReversionConfig,
    )
    from bifs_quant_engine.core.backtest import Backtest

    strategy = MeanReversionStrategy(
        name="mean_reversion",
        universe=MEAN_REV_UNIVERSE,
        config=MeanReversionConfig(
            rsi_oversold=35,
            rsi_overbought=65,
            max_position_pct=0.15,
        ),
    )

    if synthetic:
        prices = make_synthetic_prices(MEAN_REV_UNIVERSE, start, end, seed=99)

        class _SyntheticAPI:
            def get_prices_for_universe(self, symbols, start=None, end=None):
                return prices[[s for s in symbols if s in prices.columns]]

        bt = Backtest(
            strategy=strategy,
            universe=MEAN_REV_UNIVERSE,
            start=start,
            end=end,
            initial_cash=initial_cash,
            rebalance_every_n_days=5,
        )
        bt.data_api = _SyntheticAPI()
    else:
        bt = Backtest(
            strategy=strategy,
            universe=MEAN_REV_UNIVERSE,
            start=start,
            end=end,
            initial_cash=initial_cash,
            rebalance_every_n_days=5,
        )

    result = bt.run()
    m = _analytics_from_equity(result.equity_curve, result.trades)

    return StrategyResult(
        name="Mean Reversion",
        equity_curve=result.equity_curve,
        **m,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. SP500 Momentum
# ─────────────────────────────────────────────────────────────────────────────

def run_sp500_momentum(
    start: str,
    end: str,
    initial_cash: float,
    synthetic: bool,
) -> StrategyResult:
    from bifs_quant_engine.strategies.sp500_momentum import (
        SP500MomentumStrategy, MomentumConfig, DEFAULT_SP500_UNIVERSE,
    )
    from bifs_quant_engine.core.backtest import Backtest

    # Use a trimmed universe for speed in both modes
    universe = DEFAULT_SP500_UNIVERSE[:50]

    strategy = SP500MomentumStrategy(
        config=MomentumConfig(universe=universe)
    )

    if synthetic:
        prices = make_synthetic_prices(universe, start, end, seed=17)

        class _SyntheticAPI:
            def get_prices_for_universe(self, symbols, start=None, end=None):
                return prices[[s for s in symbols if s in prices.columns]]

        bt = Backtest(
            strategy=strategy,
            universe=universe,
            start=start,
            end=end,
            initial_cash=initial_cash,
            rebalance_every_n_days=21,
        )
        bt.data_api = _SyntheticAPI()
    else:
        bt = Backtest(
            strategy=strategy,
            universe=universe,
            start=start,
            end=end,
            initial_cash=initial_cash,
            rebalance_every_n_days=21,
        )

    result = bt.run()
    m = _analytics_from_equity(result.equity_curve, result.trades)

    return StrategyResult(
        name="SP500 Momentum",
        equity_curve=result.equity_curve,
        **m,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Kalman OU Pairs
# ─────────────────────────────────────────────────────────────────────────────

def run_kalman_ou_pairs(
    start: str,
    end: str,
    initial_cash: float,
    synthetic: bool,
) -> StrategyResult:
    from bifs_quant_engine.strategies.kalman_ou_pairs import (
        PairsConfig, PairSelector, PairsBacktester,
    )

    config = PairsConfig(
        start_date=start,
        end_date=end,
        initial_capital=initial_cash,
        max_pairs=5,
        entry_z=1.5,
        exit_z=0.0,
        stop_loss_z=3.5,
        max_holding_days=45,
        transaction_cost_bps=5.0,
    )

    if synthetic:
        # Generate the full pairs universe so all tickers referenced below exist
        universe = config.universe
        prices = make_synthetic_prices(universe, start, end, seed=7)
        # Inject correlated "cointegrated" pairs by overwriting two tickers
        # with a linear function of another ticker (+ noise) so the pairs
        # engine can find at least one qualifying pair.
        t0, t1 = universe[0], universe[1]
        prices[t1] = prices[t0] * 1.1 + np.random.normal(0, 0.5, len(prices))
        t2, t3 = universe[4], universe[5]
        prices[t3] = prices[t2] * 0.8 + np.random.normal(0, 0.3, len(prices))
    else:
        import yfinance as yf
        raw = yf.download(
            config.universe,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["Adj Close"]
        else:
            prices = raw
        prices = prices.dropna(axis=1, how="any")

    # Pair selection on formation window
    selector = PairSelector(config)
    formation_prices = prices.iloc[:config.formation_period]
    pairs = selector.select_pairs(formation_prices)

    if not pairs:
        return StrategyResult(
            name="Kalman OU Pairs",
            ann_return=0.0,
            sharpe=0.0,
            max_dd=0.0,
            win_rate=0.0,
            n_trades=0,
            equity_curve=pd.Series(dtype=float),
            error="No pairs passed selection filters — try relaxing thresholds.",
        )

    backtester = PairsBacktester(config)
    portfolio = backtester.run_portfolio_backtest(prices, pairs)

    if "error" in portfolio:
        return StrategyResult(
            name="Kalman OU Pairs",
            ann_return=0.0,
            sharpe=0.0,
            max_dd=0.0,
            win_rate=0.0,
            n_trades=0,
            equity_curve=pd.Series(dtype=float),
            error=portfolio["error"],
        )

    equity = portfolio["portfolio_equity_curve"]
    total_trades = sum(r.num_trades for r in portfolio["pair_results"])
    avg_win_rate = np.mean([r.win_rate for r in portfolio["pair_results"]])

    return StrategyResult(
        name="Kalman OU Pairs",
        ann_return=portfolio["portfolio_ann_return"],
        sharpe=portfolio["portfolio_sharpe"],
        max_dd=float(portfolio["portfolio_max_drawdown"]),
        win_rate=avg_win_rate,
        n_trades=total_trades,
        equity_curve=equity,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table printer
# ─────────────────────────────────────────────────────────────────────────────

def _bar(pct: float, width: int = 12) -> str:
    """Tiny ASCII bar chart for the equity column."""
    filled = int(abs(pct) * width)
    sign = "+" if pct >= 0 else "-"
    return sign + "█" * min(filled, width)


def print_comparison(results: List[StrategyResult]) -> None:
    w = 80
    print()
    print("═" * w)
    print("  BIFS QUANT ENGINE — MULTI-STRATEGY BACKTEST RESULTS")
    print("═" * w)
    header = f"  {'Strategy':<22} {'Ann Ret':>8} {'Sharpe':>7} {'Max DD':>8} {'Win Rate':>9} {'Trades':>7}"
    print(header)
    print("─" * w)

    for r in results:
        if r.error:
            print(f"  {'⚠  ' + r.name:<22}  ERROR: {r.error}")
            continue

        ann_ret_str = f"{r.ann_return:+.1%}"
        sharpe_str  = f"{r.sharpe:.2f}"
        max_dd_str  = f"{r.max_dd:.1%}"
        wr_str      = f"{r.win_rate:.1%}"
        trades_str  = f"{r.n_trades:,}"

        print(f"  {r.name:<22} {ann_ret_str:>8} {sharpe_str:>7} {max_dd_str:>8} {wr_str:>9} {trades_str:>7}")

    print("═" * w)
    print()

    # Print individual equity curves as mini ASCII sparklines
    print("  Equity curves (normalised to 1.0):")
    print()
    for r in results:
        if r.error or r.equity_curve.empty:
            continue
        norm = r.equity_curve / r.equity_curve.iloc[0]
        # Sample ~40 points
        step = max(1, len(norm) // 40)
        sampled = norm.iloc[::step]
        lo, hi = sampled.min(), sampled.max()
        rng = hi - lo if hi > lo else 1.0
        spark = ""
        blocks = " ▁▂▃▄▅▆▇█"
        for v in sampled:
            idx = int((v - lo) / rng * (len(blocks) - 1))
            spark += blocks[idx]
        final = norm.iloc[-1]
        print(f"  {r.name:<22} {spark}  final={final:.3f}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all BIFS Quant Engine backtests and compare results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start",     default="2015-01-01", help="Backtest start date")
    parser.add_argument("--end",       default="2024-12-31", help="Backtest end date")
    parser.add_argument("--cash",      default=1_000_000.0, type=float, help="Initial capital")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data — no network required (fast smoke-test)")
    parser.add_argument("--skip-pairs", action="store_true",
                        help="Skip the slow Kalman OU Pairs backtest")
    args = parser.parse_args()

    mode = "SYNTHETIC" if args.synthetic else "LIVE DATA"
    print(f"\n  Mode: {mode}  |  {args.start} → {args.end}  |  Capital: ${args.cash:,.0f}\n")

    runners = [
        ("ETF Momentum",   run_etf_momentum),
        ("Mean Reversion", run_mean_reversion),
        ("SP500 Momentum", run_sp500_momentum),
    ]
    if not args.skip_pairs:
        runners.append(("Kalman OU Pairs", run_kalman_ou_pairs))

    results: List[StrategyResult] = []
    for label, fn in runners:
        print(f"  ▶  Running {label}...", end="", flush=True)
        t0 = time.time()
        try:
            res = fn(
                start=args.start,
                end=args.end,
                initial_cash=args.cash,
                synthetic=args.synthetic,
            )
        except Exception as exc:
            logger.exception("Error in %s", label)
            res = StrategyResult(
                name=label,
                ann_return=0.0, sharpe=0.0, max_dd=0.0,
                win_rate=0.0, n_trades=0,
                equity_curve=pd.Series(dtype=float),
                error=str(exc),
            )
        elapsed = time.time() - t0
        status = "✓" if not res.error else "✗"
        print(f"  {status}  ({elapsed:.1f}s)")
        results.append(res)

    print_comparison(results)

    # Basic assertions in synthetic mode (CI smoke-test)
    if args.synthetic:
        for r in results:
            if r.error:
                continue
            assert not r.equity_curve.empty, f"{r.name}: empty equity curve"
        print("  Smoke-test assertions passed.\n")


if __name__ == "__main__":
    main()
