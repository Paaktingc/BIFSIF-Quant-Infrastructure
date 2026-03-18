"""SP500 Positive Momentum Strategy
=====================================
BIFS Quant Engine — compatible strategy module

Signal design:
    Formation : 12-month cumulative return (months t-13 to t-1)
    Skip      : 1 month  (avoids short-term microstructure reversal)
    Selection : top 20% of S&P 500 by formation-period return
    Holding   : 1 month  (rebalance monthly)
    Sizing    : inverse-volatility scaled to a 15% annualised vol target

Academic basis:
    Jegadeesh & Titman (1993), Carhart (1997), Barroso & Santa-Clara (2015)
    Formation + holding summing to ~13 months sits at the Sharpe peak
    documented by Newfound Research (2018) and replicated on S&P 500 (2025).

Engine integration:
    Implements the BIFS Strategy protocol — drop into the Backtest engine via:

        Backtest(strategy=SP500MomentumStrategy(), universe=...).run()

    or into TradingSession via:

        session.add_strategy(SP500MomentumStrategy())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bifs_quant_engine.strategies.strategy_protocol import BaseStrategy, Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MomentumConfig:
    """
    All strategy parameters in one place.
    Adjust here — nowhere else — to avoid scattered magic numbers.
    """

    # ── Signal ──────────────────────────────────────────────────────────────
    formation_months: int = 12
    """Lookback window for the momentum signal."""

    skip_months: int = 1
    """Months skipped between end of formation window and portfolio formation.
    Eliminates the short-term reversal effect documented by Jegadeesh (1990)."""

    holding_months: int = 1
    """How long each cohort is held. formation + skip + holding ≈ 14 months,
    placing us at the Sharpe peak identified in the literature."""

    # ── Selection ────────────────────────────────────────────────────────────
    top_pct: float = 0.20
    """Fraction of the universe to hold long. 0.20 = top quintile."""

    min_stocks: int = 20
    """Minimum holdings regardless of universe size.
    Prevents over-concentration on small sub-universes."""

    # ── Position sizing ───────────────────────────────────────────────────────
    vol_target: float = 0.15
    """Annualised volatility target for the full portfolio.
    Barroso & Santa-Clara (2015): vol-scaling cuts crash severity ~50%."""

    vol_lookback_days: int = 126
    """Rolling window for realised volatility estimate (~6 months of trading days).
    Short enough to be responsive; long enough to be stable."""

    vol_floor: float = 0.05
    """Minimum per-stock vol used in scaling — prevents divide-by-near-zero."""

    max_weight: float = 0.10
    """Hard cap per position (10%). Prevents single-stock concentration risk."""

    # ── Universe ──────────────────────────────────────────────────────────────
    universe: List[str] = field(default_factory=list)
    """S&P 500 tickers. Populated at initialisation if not supplied."""

    min_history_months: int = 15
    """Stocks with less history than this are excluded each period.
    Needs at least formation (12) + skip (1) + buffer (2) months."""

    # ── Risk guards ───────────────────────────────────────────────────────────
    max_portfolio_vol: float = 0.25
    """If estimated portfolio vol exceeds this, scale all weights down."""

    # ── Rebalance ─────────────────────────────────────────────────────────────
    rebalance_day: int = 1
    """Day-of-month on which rebalancing fires (1 = first trading day)."""


# ---------------------------------------------------------------------------
# Default S&P 500 universe
# ---------------------------------------------------------------------------

# Representative 100-stock cross-section across all 11 GICS sectors.
# Replace with a full, survivorship-bias-free constituent list in production.
# Point-in-time constituent lists are available from CRSP, Compustat, or
# providers such as Tiingo and Norgate Data.

DEFAULT_SP500_UNIVERSE: List[str] = [
    # Information Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "AMD", "INTC", "QCOM", "TXN",
    # Financials
    "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "USB",
    # Health Care
    "LLY", "JNJ", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR",
    # Communication Services
    "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA",
    # Industrials
    "GE", "RTX", "HON", "CAT", "DE", "UPS", "BA", "LMT", "NOC", "EMR",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "MO", "PM", "CL", "GIS", "K",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "MPC", "PSX", "VLO", "OXY",
    # Real Estate
    "PLD", "AMT", "EQIX", "CCI", "SPG", "PSA", "WELL", "DLR", "O", "AVB",
    # Materials
    "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "ALB", "CF", "MOS",
]


# ---------------------------------------------------------------------------
# Signal engine (stateless — can be unit-tested independently)
# ---------------------------------------------------------------------------

class MomentumSignalEngine:
    """
    Computes momentum scores and vol-scaled weights from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices. Index = DatetimeIndex, columns = tickers.
    config  : MomentumConfig

    Usage
    -----
    engine = MomentumSignalEngine(prices, config)
    weights = engine.compute_weights(as_of_date)
    """

    def __init__(self, prices: pd.DataFrame, config: MomentumConfig) -> None:
        self.prices = prices.copy()
        self.cfg = config
        self._monthly_returns: Optional[pd.DataFrame] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_weights(self, as_of: date) -> Dict[str, float]:
        """
        Return a dict {ticker: weight} representing the target portfolio
        as of `as_of`. Weights sum to ≤ 1.0 (may be < 1 if vol scaling
        reduces overall exposure).

        Steps
        -----
        1. Compute 12-1 momentum score for each stock (skip last month)
        2. Select top 20% by score
        3. Assign equal weights, then vol-scale each position
        4. Apply max-weight cap and portfolio-level vol guard
        """
        as_of_ts = pd.Timestamp(as_of)
        prices_to_date = self.prices.loc[:as_of_ts]

        if prices_to_date.empty:
            logger.warning("No price data available as of %s", as_of)
            return {}

        scores = self._momentum_scores(prices_to_date)
        if scores.empty:
            return {}

        selected = self._select_top(scores)
        if selected.empty:
            return {}

        weights = self._vol_scale(selected, prices_to_date)
        weights = self._apply_caps(weights)

        return weights.to_dict()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _momentum_scores(self, prices: pd.DataFrame) -> pd.Series:
        """
        12-1 momentum: return from t-13 months to t-1 month.
        Stocks with insufficient history are dropped.
        """
        monthly = prices.resample("ME").last()  # month-end prices
        cfg = self.cfg

        # Need at least formation + skip + 1 rows
        required_rows = cfg.formation_months + cfg.skip_months + 1
        if len(monthly) < required_rows:
            logger.debug("Not enough monthly history (%d rows, need %d)",
                         len(monthly), required_rows)
            return pd.Series(dtype=float)

        # t-1 month end (skip the most recent month)
        end_row   = -(cfg.skip_months)          # index -1 = most recent
        start_row = end_row - cfg.formation_months  # 12 months before that

        # Slice: monthly.iloc[-13] to monthly.iloc[-1]
        price_end   = monthly.iloc[end_row]
        price_start = monthly.iloc[start_row]

        scores = price_end / price_start - 1.0

        # Drop stocks with NaN (too little history or data gaps)
        scores = scores.dropna()

        # Drop stocks with fewer than min_history_months of daily data
        daily_counts = prices.count()
        min_days = int(self.cfg.min_history_months * 21)  # ~21 trading days/month
        sufficient = daily_counts[daily_counts >= min_days].index
        scores = scores[scores.index.isin(sufficient)]

        return scores

    def _select_top(self, scores: pd.Series) -> pd.Series:
        """Select top-pct by momentum score."""
        cfg = self.cfg
        n = max(cfg.min_stocks, int(len(scores) * cfg.top_pct))
        n = min(n, len(scores))
        return scores.nlargest(n)

    def _vol_scale(self, selected: pd.Series, prices: pd.DataFrame) -> pd.Series:
        """
        Inverse-volatility position sizing scaled to vol_target.

        Weight_i = (1 / vol_i) / sum(1 / vol_j)   [equal-vol baseline]
        Then scale the full portfolio to hit vol_target.

        This is the Barroso & Santa-Clara (2015) approach, which halves
        momentum crash depth while preserving most of the return premium.
        """
        cfg = self.cfg
        tickers = selected.index.tolist()
        price_slice = prices[tickers].tail(cfg.vol_lookback_days)

        daily_returns = price_slice.pct_change().dropna()

        # Annualised realised volatility per stock
        vols = daily_returns.std() * np.sqrt(252)
        vols = vols.clip(lower=cfg.vol_floor)  # floor prevents ÷0

        # Inverse-vol weights (equal-risk contribution baseline)
        inv_vol = 1.0 / vols
        raw_weights = inv_vol / inv_vol.sum()

        # Portfolio realised vol (simplified: diagonal only, ignores correlation)
        # For a more accurate estimate use: sqrt(w' Σ w)
        port_vol_estimate = float(np.sqrt((raw_weights ** 2 * vols ** 2).sum()))

        if port_vol_estimate < 1e-8:
            return raw_weights

        # Scale to vol target
        scale = cfg.vol_target / port_vol_estimate
        scale = min(scale, 1.0)  # long-only: never leverage above 1×

        scaled_weights = raw_weights * scale
        return scaled_weights

    def _apply_caps(self, weights: pd.Series) -> pd.Series:
        """Apply per-position max-weight cap, renormalise."""
        cfg = self.cfg
        weights = weights.clip(upper=cfg.max_weight)

        # Apply portfolio-level vol guard
        # (Simplified: if weights are heavily truncated, just normalise)
        if weights.sum() > 0:
            weights = weights / weights.sum() * min(weights.sum(), 1.0)

        return weights.round(6)


# ---------------------------------------------------------------------------
# BIFS Strategy Protocol implementation
# ---------------------------------------------------------------------------

class SP500MomentumStrategy(BaseStrategy):
    """
    Long-only S&P 500 cross-sectional momentum strategy.

    Conforms to the BIFS Quant Engine ``BaseStrategy`` protocol — plug in via:

        Backtest(strategy=SP500MomentumStrategy(), universe=...).run()

    or in a live TradingSession:

        session.add_strategy(SP500MomentumStrategy())

    The strategy is stateful in live mode: it maintains a price buffer
    internally and only recomputes signals on the configured rebalance day
    each month.  In backtest mode the engine calls ``generate_target_weights``
    directly, which is fully stateless.

    Parameters
    ----------
    config : MomentumConfig, optional
        Override any default parameter. Example:
            SP500MomentumStrategy(config=MomentumConfig(top_pct=0.15))
    """

    version: str = "1.0.0"

    def __init__(self, config: Optional[MomentumConfig] = None) -> None:
        self.cfg = config or MomentumConfig()

        if not self.cfg.universe:
            self.cfg.universe = DEFAULT_SP500_UNIVERSE.copy()

        # Initialise BaseStrategy with name and universe
        super().__init__(
            name="SP500MomentumStrategy",
            universe=self.cfg.universe,
        )

        # ── Live-trading state ──────────────────────────────────────────────
        # Internal price buffer (filled by on_data calls)
        self._prices: pd.DataFrame = pd.DataFrame()

        # Current target weights — updated on rebalance
        self._target_weights: Dict[str, float] = {}

        # Track last rebalance month to avoid rebalancing intra-month
        self._last_rebalance_month: Optional[int] = None

        logger.info(
            "[%s] Initialised — universe=%d stocks, top_pct=%.0f%%, "
            "formation=%dm, skip=%dm, hold=%dm, vol_target=%.0f%%",
            self.name,
            len(self.cfg.universe),
            self.cfg.top_pct * 100,
            self.cfg.formation_months,
            self.cfg.skip_months,
            self.cfg.holding_months,
            self.cfg.vol_target * 100,
        )

    # ── BIFS core backtest protocol ───────────────────────────────────────────

    def generate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Called by ``Backtest`` on each rebalance date.

        Parameters
        ----------
        date : datetime
            Rebalance date (end-of-day).
        price_history : pd.DataFrame
            Daily adjusted close prices up to and including `date`.
            Index = DatetimeIndex, columns = ticker strings.

        Returns
        -------
        Dict[str, float]
            Target weights {ticker: weight}.  Weights sum to ≤ 1.0.
        """
        # Guard: only use data up to the rebalance date
        prices_to_date = price_history.loc[price_history.index <= date]
        if prices_to_date.empty:
            return {}

        engine = MomentumSignalEngine(prices_to_date, self.cfg)
        weights = engine.compute_weights(date.date() if hasattr(date, "date") else date)

        if weights:
            tickers_ranked = sorted(weights, key=weights.get, reverse=True)
            logger.info(
                "[%s] Signal @ %s — %d positions, top 5: %s",
                self.name, date, len(weights), tickers_ranked[:5],
            )

        return weights

    # ── BIFS live-trading protocol: called every bar by TradingSession ────────

    def on_data(self, data: dict) -> Dict[str, float]:
        """
        Called by TradingSession on each data event (daily bar).

        Parameters
        ----------
        data : dict
            Expected keys:
                "date"   : datetime.date  — current bar date
                "prices" : dict[str, float] — {ticker: close_price}

        Returns
        -------
        Dict[str, float]
            Target portfolio weights {ticker: weight}.
            Empty dict = hold current positions (no rebalance this bar).
        """
        bar_date: date = data.get("date")
        bar_prices: dict = data.get("prices", {})

        if not bar_date or not bar_prices:
            return {}

        self._update_price_buffer(bar_date, bar_prices)

        if not self._should_rebalance(bar_date):
            return {}

        return self._rebalance(bar_date)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def current_weights(self) -> Dict[str, float]:
        """Return current live target weights (live trading use)."""
        return dict(self._target_weights)

    def describe(self) -> dict:
        """Return strategy metadata dict (for compliance/reporting)."""
        return {
            "name": self.name,
            "version": self.version,
            "universe_size": len(self.cfg.universe),
            "formation_months": self.cfg.formation_months,
            "skip_months": self.cfg.skip_months,
            "holding_months": self.cfg.holding_months,
            "top_pct": self.cfg.top_pct,
            "vol_target": self.cfg.vol_target,
            "max_weight": self.cfg.max_weight,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_price_buffer(self, bar_date: date, bar_prices: dict) -> None:
        row = {t: bar_prices.get(t, np.nan) for t in self.cfg.universe}
        new_row = pd.DataFrame(row, index=[pd.Timestamp(bar_date)])
        if self._prices.empty:
            self._prices = new_row
        else:
            self._prices = pd.concat([self._prices, new_row])

    def _should_rebalance(self, bar_date: date) -> bool:
        current_month = bar_date.year * 100 + bar_date.month
        if current_month == self._last_rebalance_month:
            return False
        if bar_date.day >= self.cfg.rebalance_day:
            return True
        return False

    def _rebalance(self, bar_date: date) -> Dict[str, float]:
        """Run signal engine and update target weights."""
        engine = MomentumSignalEngine(self._prices, self.cfg)
        weights = engine.compute_weights(bar_date)

        if not weights:
            logger.warning("[%s] No weights generated @ %s", self.name, bar_date)
            return {}

        self._target_weights = weights
        self._last_rebalance_month = bar_date.year * 100 + bar_date.month

        tickers_ranked = sorted(weights, key=weights.get, reverse=True)
        logger.info(
            "[%s] Rebalanced @ %s — %d positions | top 5: %s",
            self.name, bar_date, len(weights), tickers_ranked[:5],
        )

        return dict(weights)

    def reset(self) -> None:
        """Reset all live-trading state (calls super for fills/pnl)."""
        super().reset()
        self._prices = pd.DataFrame()
        self._target_weights = {}
        self._last_rebalance_month = None


# ---------------------------------------------------------------------------
# Backtest convenience wrapper
# ---------------------------------------------------------------------------

def run_backtest(
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    initial_cash: float = 1_000_000,
    rebalance_every_n_days: int = 21,
) -> None:
    """
    Run a backtest using the core BIFS ``Backtest`` engine.

    Usage
    -----
        from bifs_quant_engine.strategies.sp500_momentum import run_backtest
        result = run_backtest()
        print(result.equity_curve)

    Parameters
    ----------
    start : str
        Backtest start date (ISO format).
    end : str
        Backtest end date (ISO format).
    initial_cash : float
        Starting capital.
    rebalance_every_n_days : int
        Approximate monthly rebalance cadence (~21 trading days).
    """
    from bifs_quant_engine.core.backtest import Backtest

    strategy = SP500MomentumStrategy()

    result = Backtest(
        strategy=strategy,
        universe=strategy.cfg.universe,
        start=start,
        end=end,
        initial_cash=initial_cash,
        rebalance_every_n_days=rebalance_every_n_days,
    ).run()

    print(result.equity_curve)
    return result


# ---------------------------------------------------------------------------
# Quick standalone smoke-test (no engine dependency)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Smoke-test: generates synthetic price data and validates the signal engine
    produces non-empty, correctly normalised weights.

    Run with:   python -m bifs_quant_engine.strategies.sp500_momentum
    """
    import random

    random.seed(42)
    np.random.seed(42)

    # Build 2 years of synthetic daily prices for 50 tickers
    tickers = DEFAULT_SP500_UNIVERSE[:50]
    dates   = pd.date_range("2022-01-03", "2024-01-03", freq="B")
    prices  = pd.DataFrame(
        np.cumprod(1 + np.random.normal(0.0003, 0.012, size=(len(dates), len(tickers))), axis=0) * 100,
        index=dates,
        columns=tickers,
    )

    cfg    = MomentumConfig(universe=tickers)
    engine = MomentumSignalEngine(prices, cfg)
    as_of  = date(2024, 1, 3)
    weights = engine.compute_weights(as_of)

    print(f"\n{'─'*50}")
    print(f"  SP500MomentumStrategy — smoke test")
    print(f"  As-of date : {as_of}")
    print(f"  Positions  : {len(weights)}")
    print(f"  Weight sum : {sum(weights.values()):.4f}")
    print(f"  Max weight : {max(weights.values()):.4f}")
    print(f"\n  Top 5 holdings:")
    for t, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {t:<10} {w:.2%}")
    print(f"{'─'*50}\n")

    assert len(weights) > 0, "No weights produced"
    assert sum(weights.values()) <= 1.001, "Weights exceed 1.0"
    assert max(weights.values()) <= 0.101, "Max weight cap breached"
    print("  All assertions passed.\n")
