"""Backtest engine: run a strategy over historical data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd

from .data_api import MarketDataAPI
from .portfolio import Portfolio
from .broker_sim import SimulatedBroker, Order
from bifs_quant_engine.strategies.base import Strategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Dict]


class Backtest:
    """
    Simple backtest:
    - Long and short positions supported
    - Uses daily close prices
    - Rebalances every N trading days
    """

    def __init__(
        self,
        strategy: Strategy,
        universe: List[str],
        start: str = "2015-01-01",
        end: Optional[str] = None,
        initial_cash: float = 100_000.0,
        rebalance_every_n_days: int = 20,
    ):
        self.strategy = strategy
        self.universe = universe
        self.start = start
        self.end = end
        self.initial_cash = initial_cash
        self.rebalance_every_n_days = rebalance_every_n_days

        self.data_api = MarketDataAPI()

    def run(self) -> BacktestResult:
        # 1) Load data
        prices = self.data_api.get_prices_for_universe(
            symbols=self.universe,
            start=self.start,
            end=self.end,
        )

        # 2) Set up portfolio and broker
        portfolio = Portfolio(cash=self.initial_cash)
        broker = SimulatedBroker(portfolio)

        equity_values: List[float] = []
        dates = prices.index.tolist()

        for i, dt in enumerate(dates):
            today = dt.to_pydatetime()
            today_prices: Dict[str, float] = prices.loc[dt].to_dict()

            # 3) Rebalance logic
            if i % self.rebalance_every_n_days == 0:
                hist_until_today = prices.loc[:dt]

                target_weights = self.strategy.generate_target_weights(
                    today,
                    hist_until_today,
                )

                order_dicts = portfolio.weights_to_orders(
                    target_weights,
                    today_prices,
                )

                # Convert to Order objects
                orders: List[Order] = [
                    Order(
                        symbol=o["symbol"],
                        qty=int(o["qty"]),
                        side=o["side"],
                    )
                    for o in order_dicts
                ]

                if orders:
                    broker.execute_orders(orders, today_prices)

            # 4) Record equity
            equity_values.append(portfolio.value(today_prices))

        # 5) Return results
        equity_curve = pd.Series(equity_values, index=prices.index, name="equity")

        return BacktestResult(
            equity_curve=equity_curve,
            trades=broker.get_trade_log(),
        )