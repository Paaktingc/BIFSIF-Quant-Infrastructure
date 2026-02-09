"""Simulated broker: executes orders against Portfolio at given prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .portfolio import Portfolio


@dataclass
class Order:
    symbol: str
    qty: int
    side: str  # 'buy' or 'sell'


class SimulatedBroker:
    """
    Simple in-memory broker:
    - executes all orders immediately at given price
    - no slippage
    - no commission
    """

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.trade_log: List[Dict] = []

    def execute_orders(self, orders: List[Order], prices: Dict[str, float]) -> None:
        """
        Execute a batch of orders based on current prices.
        """
        for o in orders:
            symbol = o.symbol
            qty = int(o.qty)
            side = o.side.lower()

            if symbol not in prices:
                print(f"[WARN] Price for {symbol} not available. Skipping order.")
                continue

            price = prices[symbol]

            # Apply order to portfolio
            self.portfolio.apply_fill(symbol, qty, side, price)

            # Log trade
            self.trade_log.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "side": side,
                    "price": price,
                }
            )

    def get_trade_log(self) -> List[Dict]:
        return self.trade_log