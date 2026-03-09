"""Portfolio: tracks positions, cash, weights, and converts target weights into orders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import math


@dataclass
class Position:
    symbol: str
    quantity: float


@dataclass
class Portfolio:
    """
    Tracks positions and cash.
    Converts strategy target weights into executable orders.

    Supports:
    - Long and short positions
    - Negative target weights generate sell_short orders
    - Market orders executed at given prices
    """

    cash: float = 100_000.0
    positions: Dict[str, Position] = field(default_factory=dict)

    # -----------------------------------------------------
    # Portfolio valuation
    # -----------------------------------------------------

    def value(self, prices: Dict[str, float]) -> float:
        """
        Total portfolio value: cash + market value of all positions.
        Short positions contribute negative value (quantity is negative).
        """
        pos_val = sum(
            self.positions[s].quantity * prices[s]
            for s in self.positions
            if s in prices
        )
        return self.cash + pos_val

    def current_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Compute current weight of each symbol in the portfolio.
        """
        total = self.value(prices)
        if total <= 0:
            return {s: 0.0 for s in prices}

        weights: Dict[str, float] = {}
        for s in prices:
            if s in self.positions:
                weights[s] = (
                    self.positions[s].quantity * prices[s]
                ) / total
            else:
                weights[s] = 0.0

        return weights

    # -----------------------------------------------------
    # Trading logic: target weights -> orders
    # -----------------------------------------------------

    def weights_to_orders(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        min_notional: float = 0.0,
    ) -> List[Dict]:
        """
        Convert target weights into buy/sell/sell_short orders.

        Supports negative target weights for short positions.

        Returns:
            List of orders:
            [
                {"symbol": "SPY", "qty": 10, "side": "buy"},
                {"symbol": "QQQ", "qty": 5, "side": "sell"},
                {"symbol": "EUR", "qty": 20, "side": "sell_short"},
            ]
        """
        total_value = self.value(prices)
        current_w = self.current_weights(prices)

        buy_requests: List[Dict] = []
        sell_requests: List[Dict] = []

        for symbol, target_w in target_weights.items():
            curr_w = current_w.get(symbol, 0.0)
            diff_w = target_w - curr_w

            if abs(diff_w) < 1e-4:
                continue

            dollar_change = diff_w * total_value

            price = prices.get(symbol)
            if price is None or price <= 0:
                continue

            if dollar_change > 0:
                # Buy (or cover short): increase position
                qty = int(math.floor(dollar_change / price))
                if qty <= 0:
                    continue
                notional = qty * price
                if notional < min_notional:
                    continue
                buy_requests.append({"symbol": symbol, "qty": qty, "price": price, "notional": notional})
            else:
                # Sell (or sell short): decrease position
                qty = int(math.floor(abs(dollar_change) / price))
                if qty <= 0:
                    continue
                notional = qty * price
                if notional < min_notional:
                    continue

                held = self.positions[symbol].quantity if symbol in self.positions else 0.0

                if held > 0 and qty <= int(held):
                    # Selling part or all of a long position
                    sell_requests.append({"symbol": symbol, "qty": qty, "price": price,
                                          "notional": notional, "side": "sell"})
                elif held > 0 and qty > int(held):
                    # Sell entire long, then short the remainder
                    sell_requests.append({"symbol": symbol, "qty": int(held), "price": price,
                                          "notional": int(held) * price, "side": "sell"})
                    short_qty = qty - int(held)
                    if short_qty > 0:
                        sell_requests.append({"symbol": symbol, "qty": short_qty, "price": price,
                                              "notional": short_qty * price, "side": "sell_short"})
                else:
                    # No long position (or already short) — open/extend short
                    sell_requests.append({"symbol": symbol, "qty": qty, "price": price,
                                          "notional": notional, "side": "sell_short"})

        orders: List[Dict] = []

        # Process sells/shorts first (they free up cash)
        total_proceeds = 0.0
        for s in sell_requests:
            orders.append({"symbol": s["symbol"], "qty": s["qty"], "side": s["side"]})
            total_proceeds += s["notional"]

        # Compute available cash for buys
        available_cash = self.cash + total_proceeds

        total_buy_notional = sum(b["notional"] for b in buy_requests)

        if total_buy_notional <= 0:
            return orders

        scale = 1.0 if total_buy_notional <= available_cash else (available_cash / total_buy_notional)

        for b in buy_requests:
            allowed_notional = b["notional"] * scale
            qty = int(math.floor(allowed_notional / b["price"]))
            if qty <= 0:
                continue
            notional = qty * b["price"]
            if notional < min_notional:
                continue
            orders.append({"symbol": b["symbol"], "qty": qty, "side": "buy"})

        return orders

    # -----------------------------------------------------
    # Execution used by SimBroker
    # -----------------------------------------------------

    def apply_fill(self, symbol: str, qty: int, side: str, price: float):
        """
        Apply one executed trade to portfolio.

        Supports:
        - buy: purchase shares (or cover a short position)
        - sell: sell existing long shares
        - sell_short: open or extend a short position (position goes negative)
        """
        if side == "buy":
            cost = qty * price
            if cost > self.cash:
                raise ValueError("Not enough cash for trade.")

            self.cash -= cost

            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol, qty)
            else:
                self.positions[symbol].quantity += qty

            # Clean up if position is now zero
            if symbol in self.positions and self.positions[symbol].quantity == 0:
                del self.positions[symbol]

        elif side == "sell":
            if symbol not in self.positions:
                return

            held = self.positions[symbol].quantity
            actual_qty = min(qty, max(int(held), 0))
            if actual_qty <= 0:
                return

            proceeds = actual_qty * price
            self.cash += proceeds
            self.positions[symbol].quantity -= actual_qty

            if self.positions[symbol].quantity == 0:
                del self.positions[symbol]

        elif side == "sell_short":
            # Open or extend a short position
            proceeds = qty * price
            self.cash += proceeds

            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol, -qty)
            else:
                self.positions[symbol].quantity -= qty

            if symbol in self.positions and self.positions[symbol].quantity == 0:
                del self.positions[symbol]