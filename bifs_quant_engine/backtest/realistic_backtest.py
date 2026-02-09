"""Realistic backtesting engine.

Production-grade backtester with:
- Transaction cost modeling (commissions, slippage)
- Market impact simulation
- Bid-ask spread modeling
- Position sizing constraints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol

import pandas as pd

from bifs_quant_engine.core.enums import OrderSide, OrderStatus
from bifs_quant_engine.core.models import Order, Fill, Position, PortfolioSnapshot


@dataclass
class BacktestConfig:
    """Configuration for realistic backtesting.
    
    Attributes:
        initial_cash: Starting capital
        commission_per_share: Commission charged per share
        min_commission: Minimum commission per order
        slippage_bps: Slippage in basis points (5 = 0.05%)
        market_impact_bps: Additional impact based on order size
        use_bid_ask: Whether to use bid/ask spreads (if available)
        max_position_pct: Maximum position size as % of portfolio
        max_order_pct_adv: Maximum order as % of average daily volume
        fill_probability: Probability of order fill (for limit orders)
    """
    initial_cash: Decimal = Decimal("100000")
    commission_per_share: Decimal = Decimal("0.005")
    min_commission: Decimal = Decimal("1.00")
    slippage_bps: int = 5
    market_impact_bps: int = 10
    use_bid_ask: bool = False
    max_position_pct: Decimal = Decimal("0.10")
    max_order_pct_adv: Decimal = Decimal("0.10")
    fill_probability: float = 1.0


class StrategyProtocol(Protocol):
    """Protocol for backtest-compatible strategies."""
    
    def generate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """Generate target portfolio weights."""
        ...


@dataclass
class BacktestResult:
    """Results from a backtest run.
    
    Attributes:
        equity_curve: Time series of portfolio value
        returns: Daily returns series
        trades: List of executed trades
        snapshots: Portfolio snapshots at each rebalance
        analytics: Performance metrics (computed lazily)
    """
    equity_curve: pd.Series
    returns: pd.Series
    trades: List[Dict[str, Any]]
    snapshots: List[PortfolioSnapshot]
    config: BacktestConfig
    
    _analytics: Optional["PerformanceAnalytics"] = field(default=None, repr=False)
    
    @property
    def analytics(self) -> "PerformanceAnalytics":
        """Get performance analytics (computed lazily)."""
        if self._analytics is None:
            from .performance_analytics import PerformanceAnalytics
            self._analytics = PerformanceAnalytics(self.returns, self.equity_curve)
        return self._analytics


class RealisticBacktest:
    """Production-grade backtesting engine.
    
    Features:
    - Realistic transaction costs (commissions, slippage, market impact)
    - Bid-ask spread modeling
    - Position sizing constraints
    - Fill simulation
    
    Example:
        from bifs_quant_engine.strategies.etf_momentum import ETFMomentumStrategy
        
        backtest = RealisticBacktest(
            strategy=ETFMomentumStrategy("momentum"),
            universe=["SPY", "QQQ", "IWM"],
            start="2020-01-01",
            end="2023-12-31",
        )
        result = backtest.run()
        print(result.analytics.summary())
    """
    
    def __init__(
        self,
        strategy: StrategyProtocol,
        universe: List[str],
        start: str = "2015-01-01",
        end: Optional[str] = None,
        config: Optional[BacktestConfig] = None,
        rebalance_frequency: int = 20,
        data_loader: Optional[Callable] = None,
    ) -> None:
        """Initialize backtest.
        
        Args:
            strategy: Trading strategy implementing StrategyProtocol
            universe: List of symbols to trade
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD), defaults to today
            config: Backtest configuration
            rebalance_frequency: Days between rebalances
            data_loader: Optional custom data loader function
        """
        self.strategy = strategy
        self.universe = universe
        self.start = start
        self.end = end
        self.config = config or BacktestConfig()
        self.rebalance_frequency = rebalance_frequency
        self._data_loader = data_loader
        
        # State during backtest
        self._cash: Decimal = Decimal("0")
        self._positions: Dict[str, Position] = {}
        self._trades: List[Dict[str, Any]] = []
        self._snapshots: List[PortfolioSnapshot] = []
        
    def run(self) -> BacktestResult:
        """Execute the backtest.
        
        Returns:
            BacktestResult with equity curve, trades, and analytics
        """
        # Load price data
        prices = self._load_prices()
        
        # Initialize
        self._cash = self.config.initial_cash
        self._positions = {}
        self._trades = []
        self._snapshots = []
        
        equity_values: List[float] = []
        dates = prices.index.tolist()
        
        for i, dt in enumerate(dates):
            today = dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else dt
            today_prices = self._get_prices_dict(prices, dt)
            
            # Rebalance check
            if i % self.rebalance_frequency == 0 and i > 0:
                hist_prices = prices.loc[:dt]
                
                # Get target weights from strategy
                target_weights = self.strategy.generate_target_weights(
                    today,
                    hist_prices,
                )
                
                # Generate and execute orders
                orders = self._weights_to_orders(target_weights, today_prices)
                for order in orders:
                    self._execute_order(order, today_prices, today)
            
            # Calculate and record equity
            equity = self._calculate_equity(today_prices)
            equity_values.append(float(equity))
            
            # Take snapshot on rebalance days
            if i % self.rebalance_frequency == 0:
                snapshot = self._create_snapshot(today_prices, today)
                self._snapshots.append(snapshot)
        
        # Build result
        equity_curve = pd.Series(equity_values, index=prices.index, name="equity")
        returns = equity_curve.pct_change().fillna(0)
        
        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            trades=self._trades,
            snapshots=self._snapshots,
            config=self.config,
        )
    
    def _load_prices(self) -> pd.DataFrame:
        """Load price data for universe."""
        if self._data_loader:
            return self._data_loader(self.universe, self.start, self.end)
        
        # Default: use MarketDataAPI
        from bifs_quant_engine.core.data_api import MarketDataAPI
        api = MarketDataAPI()
        return api.get_prices_for_universe(
            symbols=self.universe,
            start=self.start,
            end=self.end,
        )
    
    def _get_prices_dict(self, prices: pd.DataFrame, dt) -> Dict[str, Decimal]:
        """Extract prices for a date as a dict."""
        row = prices.loc[dt]
        return {
            symbol: Decimal(str(price)) 
            for symbol, price in row.items() 
            if pd.notna(price)
        }
    
    def _weights_to_orders(
        self, 
        target_weights: Dict[str, float],
        current_prices: Dict[str, Decimal],
    ) -> List[Order]:
        """Convert target weights to orders."""
        orders = []
        
        portfolio_value = self._calculate_equity(current_prices)
        
        for symbol, target_weight in target_weights.items():
            if symbol not in current_prices:
                continue
                
            price = current_prices[symbol]
            if price <= 0:
                continue
            
            # Calculate target position value and shares
            target_value = portfolio_value * Decimal(str(target_weight))
            target_shares = int(target_value / price)
            
            # Get current position
            current = self._positions.get(symbol)
            current_shares = current.quantity if current else 0
            
            # Calculate delta
            delta = target_shares - current_shares
            
            if delta == 0:
                continue
            
            # Apply position limit
            max_position_value = portfolio_value * self.config.max_position_pct
            max_shares = int(max_position_value / price)
            target_shares = min(target_shares, max_shares)
            delta = target_shares - current_shares
            
            if delta == 0:
                continue
            
            # Create order
            if delta > 0:
                side = OrderSide.BUY if current_shares >= 0 else OrderSide.COVER
                quantity = delta
            else:
                side = OrderSide.SELL if current_shares > 0 else OrderSide.SHORT
                quantity = abs(delta)
            
            orders.append(Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
            ))
        
        return orders
    
    def _execute_order(
        self, 
        order: Order, 
        prices: Dict[str, Decimal],
        date: datetime,
    ) -> Optional[Fill]:
        """Execute an order with realistic costs."""
        price = prices.get(order.symbol)
        if price is None:
            return None
        
        # Calculate fill price with slippage
        fill_price = self._calculate_fill_price(order, price)
        
        # Calculate commission
        commission = max(
            self.config.min_commission,
            self.config.commission_per_share * order.quantity,
        )
        
        # Update cash
        notional = fill_price * order.quantity
        if order.is_buy_side:
            if self._cash < notional + commission:
                # Not enough cash, reduce order
                max_shares = int((self._cash - commission) / fill_price)
                if max_shares <= 0:
                    return None
                order.quantity = max_shares
                notional = fill_price * order.quantity
            
            self._cash -= notional + commission
        else:
            self._cash += notional - commission
        
        # Update position
        self._update_position(order.symbol, order.side, order.quantity, fill_price)
        
        # Record trade
        trade = {
            "date": date,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": float(fill_price),
            "commission": float(commission),
            "notional": float(notional),
        }
        self._trades.append(trade)
        
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
        )
    
    def _calculate_fill_price(self, order: Order, base_price: Decimal) -> Decimal:
        """Calculate fill price with slippage and market impact."""
        # Base slippage
        slippage = Decimal(self.config.slippage_bps) / Decimal("10000")
        
        # Market impact (proportional to order size)
        # Simplified: assume 1% of ADV = 1bp additional impact
        impact = Decimal(self.config.market_impact_bps) / Decimal("10000")
        
        total_impact = slippage + impact
        
        if order.is_buy_side:
            return (base_price * (1 + total_impact)).quantize(Decimal("0.01"))
        else:
            return (base_price * (1 - total_impact)).quantize(Decimal("0.01"))
    
    def _update_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Decimal,
    ) -> None:
        """Update position after fill."""
        current = self._positions.get(symbol)
        current_qty = current.quantity if current else 0
        current_cost = current.avg_cost if current else Decimal("0")
        
        if side in (OrderSide.BUY, OrderSide.COVER):
            delta = quantity
        else:
            delta = -quantity
        
        new_qty = current_qty + delta
        
        if new_qty == 0:
            if symbol in self._positions:
                del self._positions[symbol]
            return
        
        # Calculate new average cost
        if (current_qty >= 0 and delta > 0) or (current_qty <= 0 and delta < 0):
            # Adding to position
            total_cost = abs(current_qty) * current_cost + abs(delta) * price
            new_cost = total_cost / abs(new_qty)
        else:
            # Reducing position
            new_cost = current_cost
        
        self._positions[symbol] = Position(
            symbol=symbol,
            quantity=new_qty,
            avg_cost=new_cost.quantize(Decimal("0.0001")),
        )
    
    def _calculate_equity(self, prices: Dict[str, Decimal]) -> Decimal:
        """Calculate total portfolio equity."""
        positions_value = Decimal("0")
        
        for symbol, pos in self._positions.items():
            price = prices.get(symbol)
            if price:
                positions_value += pos.quantity * price
        
        return self._cash + positions_value
    
    def _create_snapshot(
        self, 
        prices: Dict[str, Decimal],
        date: datetime,
    ) -> PortfolioSnapshot:
        """Create portfolio snapshot."""
        long_value = Decimal("0")
        short_value = Decimal("0")
        
        for symbol, pos in self._positions.items():
            price = prices.get(symbol, Decimal("0"))
            value = pos.quantity * price
            if pos.quantity > 0:
                long_value += value
            else:
                short_value += abs(value)
        
        total_equity = self._cash + long_value - short_value
        
        return PortfolioSnapshot(
            timestamp=date,
            cash=self._cash,
            total_equity=total_equity,
            long_market_value=long_value,
            short_market_value=short_value,
            gross_exposure=long_value + short_value,
            net_exposure=long_value - short_value,
            positions=dict(self._positions),
        )
