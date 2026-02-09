#!/usr/bin/env python3
"""Paper Trading Demo

Demonstrates how to run a momentum strategy in paper trading mode
with the BIFS Quant Engine.

Usage:
    python examples/paper_trading_demo.py
"""

import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

from bifs_quant_engine.core.enums import OrderSide
from bifs_quant_engine.core.models import Order
from bifs_quant_engine.brokers.paper_broker import PaperBroker, PaperBrokerConfig
from bifs_quant_engine.strategies.strategy_protocol import BaseStrategy, Signal, SignalType
from bifs_quant_engine.trading.session import TradingSession, SessionConfig


class MomentumStrategy(BaseStrategy):
    """Simple 20-day momentum strategy.
    
    Goes long stocks with positive 20-day returns,
    short stocks with negative 20-day returns.
    """
    
    def __init__(self, symbols: list, lookback: int = 20):
        super().__init__("momentum", symbols)
        self.lookback = lookback
    
    def generate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> dict:
        """Generate target weights based on momentum."""
        if len(price_history) < self.lookback:
            return {}
        
        weights = {}
        for symbol in self._universe:
            if symbol not in price_history.columns:
                continue
            
            # Calculate momentum (percentage return over lookback period)
            returns = price_history[symbol].pct_change(self.lookback).iloc[-1]
            
            if pd.notna(returns):
                # Equal weight long/short based on momentum sign
                weights[symbol] = float(np.sign(returns) * 0.1)
        
        return weights
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: dict,
    ) -> list:
        """Convert weights to signals."""
        weights = self.generate_target_weights(datetime.now(), market_data)
        signals = []
        
        for symbol, weight in weights.items():
            if weight > 0.01:
                signals.append(Signal(
                    strategy_id=self.name,
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    weight=weight,
                ))
            elif weight < -0.01:
                signals.append(Signal(
                    strategy_id=self.name,
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    weight=abs(weight),
                ))
        
        return signals


def generate_sample_prices(symbols: list, days: int = 60) -> pd.DataFrame:
    """Generate sample historical prices for demo."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    
    data = {}
    for i, symbol in enumerate(symbols):
        base = 100 + i * 50  # Different base prices
        drift = 0.0005 * (1 if i % 2 == 0 else -1)  # Alternating trends
        volatility = 0.02
        
        returns = np.random.randn(days) * volatility + drift
        prices = base * np.cumprod(1 + returns)
        data[symbol] = prices
    
    return pd.DataFrame(data, index=dates)


def main():
    """Run paper trading demo."""
    print("=" * 60)
    print("  BIFS Quant Engine - Paper Trading Demo")
    print("=" * 60)
    
    # Configuration
    symbols = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
    initial_capital = Decimal("100000")
    
    # Create broker with realistic settings
    broker_config = PaperBrokerConfig(
        initial_cash=initial_capital,
        latency_ms=50,  # 50ms simulated latency
        slippage_bps=5,  # 5 basis points slippage
        commission_per_share=Decimal("0.01"),
        min_commission=Decimal("1.00"),
    )
    broker = PaperBroker(broker_config)
    
    # Create trading session
    session_config = SessionConfig(
        initial_capital=initial_capital,
        max_position_pct=0.15,
        max_daily_loss=Decimal("-2000"),
        run_interval_seconds=5,  # 5 second interval for demo
    )
    session = TradingSession(broker, session_config) 
    
    # Add momentum strategy
    strategy = MomentumStrategy(symbols, lookback=10)
    session.add_strategy(strategy)
    
    # Generate sample prices  
    print("\n📊 Generating sample price history...")
    prices = generate_sample_prices(symbols, days=60)
    session.set_price_history(prices)
    
    # Set current quotes (last prices from history)
    quotes = {
        symbol: {"bid": prices[symbol].iloc[-1] * 0.999,
                 "ask": prices[symbol].iloc[-1] * 1.001,
                 "last": prices[symbol].iloc[-1]}
        for symbol in symbols
    }
    session.update_quotes(quotes)
    
    print(f"\n💰 Starting capital: ${initial_capital:,.2f}")
    print(f"📈 Trading symbols: {', '.join(symbols)}")
    print(f"🎯 Strategy: {strategy.name}")
    
    # Run one trading cycle manually (demo mode)
    print("\n" + "-" * 60)
    print("Running single trading cycle...")
    print("-" * 60)
    
    # Connect broker manually for demo
    broker.connect()
    
    # Generate signals and execute
    signals = strategy.generate_signals(prices, {})
    print(f"\n📡 Generated {len(signals)} signals:")
    
    for signal in signals:
        direction = "LONG" if signal.signal_type == SignalType.LONG else "SHORT"
        print(f"   {signal.symbol}: {direction} (weight: {signal.weight:.2%})")
    
    # Convert signals to orders and execute
    for signal in signals:
        quote = quotes[signal.symbol]
        price = Decimal(str(quote["last"]))
        
        # Calculate order quantity (10% of capital per position)
        position_value = initial_capital * Decimal(str(signal.weight))
        quantity = int(position_value / price)
        
        if quantity > 0:
            side = OrderSide.BUY if signal.signal_type == SignalType.LONG else OrderSide.SHORT
            order = Order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                strategy_id=signal.strategy_id,
            )
            
            result = broker.submit_order(order)
            
            if result.avg_fill_price:
                print(f"\n✅ {side.value.upper()} {quantity} {signal.symbol}")
                print(f"   Fill price: ${result.avg_fill_price:.2f}")
    
    # Display final portfolio state
    print("\n" + "=" * 60)
    print("  Portfolio Summary")
    print("=" * 60)
    
    cash = broker.get_cash_balance()
    positions = broker.get_positions()
    
    print(f"\n💵 Cash: ${cash:,.2f}")
    print(f"📦 Positions: {len(positions)}")
    
    if positions:
        print("\n   Symbol    Qty    Avg Cost    Market Value")
        print("   " + "-" * 44)
        
        total_value = cash
        for symbol, pos in positions.items():
            price = Decimal(str(quotes[symbol]["last"]))
            market_value = pos.quantity * price
            total_value += market_value
            print(f"   {symbol:<8} {pos.quantity:>6}  ${pos.avg_cost:>8.2f}  ${market_value:>12,.2f}")
    
    total_value = cash + sum(
        p.quantity * Decimal(str(quotes[p.symbol]["last"]))
        for p in positions.values()
    )
    pnl = total_value - initial_capital
    pnl_pct = (pnl / initial_capital) * 100
    
    print(f"\n📊 Total Value: ${total_value:,.2f}")
    print(f"📈 P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
    
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print("\nTo run continuous paper trading, use:")
    print("  session.start()  # Start background trading")
    print("  session.stop()   # Stop trading")


if __name__ == "__main__":
    main()
