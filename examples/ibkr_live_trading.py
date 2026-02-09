"""IBKR Live Trading Runner.

Runs a simple Momentum Strategy on Interactive Brokers live/paper account.
"""

import time
import logging
import signal
import sys
from decimal import Decimal

from bifs_quant_engine.brokers import IBKRBroker, IBKRConfig
from bifs_quant_engine.trading.session import TradingSession, SessionConfig
from bifs_quant_engine.strategies.strategy_protocol import BaseStrategy, Signal, SignalType
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Simple Strategy for Demo ---
class InteractiveBrokersMomentum(BaseStrategy):
    """Simple strategy that buys AAPL if price > 150 (dummy logic for test)."""
    
    def __init__(self):
        super().__init__("ibkr_momentum", ["AAPL"])
        
    def generate_signals(self, market_data: pd.DataFrame, current_positions: dict) -> list:
        # In a real strategy, you'd use market_data here.
        # For this connectivity demo, we'll just log and maybe emit a signal.
        logger.info("Analyzing market data...")
        return []

def main():
    print("=" * 60)
    print("  BIFS Quant Engine - IBKR Live Trading")
    print("=" * 60)

    # 1. Configure Broker
    # NOTE: Set port to 7496 for LIVE trading, 7497 for PAPER
    config = IBKRConfig(
        host="127.0.0.1",
        port=7497,
        client_id=1
    )
    broker = IBKRBroker(config)
    
    # 2. Configure Session
    session_config = SessionConfig(
        initial_capital=Decimal("100000"),
        run_interval_seconds=10,  # fast interval for testing
    )
    
    session = TradingSession(broker, session_config)
    
    # 3. Add Strategy
    strategy = InteractiveBrokersMomentum()
    session.add_strategy(strategy)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nStopping session...")
        session.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("Connecting to IBKR...")
        session.start()
        
        print("\nTrading session running. Press Ctrl+C to stop.")
        while session.is_running:
            # The session runs in a background thread, so we just keep main alive
            # and print status occasionally
            status = session.get_status()
            print(f"Status: {status['state']} | NAV: {status.get('nav', 'N/A')}")
            time.sleep(5)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        session.stop()

if __name__ == "__main__":
    main()
