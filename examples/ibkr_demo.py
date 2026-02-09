"""Interactive Brokers integration demo.

Requires TWS or IB Gateway to be running and configured to accept API connections.
Default port: 7497 (Paper Trading)
"""

import time
import logging
from bifs_quant_engine.brokers.ibkr_broker import IBKRBroker, IBKRConfig
from bifs_quant_engine.core.models import Order
from bifs_quant_engine.core.enums import OrderSide, OrderType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting IBKR Demo...")
    
    # create config
    config = IBKRConfig(
        host="127.0.0.1",
        port=7497,  # Paper trading port
        client_id=1
    )
    
    broker = IBKRBroker(config)
    
    try:
        logger.info("Connecting to IBKR...")
        broker.connect()
        logger.info("Connected!")
        
        # Check balance
        cash = broker.get_cash_balance()
        logger.info(f"Cash Balance: ${cash:,.2f}")
        
        # Get Quote
        symbol = "AAPL"
        logger.info(f"Getting quote for {symbol}...")
        quote = broker.get_quote(symbol)
        if quote:
            logger.info(f"Quote: {quote}")
        else:
            logger.warning("No quote received")
            
        # Submit Order (Commented out to prevent accidental firing, uncomment to test)
        # logger.info("Submitting Market Buy Order for 1 AAPL...")
        # order = Order(
        #     symbol="AAPL",
        #     side=OrderSide.BUY,
        #     quantity=1,
        #     order_type=OrderType.MARKET
        # )
        # filled_order = broker.submit_order(order)
        # logger.info(f"Order Status: {filled_order.status}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.info("Ensure TWS or IB Gateway is running and API connections are enabled.")
    finally:
        if broker.is_connected:
            broker.disconnect()
            logger.info("Disconnected.")

if __name__ == "__main__":
    main()
