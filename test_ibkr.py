import sys
import os

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from bifs_quant_engine.brokers.ibkr_broker import IBKRBroker, IBKRConfig
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def test_connection():
    broker = IBKRBroker()
    print("Attempting to connect to IBKR on 127.0.0.1:7497...")
    try:
        broker.connect()
        print(f"Success! Status: is_connected={broker.is_connected}")
        
        balance = broker.get_cash_balance()
        print(f"Cash Balance: ${balance}")
        
        bp = broker.get_buying_power()
        print(f"Buying Power: ${bp}")
        
        quote = broker.get_quote("AAPL")
        print(f"AAPL Quote: {quote}")
        
    except Exception as e:
        print(f"Failed to connect or fetch data: {type(e).__name__} - {e}")
        print("Make sure TWS or IB Gateway is running and API is enabled on port 7497.")
    finally:
        broker.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    test_connection()
