# BIFS Quant Engine — Step-by-Step Startup Guide

A hands-on walkthrough to get the engine running, from zero to live signals.

---

## Step 1: Prerequisites

| Requirement | Minimum |
|---|---|
| Python | 3.10+ |
| pip | latest |
| OS | macOS / Linux |
| Internet | Required for live data (not needed in `--synthetic` mode) |

---

## Step 2: Set Up Your Environment

```bash
# Navigate to the project root
cd /path/to/BIFSIF-Quant-Infrastructure

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r bifs_quant_engine/requirements.txt
```

### Key dependencies installed:
- **Core**: `numpy`, `pandas`, `PyYAML`, `matplotlib`
- **Market data**: `yfinance`, `requests`
- **Live trading**: `ib_insync` (IBKR), `alpaca-trade-api`
- **Research/NLP**: `statsmodels`, `sentence-transformers`, `feedparser`
- **Crypto**: `py-clob-client`, `websockets`, `httpx`

---

## Step 3: Review & Customise Configuration

All config files live in `bifs_quant_engine/config/`. Edit these before running:

### 3a. Fund Mandate — `fund_mandate.yaml`
```yaml
fund_name: BIFS Quant Fund
base_currency: USD
rebalance_frequency: monthly
max_positions: 5
risk_limits:
  max_drawdown: 0.2
  var_confidence: 0.95
```

### 3b. Universe — `universe.yaml`
```yaml
name: default_etf_universe
tickers:
  - SPY
  - QQQ
  - IWM
  - EFA
  - EEM
```

### 3c. Strategies — `strategies.yaml`
Defines which strategies are active and their parameters (ETF momentum, Polymarket arb, vol straddle, etc.).

### 3d. Broker — `broker_config.yaml`
```yaml
broker:
  mode: paper          # Change to "live" for real execution
  paper:
    initial_cash: 100000
    slippage_bps: 5
```

### 3e. Risk Limits — `risk_limits.yaml`
Position sizing, exposure limits, and circuit breakers. **Review these carefully before going live.**

---

## Step 4: Run a Quick Smoke Test (No Network Required)

Verify everything is installed and working using synthetic data:

```bash
python run_all_backtests.py --synthetic
```

Expected output: a comparison table for all four strategies (ETF Momentum, Mean Reversion, SP500 Momentum, Kalman OU Pairs) with annualised return, Sharpe, max drawdown, and ASCII equity curves.

---

## Step 5: Run a Full Backtest (Live Data)

Download real prices via `yfinance` and run the full multi-strategy backtest:

```bash
# Default range (2015–2024, $1M capital)
python run_all_backtests.py

# Custom date range and capital
python run_all_backtests.py --start 2018-01-01 --end 2023-12-31 --cash 500000

# Skip the slower Kalman OU Pairs strategy
python run_all_backtests.py --skip-pairs
```

### Run a single strategy backtest

```bash
# ETF Momentum only (standalone script)
python bifs_quant_engine/backtest_strategy.py
```

---

## Step 6: Run the Fund Engine Loop

The fund engine evaluates strategies against your mandate and writes reports:

```bash
python bifs_quant_engine/run_fund.py
```

> [!TIP]
> You can override config paths:
> ```bash
> python bifs_quant_engine/run_fund.py \
>   --config   bifs_quant_engine/config/fund_mandate.yaml \
>   --universe bifs_quant_engine/config/universe.yaml \
>   --strategies bifs_quant_engine/config/strategies.yaml
> ```

Output is written to `bifs_quant_engine/reports/last_run.txt`.

---

## Step 7: Paper Trading (Simulated Live)

Run a momentum strategy against simulated execution:

```bash
python examples/paper_trading_demo.py
```

Or programmatically:

```python
from bifs_quant_engine.brokers.paper_broker import PaperBroker, PaperBrokerConfig
from bifs_quant_engine.trading.session import TradingSession, SessionConfig

broker = PaperBroker(PaperBrokerConfig(initial_cash=100000))
session = TradingSession(broker, SessionConfig())

# Add a strategy, set price data, then:
session.start()   # Start background trading loop
session.stop()    # Stop when done
```

---

## Step 8: Set Up for Live Trading

### 8a. Alpaca (US Equities)
```bash
export ALPACA_API_KEY=your_api_key
export ALPACA_API_SECRET=your_api_secret
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 8b. Interactive Brokers
Requires TWS or IB Gateway running locally. Test connectivity:
```bash
python test_ibkr.py
```

### 8c. Polymarket (Event Contracts)
```bash
export POLYMARKET_PRIVATE_KEY=your_key
export POLYMARKET_FUNDER=your_funder_address
```

> [!CAUTION]
> Always validate with paper trading before switching `broker_config.yaml` to `mode: live`.

---

## Step 9: Start the Monitoring Dashboard

```python
from bifs_quant_engine.dashboard import DashboardApp
from bifs_quant_engine.monitoring.metrics import MetricsCollector

metrics = MetricsCollector(initial_nav=100000)
app = DashboardApp(metrics_collector=metrics)
app.run(port=5000)
```

Open **http://localhost:5000** to view portfolio NAV, positions, strategy performance, and active alerts.

---

## Step 10: Verify & Monitor

```bash
# Run the test suite
pytest tests/ -v

# Check a specific test layer
pytest tests/unit/ -v
pytest tests/integration/ -v
```

---

## Quick Reference: All Entry Points

| What | Command |
|---|---|
| Smoke test (no network) | `python run_all_backtests.py --synthetic` |
| Full multi-strategy backtest | `python run_all_backtests.py` |
| Single ETF momentum backtest | `python bifs_quant_engine/backtest_strategy.py` |
| Fund engine loop | `python bifs_quant_engine/run_fund.py` |
| Paper trading demo | `python examples/paper_trading_demo.py` |
| IBKR connectivity test | `python test_ibkr.py` |
| Test suite | `pytest tests/ -v` |

---

## Key File Locations

```
BIFSIF Quant Infrastructure/
├── run_all_backtests.py              ← Main multi-strategy runner
├── test_ibkr.py                      ← IBKR connectivity test
├── bifs_quant_engine/
│   ├── run_fund.py                   ← Fund engine entry point
│   ├── backtest_strategy.py          ← Single strategy backtest
│   ├── config/                       ← All YAML configurations
│   │   ├── fund_mandate.yaml
│   │   ├── universe.yaml
│   │   ├── strategies.yaml
│   │   ├── broker_config.yaml
│   │   └── risk_limits.yaml
│   ├── core/                         ← Engine core (backtest, data, risk)
│   ├── strategies/                   ← Strategy implementations
│   ├── brokers/                      ← Broker adapters (Paper, Alpaca)
│   ├── trading/                      ← Live trading session management
│   ├── dashboard/                    ← Web monitoring UI
│   └── reports/                      ← Generated reports
├── examples/                         ← Demo scripts
│   ├── paper_trading_demo.py
│   ├── ibkr_demo.py
│   └── ibkr_live_trading.py
└── docs/                             ← Documentation
```
