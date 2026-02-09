# BIFS Quant Engine

This repository contains a minimal scaffold for a quantitative fund/backtesting engine. It includes:

- Config files (YAML) for universes, fund mandates, and strategies
- Core modules for data access, backtesting, portfolio/risk, and broker sims
- Strategy examples (ETF momentum)
- Entrypoint scripts to run a fund loop or backtest a strategy

Quick start
- Create a virtual environment and install requirements
  - pip install -r requirements.txt
- Edit files in config/ to suit your data and preferences
- Run a quick backtest
  - python backtest_strategy.py --strategy etf_momentum

Directory layout
bifs_quant_engine/
├── requirements.txt
├── README.md
├── run_fund.py
├── backtest_strategy.py
├── config/
│   ├── universe.yaml
│   ├── fund_mandate.yaml
│   └── strategies.yaml
├── core/
│   ├── __init__.py
│   ├── data_api.py
│   ├── backtest.py
│   ├── portfolio.py
│   ├── risk.py
│   ├── broker_sim.py
│   ├── broker_ibkr.py
│   ├── fund_engine.py
│   └── utils.py
├── strategies/
│   ├── __init__.py
│   ├── base.py
│   └── etf_momentum.py
├── research/
│   ├── etf_momentum_research.ipynb
│   └── .gitkeep
├── reports/
│   └── .gitkeep
└── data/
    └── .gitkeep
