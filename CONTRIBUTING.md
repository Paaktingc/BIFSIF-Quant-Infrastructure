# Contributing

Thanks for contributing to the BIFS Quant Engine. This file outlines the basics to get started.

Development setup
- Create a Python virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Testing
- Run tests:

```bash
PYTHONPATH=. pytest -q
```

Code style
- We recommend `black`, `isort`, and `flake8`. A `.pre-commit-config.yaml` is provided; install pre-commit and run `pre-commit install`.

Adding strategies
- Place new strategy modules under `bifs_quant_engine/strategies/` and follow the `Strategy` interface in `base.py`.
