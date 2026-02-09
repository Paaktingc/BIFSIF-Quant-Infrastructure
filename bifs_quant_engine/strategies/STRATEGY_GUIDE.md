# Strategy Developer Guide

This short guide explains how to create strategies for the BIFS Quant Engine.

Contract for strategies
- Implement a class that inherits from `Strategy` (see `strategies/base.py`).
- Implement `generate_target_weights(self, date: datetime, price_history: pd.DataFrame) -> Dict[str, float]`.
  - `date`: the rebalance date (end-of-day)
  - `price_history`: DataFrame with index = DateTimeIndex, columns = symbols, values = prices
  - Return a dict mapping symbol -> target weight (floats summing to <= 1.0)

Example
- See `example_strategy.py` for a minimal equal-weight template.

Best practices
- Keep strategy pure: only read `price_history` and compute weights.
- Do not modify global state in the strategy.
- Return weights for all symbols in your universe (zero for symbols you do not want).
- Add unit tests for edge cases: insufficient history, NaNs, zero-length universe.

Running and testing
- Backtests call `generate_target_weights` on rebalance dates and expect a dict of weights.
- Use `tests/` to add strategy unit tests.
