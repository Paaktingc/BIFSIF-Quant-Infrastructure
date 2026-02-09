"""Risk model stubs."""

from __future__ import annotations

from typing import Dict, Any


def simple_var(returns: list[float], conf: float = 0.95) -> float:
    if not returns:
        return 0.0
    # Placeholder: not a real VaR, just a scaled std proxy
    import statistics as stats
    try:
        sigma = stats.pstdev(returns)
    except stats.StatisticsError:
        sigma = 0.0
    z = 1.65 if conf >= 0.95 else 1.28
    return z * sigma


def risk_report(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {"risk": "ok", **metrics}
