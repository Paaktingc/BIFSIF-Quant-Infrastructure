"""Risk management module for BIFS Quant Engine.

This module provides production-grade risk controls including:
- Position and exposure limits
- Circuit breakers for daily loss and drawdown
- Pre-trade validation
- Risk state tracking
"""

from bifs_quant_engine.risk.limits import (
    PositionLimits,
    ExposureLimits,
    check_position_limit,
    check_exposure_limits,
)
from bifs_quant_engine.risk.circuit_breakers import (
    CircuitBreakerConfig,
    check_circuit_breakers,
)
from bifs_quant_engine.risk.state import RiskStateTracker
from bifs_quant_engine.risk.validators import OrderValidator
from bifs_quant_engine.risk.manager import ProductionRiskManager

__all__ = [
    "PositionLimits",
    "ExposureLimits",
    "check_position_limit",
    "check_exposure_limits",
    "CircuitBreakerConfig",
    "check_circuit_breakers",
    "RiskStateTracker",
    "OrderValidator",
    "ProductionRiskManager",
]
