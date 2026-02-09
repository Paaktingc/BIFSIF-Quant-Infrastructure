"""Production RiskManager implementation.

This module provides the full RiskManager that combines all risk controls:
- Pre-trade order validation
- Position and exposure limits
- Circuit breakers
- Risk state tracking
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional
import yaml
from pathlib import Path

from bifs_quant_engine.core.enums import RiskAction
from bifs_quant_engine.core.models import (
    Order,
    Position,
    PortfolioSnapshot,
    RiskDecision,
    RiskState,
)
from bifs_quant_engine.core.protocols import RiskManager
from bifs_quant_engine.risk.limits import (
    PositionLimits,
    ExposureLimits,
    check_position_limit,
    check_exposure_limits,
)
from bifs_quant_engine.risk.circuit_breakers import (
    CircuitBreakerConfig,
    check_circuit_breakers,
    reset_circuit_breakers,
)
from bifs_quant_engine.risk.state import RiskStateTracker
from bifs_quant_engine.risk.validators import OrderValidator, ValidationConfig


class ProductionRiskManager(RiskManager):
    """Full implementation of the RiskManager protocol.
    
    Provides production-grade risk controls by combining:
    - OrderValidator for basic validation
    - Position limits (per-symbol concentration)
    - Exposure limits (gross/net exposure)
    - Circuit breakers (daily loss, drawdown)
    
    Example:
        risk_manager = ProductionRiskManager.from_config_file("config/risk_limits.yaml")
        
        # Validate an order
        decision = risk_manager.validate_order(order)
        if decision.action == RiskAction.ALLOW:
            execution_engine.submit(order)
        
        # Check if trading is allowed
        if not risk_manager.can_trade():
            logger.warning("Trading halted due to circuit breaker")
    """
    
    def __init__(
        self,
        position_limits: Optional[PositionLimits] = None,
        exposure_limits: Optional[ExposureLimits] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
        starting_equity: Decimal = Decimal("0"),
    ) -> None:
        """Initialize the risk manager.
        
        Args:
            position_limits: Position limit configuration
            exposure_limits: Exposure limit configuration
            circuit_breaker_config: Circuit breaker configuration
            validation_config: Order validation configuration
            starting_equity: Starting portfolio equity
        """
        self._position_limits = position_limits or PositionLimits()
        self._exposure_limits = exposure_limits or ExposureLimits()
        self._breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self._validator = OrderValidator(validation_config)
        self._state_tracker = RiskStateTracker(starting_equity)
        
        # Current portfolio state for limit checks
        self._current_positions: Dict[str, Position] = {}
        self._current_prices: Dict[str, Decimal] = {}
        self._portfolio_equity: Decimal = starting_equity
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "ProductionRiskManager":
        """Create a RiskManager from a YAML config file.
        
        Args:
            config_path: Path to risk_limits.yaml
            
        Returns:
            Configured ProductionRiskManager instance
        """
        path = Path(config_path)
        if not path.exists():
            # Return with defaults if no config file
            return cls()
        
        with open(path) as f:
            config = yaml.safe_load(f)
        
        position_cfg = config.get("position_limits", {})
        position_limits = PositionLimits(
            max_position_pct=Decimal(str(position_cfg.get("max_position_pct", 0.10))),
            max_position_notional=Decimal(str(position_cfg.get("max_position_notional", 50000))),
            max_shares_per_order=position_cfg.get("max_shares_per_order", 10000),
        )
        
        exposure_cfg = config.get("exposure_limits", {})
        exposure_limits = ExposureLimits(
            max_gross_exposure=Decimal(str(exposure_cfg.get("max_gross_exposure", 2.0))),
            max_net_exposure=Decimal(str(exposure_cfg.get("max_net_exposure", 0.20))),
            max_long_exposure=Decimal(str(exposure_cfg.get("max_long_exposure", 1.5))),
            max_short_exposure=Decimal(str(exposure_cfg.get("max_short_exposure", 0.5))),
        )
        
        breaker_cfg = config.get("circuit_breakers", {})
        breaker_config = CircuitBreakerConfig(
            daily_loss_limit_pct=Decimal(str(breaker_cfg.get("daily_loss_limit_pct", 0.02))),
            drawdown_limit_pct=Decimal(str(breaker_cfg.get("drawdown_limit_pct", 0.10))),
            enabled=breaker_cfg.get("enabled", True),
        )
        
        return cls(
            position_limits=position_limits,
            exposure_limits=exposure_limits,
            circuit_breaker_config=breaker_config,
        )
    
    def validate_order(self, order: Order) -> RiskDecision:
        """Validate a single order against all risk controls.
        
        Performs checks in order:
        1. Circuit breaker check (halt all trading)
        2. Basic order validation
        3. Position limits
        
        Args:
            order: The order to validate
            
        Returns:
            RiskDecision indicating if order is allowed
        """
        # Check circuit breakers first
        current_state = self._state_tracker.get_state()
        if current_state.trading_halted:
            return RiskDecision(
                action=RiskAction.HALT,
                order_id=order.order_id,
                original_quantity=order.quantity,
                approved_quantity=0,
                reason="Trading halted due to circuit breaker",
                violated_limits=self._get_triggered_breakers(current_state),
            )
        
        # Basic validation
        validation_result = self._validator.validate(order)
        if validation_result.action == RiskAction.REJECT:
            return validation_result
        
        # Position limits
        position_result = check_position_limit(
            order=order,
            current_positions=self._current_positions,
            current_prices=self._current_prices,
            portfolio_equity=self._portfolio_equity,
            limits=self._position_limits,
        )
        
        if position_result.action != RiskAction.ALLOW:
            return position_result
        
        # All checks passed
        return RiskDecision(
            action=RiskAction.ALLOW,
            order_id=order.order_id,
            original_quantity=order.quantity,
            approved_quantity=order.quantity,
            reason="Order approved by risk manager",
        )
    
    def validate_orders(self, orders: List[Order]) -> List[RiskDecision]:
        """Validate multiple orders.
        
        Args:
            orders: List of orders to validate
            
        Returns:
            List of RiskDecisions, one per order
        """
        return [self.validate_order(order) for order in orders]
    
    def check_circuit_breakers(self) -> RiskState:
        """Check all circuit breakers and update state.
        
        Returns:
            Current risk state with updated breaker flags
        """
        current_state = self._state_tracker.get_state()
        return check_circuit_breakers(current_state, self._breaker_config)
    
    def update_state(self, snapshot: PortfolioSnapshot) -> RiskState:
        """Update risk state from portfolio snapshot.
        
        Also updates internal position tracking for limit checks.
        
        Args:
            snapshot: Current portfolio snapshot
            
        Returns:
            Updated risk state
        """
        # Update internal tracking
        self._current_positions = snapshot.positions.copy()
        self._portfolio_equity = snapshot.total_equity
        
        # Update risk state
        state = self._state_tracker.update_from_snapshot(snapshot)
        
        # Check circuit breakers after update
        return check_circuit_breakers(state, self._breaker_config)
    
    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        """Update current market prices for limit calculations.
        
        Args:
            prices: Dict mapping symbol to current price
        """
        self._current_prices.update(prices)
    
    def get_risk_state(self) -> RiskState:
        """Get current risk state."""
        return self._state_tracker.get_state()
    
    def reset_daily_state(self) -> None:
        """Reset for new trading day.
        
        Resets daily P&L tracking and daily loss circuit breaker.
        """
        self._state_tracker.reset_daily(self._portfolio_equity)
    
    def can_trade(self) -> bool:
        """Check if trading is currently allowed.
        
        Returns:
            True if no circuit breakers are triggered
        """
        return not self._state_tracker.get_state().trading_halted
    
    def _get_triggered_breakers(self, state: RiskState) -> List[str]:
        """Get list of triggered circuit breaker names."""
        triggered = []
        if state.daily_loss_breaker_triggered:
            triggered.append("daily_loss")
        if state.drawdown_breaker_triggered:
            triggered.append("drawdown")
        if state.volatility_breaker_triggered:
            triggered.append("volatility")
        return triggered
