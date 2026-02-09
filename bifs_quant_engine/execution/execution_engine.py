"""Execution engine implementation.

Manages order lifecycle, position reconciliation, and fill handling.
Implements the ExecutionEngine protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set
from uuid import UUID
import logging

from bifs_quant_engine.core.enums import OrderStatus
from bifs_quant_engine.core.models import Order, Fill, Position
from bifs_quant_engine.core.protocols import BrokerAdapter, ExecutionEngine

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for execution engine.
    
    Attributes:
        max_retries: Maximum retry attempts for failed orders
        retry_delay_ms: Delay between retries in milliseconds
        reconcile_interval_seconds: How often to auto-reconcile
    """
    max_retries: int = 3
    retry_delay_ms: int = 1000
    reconcile_interval_seconds: int = 60


class ExecutionEngineImpl(ExecutionEngine):
    """Production execution engine implementation.
    
    Manages order lifecycle with:
    - Idempotent order submission via client_order_id tracking
    - Order state machine with valid transitions
    - Retry logic for transient failures
    - Position reconciliation with broker
    - Fill event handling and aggregation
    
    Example:
        broker = PaperBroker()
        engine = ExecutionEngineImpl(broker)
        
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        result = engine.submit_orders([order])
    """
    
    def __init__(
        self, 
        broker: BrokerAdapter,
        config: Optional[ExecutionConfig] = None,
    ) -> None:
        """Initialize execution engine.
        
        Args:
            broker: Broker adapter for order routing
            config: Engine configuration
        """
        self._broker = broker
        self._config = config or ExecutionConfig()
        
        # Order tracking (keyed by order_id for idempotency)
        self._orders: Dict[UUID, Order] = {}
        self._client_order_ids: Set[UUID] = set()
        
        # Fill tracking
        self._fills: List[Fill] = []
        self._fills_by_order: Dict[UUID, List[Fill]] = {}
        
        # Position cache (broker is source of truth)
        self._positions: Dict[str, Position] = {}
        
        # Retry tracking
        self._retry_counts: Dict[UUID, int] = {}
        
    def submit_orders(self, orders: List[Order]) -> List[Order]:
        """Submit a batch of orders.
        
        Idempotent: resubmitting same order_id returns existing order.
        
        Args:
            orders: Orders to submit
            
        Returns:
            Orders with updated status
        """
        results = []
        
        for order in orders:
            # Check for idempotency
            if order.order_id in self._client_order_ids:
                existing = self._orders.get(order.order_id)
                if existing:
                    logger.info(f"Order {order.order_id} already submitted, returning existing")
                    results.append(existing)
                    continue
            
            # Track new order
            self._client_order_ids.add(order.order_id)
            self._orders[order.order_id] = order
            
            # Submit to broker with retry
            submitted = self._submit_with_retry(order)
            results.append(submitted)
            
        return results
    
    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel a specific order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation succeeded
        """
        order = self._orders.get(order_id)
        if order is None:
            return False
            
        if order.is_terminal:
            return False
            
        success = self._broker.cancel_order(order_id)
        if success:
            order.status = OrderStatus.CANCELLED
            
        return success
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders.
        
        Returns:
            Count of orders cancelled
        """
        count = 0
        for order_id, order in self._orders.items():
            if not order.is_terminal:
                if self.cancel_order(order_id):
                    count += 1
        return count
    
    def get_order(self, order_id: UUID) -> Optional[Order]:
        """Get order by ID from internal tracking.
        
        Args:
            order_id: Order ID to look up
            
        Returns:
            Order or None if not found
        """
        return self._orders.get(order_id)
    
    def get_open_orders(self) -> List[Order]:
        """Get all non-terminal orders."""
        return [o for o in self._orders.values() if not o.is_terminal]
    
    def get_fills(self, since: Optional[datetime] = None) -> List[Fill]:
        """Get fills, optionally since a timestamp.
        
        Args:
            since: Optional cutoff timestamp
            
        Returns:
            List of fills
        """
        if since is None:
            return list(self._fills)
        return [f for f in self._fills if f.filled_at >= since]
    
    def reconcile(self) -> Dict[str, Any]:
        """Reconcile internal state with broker.
        
        Returns:
            Reconciliation report with discrepancies and actions
        """
        report = {
            "position_discrepancies": [],
            "order_discrepancies": [],
            "corrective_actions": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Sync positions from broker
        broker_positions = self._broker.get_positions()
        
        # Find position discrepancies
        all_symbols = set(self._positions.keys()) | set(broker_positions.keys())
        
        for symbol in all_symbols:
            local = self._positions.get(symbol)
            broker = broker_positions.get(symbol)
            
            local_qty = local.quantity if local else 0
            broker_qty = broker.quantity if broker else 0
            
            if local_qty != broker_qty:
                report["position_discrepancies"].append({
                    "symbol": symbol,
                    "local_qty": local_qty,
                    "broker_qty": broker_qty,
                    "diff": broker_qty - local_qty,
                })
                
                # Correct by adopting broker as source of truth
                if broker:
                    self._positions[symbol] = broker
                elif symbol in self._positions:
                    del self._positions[symbol]
                    
                report["corrective_actions"].append({
                    "action": "position_sync",
                    "symbol": symbol,
                    "new_qty": broker_qty,
                })
        
        # Sync open orders
        broker_orders = self._broker.get_open_orders()
        broker_order_ids = {o.order_id for o in broker_orders}
        local_open_ids = {o.order_id for o in self.get_open_orders()}
        
        # Orders in broker but not local
        for order in broker_orders:
            if order.order_id not in local_open_ids:
                report["order_discrepancies"].append({
                    "type": "missing_local",
                    "order_id": str(order.order_id),
                    "symbol": order.symbol,
                })
                # Add to local tracking
                self._orders[order.order_id] = order
                self._client_order_ids.add(order.order_id)
        
        # Update positions from broker
        self._positions = broker_positions
        
        logger.info(f"Reconciliation complete: {len(report['position_discrepancies'])} position discrepancies")
        
        return report
    
    def sync_positions(self) -> Dict[str, Position]:
        """Fetch positions from broker and update internal state.
        
        Returns:
            Current positions (broker is source of truth)
        """
        self._positions = self._broker.get_positions()
        return self._positions
    
    def add_fill(self, fill: Fill) -> None:
        """Record a fill event.
        
        Args:
            fill: Fill to record
        """
        self._fills.append(fill)
        
        if fill.order_id not in self._fills_by_order:
            self._fills_by_order[fill.order_id] = []
        self._fills_by_order[fill.order_id].append(fill)
        
        # Update order state
        order = self._orders.get(fill.order_id)
        if order:
            total_filled = sum(f.quantity for f in self._fills_by_order[fill.order_id])
            order.filled_quantity = total_filled
            
            if total_filled >= order.quantity:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
    
    def _submit_with_retry(self, order: Order) -> Order:
        """Submit order with retry logic.
        
        Args:
            order: Order to submit
            
        Returns:
            Submitted order
        """
        retries = 0
        last_error = None
        
        while retries <= self._config.max_retries:
            try:
                result = self._broker.submit_order(order)
                
                # If rejected due to transient error, retry
                if result.status == OrderStatus.REJECTED:
                    reject_reason = result.reject_reason or ""
                    if self._is_transient_error(reject_reason):
                        retries += 1
                        self._retry_counts[order.order_id] = retries
                        continue
                
                # Record fills if order was filled
                if result.status == OrderStatus.FILLED and result.avg_fill_price:
                    fill = Fill(
                        order_id=result.order_id,
                        symbol=result.symbol,
                        side=result.side,
                        quantity=result.filled_quantity,
                        price=result.avg_fill_price,
                        filled_at=datetime.now(timezone.utc),
                    )
                    self.add_fill(fill)
                
                return result
                
            except Exception as e:
                last_error = e
                retries += 1
                self._retry_counts[order.order_id] = retries
                logger.warning(f"Order {order.order_id} submission failed (attempt {retries}): {e}")
        
        # All retries exhausted
        order.status = OrderStatus.REJECTED
        order.reject_reason = f"Max retries exceeded: {last_error}"
        return order
    
    def _is_transient_error(self, error_msg: str) -> bool:
        """Check if error is transient and worth retrying.
        
        Args:
            error_msg: Error message
            
        Returns:
            True if error is transient
        """
        transient_patterns = [
            "timeout",
            "connection",
            "temporarily unavailable",
            "rate limit",
            "try again",
        ]
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in transient_patterns)
