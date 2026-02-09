"""Audit logging for trade decisions and system events.

Provides immutable logging of:
- Order submissions and fills
- Risk manager decisions
- Strategy signals
- System events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Type of audit event."""
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    RISK_DECISION = "risk_decision"
    STRATEGY_SIGNAL = "strategy_signal"
    POSITION_CHANGE = "position_change"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    ALERT_TRIGGERED = "alert_triggered"


@dataclass
class AuditEntry:
    """Immutable audit log entry.
    
    Attributes:
        entry_id: Unique identifier
        event_type: Type of event
        timestamp: When event occurred
        data: Event-specific data
        user_id: User who triggered event (if applicable)
        strategy_id: Related strategy (if applicable)
        order_id: Related order (if applicable)
        hash: Hash of entry for integrity verification
        prev_hash: Hash of previous entry (blockchain-style chain)
    """
    entry_id: UUID = field(default_factory=uuid4)
    event_type: AuditEventType = AuditEventType.ORDER_SUBMITTED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    strategy_id: Optional[str] = None
    order_id: Optional[UUID] = None
    hash: str = ""
    prev_hash: str = ""
    
    def calculate_hash(self, prev_hash: str = "") -> str:
        """Calculate entry hash for integrity."""
        content = (
            str(self.entry_id) +
            self.event_type.value +
            self.timestamp.isoformat() +
            json.dumps(self.data, sort_keys=True, default=str) +
            (self.user_id or "") +
            (self.strategy_id or "") +
            (str(self.order_id) if self.order_id else "") +
            prev_hash
        )
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": str(self.entry_id),
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "user_id": self.user_id,
            "strategy_id": self.strategy_id,
            "order_id": str(self.order_id) if self.order_id else None,
            "hash": self.hash,
            "prev_hash": self.prev_hash,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        return cls(
            entry_id=UUID(d["entry_id"]),
            event_type=AuditEventType(d["event_type"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            data=d.get("data", {}),
            user_id=d.get("user_id"),
            strategy_id=d.get("strategy_id"),
            order_id=UUID(d["order_id"]) if d.get("order_id") else None,
            hash=d.get("hash", ""),
            prev_hash=d.get("prev_hash", ""),
        )


class AuditLogger:
    """Immutable audit logger with integrity verification.
    
    Features:
    - Hash-chained entries for tamper detection
    - File and database persistence
    - Query by event type, time range, order
    
    Example:
        logger = AuditLogger(log_dir=Path("./audit_logs"))
        logger.log_order_submitted(order, strategy_id="momentum")
        logger.log_risk_decision(
            order_id=order.order_id,
            decision="approved",
            reason="Within limits",
        )
    """
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        persist_to_db: bool = False,
    ) -> None:
        """Initialize audit logger.
        
        Args:
            log_dir: Directory for log files
            persist_to_db: Also persist to database
        """
        self._log_dir = log_dir
        self._persist_to_db = persist_to_db
        
        # In-memory log
        self._entries: List[AuditEntry] = []
        self._last_hash = ""
        
        # Create log directory
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log system start
        self._add_entry(AuditEntry(
            event_type=AuditEventType.SYSTEM_START,
            data={"message": "Audit logger initialized"},
        ))
    
    def log_order_submitted(
        self,
        order: "Order",
        strategy_id: Optional[str] = None,
    ) -> AuditEntry:
        """Log order submission.
        
        Args:
            order: Submitted order
            strategy_id: Originating strategy
            
        Returns:
            Created audit entry
        """
        from bifs_quant_engine.core.models import Order
        
        entry = AuditEntry(
            event_type=AuditEventType.ORDER_SUBMITTED,
            order_id=order.order_id,
            strategy_id=strategy_id,
            data={
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "order_type": order.order_type.value,
                "limit_price": str(order.limit_price) if order.limit_price else None,
            },
        )
        return self._add_entry(entry)
    
    def log_order_filled(
        self,
        order: "Order",
        fill_price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> AuditEntry:
        """Log order fill.
        
        Args:
            order: Filled order
            fill_price: Execution price
            commission: Commission charged
            
        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            event_type=AuditEventType.ORDER_FILLED,
            order_id=order.order_id,
            data={
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.filled_quantity,
                "fill_price": str(fill_price),
                "commission": str(commission),
            },
        )
        return self._add_entry(entry)
    
    def log_order_rejected(
        self,
        order: "Order",
        reason: str,
        rejected_by: str = "risk_manager",
    ) -> AuditEntry:
        """Log order rejection.
        
        Args:
            order: Rejected order
            reason: Rejection reason
            rejected_by: Component that rejected
            
        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            event_type=AuditEventType.ORDER_REJECTED,
            order_id=order.order_id,
            data={
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "reason": reason,
                "rejected_by": rejected_by,
            },
        )
        return self._add_entry(entry)
    
    def log_risk_decision(
        self,
        order_id: UUID,
        decision: str,
        reason: str,
        risk_metrics: Optional[Dict] = None,
    ) -> AuditEntry:
        """Log risk manager decision.
        
        Args:
            order_id: Related order
            decision: Decision made (approved, modified, rejected)
            reason: Reason for decision
            risk_metrics: Relevant risk metrics at time of decision
            
        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            event_type=AuditEventType.RISK_DECISION,
            order_id=order_id,
            data={
                "decision": decision,
                "reason": reason,
                "risk_metrics": risk_metrics or {},
            },
        )
        return self._add_entry(entry)
    
    def log_strategy_signal(
        self,
        strategy_id: str,
        signal_type: str,
        symbol: str,
        weight: float,
        metadata: Optional[Dict] = None,
    ) -> AuditEntry:
        """Log strategy signal generation.
        
        Args:
            strategy_id: Strategy identifier
            signal_type: Type of signal
            symbol: Target symbol
            weight: Signal weight
            metadata: Additional signal data
            
        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            event_type=AuditEventType.STRATEGY_SIGNAL,
            strategy_id=strategy_id,
            data={
                "signal_type": signal_type,
                "symbol": symbol,
                "weight": weight,
                **({} if metadata is None else metadata),
            },
        )
        return self._add_entry(entry)
    
    def log_position_change(
        self,
        symbol: str,
        old_quantity: int,
        new_quantity: int,
        reason: str = "",
    ) -> AuditEntry:
        """Log position change.
        
        Args:
            symbol: Symbol changed
            old_quantity: Previous quantity
            new_quantity: New quantity
            reason: Reason for change
            
        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            event_type=AuditEventType.POSITION_CHANGE,
            data={
                "symbol": symbol,
                "old_quantity": old_quantity,
                "new_quantity": new_quantity,
                "delta": new_quantity - old_quantity,
                "reason": reason,
            },
        )
        return self._add_entry(entry)
    
    def log_alert(
        self,
        alert_type: str,
        level: str,
        message: str,
        metric_value: Optional[float] = None,
    ) -> AuditEntry:
        """Log alert triggered.
        
        Args:
            alert_type: Type of alert
            level: Severity level
            message: Alert message
            metric_value: Value that triggered alert
            
        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            event_type=AuditEventType.ALERT_TRIGGERED,
            data={
                "alert_type": alert_type,
                "level": level,
                "message": message,
                "metric_value": metric_value,
            },
        )
        return self._add_entry(entry)
    
    def get_entries(
        self,
        event_type: Optional[AuditEventType] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        order_id: Optional[UUID] = None,
        strategy_id: Optional[str] = None,
    ) -> List[AuditEntry]:
        """Query audit entries.
        
        Args:
            event_type: Filter by event type
            since: Filter by start time
            until: Filter by end time
            order_id: Filter by order
            strategy_id: Filter by strategy
            
        Returns:
            Matching entries
        """
        entries = self._entries
        
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        
        if until:
            entries = [e for e in entries if e.timestamp <= until]
        
        if order_id:
            entries = [e for e in entries if e.order_id == order_id]
        
        if strategy_id:
            entries = [e for e in entries if e.strategy_id == strategy_id]
        
        return entries
    
    def verify_integrity(self) -> bool:
        """Verify hash chain integrity.
        
        Returns:
            True if all hashes valid
        """
        prev_hash = ""
        
        for entry in self._entries:
            calculated = entry.calculate_hash(prev_hash)
            if entry.hash != calculated:
                logger.error(f"Integrity violation at {entry.entry_id}")
                return False
            prev_hash = entry.hash
        
        return True
    
    def export_to_file(self, filepath: Path) -> None:
        """Export log to JSON file.
        
        Args:
            filepath: Output file path
        """
        data = [e.to_dict() for e in self._entries]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(data)} entries to {filepath}")
    
    def _add_entry(self, entry: AuditEntry) -> AuditEntry:
        """Add entry with hash chain."""
        entry.prev_hash = self._last_hash
        entry.hash = entry.calculate_hash(self._last_hash)
        self._last_hash = entry.hash
        
        self._entries.append(entry)
        
        # Persist to file
        if self._log_dir:
            self._persist_entry(entry)
        
        return entry
    
    def _persist_entry(self, entry: AuditEntry) -> None:
        """Persist single entry to daily log file."""
        date_str = entry.timestamp.strftime("%Y-%m-%d")
        filepath = self._log_dir / f"audit_{date_str}.jsonl"
        
        with open(filepath, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
