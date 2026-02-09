"""Alert management for trading system.

Provides configurable alerting with:
- Multiple alert levels (info, warning, critical)
- Threshold-based triggers
- Webhook and email delivery
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Callable, Dict, List, Optional
from uuid import UUID, uuid4
import logging
import json

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity level."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Type of alert trigger."""
    DRAWDOWN = "drawdown"
    POSITION_SIZE = "position_size"
    EXPOSURE = "exposure"
    PNL = "pnl"
    VOLATILITY = "volatility"
    FILL_RATE = "fill_rate"
    CUSTOM = "custom"


@dataclass
class Alert:
    """An alert instance.
    
    Attributes:
        alert_id: Unique identifier
        alert_type: Type of alert
        level: Severity level
        message: Human-readable description
        timestamp: When alert was triggered
        metric_value: Current metric value
        threshold: Threshold that was breached
        acknowledged: Whether alert has been acknowledged
    """
    alert_id: UUID = field(default_factory=uuid4)
    alert_type: AlertType = AlertType.CUSTOM
    level: AlertLevel = AlertLevel.INFO
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    acknowledged: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class AlertConfig:
    """Configuration for an alert rule.
    
    Attributes:
        name: Rule name
        alert_type: Type of condition
        level: Severity level
        threshold: Trigger threshold
        comparison: 'above' or 'below'
        cooldown_minutes: Minimum time between alerts
        enabled: Whether rule is active
    """
    name: str
    alert_type: AlertType
    level: AlertLevel
    threshold: float
    comparison: str = "above"  # "above" or "below"
    cooldown_minutes: int = 15
    enabled: bool = True


class AlertManager:
    """Manages alerting for the trading system.
    
    Features:
    - Configure alert rules with thresholds
    - Cooldown to prevent alert spam
    - Multiple delivery methods (webhook, email, callback)
    - Alert history tracking
    
    Example:
        manager = AlertManager()
        manager.add_rule(AlertConfig(
            name="big_drawdown",
            alert_type=AlertType.DRAWDOWN,
            level=AlertLevel.CRITICAL,
            threshold=-0.10,  # 10% drawdown
            comparison="below",
        ))
        manager.check_metric(AlertType.DRAWDOWN, -0.12)  # Triggers alert
    """
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        email_config: Optional[Dict] = None,
    ) -> None:
        """Initialize alert manager.
        
        Args:
            webhook_url: URL for webhook notifications
            email_config: Email configuration dict
        """
        self._webhook_url = webhook_url
        self._email_config = email_config
        
        # Alert rules
        self._rules: Dict[str, AlertConfig] = {}
        
        # Alert history
        self._alerts: List[Alert] = []
        self._active_alerts: Dict[str, Alert] = {}
        
        # Last trigger times for cooldown
        self._last_triggered: Dict[str, datetime] = {}
        
        # Custom handlers
        self._handlers: List[Callable[[Alert], None]] = []
    
    def add_rule(self, config: AlertConfig) -> None:
        """Add an alert rule.
        
        Args:
            config: Alert configuration
        """
        self._rules[config.name] = config
        logger.info(f"Added alert rule: {config.name}")
    
    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule.
        
        Args:
            name: Rule name
            
        Returns:
            True if removed
        """
        if name in self._rules:
            del self._rules[name]
            return True
        return False
    
    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a custom alert handler.
        
        Args:
            handler: Callback function that receives Alert
        """
        self._handlers.append(handler)
    
    def check_metric(
        self, 
        alert_type: AlertType, 
        value: float,
        metadata: Optional[Dict] = None,
    ) -> List[Alert]:
        """Check a metric against all applicable rules.
        
        Args:
            alert_type: Type of metric
            value: Current metric value
            metadata: Optional additional data
            
        Returns:
            List of triggered alerts
        """
        triggered = []
        
        for name, rule in self._rules.items():
            if not rule.enabled:
                continue
            
            if rule.alert_type != alert_type:
                continue
            
            if not self._should_trigger(rule, value):
                continue
            
            if not self._check_cooldown(name, rule.cooldown_minutes):
                continue
            
            # Create and deliver alert
            alert = self._create_alert(rule, value, metadata)
            self._deliver_alert(alert)
            triggered.append(alert)
            
            # Update tracking
            self._last_triggered[name] = datetime.now(timezone.utc)
            self._alerts.append(alert)
            self._active_alerts[name] = alert
        
        return triggered
    
    def trigger_custom(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        metadata: Optional[Dict] = None,
    ) -> Alert:
        """Trigger a custom alert.
        
        Args:
            message: Alert message
            level: Severity level
            metadata: Optional data
            
        Returns:
            Created alert
        """
        alert = Alert(
            alert_type=AlertType.CUSTOM,
            level=level,
            message=message,
            metadata=metadata or {},
        )
        
        self._deliver_alert(alert)
        self._alerts.append(alert)
        
        return alert
    
    def acknowledge(self, alert_id: UUID) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert to acknowledge
            
        Returns:
            True if found and acknowledged
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts."""
        return [a for a in self._alerts if not a.acknowledged]
    
    def get_alert_history(
        self, 
        since: Optional[datetime] = None,
        level: Optional[AlertLevel] = None,
    ) -> List[Alert]:
        """Get alert history with optional filters.
        
        Args:
            since: Only alerts after this time
            level: Only alerts of this level
            
        Returns:
            Filtered alert list
        """
        alerts = self._alerts
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts
    
    def _should_trigger(self, rule: AlertConfig, value: float) -> bool:
        """Check if value breaches threshold."""
        if rule.comparison == "above":
            return value > rule.threshold
        elif rule.comparison == "below":
            return value < rule.threshold
        return False
    
    def _check_cooldown(self, rule_name: str, cooldown_minutes: int) -> bool:
        """Check if rule is past cooldown period."""
        last = self._last_triggered.get(rule_name)
        if last is None:
            return True
        
        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 60
        return elapsed >= cooldown_minutes
    
    def _create_alert(
        self, 
        rule: AlertConfig, 
        value: float,
        metadata: Optional[Dict],
    ) -> Alert:
        """Create an alert from a rule breach."""
        comparison_word = "exceeded" if rule.comparison == "above" else "dropped below"
        message = f"{rule.name}: {rule.alert_type.value} {comparison_word} {rule.threshold} (current: {value:.4f})"
        
        return Alert(
            alert_type=rule.alert_type,
            level=rule.level,
            message=message,
            metric_value=value,
            threshold=rule.threshold,
            metadata=metadata or {},
        )
    
    def _deliver_alert(self, alert: Alert) -> None:
        """Deliver alert via configured channels."""
        # Log
        log_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }.get(alert.level, logger.info)
        
        log_method(f"ALERT [{alert.level.value}]: {alert.message}")
        
        # Webhook
        if self._webhook_url:
            self._send_webhook(alert)
        
        # Email
        if self._email_config:
            self._send_email(alert)
        
        # Custom handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def _send_webhook(self, alert: Alert) -> None:
        """Send alert via webhook."""
        try:
            import urllib.request
            
            payload = {
                "alert_id": str(alert.alert_id),
                "type": alert.alert_type.value,
                "level": alert.level.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "value": alert.metric_value,
                "threshold": alert.threshold,
            }
            
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self._webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'},
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status != 200:
                    logger.warning(f"Webhook returned {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
    
    def _send_email(self, alert: Alert) -> None:
        """Send alert via email."""
        # Email sending would require smtplib configuration
        # This is a placeholder for the implementation
        logger.debug(f"Email alert: {alert.message}")


# Default alert rules
DEFAULT_RULES = [
    AlertConfig(
        name="drawdown_warning",
        alert_type=AlertType.DRAWDOWN,
        level=AlertLevel.WARNING,
        threshold=-0.05,
        comparison="below",
    ),
    AlertConfig(
        name="drawdown_critical",
        alert_type=AlertType.DRAWDOWN,
        level=AlertLevel.CRITICAL,
        threshold=-0.10,
        comparison="below",
    ),
    AlertConfig(
        name="high_exposure",
        alert_type=AlertType.EXPOSURE,
        level=AlertLevel.WARNING,
        threshold=2.0,
        comparison="above",
    ),
    AlertConfig(
        name="large_position",
        alert_type=AlertType.POSITION_SIZE,
        level=AlertLevel.WARNING,
        threshold=0.15,
        comparison="above",
    ),
]
