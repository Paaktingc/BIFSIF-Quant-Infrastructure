"""Compliance reporting and export.

Generates regulatory-compliant reports:
- Trade logs (CSV, JSON)
- Position history
- Exposure reports
- Daily summaries
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import json

from bifs_quant_engine.compliance.audit_logger import AuditLogger, AuditEntry, AuditEventType


class ReportFormat(Enum):
    """Report output format."""
    CSV = "csv"
    JSON = "json"
    HTML = "html"


@dataclass
class TradeRecord:
    """Standardized trade record for reporting.
    
    Attributes:
        trade_date: Date of trade execution
        trade_time: Time of execution
        symbol: Traded symbol
        side: Buy/Sell/Short/Cover
        quantity: Number of shares
        price: Execution price
        commission: Commission paid
        notional: Trade value (price * quantity)
        strategy: Strategy that generated trade
        order_id: Internal order ID
    """
    trade_date: str
    trade_time: str
    symbol: str
    side: str
    quantity: int
    price: Decimal
    commission: Decimal
    notional: Decimal
    strategy: str
    order_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_date": self.trade_date,
            "trade_time": self.trade_time,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": str(self.price),
            "commission": str(self.commission),
            "notional": str(self.notional),
            "strategy": self.strategy,
            "order_id": self.order_id,
        }


class ComplianceReporter:
    """Generates compliance reports from audit log.
    
    Features:
    - Trade log export (CSV, JSON)
    - Position history reports
    - Daily summary reports
    - Customizable date ranges
    
    Example:
        reporter = ComplianceReporter(audit_logger)
        reporter.generate_trade_log(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
            output=Path("./reports/trades_2024.csv"),
            format=ReportFormat.CSV,
        )
    """
    
    def __init__(self, audit_logger: AuditLogger) -> None:
        """Initialize reporter.
        
        Args:
            audit_logger: Source of audit data
        """
        self._logger = audit_logger
    
    def generate_trade_log(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        strategy: Optional[str] = None,
        output: Optional[Path] = None,
        format: ReportFormat = ReportFormat.CSV,
    ) -> List[TradeRecord]:
        """Generate trade log report.
        
        Args:
            start: Start date filter
            end: End date filter
            strategy: Strategy filter
            output: Output file path
            format: Output format
            
        Returns:
            List of trade records
        """
        # Get fill entries
        entries = self._logger.get_entries(
            event_type=AuditEventType.ORDER_FILLED,
            since=start,
            until=end,
            strategy_id=strategy,
        )
        
        # Convert to trade records
        records = []
        for entry in entries:
            data = entry.data
            
            price = Decimal(data.get("fill_price", "0"))
            quantity = int(data.get("quantity", 0))
            commission = Decimal(data.get("commission", "0"))
            
            record = TradeRecord(
                trade_date=entry.timestamp.strftime("%Y-%m-%d"),
                trade_time=entry.timestamp.strftime("%H:%M:%S"),
                symbol=data.get("symbol", ""),
                side=data.get("side", ""),
                quantity=quantity,
                price=price,
                commission=commission,
                notional=price * quantity,
                strategy=entry.strategy_id or "",
                order_id=str(entry.order_id) if entry.order_id else "",
            )
            records.append(record)
        
        # Export if output specified
        if output:
            self._export(records, output, format)
        
        return records
    
    def generate_position_history(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        symbol: Optional[str] = None,
        output: Optional[Path] = None,
        format: ReportFormat = ReportFormat.CSV,
    ) -> List[Dict]:
        """Generate position change history.
        
        Args:
            start: Start date filter
            end: End date filter
            symbol: Symbol filter
            output: Output file path
            format: Output format
            
        Returns:
            List of position change records
        """
        entries = self._logger.get_entries(
            event_type=AuditEventType.POSITION_CHANGE,
            since=start,
            until=end,
        )
        
        records = []
        for entry in entries:
            data = entry.data
            
            if symbol and data.get("symbol") != symbol:
                continue
            
            records.append({
                "date": entry.timestamp.strftime("%Y-%m-%d"),
                "time": entry.timestamp.strftime("%H:%M:%S"),
                "symbol": data.get("symbol", ""),
                "old_quantity": data.get("old_quantity", 0),
                "new_quantity": data.get("new_quantity", 0),
                "delta": data.get("delta", 0),
                "reason": data.get("reason", ""),
            })
        
        if output:
            self._export_dicts(records, output, format)
        
        return records
    
    def generate_risk_decision_log(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        output: Optional[Path] = None,
        format: ReportFormat = ReportFormat.CSV,
    ) -> List[Dict]:
        """Generate risk decision audit log.
        
        Args:
            start: Start date filter
            end: End date filter
            output: Output file path
            format: Output format
            
        Returns:
            List of risk decision records
        """
        entries = self._logger.get_entries(
            event_type=AuditEventType.RISK_DECISION,
            since=start,
            until=end,
        )
        
        records = []
        for entry in entries:
            data = entry.data
            records.append({
                "timestamp": entry.timestamp.isoformat(),
                "order_id": str(entry.order_id) if entry.order_id else "",
                "decision": data.get("decision", ""),
                "reason": data.get("reason", ""),
                "risk_metrics": json.dumps(data.get("risk_metrics", {})),
            })
        
        if output:
            self._export_dicts(records, output, format)
        
        return records
    
    def generate_daily_summary(
        self,
        date: datetime,
        output: Optional[Path] = None,
        format: ReportFormat = ReportFormat.JSON,
    ) -> Dict:
        """Generate daily trading summary.
        
        Args:
            date: Date to summarize
            output: Output file path
            format: Output format
            
        Returns:
            Summary dictionary
        """
        start = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(date.year, date.month, date.day, 23, 59, 59, tzinfo=timezone.utc)
        
        # Get all entries for the day
        fills = self._logger.get_entries(
            event_type=AuditEventType.ORDER_FILLED,
            since=start,
            until=end,
        )
        
        rejections = self._logger.get_entries(
            event_type=AuditEventType.ORDER_REJECTED,
            since=start,
            until=end,
        )
        
        # Calculate statistics
        total_trades = len(fills)
        total_rejected = len(rejections)
        total_volume = sum(int(f.data.get("quantity", 0)) for f in fills)
        total_notional = sum(
            Decimal(f.data.get("fill_price", "0")) * int(f.data.get("quantity", 0))
            for f in fills
        )
        total_commissions = sum(
            Decimal(f.data.get("commission", "0")) for f in fills
        )
        
        # Unique symbols traded
        symbols = set(f.data.get("symbol") for f in fills)
        
        summary = {
            "date": date.strftime("%Y-%m-%d"),
            "total_trades": total_trades,
            "total_rejected": total_rejected,
            "rejection_rate": total_rejected / (total_trades + total_rejected) if (total_trades + total_rejected) > 0 else 0,
            "total_volume": total_volume,
            "total_notional": str(total_notional),
            "total_commissions": str(total_commissions),
            "unique_symbols": len(symbols),
            "symbols_traded": list(symbols),
        }
        
        if output:
            with open(output, 'w') as f:
                json.dump(summary, f, indent=2)
        
        return summary
    
    def _export(
        self, 
        records: List[TradeRecord], 
        output: Path,
        format: ReportFormat,
    ) -> None:
        """Export trade records to file."""
        if format == ReportFormat.CSV:
            self._export_csv(records, output)
        elif format == ReportFormat.JSON:
            self._export_json(records, output)
        elif format == ReportFormat.HTML:
            self._export_html(records, output)
    
    def _export_csv(self, records: List[TradeRecord], output: Path) -> None:
        """Export to CSV."""
        if not records:
            return
        
        fieldnames = list(records[0].to_dict().keys())
        
        with open(output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_dict())
    
    def _export_json(self, records: List[TradeRecord], output: Path) -> None:
        """Export to JSON."""
        data = [r.to_dict() for r in records]
        
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_html(self, records: List[TradeRecord], output: Path) -> None:
        """Export to HTML table."""
        if not records:
            html = "<html><body><p>No trades</p></body></html>"
        else:
            headers = list(records[0].to_dict().keys())
            
            html = """
<html>
<head>
    <style>
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Trade Log</h1>
    <table>
        <tr>"""
            
            for h in headers:
                html += f"<th>{h}</th>"
            html += "</tr>"
            
            for record in records:
                html += "<tr>"
                for v in record.to_dict().values():
                    html += f"<td>{v}</td>"
                html += "</tr>"
            
            html += """
    </table>
</body>
</html>"""
        
        with open(output, 'w') as f:
            f.write(html)
    
    def _export_dicts(
        self, 
        records: List[Dict], 
        output: Path,
        format: ReportFormat,
    ) -> None:
        """Export dict records to file."""
        if format == ReportFormat.CSV:
            if not records:
                return
            
            fieldnames = list(records[0].keys())
            
            with open(output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
                
        elif format == ReportFormat.JSON:
            with open(output, 'w') as f:
                json.dump(records, f, indent=2)
