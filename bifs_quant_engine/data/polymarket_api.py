"""Polymarket data API client — data types and HTTP client."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional


@dataclass
class PolymarketToken:
    """A YES or NO token for a Polymarket binary market."""
    token_id: str
    outcome: str   # "Yes" or "No"
    price: Decimal = Decimal("0.5")


@dataclass
class PolymarketMarket:
    """A Polymarket prediction market."""
    condition_id: str
    question: str
    tokens: List[PolymarketToken] = field(default_factory=list)
    is_active: bool = True
    end_date_iso: Optional[str] = None

    def get_token(self, outcome: str) -> Optional[PolymarketToken]:
        """Return the token matching the given outcome ('Yes' or 'No')."""
        for t in self.tokens:
            if t.outcome.lower() == outcome.lower():
                return t
        return None

    @property
    def yes_price(self) -> Decimal:
        tok = self.get_token("Yes")
        return tok.price if tok else Decimal("0.5")

    @property
    def no_price(self) -> Decimal:
        tok = self.get_token("No")
        return tok.price if tok else Decimal("0.5")


class PolymarketDataAPI:
    """
    Thin client for the Polymarket public API.
    Provides market data fetching; real implementation would use HTTP.
    """

    def __init__(self, base_url: str = "https://clob.polymarket.com") -> None:
        self.base_url = base_url

    def get_active_markets(self, limit: int = 100) -> List[PolymarketMarket]:
        """Fetch a list of active binary markets. Stub for testing."""
        return []

    def get_market(self, condition_id: str) -> Optional[PolymarketMarket]:
        """Fetch a single market by condition ID."""
        return None
