"""News monitoring data types and feed source definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class FeedSource:
    """Configuration for a single news RSS/API feed source."""
    name: str
    url: str
    feed_type: str = "rss"  # "rss" | "api"
    api_key: Optional[str] = None


@dataclass
class NewsItem:
    """A single news item retrieved from a feed source."""
    source: str
    title: str
    summary: Optional[str] = None
    url: Optional[str] = None
    published_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: Optional[dict] = None

    def full_text(self) -> str:
        """Return concatenated title and summary for NLP processing."""
        parts = [self.title]
        if self.summary:
            parts.append(self.summary)
        return " ".join(parts)

    @property
    def raw_text(self) -> str:
        """Alias for full_text() for compatibility with entity_extractor."""
        return self.full_text()


class NewsMonitor:
    """
    Manages multiple news feed sources and polls them for new items.

    In production this fetches RSS/API feeds. The stub implementation
    returns an empty list and is safe to use in tests without network access.
    """

    def __init__(self, sources: List[FeedSource]) -> None:
        self._sources = sources
        self._seen_urls: set = set()

    @property
    def sources(self) -> List[FeedSource]:
        return list(self._sources)

    def add_source(self, source: FeedSource) -> None:
        self._sources.append(source)

    def poll_all(self) -> List[NewsItem]:
        """
        Poll all configured feed sources and return new NewsItems.
        Returns an empty list when no network/RSS library is available.
        """
        items: List[NewsItem] = []
        for source in self._sources:
            try:
                new = self._fetch_source(source)
                items.extend(new)
            except Exception:
                pass
        return items

    def _fetch_source(self, source: FeedSource) -> List[NewsItem]:
        """Fetch items from a single source. Override in subclasses."""
        return []

