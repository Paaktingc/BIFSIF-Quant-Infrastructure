"""Tests for the NLP entity extractor."""

from __future__ import annotations

import pytest

from bifs_quant_engine.data.news_monitor import NewsItem
from bifs_quant_engine.nlp.entity_extractor import EntityExtractor, ExtractionResult


@pytest.fixture
def extractor() -> EntityExtractor:
    """Create an extractor with models disabled (rule-based only)."""
    return EntityExtractor(lazy_load=True)


class TestInstitutionExtraction:
    def test_ecb_detected(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="ECB decides to maintain interest rates",
            summary="The European Central Bank held rates steady.",
        )
        result = extractor.extract(item)
        assert "ECB" in result.entities

    def test_boj_detected(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="Bank of Japan raises interest rates by 25 basis points",
        )
        result = extractor.extract(item)
        assert "BOJ" in result.entities

    def test_fed_detected(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="Federal Reserve holds rates at current level",
        )
        result = extractor.extract(item)
        assert "Fed" in result.entities

    def test_opec_detected(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="OPEC+ agrees to cut oil production by 1 million barrels",
        )
        result = extractor.extract(item)
        assert "OPEC" in result.entities

    def test_multiple_institutions(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="ECB follows Fed in rate decision",
        )
        result = extractor.extract(item)
        assert "ECB" in result.entities
        assert "Fed" in result.entities

    def test_no_institution(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="Weather forecast: sunny skies expected",
        )
        result = extractor.extract(item)
        assert len(result.entities) == 0


class TestEventExtraction:
    def test_rate_hike(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="BOJ decided to raise interest rates by 0.25",
        )
        result = extractor.extract(item)
        assert result.event_type == "rate_hike"
        assert result.key_facts.get("amount") == "0.25"

    def test_rate_cut(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="ECB cuts rates by 0.50 percentage points",
        )
        result = extractor.extract(item)
        assert result.event_type == "rate_cut"

    def test_rate_hold(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="Fed decided to maintain rates unchanged at 5.25",
        )
        result = extractor.extract(item)
        assert result.event_type == "rate_hold"

    def test_vote_result(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="Parliament approved the bill by 320 to 150",
        )
        result = extractor.extract(item)
        assert result.event_type == "vote_result"
        assert result.key_facts.get("votes_for") == "320"
        assert result.key_facts.get("votes_against") == "150"

    def test_vote_rejected(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="Senate rejected the proposal by 55 to 45",
        )
        result = extractor.extract(item)
        assert result.event_type == "vote_rejected"

    def test_unknown_event(self, extractor: EntityExtractor) -> None:
        item = NewsItem(
            source="test",
            title="New trade agreement signed between EU and Japan",
        )
        result = extractor.extract(item)
        assert result.event_type == "unknown"


class TestExtractionResult:
    def test_result_has_news_item(self, extractor: EntityExtractor) -> None:
        item = NewsItem(source="test", title="Test headline")
        result = extractor.extract(item)
        assert result.news_item is item

    def test_default_sentiment_neutral(self, extractor: EntityExtractor) -> None:
        # Without transformers installed, sentiment defaults to neutral
        item = NewsItem(source="test", title="Some news")
        result = extractor.extract(item)
        assert result.sentiment == "neutral"
        assert result.sentiment_score == 0.0
