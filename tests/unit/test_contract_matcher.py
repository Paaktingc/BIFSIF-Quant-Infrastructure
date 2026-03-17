"""Tests for the Polymarket contract matcher."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pytest

from bifs_quant_engine.data.news_monitor import NewsItem
from bifs_quant_engine.data.polymarket_api import (
    PolymarketDataAPI,
    PolymarketMarket,
    PolymarketToken,
)
from bifs_quant_engine.nlp.contract_matcher import ContractMatcher
from bifs_quant_engine.nlp.entity_extractor import ExtractionResult


@pytest.fixture
def mock_api() -> MagicMock:
    return MagicMock(spec=PolymarketDataAPI)


@pytest.fixture
def matcher(mock_api: MagicMock) -> ContractMatcher:
    return ContractMatcher(mock_api, similarity_threshold=0.5)


def _make_market(
    cid: str, question: str, yes_price: float = 0.50
) -> PolymarketMarket:
    return PolymarketMarket(
        condition_id=cid,
        question=question,
        tokens=[
            PolymarketToken(token_id=f"{cid}_yes", outcome="Yes", price=Decimal(str(yes_price))),
            PolymarketToken(token_id=f"{cid}_no", outcome="No", price=Decimal(str(1 - yes_price))),
        ],
    )


def _make_extraction(
    sentiment: str = "positive",
    score: float = 0.9,
    embedding: np.ndarray | None = None,
) -> ExtractionResult:
    if embedding is None:
        embedding = np.random.randn(384).astype(np.float32)
    return ExtractionResult(
        news_item=NewsItem(source="test", title="Test"),
        sentiment=sentiment,
        sentiment_score=score,
        embedding=embedding,
        event_type="rate_cut",
    )


class TestCosineSimlarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert ContractMatcher._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert ContractMatcher._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert ContractMatcher._cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert ContractMatcher._cosine_similarity(a, b) == 0.0


class TestContractMatching:
    def test_empty_cache_returns_no_matches(self, matcher: ContractMatcher) -> None:
        extraction = _make_extraction()
        matches = matcher.find_matches(extraction)
        assert matches == []

    def test_no_embedding_returns_no_matches(self, matcher: ContractMatcher) -> None:
        extraction = _make_extraction()
        extraction.embedding = None
        matches = matcher.find_matches(extraction)
        assert matches == []

    def test_matches_similar_contract(self, matcher: ContractMatcher) -> None:
        # Manually populate cache with matching embedding
        market = _make_market("c1", "Will ECB cut rates?", 0.30)
        embedding = np.random.randn(384).astype(np.float32)
        matcher._cache["c1"] = (market, embedding)

        # Use same embedding for high similarity
        extraction = _make_extraction(embedding=embedding)
        matches = matcher.find_matches(extraction)

        assert len(matches) == 1
        assert matches[0].market.condition_id == "c1"
        assert matches[0].similarity_score == pytest.approx(1.0)

    def test_filters_below_threshold(self, matcher: ContractMatcher) -> None:
        market = _make_market("c1", "Will ECB cut rates?")
        matcher._cache["c1"] = (market, np.array([1.0, 0.0, 0.0]))

        # Orthogonal vector -> similarity = 0
        extraction = _make_extraction(embedding=np.array([0.0, 1.0, 0.0]))
        matches = matcher.find_matches(extraction)

        assert len(matches) == 0

    def test_top_k_limiting(self, matcher: ContractMatcher) -> None:
        base = np.random.randn(10).astype(np.float32)

        for i in range(10):
            market = _make_market(f"c{i}", f"Market {i}")
            # Slight variations so all are similar
            emb = base + np.random.randn(10).astype(np.float32) * 0.01
            matcher._cache[f"c{i}"] = (market, emb)

        extraction = _make_extraction(embedding=base)
        matches = matcher.find_matches(extraction, top_k=3)
        assert len(matches) <= 3

    def test_match_has_edge(self, matcher: ContractMatcher) -> None:
        market = _make_market("c1", "Will ECB cut rates?", yes_price=0.20)
        emb = np.array([1.0, 0.0])
        matcher._cache["c1"] = (market, emb)

        extraction = _make_extraction(
            sentiment="positive", score=0.9, embedding=np.array([1.0, 0.0])
        )
        matches = matcher.find_matches(extraction)
        assert len(matches) == 1
        # Positive sentiment -> "yes" direction, so edge = fair_value - 0.20
        assert float(matches[0].edge) > 0


class TestKeywordMatching:
    def test_match_by_keywords(self, matcher: ContractMatcher) -> None:
        m1 = _make_market("c1", "Will ECB cut interest rates in March?")
        m2 = _make_market("c2", "Will Bitcoin reach $100k?")
        matcher._cache["c1"] = (m1, np.zeros(2))
        matcher._cache["c2"] = (m2, np.zeros(2))

        results = matcher.match_by_keywords(["ECB", "rates"])
        assert len(results) == 1
        assert results[0].condition_id == "c1"

    def test_case_insensitive(self, matcher: ContractMatcher) -> None:
        m = _make_market("c1", "Will the ECB cut rates?")
        matcher._cache["c1"] = (m, np.zeros(2))

        results = matcher.match_by_keywords(["ecb"])
        assert len(results) == 1


class TestCachedMarketCount:
    def test_empty_cache(self, matcher: ContractMatcher) -> None:
        assert matcher.cached_market_count == 0

    def test_populated_cache(self, matcher: ContractMatcher) -> None:
        m = _make_market("c1", "Test?")
        matcher._cache["c1"] = (m, np.zeros(2))
        assert matcher.cached_market_count == 1
