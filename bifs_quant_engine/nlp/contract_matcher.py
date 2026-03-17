"""Contract matcher for linking news events to Polymarket markets.

Uses semantic similarity (cosine distance) between news embeddings and
Polymarket contract question embeddings to find matching markets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np

from bifs_quant_engine.data.polymarket_api import PolymarketDataAPI, PolymarketMarket
from bifs_quant_engine.nlp.entity_extractor import ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class ContractMatch:
    """A matched Polymarket contract for a news event.

    Attributes:
        market: The matched Polymarket market.
        similarity_score: Cosine similarity between news and contract.
        implied_direction: Whether the news implies "yes" or "no" outcome.
        current_yes_price: Current YES token price.
        estimated_fair_value: Estimated true probability based on news.
        edge: Difference between fair value and current price.
    """

    market: PolymarketMarket
    similarity_score: float
    implied_direction: str  # "yes" or "no"
    current_yes_price: Decimal = Decimal("0.50")
    estimated_fair_value: Decimal = Decimal("0.50")
    edge: Decimal = Decimal("0")


class ContractMatcher:
    """Matches news extraction results to active Polymarket contracts.

    Maintains an embedding cache of active contract questions and
    computes cosine similarity to find relevant matches.

    Example:
        api = PolymarketDataAPI()
        matcher = ContractMatcher(api)
        matcher.refresh_contract_embeddings()

        matches = matcher.find_matches(extraction_result)
        for m in matches:
            print(f"{m.market.question}: sim={m.similarity_score:.3f}, edge={m.edge}")
    """

    def __init__(
        self,
        polymarket_api: PolymarketDataAPI,
        similarity_threshold: float = 0.65,
    ) -> None:
        """Initialize contract matcher.

        Args:
            polymarket_api: API client for fetching markets.
            similarity_threshold: Minimum cosine similarity for a match.
        """
        self._api = polymarket_api
        self._threshold = similarity_threshold

        # Cache: condition_id -> (market, embedding)
        self._cache: Dict[str, Tuple[PolymarketMarket, np.ndarray]] = {}
        self._embedder = None

    @property
    def cached_market_count(self) -> int:
        """Number of markets with cached embeddings."""
        return len(self._cache)

    def refresh_contract_embeddings(
        self,
        categories: Optional[List[str]] = None,
        limit: int = 200,
    ) -> int:
        """Refresh the embedding cache with active markets.

        Args:
            categories: Filter by categories. None fetches all.
            limit: Maximum markets to fetch.

        Returns:
            Number of markets cached.
        """
        self._ensure_embedder()
        if self._embedder is None:
            logger.warning("Embedder not available; cannot refresh cache")
            return 0

        markets: List[PolymarketMarket] = []
        if categories:
            for cat in categories:
                markets.extend(self._api.get_active_markets(
                    category=cat, limit=limit // len(categories)
                ))
        else:
            markets = self._api.get_active_markets(limit=limit)

        cached = 0
        for market in markets:
            if not market.question:
                continue

            text = f"{market.question} {market.description[:200]}"
            embedding = self._embedder.encode(text, show_progress_bar=False)

            self._cache[market.condition_id] = (market, embedding)
            cached += 1

        logger.info("Cached embeddings for %d markets", cached)
        return cached

    def find_matches(
        self,
        extraction: ExtractionResult,
        top_k: int = 5,
    ) -> List[ContractMatch]:
        """Find Polymarket contracts matching a news extraction.

        Args:
            extraction: NLP extraction result with embedding.
            top_k: Maximum number of matches to return.

        Returns:
            List of ContractMatch sorted by similarity (descending).
        """
        if extraction.embedding is None:
            logger.warning("No embedding in extraction; cannot match")
            return []

        if not self._cache:
            logger.warning("Contract cache empty; call refresh_contract_embeddings()")
            return []

        scores: List[Tuple[str, float]] = []

        for cid, (market, emb) in self._cache.items():
            sim = self._cosine_similarity(extraction.embedding, emb)
            if sim >= self._threshold:
                scores.append((cid, sim))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:top_k]

        matches: List[ContractMatch] = []
        for cid, sim in scores:
            market, _ = self._cache[cid]

            # Determine direction from sentiment
            direction = self._infer_direction(extraction, market)

            # Get current YES price
            yes_price = Decimal("0.50")
            if market.tokens:
                yes_price = market.tokens[0].price

            # Estimate fair value from sentiment
            fair_value = self._estimate_fair_value(extraction, direction)

            # Calculate edge
            if direction == "yes":
                edge = fair_value - yes_price
            else:
                edge = (Decimal("1") - fair_value) - (Decimal("1") - yes_price)

            matches.append(ContractMatch(
                market=market,
                similarity_score=sim,
                implied_direction=direction,
                current_yes_price=yes_price,
                estimated_fair_value=fair_value,
                edge=edge,
            ))

        return matches

    def match_by_keywords(
        self,
        keywords: List[str],
    ) -> List[PolymarketMarket]:
        """Find markets matching keywords in their question text.

        A simpler alternative to embedding-based matching for cases
        where exact keyword matching is sufficient.

        Args:
            keywords: List of keywords to match.

        Returns:
            List of matching markets.
        """
        results = []
        for _cid, (market, _emb) in self._cache.items():
            question_lower = market.question.lower()
            if any(kw.lower() in question_lower for kw in keywords):
                results.append(market)
        return results

    def _infer_direction(
        self,
        extraction: ExtractionResult,
        market: PolymarketMarket,
    ) -> str:
        """Infer whether news implies YES or NO outcome.

        Args:
            extraction: NLP extraction result.
            market: The target market.

        Returns:
            "yes" or "no".
        """
        # Simple heuristic: positive sentiment -> "yes", negative -> "no"
        # This works for questions framed as "Will X happen?"
        if extraction.sentiment == "positive":
            return "yes"
        elif extraction.sentiment == "negative":
            return "no"

        # For rate decisions, map to common contract framing
        if extraction.event_type == "rate_cut":
            question_lower = market.question.lower()
            if "cut" in question_lower or "lower" in question_lower:
                return "yes"
            elif "hold" in question_lower or "unchanged" in question_lower:
                return "no"

        if extraction.event_type == "rate_hold":
            question_lower = market.question.lower()
            if "hold" in question_lower or "unchanged" in question_lower:
                return "yes"
            elif "cut" in question_lower or "hike" in question_lower:
                return "no"

        return "yes"

    def _estimate_fair_value(
        self,
        extraction: ExtractionResult,
        direction: str,
    ) -> Decimal:
        """Estimate the fair probability based on extraction confidence.

        Args:
            extraction: NLP extraction result.
            direction: Inferred direction ("yes" or "no").

        Returns:
            Estimated fair probability for the YES outcome.
        """
        confidence = abs(extraction.sentiment_score)

        # Scale confidence to a probability estimate
        # High confidence (>0.8) -> near certainty (0.90-0.95)
        # Medium confidence (0.5-0.8) -> moderate (0.70-0.85)
        # Low confidence (<0.5) -> weak signal (0.55-0.65)
        if confidence > 0.8:
            fair = Decimal(str(0.85 + confidence * 0.10))
        elif confidence > 0.5:
            fair = Decimal(str(0.60 + confidence * 0.25))
        else:
            fair = Decimal(str(0.50 + confidence * 0.15))

        # Clamp to valid range
        fair = max(Decimal("0.05"), min(Decimal("0.95"), fair))

        if direction == "no":
            fair = Decimal("1") - fair

        return fair.quantize(Decimal("0.01"))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity (-1 to 1).
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _ensure_embedder(self) -> None:
        """Lazy-load embedding model."""
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("ContractMatcher loaded embedding model")
        except ImportError:
            logger.warning("sentence-transformers not installed")
        except Exception as e:
            logger.warning("Failed to load embedding model: %s", e)
