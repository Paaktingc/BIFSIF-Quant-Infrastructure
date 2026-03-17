"""Entity extraction and sentiment analysis for financial news.

Uses FinBERT for sentiment classification and sentence-transformers for
semantic embeddings. Includes regex-based extractors for structured
central bank and parliamentary outputs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from bifs_quant_engine.data.news_monitor import NewsItem

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of NLP extraction on a news item.

    Attributes:
        news_item: The original news item.
        sentiment: Overall sentiment ("positive", "negative", "neutral").
        sentiment_score: Sentiment confidence (-1.0 to 1.0).
        entities: Extracted entities (institution, action, topic).
        embedding: Sentence embedding vector for contract matching.
        event_type: Classified event type.
        key_facts: Structured facts extracted from the text.
    """

    news_item: NewsItem
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    entities: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    event_type: str = "unknown"
    key_facts: Dict[str, str] = field(default_factory=dict)


# Regex patterns for structured financial text
RATE_DECISION_PATTERNS = [
    (r"(?:decided to )?(?:raise|increase|hike)\w* .*?(?:rate|rates).*?(?:by |to )(\d+\.?\d*)",
     "rate_hike"),
    (r"(?:decided to )?(?:cut|lower|reduce|decrease)\w* .*?(?:rate|rates).*?(?:by |to )(\d+\.?\d*)",
     "rate_cut"),
    (r"(?:decided to )?(?:maintain|hold|keep|leave)\w* .*?(?:rate|rates).*?(?:unchanged|steady|at)",
     "rate_hold"),
]

VOTE_PATTERNS = [
    (r"(?:voted|approved|passed|adopted)\b.*?(?:by\s+)?(\d+)\s*(?:to|[-–])\s*(\d+)",
     "vote_result"),
    (r"(?:rejected|defeated|voted down)\b.*?(?:by\s+)?(\d+)\s*(?:to|[-–])\s*(\d+)",
     "vote_rejected"),
]

INSTITUTION_PATTERNS = {
    "ECB": r"\b(?:ECB|European Central Bank)\b",
    "BOJ": r"\b(?:BOJ|Bank of Japan)\b",
    "RBA": r"\b(?:RBA|Reserve Bank of Australia)\b",
    "BOE": r"\b(?:BOE|Bank of England)\b",
    "Fed": r"\b(?:Fed|Federal Reserve|FOMC)\b",
    "OPEC": r"\b(?:OPEC|OPEC\+)\b",
    "EU_Parliament": r"\b(?:European Parliament|EU Parliament|MEPs)\b",
    "UK_Parliament": r"\b(?:House of Commons|UK Parliament|MPs voted)\b",
    "UN": r"\b(?:United Nations|UN Security Council|UNSC)\b",
    "ICJ": r"\b(?:ICJ|International Court of Justice)\b",
}


class EntityExtractor:
    """Extracts financial entities, sentiment, and embeddings from news text.

    Combines rule-based pattern matching for structured financial text with
    transformer-based models for sentiment and embedding generation.

    Example:
        extractor = EntityExtractor()
        result = extractor.extract(news_item)
        print(f"Sentiment: {result.sentiment} ({result.sentiment_score:.2f})")
        print(f"Entities: {result.entities}")
        print(f"Event type: {result.event_type}")
    """

    def __init__(
        self,
        sentiment_model: str = "ProsusAI/finbert",
        embedding_model: str = "all-MiniLM-L6-v2",
        lazy_load: bool = True,
    ) -> None:
        """Initialize extractor.

        Args:
            sentiment_model: HuggingFace model for sentiment analysis.
            embedding_model: Sentence-transformer model for embeddings.
            lazy_load: If True, defer model loading until first use.
        """
        self._sentiment_model_name = sentiment_model
        self._embedding_model_name = embedding_model
        self._sentiment_pipeline = None
        self._embedder = None

        if not lazy_load:
            self._load_models()

    def extract(self, news_item: NewsItem) -> ExtractionResult:
        """Extract entities, sentiment, and embedding from a news item.

        Args:
            news_item: The news item to process.

        Returns:
            ExtractionResult with extracted information.
        """
        text = news_item.raw_text or f"{news_item.title} {news_item.summary}"

        # Rule-based extraction
        entities = self._extract_institutions(text)
        event_type, key_facts = self._extract_event(text)

        # Sentiment analysis
        sentiment, sentiment_score = self._analyze_sentiment(text)

        # Embedding
        embedding = self.get_embedding(text)

        return ExtractionResult(
            news_item=news_item,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            entities=entities,
            embedding=embedding,
            event_type=event_type,
            key_facts=key_facts,
        )

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate sentence embedding for text.

        Args:
            text: Input text.

        Returns:
            Embedding vector or None if model unavailable.
        """
        try:
            self._ensure_embedder()
            if self._embedder is not None:
                return self._embedder.encode(text, show_progress_bar=False)
        except Exception as e:
            logger.warning("Embedding generation failed: %s", e)
        return None

    def _analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment using FinBERT.

        Args:
            text: Input text.

        Returns:
            Tuple of (sentiment label, score from -1 to 1).
        """
        try:
            self._ensure_sentiment_pipeline()
            if self._sentiment_pipeline is not None:
                # Truncate to model max length
                truncated = text[:512]
                result = self._sentiment_pipeline(truncated)[0]
                label = result["label"].lower()
                score = result["score"]

                if label == "negative":
                    return "negative", -score
                elif label == "positive":
                    return "positive", score
                else:
                    return "neutral", 0.0
        except Exception as e:
            logger.warning("Sentiment analysis failed: %s", e)

        return "neutral", 0.0

    def _extract_institutions(self, text: str) -> List[str]:
        """Extract financial institution references from text.

        Args:
            text: Input text.

        Returns:
            List of matched institution codes.
        """
        found = []
        for code, pattern in INSTITUTION_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                found.append(code)
        return found

    def _extract_event(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Extract structured event information from text.

        Args:
            text: Input text.

        Returns:
            Tuple of (event_type, key_facts dict).
        """
        facts: Dict[str, str] = {}

        # Check rate decision patterns
        for pattern, event_type in RATE_DECISION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if match.groups():
                    facts["amount"] = match.group(1)
                return event_type, facts

        # Check vote patterns
        for pattern, event_type in VOTE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    facts["votes_for"] = match.group(1)
                    facts["votes_against"] = match.group(2)
                return event_type, facts

        return "unknown", facts

    def _ensure_sentiment_pipeline(self) -> None:
        """Lazy-load sentiment model."""
        if self._sentiment_pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self._sentiment_model_name,
                truncation=True,
            )
            logger.info("Loaded sentiment model: %s", self._sentiment_model_name)
        except ImportError:
            logger.warning("transformers not installed; sentiment disabled")
        except Exception as e:
            logger.warning("Failed to load sentiment model: %s", e)

    def _ensure_embedder(self) -> None:
        """Lazy-load embedding model."""
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model_name)
            logger.info("Loaded embedding model: %s", self._embedding_model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed; embeddings disabled")
        except Exception as e:
            logger.warning("Failed to load embedding model: %s", e)

    def _load_models(self) -> None:
        """Eagerly load all models."""
        self._ensure_sentiment_pipeline()
        self._ensure_embedder()
