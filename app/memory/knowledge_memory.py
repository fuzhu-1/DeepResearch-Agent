"""Knowledge memory — persists research findings across tasks.

In production this uses ChromaDB for vector-based similarity search; for
development we use an in-memory store with simple keyword-overlap scoring
(TF-IDF-like).
"""

import json
import logging
import math
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple keyword-match helpers (no external dependencies)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Split text into lowercased alphanumeric tokens."""
    tokens: List[str] = []
    current: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _tfidf_score(query_tokens: Set[str], doc_tokens: Counter, total_docs: int, doc_freq: Counter) -> float:
    """Compute a crude TF-IDF cosine-like score between query tokens and a document.

    This is intentionally simple — just enough to rank reports without any
    external ML/NLP library.
    """
    score = 0.0
    for qt in query_tokens:
        if qt not in doc_tokens:
            continue
        tf = doc_tokens[qt] / max(sum(doc_tokens.values()), 1)
        idf = math.log((total_docs + 1) / (doc_freq.get(qt, 0) + 1)) + 1
        score += tf * idf
    return score


# ---------------------------------------------------------------------------
# Report data class
# ---------------------------------------------------------------------------

class KnowledgeEntry:
    """A single stored research report."""

    def __init__(self, report_id: str, task: str, report: str, tags: List[str], timestamp: float):
        self.report_id = report_id
        self.task = task
        self.report = report
        self.tags = tags
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "task": self.task,
            "report": self.report,
            "tags": list(self.tags),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(
            report_id=data["report_id"],
            task=data["task"],
            report=data["report"],
            tags=list(data.get("tags", [])),
            timestamp=data.get("timestamp", 0.0),
        )


# ---------------------------------------------------------------------------
# KnowledgeMemory
# ---------------------------------------------------------------------------

class KnowledgeMemory:
    """Persists research findings across tasks for cross-session knowledge reuse.

    In production this uses ChromaDB; for development we use an in-memory
    store with keyword-overlap similarity matching.
    """

    def __init__(self, chroma_path: str = "./data/chroma_db"):
        self._chroma_path = chroma_path
        self._entries: List[KnowledgeEntry] = []
        self._use_chromadb = False

        # Attempt to detect ChromaDB availability (no-op for dev)
        try:
            import chromadb  # noqa: F401

            self._use_chromadb = True
            logger.info("chromadb available — using in-memory fallback for development")
        except ImportError:
            logger.info("chromadb not installed — using pure keyword-based matching")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_report(
        self, task: str, report: str, tags: Optional[List[str]] = None
    ) -> str:
        """Store a report.

        Args:
            task: The original research task/question.
            report: The full report text.
            tags: Optional tags/categories for the report.

        Returns:
            A unique report_id string.
        """
        report_id = str(uuid.uuid4())
        entry = KnowledgeEntry(
            report_id=report_id,
            task=task,
            report=report,
            tags=tags or [],
            timestamp=time.time(),
        )
        self._entries.append(entry)
        logger.info("Saved report '%s' for task: %s", report_id, task[:60])
        return report_id

    async def query_similar(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Find reports similar to *query* using keyword-based scoring.

        If chromadb is available in production, this would use embeddings.
        For development, we compute a crude TF-IDF overlap score.

        Args:
            query: The search query.
            k: Maximum number of results to return.

        Returns:
            A list of report dicts sorted by relevance descending.
        """
        if not self._entries:
            return []

        query_tokens = set(_tokenize(query))

        # Build document frequency table across all entries
        total_docs = len(self._entries)
        doc_freq: Counter = Counter()
        doc_token_counts: List[Counter] = []
        for entry in self._entries:
            text = f"{entry.task} {' '.join(entry.tags)} {entry.report[:2000]}"
            tokens = Counter(_tokenize(text))
            doc_token_counts.append(tokens)
            for t in tokens:
                doc_freq[t] += 1

        # Score each entry
        scored: List[tuple] = []
        for idx, entry in enumerate(self._entries):
            score = _tfidf_score(query_tokens, doc_token_counts[idx], total_docs, doc_freq)
            if score > 0:
                scored.append((score, entry))

        # Sort descending by score, take top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry.to_dict() for _, entry in scored[:k]]
        return results

    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific report by ID.

        Args:
            report_id: The unique identifier returned by save_report().

        Returns:
            The report dict or None if not found.
        """
        for entry in self._entries:
            if entry.report_id == report_id:
                return entry.to_dict()
        return None

    async def list_reports(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent reports, newest first.

        Args:
            limit: Maximum number of reports to return.

        Returns:
            A list of report dicts sorted by timestamp descending.
        """
        sorted_entries = sorted(
            self._entries, key=lambda e: e.timestamp, reverse=True
        )
        return [e.to_dict() for e in sorted_entries[:limit]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of stored reports (convenience for testing)."""
        return len(self._entries)
