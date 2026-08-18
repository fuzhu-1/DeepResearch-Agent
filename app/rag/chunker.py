"""Text chunker — splits documents into small overlapping chunks for embedding."""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TextChunker:
    """Split documents into chunks for embedding.

    Uses a recursive character splitting strategy:
      1. Split by paragraph boundaries (``\\n\\n``)
      2. If a paragraph exceeds *chunk_size*, split by sentence boundaries
         (``.``, ``!``, ``?``)
      3. If a sentence still exceeds *chunk_size*, split by character with
         overlap.

    Each chunk is returned as a dict with ``text``, ``metadata``, and
    ``chunk_id`` fields.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        """
        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters to overlap between consecutive
                chunks (applied at each splitting level).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Split *text* into overlapping chunks.

        Each chunk dict contains:
          - ``text``: the chunk content
          - ``metadata``: a copy of the user-supplied *metadata* dict (or empty)
          - ``chunk_id``: a hex digest (SHA-256) unique to this chunk

        Args:
            text: The document text to split.
            metadata: Optional metadata to attach to every chunk (e.g. source,
                page number).

        Returns:
            A list of chunk dicts.
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        chunks: List[Dict[str, Any]] = []

        # Level 1: split by paragraphs
        paragraphs = self._split_paragraphs(text)
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                chunks.append(self._make_chunk(para, metadata))
            else:
                # Level 2: split large paragraphs by sentences
                sentences = self._split_sentences(para)
                buffer = ""
                for sentence in sentences:
                    if len(buffer) + len(sentence) + 1 <= self.chunk_size:
                        buffer = (buffer + " " + sentence).strip()
                    else:
                        if buffer:
                            chunks.append(self._make_chunk(buffer, metadata))
                        # Level 3: if a single sentence is too long, chunk by character
                        if len(sentence) > self.chunk_size:
                            self._chunk_by_char(sentence, chunks, metadata)
                        else:
                            buffer = sentence
                if buffer:
                    chunks.append(self._make_chunk(buffer, metadata))

        # Apply overlap: create overlapping windows for consecutive chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        logger.debug("Chunked text into %d chunks (size=%d, overlap=%d)",
                     len(chunks), self.chunk_size, self.chunk_overlap)
        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents at once.

        Each document must have at least a ``text`` key.  An optional
        ``metadata`` key will be merged into each chunk.

        Args:
            documents: List of dicts, each with ``text`` and optionally
                ``metadata``.

        Returns:
            A flat list of chunk dicts.
        """
        all_chunks: List[Dict[str, Any]] = []
        for doc in documents:
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            chunks = self.chunk_text(text, metadata=meta)
            all_chunks.extend(chunks)
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split by double newlines, filtering empty results."""
        raw = re.split(r"\n\s*\n", text)
        return [p.strip() for p in raw if p.strip()]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences.

        Uses a simple heuristic: split on ``.``, ``!``, ``?`` followed by
        whitespace.  Does **not** handle all edge cases (e.g. "Mr. Smith").
        """
        # Split on sentence-ending punctuation followed by space or end-of-string
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in parts if s.strip()]

    @staticmethod
    def _make_chunk(text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build a single chunk dict with a deterministic ``chunk_id``."""
        chunk_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return {
            "text": text,
            "metadata": dict(metadata),
            "chunk_id": chunk_id,
        }

    def _chunk_by_char(
        self,
        text: str,
        chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> None:
        """Split *text* into fixed-size character chunks with overlap."""
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(self._make_chunk(chunk_text, metadata))
            start += self.chunk_size - self.chunk_overlap

    def _apply_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create overlapping windows between consecutive chunks.

        This works by re-splitting chunk boundaries so each chunk boundary
        overlaps with its neighbours by *chunk_overlap* characters.
        """
        overlapped: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            meta = chunk["metadata"]

            # For the first chunk, keep as-is
            if i == 0:
                overlapped.append(chunk)
                continue

            # For subsequent chunks, prepend the tail of the previous chunk
            prev_text = chunks[i - 1]["text"]
            overlap_text = prev_text[-self.chunk_overlap:] if len(prev_text) > self.chunk_overlap else prev_text
            new_text = (overlap_text + " " + text).strip()
            overlapped.append(self._make_chunk(new_text, meta))

        return overlapped
