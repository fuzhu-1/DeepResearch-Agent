"""Embedder — generates embedding vectors for text chunks.

Uses OpenAI embeddings (``text-embedding-3-small``) when ``OPENAI_API_KEY``
is set.  Otherwise falls back to a TF-IDF vectorizer via scikit-learn, or a
simple word-count vector approach if sklearn is unavailable.

Also builds a BM25 sparse index for hybrid retrieval.
"""

import logging
import math
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embedding vectors for text.

    Strategy resolution (first available wins):
      1. DashScope via DASHSCOPE_API_KEY
      2. OpenAI text-embedding-3-small via OPENAI_API_KEY
      3. scikit-learn TfidfVectorizer (bag-of-words, lightweight)
      4. Simple word-count / character-trigram vector (no external deps)
    """

    def __init__(self, model: str = "text-embedding-v3"):
        self.model = model
        self._openai_client = None
        self._sklearn_vectorizer = None
        self._vocab: list[str] = []
        self._dimension = 0
        self._cfg_signature: Optional[tuple] = None
        self._mode = self._resolve_backend()

        # BM25 state
        self._bm25 = None
        self._bm25_chunks: list[str] = []
        self._chunk_metadata: list[dict] = []
        self._bm25_chunk_ids: list[str] = []

    @property
    def has_bm25_index(self) -> bool:
        """Whether a BM25 index has been built (and rank-bm25 is available)."""
        return self._bm25 is not None

    def _resolve_backend(self) -> str:
        # 0. Saved runtime config (OpenAI-compatible, incl. DashScope)
        from app.services.config_service import load_runtime_config

        try:
            rt = load_runtime_config()
        except Exception:
            rt = None
        if rt and (rt.embedding_api_key or rt.api_key):
            key = rt.embedding_api_key or rt.api_key
            base = rt.embedding_base_url or rt.base_url or ""
            model = rt.embedding_model or (
                "text-embedding-v3" if "dashscope" in base.lower() else "text-embedding-3-small"
            )
            try:
                from openai import AsyncOpenAI

                self._openai_client = AsyncOpenAI(
                    api_key=key,
                    base_url=base or "https://api.openai.com/v1",
                )
                self.model = model
                self._cfg_signature = (key, base, model)
                logger.info("Embedder: runtime config — %s", self.model)
                return "openai"
            except Exception as exc:
                logger.warning("Runtime config embedder init failed: %s", exc)

        # 0.5 EMBEDDING_* env vars (OpenAI-compatible, e.g. OpenRouter)
        try:
            from app.config import settings as _settings

            emb_key = _settings.EMBEDDING_API_KEY or os.environ.get(
                "EMBEDDING_API_KEY", ""
            )
            emb_base = _settings.EMBEDDING_BASE_URL or os.environ.get(
                "EMBEDDING_BASE_URL", ""
            )
            emb_model = _settings.EMBEDDING_MODEL or os.environ.get(
                "EMBEDDING_MODEL", "text-embedding-3-small"
            )
        except Exception:
            emb_key = os.environ.get("EMBEDDING_API_KEY", "")
            emb_base = os.environ.get("EMBEDDING_BASE_URL", "")
            emb_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        if emb_key and emb_base:
            try:
                from openai import AsyncOpenAI

                self._openai_client = AsyncOpenAI(api_key=emb_key, base_url=emb_base)
                self.model = emb_model
                self._cfg_signature = (emb_key, emb_base, emb_model)
                logger.info("Embedder: EMBEDDING_* env — %s", self.model)
                return "openai"
            except Exception as exc:
                logger.warning("EMBEDDING_* env init failed: %s", exc)

        # 1. DashScope
        try:
            from app.config import settings

            api_key = settings.DASHSCOPE_API_KEY or os.environ.get("DASHSCOPE_API_KEY", "")
        except Exception:
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if api_key:
            try:
                from openai import AsyncOpenAI

                self._openai_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                self.model = "text-embedding-v3"
                self._cfg_signature = (
                    api_key,
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "text-embedding-v3",
                )
                logger.info("Embedder: DashScope — %s", self.model)
                return "openai"
            except Exception as exc:
                logger.warning("DashScope init failed: %s", exc)

        # 2. OpenAI
        try:
            from app.config import settings

            api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        except Exception:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            try:
                from openai import AsyncOpenAI

                self._openai_client = AsyncOpenAI(api_key=api_key)
                self.model = "text-embedding-3-small"
                self._cfg_signature = (api_key, "", "text-embedding-3-small")
                logger.info("Embedder: OpenAI — %s", self.model)
                return "openai"
            except Exception as exc:
                logger.warning("OpenAI init failed: %s", exc)

        # 3. sklearn TF-IDF
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._sklearn_vectorizer = TfidfVectorizer(
                max_features=256, analyzer="char", ngram_range=(2, 4)
            )
            if not api_key:
                logger.info("Embedder: sklearn TF-IDF (no API key configured)")
                return "sklearn"
        except ImportError:
            pass

        # 4. Simple fallback
        logger.info("Embedder: simple word-count")
        return "simple"

    # ------------------------------------------------------------------
    # Public embedding API (unchanged signatures)
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding vector for a single text string."""
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for multiple texts (batched)."""
        if not texts:
            return []

        self._ensure_config()

        if self._mode == "openai":
            return await self._embed_openai(texts)
        elif self._mode == "sklearn":
            return self._embed_sklearn(texts)
        else:
            return self._embed_simple(texts)

    def _ensure_config(self) -> None:
        """Rebuild the OpenAI client/model when the saved embedding config changes."""
        if self._mode != "openai":
            return
        from app.services.config_service import load_runtime_config

        try:
            rt = load_runtime_config()
        except Exception:
            return
        if not rt or not (rt.embedding_api_key or rt.api_key):
            return
        key = rt.embedding_api_key or rt.api_key
        base = rt.embedding_base_url or rt.base_url or ""
        model = rt.embedding_model or (
            "text-embedding-v3" if "dashscope" in base.lower() else "text-embedding-3-small"
        )
        signature = (key, base, model)
        if signature != self._cfg_signature:
            logger.info("Embedder config changed, rebuilding client (model=%s)", model)
            self._openai_client = None
            self.model = model
            self._mode = self._resolve_backend()

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    # DashScope compatible-mode / OpenAI embedding batch size cap.
    # DashScope rejects batches larger than 10; OpenAI tolerates up to 2048.
    # Splitting by 10 is safe for both providers.
    _MAX_EMBED_BATCH = 10

    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        client = self._openai_client
        assert client is not None

        vectors: List[List[float]] = []
        try:
            for start in range(0, len(texts), self._MAX_EMBED_BATCH):
                batch = texts[start : start + self._MAX_EMBED_BATCH]
                resp = await client.embeddings.create(
                    model=self.model, input=batch, encoding_format="float"
                )
                vectors.extend(item.embedding for item in resp.data)
            logger.debug(
                "OpenAI embedding returned %d vectors (dim=%d)",
                len(vectors),
                len(vectors[0]) if vectors else 0,
            )
            return vectors
        except Exception as exc:
            logger.error("OpenAI embedding failed: %s", exc)
            self._mode = self._resolve_backend()
            if self._mode != "openai":
                logger.warning("Falling back to %s backend after OpenAI error", self._mode)
                return await self.embed_batch(texts)
            raise

    # ------------------------------------------------------------------
    # scikit-learn TF-IDF backend
    # ------------------------------------------------------------------

    def _embed_sklearn(self, texts: List[str]) -> List[List[float]]:
        vectorizer = self._sklearn_vectorizer
        assert vectorizer is not None

        if not hasattr(vectorizer, "vocabulary_") or not vectorizer.vocabulary_:
            logger.debug("Fitting TF-IDF vectorizer on first batch (%d texts)", len(texts))
            matrix = vectorizer.fit_transform(texts)
        else:
            matrix = vectorizer.transform(texts)

        vectors: List[List[float]] = []
        for i in range(matrix.shape[0]):
            row = matrix[i].toarray().flatten().astype(np.float64)
            norm = np.linalg.norm(row)
            if norm > 0:
                row = row / norm
            vectors.append(row.tolist())
        return vectors

    # ------------------------------------------------------------------
    # Simple word-count fallback backend
    # ------------------------------------------------------------------

    def _build_vocab(self, texts: List[str]) -> None:
        counter: Counter = Counter()
        for t in texts:
            tokens = self._tokenize_english(t)
            counter.update(tokens)
        top = counter.most_common(256)
        self._vocab = [word for word, _ in top]
        self._dimension = len(self._vocab)
        logger.debug("Simple embedder vocab built: %d terms", self._dimension)

    def _embed_simple(self, texts: List[str]) -> List[List[float]]:
        if not self._vocab:
            self._build_vocab(texts)
        if self._dimension == 0:
            return [[0.0] * 8 for _ in texts]

        word_set = set(self._vocab)
        vectors: List[List[float]] = []
        for text in texts:
            tokens = self._tokenize_english(text)
            counter = Counter(tokens)
            vec = [0.0] * self._dimension
            for i, word in enumerate(self._vocab):
                count = counter.get(word, 0)
                if count > 0:
                    vec[i] = 1.0 + math.log(count)
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors

    @staticmethod
    def _tokenize_english(text: str) -> List[str]:
        """Lowercase and split on non-alphanumeric characters (English only)."""
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

    # ------------------------------------------------------------------
    # BM25 sparse index
    # ------------------------------------------------------------------

    async def build_bm25(self, texts: List[str], metadata: List[dict], chunk_ids: Optional[List[str]] = None) -> None:
        """Build a BM25 sparse index from the given document texts.

        This is called after embedding documents.  The BM25 index is used
        by ``search_bm25()`` for keyword-based retrieval.

        Args:
            texts: Document chunk texts.
            metadata: Corresponding metadata dicts.
            chunk_ids: Corresponding chunk IDs (used to align BM25 results
                with the vector store).
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed — BM25 index unavailable")
            self._bm25 = None
            return

        if not texts:
            self._bm25 = None
            self._bm25_chunk_ids = []
            return

        tokenized = [self._tokenize_cjk(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_chunks = list(texts)
        self._chunk_metadata = list(metadata)
        self._bm25_chunk_ids = list(chunk_ids or [])
        logger.info("BM25 index built: %d chunks", len(texts))

    def search_bm25(self, query: str, top_k: int = 20) -> List[tuple[int, float]]:
        """BM25 keyword search.

        Returns list of (chunk_index, score) sorted by descending score.
        Use :meth:`get_chunk_id` to resolve an index to its vector-store
        chunk ID, or :meth:`get_chunk_text` for the raw text.
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built — returning empty results")
            return []

        tokenized_query = self._tokenize_cjk(query)
        scores = self._bm25.get_scores(tokenized_query)
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in indices if scores[idx] > 0]

    def get_chunk_id(self, index: int) -> Optional[str]:
        """Return the vector-store chunk ID for a BM25 index position."""
        if 0 <= index < len(self._bm25_chunk_ids):
            return self._bm25_chunk_ids[index]
        return None

    def get_chunk_text(self, index: int) -> str:
        """Return chunk text by index (for retrieval result lookup)."""
        if 0 <= index < len(self._bm25_chunks):
            return self._bm25_chunks[index]
        return ""

    def get_chunk_metadata(self, index: int) -> dict:
        """Return chunk metadata by index."""
        if 0 <= index < len(self._chunk_metadata):
            return dict(self._chunk_metadata[index])
        return {}

    def save_bm25(self, path: str) -> None:
        """Persist BM25 index and chunk data to disk."""
        data = {
            "chunks": self._bm25_chunks,
            "metadata": self._chunk_metadata,
            "chunk_ids": self._bm25_chunk_ids,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("BM25 index saved to %s (%d chunks)", path, len(self._bm25_chunks))

    def load_bm25(self, path: str) -> None:
        """Load BM25 index and chunk data from disk."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed, cannot load BM25 index")
            return

        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25_chunks = data["chunks"]
        self._chunk_metadata = data["metadata"]
        self._bm25_chunk_ids = data.get("chunk_ids", [])
        tokenized = [self._tokenize_cjk(t) for t in self._bm25_chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index loaded from %s (%d chunks)", path, len(self._bm25_chunks))

    # ------------------------------------------------------------------
    # CJK-aware tokenizer
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_cjk(text: str) -> List[str]:
        """Character-level for CJK, word-level for English.

        Detects the ratio of CJK characters per segment and splits
        accordingly:
        - Segments with >30% CJK chars → character-level split
        - Otherwise → whitespace split
        """
        import re

        tokens: list[str] = []
        for part in re.split(r"(\s+)", text):
            if part.strip():
                cjk_count = sum(1 for c in part if "一" <= c <= "鿿")
                if cjk_count > len(part) * 0.3:
                    tokens.extend(list(part))
                else:
                    tokens.extend(part.split())
        return [t.lower() for t in tokens if t.strip()]
