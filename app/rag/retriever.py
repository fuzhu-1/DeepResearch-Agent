"""RAGRetriever — high-level interface that combines chunking, embedding, and vector search.

This is the primary entry point for RAG operations in the research agent.
It orchestrates:

  1. :class:`TextChunker`  — splits documents into chunks
  2. :class:`Embedder`     — generates vector embeddings for chunks
  3. :class:`VectorStore`  — persists and queries chunk embeddings

Also supports hybrid retrieval (dense + BM25 fusion) and optional
reranking (OpenRouter or DashScope) when enabled via settings.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from app.rag.chunker import TextChunker
from app.rag.embedder import Embedder
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenRouter Reranker
# ---------------------------------------------------------------------------


class OpenRouterReranker:
    """Reranker via OpenRouter's ``POST /api/v1/rerank`` endpoint.

    Args:
        api_key: OpenRouter API key.
        model_name: Reranker model id (e.g. ``nvidia/llama-nemotron-rerank-vl-1b-v2:free``).
        api_base: Reranker API endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        api_base: str = "https://openrouter.ai/api/v1/rerank",
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base

    def compute_scores(
        self, query: str, documents: list[str], top_n: int = 5
    ) -> list[float]:
        """Compute relevance scores for documents against query.

        Returns a score per document, aligned with the input order.
        """
        import httpx

        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        for attempt in range(3):
            try:
                resp = httpx.post(
                    self.api_base,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                score_map = {
                    r["index"]: r["relevance_score"] for r in results
                }
                return [score_map.get(i, 0.0) for i in range(len(documents))]
            except Exception as e:
                logger.warning(
                    "OpenRouter reranker attempt %d/3: %s", attempt + 1, e
                )
                if attempt < 2:
                    time.sleep(2**attempt)

        return [0.0] * len(documents)


# ---------------------------------------------------------------------------
# DashScope Reranker
# ---------------------------------------------------------------------------


class DashScopeReranker:
    """DashScope reranker via HTTP API.

    Args:
        api_base: Reranker API endpoint.
        api_key: DashScope API key.
        model_name: Reranker model name (default: ``gte-rerank``).
    """

    def __init__(self, api_base: str, api_key: str, model_name: str = "gte-rerank"):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name

    def compute_scores(
        self, query: str, documents: list[str], top_n: int = 5
    ) -> list[float]:
        """Compute relevance scores for documents against query.

        Returns a score per document, aligned with the input order.
        """
        import httpx

        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n},
        }

        for attempt in range(3):
            try:
                resp = httpx.post(
                    self.api_base,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("output", {}).get("results", [])
                score_map = {r["index"]: r["relevance_score"] for r in results}
                return [score_map.get(i, 0.0) for i in range(len(documents))]
            except Exception as e:
                logger.warning("Reranker API attempt %d/3: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2**attempt)

        return [0.0] * len(documents)


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------


class RAGRetriever:
    """High-level retriever that combines chunking, embedding, and vector search.

    Typical usage::

        retriever = RAGRetriever()
        chunk_ids = await retriever.ingest_document(
            content="Some long document text...",
            source="report_2024.pdf",
            doc_type="pdf",
        )
        results = await retriever.retrieve("quantum computing advances")
        # Or with hybrid search:
        results = await retriever.hybrid_retrieve("quantum computing advances")
    """

    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

        # Reranker (lazy, provider-aware)
        self._reranker: Optional[DashScopeReranker] = None
        self._reranker_sig: Optional[tuple] = None

        # Accumulated chunk corpus for BM25 (persists across ingests)
        self._bm25_chunks: list[dict] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest_document(
        self,
        content: str,
        source: str,
        doc_type: str = "text",
    ) -> List[str]:
        """Process a document and store its chunks in the vector store.

        When hybrid search is enabled, the BM25 index is rebuilt over the
        *entire* accumulated corpus so earlier documents stay searchable.
        """
        metadata = {"source": source, "doc_type": doc_type}
        chunks = self.chunker.chunk_text(content, metadata=metadata)
        if not chunks:
            logger.warning("Document '%s' produced no chunks", source)
            return []

        chunk_ids = await self.vector_store.add_documents(
            chunks, embedder=self.embedder
        )

        # Rebuild BM25 over the full corpus so multiple documents fuse correctly
        try:
            from app.config import settings

            if settings.HYBRID_SEARCH_ENABLED:
                self._bm25_chunks.extend(chunks)
                await self._rebuild_bm25()
        except Exception:
            pass

        logger.info("Ingested '%s': %d chunks", source, len(chunk_ids))
        return chunk_ids

    async def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from the accumulated chunk corpus."""
        if not self._bm25_chunks:
            return
        texts = [c["text"] for c in self._bm25_chunks]
        metas = [c.get("metadata", {}) for c in self._bm25_chunks]
        ids = [c.get("chunk_id", "") for c in self._bm25_chunks]
        await self.embedder.build_bm25(texts, metas, chunk_ids=ids)

    async def rebuild_bm25_from_store(self) -> None:
        """Rebuild the BM25 index from all chunks persisted in the vector store.

        Called at server startup: the BM25 index lives in memory only, so after
        a restart it must be reconstructed from ChromaDB so hybrid retrieval
        keeps working without re-ingesting documents.
        """
        try:
            from app.config import settings

            if not settings.HYBRID_SEARCH_ENABLED:
                return
        except Exception:
            return

        chunks = await self.vector_store.get_all_chunks()
        if not chunks:
            logger.info("No persisted chunks to rebuild BM25 index from")
            return
        self._bm25_chunks = chunks
        await self._rebuild_bm25()
        logger.info("Rebuilt BM25 index from store: %d chunks", len(chunks))

    # ------------------------------------------------------------------
    # Retrieval (existing API — unchanged)
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks for query (dense only)."""
        results = await self.vector_store.similarity_search(
            query, k=k, embedder=self.embedder
        )
        logger.debug("Retrieved %d results for '%s' (k=%d)", len(results), query, k)
        return results

    async def retrieve_with_scores(
        self, query: str, k: int = 5, score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks with minimum score (dense only)."""
        all_results = await self.retrieve(query, k=k)
        filtered = [
            r for r in all_results if r.get("score", 0.0) >= score_threshold
        ]
        logger.debug("%d results after threshold %.2f", len(filtered), score_threshold)
        return filtered

    # ------------------------------------------------------------------
    # Hybrid retrieval (new)
    # ------------------------------------------------------------------

    async def hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Dense ANN + BM25 weighted fusion + optional reranker.

        Falls back to dense-only :meth:`retrieve` when hybrid search
        is disabled or BM25 is unavailable.

        Dense and sparse results are aligned by ``chunk_id`` (not position),
        so the weighted fusion is correct across both retrievers.

        Args:
            query: Search query.
            top_k: Number of final results to return.

        Returns:
            List of result dicts with ``text``, ``metadata``, ``score``,
            and ``source_type`` (``"dense"``, ``"hybrid"``, or ``"reranked"``).
        """
        dense_results = await self.vector_store.similarity_search(
            query, k=top_k * 4, embedder=self.embedder
        )

        try:
            from app.config import settings

            hybrid_enabled = settings.HYBRID_SEARCH_ENABLED
            vector_weight = settings.VECTOR_WEIGHT
            bm25_weight = settings.BM25_WEIGHT
        except Exception:
            hybrid_enabled = False
            vector_weight = 0.7
            bm25_weight = 0.3

        if not hybrid_enabled or not self.embedder.has_bm25_index:
            return self._format_dense(dense_results, top_k)

        bm25_results = self.embedder.search_bm25(query, top_k=top_k * 4)
        if not bm25_results:
            return self._format_dense(dense_results, top_k)

        # --- Rank-Biased Fusion (Dense leads, BM25 supplements) ---
        # Strategy: Dense controls ranking; BM25 expands the candidate pool
        # with chunk_ids that matched on keywords but missed by vector.
        # This preserves semantic quality on natural-language queries while
        # letting BM25 rescue precise-term queries (e.g. "PQ", "StateGraph").
        #
        # Implementation: build a priority map.  A chunk's fused rank is its
        # minimum (best) rank across both lists, weighted by how each list
        # ranked it.  Chunks only in BM25 get appended at end — they only
        # appear when Dense has nothing good.
        dense_order: dict[str, int] = {}
        for rank, r in enumerate(dense_results, start=1):
            cid = r.get("chunk_id")
            if cid:
                dense_order[cid] = rank

        bm25_order: dict[str, int] = {}
        for rank, (idx, _score) in enumerate(bm25_results, start=1):
            cid = self.embedder.get_chunk_id(idx)
            if cid:
                bm25_order[cid] = rank

        # Build ordered result list: Dense top-k first, then sprinkle in
        # BM25-only chunks that weren't in Dense at all (break ties by
        # BM25 rank).
        seen: set[str] = set()
        ordered: list[str] = []
        for r in dense_results:
            cid = r.get("chunk_id")
            if cid:
                ordered.append(cid)
                seen.add(cid)
        for rank, (idx, _score) in enumerate(bm25_results, start=1):
            cid = self.embedder.get_chunk_id(idx)
            if cid and cid not in seen:
                ordered.append(cid)
                seen.add(cid)

        # Truncate to candidates for rerank / output
        candidate_ids = ordered[: top_k * 4]
        if not candidate_ids:
            return self._format_dense(dense_results, top_k)

        fused: dict[str, float] = {}
        for cid in candidate_ids:
            score = 0.0
            if cid in dense_order:
                # Dense gets full weight; BM25 gets residual weight
                dr = dense_order[cid]
                score += vector_weight / dr
            if cid in bm25_order:
                br = bm25_order[cid]
                score += bm25_weight / br
            fused[cid] = score

        ranked_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        dense_by_id = {r["chunk_id"]: r for r in dense_results if r.get("chunk_id")}

        # --- reranker (optional) ---
        try:
            from app.config import settings

            reranker_enabled = settings.RERANKER_ENABLED
            from app.services.config_service import load_runtime_config

            rt = load_runtime_config()
            if rt and rt.reranker_enabled and rt.reranker_api_key:
                reranker_enabled = True
        except Exception:
            reranker_enabled = False

        if reranker_enabled:
            return self._rerank_by_id(query, ranked_ids, top_k=top_k)

        results: List[Dict[str, Any]] = []
        for cid, score in ranked_ids[:top_k]:
            if cid in dense_by_id:
                r = dense_by_id[cid]
                results.append(
                    {
                        "text": r["text"],
                        "metadata": r.get("metadata", {}),
                        "score": float(score),
                        "source_type": "hybrid",
                        "chunk_id": cid,
                    }
                )
            else:
                idx = self._chunk_id_to_bm25_index(cid)
                results.append(
                    {
                        "text": self.embedder.get_chunk_text(idx),
                        "metadata": self.embedder.get_chunk_metadata(idx),
                        "score": float(score),
                        "source_type": "hybrid",
                        "chunk_id": cid,
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_dense(
        results: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Normalise dense results to the hybrid output shape."""
        return [
            {
                "text": r["text"],
                "metadata": r.get("metadata", {}),
                "score": r.get("score", 0.0),
                "source_type": "dense",
                "chunk_id": r.get("chunk_id"),
            }
            for r in results[:top_k]
        ]

    def _chunk_id_to_bm25_index(self, chunk_id: str) -> int:
        """Map a chunk ID back to its BM25 index position (or 0 if missing)."""
        try:
            return self.embedder._bm25_chunk_ids.index(chunk_id)  # type: ignore[attr-defined]
        except (ValueError, AttributeError):
            return 0

    def _rerank_by_id(
        self,
        query: str,
        ranked_ids: list[tuple[str, float]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank hybrid candidates using the DashScope reranker."""
        documents: list[str] = []
        valid_ids: list[str] = []
        for cid, _score in ranked_ids:
            text = self._chunk_text_by_id(cid)
            if text:
                documents.append(text)
                valid_ids.append(cid)

        if not documents:
            return []

        try:
            from app.config import settings

            reranker = self._get_reranker(settings)
            rerank_scores = reranker.compute_scores(
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
            )
        except Exception as e:
            logger.warning("Reranker failed, using fusion scores: %s", e)
            return [
                {
                    "text": self._chunk_text_by_id(cid),
                    "metadata": self._chunk_meta_by_id(cid),
                    "score": float(score),
                    "source_type": "hybrid",
                    "chunk_id": cid,
                }
                for cid, score in ranked_ids[:top_k]
            ]

        scored = list(zip(valid_ids, rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for cid, rerank_score in scored[:top_k]:
            results.append(
                {
                    "text": self._chunk_text_by_id(cid),
                    "metadata": self._chunk_meta_by_id(cid),
                    "score": float(rerank_score),
                    "source_type": "reranked",
                    "chunk_id": cid,
                }
            )
        return results

    def _chunk_text_by_id(self, chunk_id: str) -> str:
        """Return chunk text by chunk ID."""
        try:
            idx = self.embedder._bm25_chunk_ids.index(chunk_id)  # type: ignore[attr-defined]
            return self.embedder.get_chunk_text(idx)
        except (ValueError, AttributeError):
            return ""

    def _chunk_meta_by_id(self, chunk_id: str) -> dict:
        """Return chunk metadata by chunk ID."""
        try:
            idx = self.embedder._bm25_chunk_ids.index(chunk_id)  # type: ignore[attr-defined]
            return self.embedder.get_chunk_metadata(idx)
        except (ValueError, AttributeError):
            return {}

    def _get_reranker(self, settings) -> DashScopeReranker:
        """Lazy-init the reranker, preferring runtime config (OpenRouter).

        Resolution order:
          1. Saved runtime config with ``reranker_api_key`` -> OpenRouter
          2. ``RERANKER_PROVIDER=openrouter`` in env -> OpenRouter
          3. DashScope (default) via DASHSCOPE_API_KEY
        """
        try:
            from app.services.config_service import load_runtime_config

            rt = load_runtime_config()
        except Exception:
            rt = None

        # 1. Runtime config (OpenRouter, matches the LLM settings page)
        if rt and rt.reranker_api_key:
            api_key = rt.reranker_api_key
            api_base = (
                rt.reranker_base_url
                or "https://openrouter.ai/api/v1/rerank"
            )
            model = (
                rt.reranker_model
                or "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
            )
            sig = ("openrouter", api_key, api_base, model)
            if self._reranker_sig != sig:
                self._reranker = OpenRouterReranker(
                    api_key=api_key, model_name=model, api_base=api_base
                )
                self._reranker_sig = sig
                logger.info("OpenRouter reranker initialized: %s", model)
            return self._reranker

        # 2. Env-driven OpenRouter
        env_key = settings.RERANKER_API_KEY or os.environ.get(
            "RERANKER_API_KEY", ""
        )
        if settings.RERANKER_PROVIDER == "openrouter" and env_key:
            api_base = settings.RERANKER_API_BASE or (
                "https://openrouter.ai/api/v1/rerank"
            )
            model = settings.RERANKER_MODEL
            sig = ("openrouter", env_key, api_base, model)
            if self._reranker_sig != sig:
                self._reranker = OpenRouterReranker(
                    api_key=env_key, model_name=model, api_base=api_base
                )
                self._reranker_sig = sig
                logger.info("OpenRouter reranker initialized: %s", model)
            return self._reranker

        # 3. DashScope fallback
        api_key = settings.DASHSCOPE_API_KEY or os.environ.get(
            "DASHSCOPE_API_KEY", ""
        )
        sig = (
            "dashscope",
            api_key,
            settings.RERANKER_API_BASE,
            settings.RERANKER_MODEL,
        )
        if self._reranker_sig != sig:
            self._reranker = DashScopeReranker(
                api_base=settings.RERANKER_API_BASE,
                api_key=api_key,
                model_name=settings.RERANKER_MODEL,
            )
            self._reranker_sig = sig
            logger.info("DashScope reranker initialized: %s", settings.RERANKER_MODEL)
        return self._reranker
