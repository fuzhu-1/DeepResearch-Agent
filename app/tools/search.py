"""SearchTool — multi-source web search with parallel aggregation.

Runs Tavily, DuckDuckGo, and GitHub searches in parallel, merges results,
and deduplicates by URL. Each source runs independently so one failure
does not block the others.
"""

import asyncio
import logging

from app.config import settings
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query string"},
        "max_results": {"type": "integer", "description": "Max results per source", "default": 5},
    },
    "required": ["query"],
}


class SearchTool(BaseTool):
    name = "search"
    description = "Search the web for information on a given query (aggregates configured backends)"
    parameters = SEARCH_PARAMETERS

    # ------------------------------------------------------------------
    # Mock fallback (last resort)
    # ------------------------------------------------------------------

    async def _mock_results(self, query: str, max_results: int) -> list:
        return [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/results/{i+1}",
                "snippet": "⚠ [MOCK DATA - SEARCH UNAVAILABLE] Search backends are not configured or all failed. This is placeholder data, not real results.",
                "source": "mock",
            }
            for i in range(max_results)
        ]

    # ------------------------------------------------------------------
    # Multi-source parallel execution
    # ------------------------------------------------------------------

    async def execute(self, query: str, max_results: int = 5, **_kwargs):
        """Run all configured backends in parallel and merge results."""
        from app.tools.search_backends import build_backends

        backends = build_backends()
        if not backends:
            logger.warning("No search backends configured")
            return await self._empty_or_mock(query, max_results, "未配置任何搜索后端")

        tasks = [b.search(query, max_results) for b in backends]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        seen_urls = set()
        for rl in results_lists:
            if isinstance(rl, Exception):
                logger.warning("Search backend failed: %s", rl)
                continue
            for item in rl:
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(item)

        if not all_results:
            return await self._empty_or_mock(query, max_results, "所有搜索后端均失败，无可用结果")

        source_order = {"tavily": 0, "exa": 1, "github": 2, "duckduckgo": 3, "mock": 4}
        all_results.sort(key=lambda x: source_order.get(x.get("source", ""), 99))
        all_results = all_results[:max_results * 2]

        return ToolResult(
            success=True,
            data=all_results,
            metadata={
                "source": ",".join(sorted(set(r.get("source", "?") for r in all_results))),
                "result_count": len(all_results),
            },
        )

    async def _empty_or_mock(self, query: str, max_results: int, reason: str) -> ToolResult:
        if settings.SEARCH_MOCK_FALLBACK:
            mock = await self._mock_results(query, max_results)
            return ToolResult(success=True, data=mock, metadata={"source": "mock", "result_count": len(mock)})
        return ToolResult(
            success=False,
            data=[],
            error=reason,
            metadata={"source": "none", "result_count": 0},
        )
