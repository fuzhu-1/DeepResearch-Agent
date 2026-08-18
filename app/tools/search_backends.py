"""Pluggable search backends for SearchTool."""

import asyncio
import logging
from typing import Any, Dict, List, Protocol

from app.config import settings

logger = logging.getLogger(__name__)


class SearchBackend(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        ...


class TavilyBackend:
    name = "tavily"

    async def search(self, query: str, max_results: int) -> list:
        api_key = settings.TAVILY_API_KEY
        if not api_key:
            return []
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = await asyncio.to_thread(
            client.search, query=query, search_depth="advanced", max_results=max_results
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", r.get("snippet", "")),
                "source": "tavily",
            }
            for r in response.get("results", [])
        ]


class DuckDuckGoBackend:
    name = "duckduckgo"

    async def search(self, query: str, max_results: int) -> list:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo_search not installed")
            return []

        def _search():
            return list(DDGS().text(query, max_results=max_results))

        results = await asyncio.to_thread(_search)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", "")),
                "source": "duckduckgo",
            }
            for r in results
        ]


class GitHubBackend:
    name = "github"

    async def search(self, query: str, max_results: int) -> list:
        import httpx

        headers = {"Accept": "application/vnd.github.v3+json"}
        if settings.GITHUB_TOKEN:
            headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.github.com/search/repositories",
                    params={"q": query, "per_page": max_results, "sort": "stars"},
                    headers=headers,
                )
                if resp.status_code != 200:
                    logger.warning("GitHub search returned %d", resp.status_code)
                    return []
                results = []
                for r in resp.json().get("items", [])[:max_results]:
                    desc = r.get("description") or ""
                    lang = r.get("language") or ""
                    stars = r.get("stargazers_count", 0)
                    snippet = f"[{lang}] {desc}" if lang else desc
                    if stars:
                        snippet = f"⭐ {stars} ⭐ {snippet}" if snippet else f"⭐ {stars} stars"
                    results.append({
                        "title": r.get("full_name", r.get("name", "")),
                        "url": r.get("html_url", ""),
                        "snippet": snippet[:500],
                        "source": "github",
                    })
                return results
        except Exception as exc:
            logger.warning("GitHub search failed: %s", exc)
            return []


class ExaBackend:
    name = "exa"

    async def search(self, query: str, max_results: int) -> list:
        api_key = settings.EXA_API_KEY
        if not api_key:
            return []
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.exa.ai/search",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json={"query": query, "numResults": max_results, "contents": {"text": True}},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": (r.get("text") or "")[:300],
                    "source": "exa",
                }
                for r in results
            ]


BACKEND_REGISTRY: Dict[str, type] = {
    "tavily": TavilyBackend,
    "duckduckgo": DuckDuckGoBackend,
    "github": GitHubBackend,
    "exa": ExaBackend,
}


def build_backends() -> List[SearchBackend]:
    names = [n.strip().lower() for n in settings.SEARCH_BACKENDS.split(",") if n.strip()]
    backends = []
    for name in names:
        cls = BACKEND_REGISTRY.get(name)
        if cls is None:
            logger.warning("Unknown search backend: %s", name)
            continue
        try:
            backends.append(cls())
        except Exception as exc:
            logger.warning("Failed to init backend %s: %s", name, exc)
    return backends
