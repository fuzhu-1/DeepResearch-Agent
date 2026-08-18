"""Tests for the search backend registry."""

import pytest

from app.tools.search_backends import BACKEND_REGISTRY, build_backends


def test_build_backends_filters_unknown(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "SEARCH_BACKENDS", "tavily,unknown,github")
    backends = build_backends()
    names = {b.name for b in backends}
    assert names == {"tavily", "github"}


@pytest.mark.asyncio
async def test_search_uses_registry_backends(monkeypatch):
    from app.config import settings
    from app.tools.search import SearchTool

    class FakeBackend:
        name = "fake"

        async def search(self, query, max_results):
            return [
                {"title": "t", "url": "https://fake.com", "snippet": "s", "source": "fake"}
            ]

    BACKEND_REGISTRY["fake"] = FakeBackend
    monkeypatch.setattr(settings, "SEARCH_BACKENDS", "fake")
    tool = SearchTool()
    result = await tool.execute(query="q", max_results=2)
    assert result.success is True
    assert result.data[0]["url"] == "https://fake.com"
    BACKEND_REGISTRY.pop("fake")
