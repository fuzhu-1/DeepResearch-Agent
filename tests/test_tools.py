"""Tests for the tool system (BaseTool, ToolResult, ToolRouter, and all tool implementations).

Note: MemoryTool tests are now in tests/test_memory.py — this file tests the
old action names for backward compatibility if needed, and keeps the original
structure for SearchTool, BrowserTool, and PythonTool.
"""

import pytest

from app.tools.base import ToolResult
from app.tools.browser import BrowserTool
from app.tools.memory import MemoryTool
from app.tools.python_executor import PythonTool
from app.tools.router import ToolRouter
from app.tools.search import SearchTool

# ======================================================================
# ToolRouter
# ======================================================================

class TestToolRouter:
    """ToolRouter initialisation, registration, lookup and execution."""

    def test_init_registers_defaults(self):
        router = ToolRouter()
        names = {t.name for t in router._tools.values()}
        assert names == {"search", "browse", "analyze", "memory", "rag_retrieve"}

    def test_get_tool_known(self):
        router = ToolRouter()
        tool = router.get_tool("search")
        assert isinstance(tool, SearchTool)

    def test_get_tool_unknown_raises(self):
        router = ToolRouter()
        with pytest.raises(ValueError, match="Tool 'nope' not found"):
            router.get_tool("nope")

    def test_register_custom(self):
        router = ToolRouter()
        dummy = SearchTool()
        router.register("custom", dummy)
        assert router.get_tool("custom") is dummy

    def test_list_tools(self):
        router = ToolRouter()
        summary = router.list_tools()
        assert isinstance(summary, list)
        names = {s["name"] for s in summary}
        assert names == {"search", "browse", "analyze", "memory", "rag_retrieve"}

    @pytest.mark.asyncio
    async def test_execute_unknown_raises(self):
        router = ToolRouter()
        with pytest.raises(ValueError, match="Tool 'nope' not found"):
            await router.execute("nope")

    @pytest.mark.asyncio
    async def test_analyze_disabled_by_default(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "ENABLE_PYTHON_TOOL", False)
        router = ToolRouter()
        result = await router.execute("analyze", code="print(1)")
        assert result.success is False
        assert "已禁用" in result.error

    @pytest.mark.asyncio
    async def test_analyze_blocks_dangerous_modules_when_enabled(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "ENABLE_PYTHON_TOOL", True)
        router = ToolRouter()
        result = await router.execute(
            "analyze", code="import os\nprint(os.getcwd())"
        )
        assert result.success is True  # errors are captured, not raised
        assert "blocked" in result.data["stderr"]

    @pytest.mark.asyncio
    async def test_execute_search_mock(self, monkeypatch):
        """ToolRouter should route to SearchTool and return merged results."""
        class FakeBackend:
            name = "fake"

            async def search(self, query, max_results):
                return [{"title": "t", "url": "https://github.com/x", "snippet": "s", "source": "github"}]

        monkeypatch.setattr(
            "app.tools.search_backends.build_backends",
            lambda: [FakeBackend()],
        )
        router = ToolRouter()
        result = await router.execute("search", query="test query", max_results=2)
        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["url"] == "https://github.com/x"

    @pytest.mark.asyncio
    async def test_execute_search_defaults(self, monkeypatch):
        """Default max_results should be 5 (deterministic with mocked backend)."""
        class FakeBackend:
            name = "fake"

            async def search(self, query, max_results):
                return [
                    {"title": f"r{i}", "url": f"https://github.com/{i}", "snippet": "s", "source": "github"}
                    for i in range(3)
                ]

        monkeypatch.setattr(
            "app.tools.search_backends.build_backends",
            lambda: [FakeBackend()],
        )
        router = ToolRouter()
        result = await router.execute("search", query="hello")
        assert result.success is True
        assert len(result.data) <= 5

    def test_resolve_tool_name_direct(self):
        assert ToolRouter.resolve_tool_name("search") == "search"
        assert ToolRouter.resolve_tool_name("browse") == "browse"
        assert ToolRouter.resolve_tool_name("analyze") == "analyze"

    def test_resolve_tool_name_unknown(self):
        """Unknown tool values should pass through unchanged."""
        assert ToolRouter.resolve_tool_name("foo") == "foo"


# ======================================================================
# BaseTool / ToolResult
# ======================================================================

class TestBaseTool:
    """BaseTool abstract interface contract."""

    def test_tool_result_defaults(self):
        r = ToolResult()
        assert r.success is True
        assert r.data is None
        assert r.error is None
        assert r.metadata == {}

    def test_tool_result_custom(self):
        r = ToolResult(success=False, data=[1, 2], error="boom", metadata={"a": 1})
        assert r.success is False
        assert r.data == [1, 2]
        assert r.error == "boom"
        assert r.metadata == {"a": 1}

    def test_concrete_tool_has_required_attrs(self):
        for tool_cls in (SearchTool, BrowserTool, PythonTool, MemoryTool):
            inst = tool_cls() if tool_cls is not MemoryTool else tool_cls()
            assert hasattr(inst, "name")
            assert hasattr(inst, "description")
            assert hasattr(inst, "parameters")


# ======================================================================
# SearchTool
# ======================================================================

class TestSearchTool:
    """SearchTool with fallback chain: Tavily → DuckDuckGo → mock."""

    @pytest.mark.asyncio
    async def test_all_backends_fail_returns_failure(self, monkeypatch):
        """Without any backend results, search must fail, not return mock data."""
        monkeypatch.setattr("app.tools.search_backends.build_backends", lambda: [])
        tool = SearchTool()
        result = await tool.execute(query="ai agent industry trend", max_results=3)
        assert result.success is False
        assert result.data == []
        assert result.metadata.get("source") == "none"

    @pytest.mark.asyncio
    async def test_mock_only_when_enabled(self, monkeypatch):
        """Mock fallback must require SEARCH_MOCK_FALLBACK=true."""
        from app.config import settings

        monkeypatch.setattr(settings, "SEARCH_MOCK_FALLBACK", True)
        monkeypatch.setattr("app.tools.search_backends.build_backends", lambda: [])
        tool = SearchTool()
        result = await tool.execute(query="test query", max_results=2)
        assert result.success is True
        assert len(result.data) == 2
        assert all(r["source"] == "mock" for r in result.data)


# ======================================================================
# BrowserTool
# ======================================================================

class TestBrowserTool:
    """BrowserTool with a real HTTP request."""

    @pytest.mark.asyncio
    async def test_invalid_url(self):
        tool = BrowserTool()
        result = await tool.execute(url="")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_valid_http_page(self):
        """Fetch httpbin.org/html — requires network."""
        pytest.skip("Requires network access")
        tool = BrowserTool()
        result = await tool.execute(url="https://httpbin.org/html")
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["url"] == "https://httpbin.org/html"
        assert len(result.data["content"]) > 0

    @pytest.mark.asyncio
    async def test_fetch_error(self):
        """A non-existent domain should be handled gracefully."""
        tool = BrowserTool()
        result = await tool.execute(url="https://this-domain-does-not-exist-12345.com")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_httpx_fallback_when_playwright_disabled(self, monkeypatch):
        """BROWSER_USE_PLAYWRIGHT=false must use the httpx path."""
        from app.config import settings

        monkeypatch.setattr(settings, "BROWSER_USE_PLAYWRIGHT", False)
        tool = BrowserTool()

        async def fake_fetch(url, attempt=0):
            return ToolResult(
                success=True,
                data={"title": "t", "content": "c", "url": url},
                metadata={"engine": "httpx"},
            )

        monkeypatch.setattr(tool, "_fetch_and_extract", fake_fetch)
        result = await tool.execute(url="https://example.com")
        assert result.success is True
        assert result.data["title"] == "t"

    @pytest.mark.asyncio
    async def test_playwright_failure_falls_back_to_httpx(self, monkeypatch):
        """A failing Playwright fetch must fall back to httpx, not error out."""
        from app.config import settings

        monkeypatch.setattr(settings, "BROWSER_USE_PLAYWRIGHT", True)
        tool = BrowserTool()
        monkeypatch.setattr(tool, "_playwright_available", lambda: True)

        async def failing_playwright(url):
            raise RuntimeError("chromium launch failed")

        async def fake_fetch(url, attempt=0):
            return ToolResult(
                success=True,
                data={"title": "fallback", "content": "c", "url": url},
                metadata={"engine": "httpx"},
            )

        monkeypatch.setattr(tool, "_fetch_with_playwright", failing_playwright)
        monkeypatch.setattr(tool, "_fetch_and_extract", fake_fetch)
        result = await tool.execute(url="https://example.com")
        assert result.success is True
        assert result.data["title"] == "fallback"

    def test_extract_pdf_text_graceful(self):
        """Invalid PDF bytes must not raise; returns a placeholder."""
        from app.tools.browser import _extract_pdf_text

        text = _extract_pdf_text(b"not a real pdf")
        assert "PDF" in text


# ======================================================================
# PythonTool
# ======================================================================

class TestPythonTool:
    """PythonTool sandbox execution."""

    @pytest.mark.asyncio
    async def test_simple_code(self):
        tool = PythonTool()
        result = await tool.execute(code="print('hello world')")
        assert result.success is True
        assert "hello world" in result.data["stdout"]
        assert result.data["stderr"] == ""

    @pytest.mark.asyncio
    async def test_stdout_capture(self):
        tool = PythonTool()
        code = """
for i in range(3):
    print(f"line {i}")
"""
        result = await tool.execute(code=code)
        assert result.success is True
        stdout = result.data["stdout"]
        assert "line 0" in stdout
        assert "line 1" in stdout
        assert "line 2" in stdout

    @pytest.mark.asyncio
    async def test_stderr_capture(self):
        tool = PythonTool()
        code = "import sys; print('error', file=sys.stderr)"
        result = await tool.execute(code=code)
        assert result.success is True
        assert "error" in result.data["stderr"]

    @pytest.mark.asyncio
    async def test_execution_error(self):
        tool = PythonTool()
        code = "1 / 0"
        result = await tool.execute(code=code)
        assert result.success is True  # execution doesn't raise, error is captured
        assert "ZeroDivisionError" in result.data["stderr"]

    @pytest.mark.asyncio
    async def test_execution_time(self):
        tool = PythonTool()
        result = await tool.execute(code="print('fast')")
        assert result.success is True
        assert result.data["execution_time"] >= 0

    @pytest.mark.asyncio
    async def test_empty_code(self):
        tool = PythonTool()
        result = await tool.execute(code="")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_timeout(self):
        tool = PythonTool()
        code = "import time; time.sleep(5)"
        result = await tool.execute(code=code, timeout=1)
        assert result.success is False
        assert "timed out" in result.error.lower()


# ======================================================================
# MemoryTool (backward-compatible tests)
# ======================================================================

class TestMemoryTool:
    """MemoryTool — backward-compatible tests for the new delegate-based implementation.

    The new MemoryTool delegates to SessionMemory and KnowledgeMemory.
    This test class verifies that the old-style simple key-value actions
    ('save' / 'load' / 'list') are *not* available and returns a clear error.
    """

    @pytest.mark.asyncio
    async def test_save_returns_clear_error(self):
        """Old 'save' action is no longer supported."""
        tool = MemoryTool()
        result = await tool.execute(action="save", key="name", value="Alice")
        assert result.success is False
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_load_returns_clear_error(self):
        """Old 'load' action is no longer supported."""
        tool = MemoryTool()
        result = await tool.execute(action="load", key="name")
        assert result.success is False
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_list_returns_clear_error(self):
        """Old 'list' action is no longer supported."""
        tool = MemoryTool()
        result = await tool.execute(action="list")
        assert result.success is False
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_new_session_save_and_load(self):
        """New session_save/session_load actions work correctly."""
        tool = MemoryTool()
        state_dict = {"task": "T", "plan": [], "research_data": [], "errors": []}
        save_result = await tool.execute(action="session_save", task_id="t1", state=state_dict)
        assert save_result.success is True

        load_result = await tool.execute(action="session_load", task_id="t1")
        assert load_result.success is True
        assert load_result.data["state"]["task"] == "T"

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = MemoryTool()
        result = await tool.execute(action="unknown")
        assert result.success is False
        assert "Unknown action" in result.error
