"""Tests for WorkspaceReaderTool (read_workspace): reads reference files from a task workspace."""

import pytest

from app.tools.workspace_reader import WorkspaceReaderTool


@pytest.fixture
def reader_ws(tmp_path):
    """A temp workspace dir containing a sample reference file."""
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    (ws_dir / "ref.md").write_text(
        "# Reference\n\nSome reference content here.", encoding="utf-8"
    )
    return ws_dir


class TestWorkspaceReader:
    @pytest.mark.asyncio
    async def test_read_workspace_reads_file(self, reader_ws):
        tool = WorkspaceReaderTool()
        result = await tool.execute(filename="ref.md", workspace_dir=str(reader_ws))
        assert result.success is True
        assert result.data["filename"] == "ref.md"
        assert "Some reference content here." in result.data["content"]

    @pytest.mark.asyncio
    async def test_read_workspace_rejects_traversal(self, reader_ws):
        tool = WorkspaceReaderTool()
        result = await tool.execute(filename="../evil.txt", workspace_dir=str(reader_ws))
        assert result.success is False
        assert result.data is None

    @pytest.mark.asyncio
    async def test_read_workspace_rejects_nonexistent(self, reader_ws):
        tool = WorkspaceReaderTool()
        result = await tool.execute(filename="missing.md", workspace_dir=str(reader_ws))
        assert result.success is False
        assert "文件不存在" in result.error

    @pytest.mark.asyncio
    async def test_read_workspace_rejects_absolute_path(self, reader_ws):
        tool = WorkspaceReaderTool()
        result = await tool.execute(filename="/etc/passwd", workspace_dir=str(reader_ws))
        assert result.success is False

    @pytest.mark.asyncio
    async def test_read_workspace_truncates_long_content(self, reader_ws):
        from app.tools.workspace_reader import MAX_CONTENT_CHARS

        (reader_ws / "big.txt").write_text("x" * (MAX_CONTENT_CHARS + 100), encoding="utf-8")
        tool = WorkspaceReaderTool()
        result = await tool.execute(filename="big.txt", workspace_dir=str(reader_ws))
        assert result.success is True
        assert len(result.data["content"]) <= MAX_CONTENT_CHARS + 100
        assert result.metadata.get("truncated") is True
        assert "截断" in result.data["content"]
