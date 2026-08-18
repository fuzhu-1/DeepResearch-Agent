"""Tests for WorkspaceManager: directory lifecycle and safety checks."""

import os

import pytest

from app.config import settings
from app.services.workspace import WorkspaceManager


@pytest.fixture
def ws(tmp_path):
    """WorkspaceManager rooted at a temp dir."""
    return WorkspaceManager(root_dir=str(tmp_path / "workspaces"))


class TestWorkspaceLifecycle:
    @pytest.mark.asyncio
    async def test_create_makes_isolated_dir(self, ws):
        await ws.ensure_workspace("task_abc")
        path = ws.workspace_path("task_abc")
        assert os.path.isdir(path)
        assert os.path.realpath(path).startswith(os.path.realpath(ws.root_dir))

    @pytest.mark.asyncio
    async def test_list_files_empty(self, ws):
        await ws.ensure_workspace("task_empty")
        assert ws.list_files("task_empty") == []

    @pytest.mark.asyncio
    async def test_cleanup_removes_dir(self, ws):
        await ws.ensure_workspace("task_clean")
        ws.cleanup("task_clean")
        assert not os.path.exists(ws.workspace_path("task_clean"))

    def test_path_traversal_rejected(self, ws):
        with pytest.raises(ValueError, match="非法 task_id"):
            ws.workspace_path("../evil")

    @pytest.mark.asyncio
    async def test_save_upload_valid_file(self, ws):
        await ws.ensure_workspace("task_upload")
        meta = await ws.save_upload(
            "task_upload",
            filename="ref.md",
            content=b"# Ref",
        )
        assert meta["name"] == "ref.md"
        files = ws.list_files("task_upload")
        assert [f["name"] for f in files] == ["ref.md"]

    @pytest.mark.asyncio
    async def test_save_upload_rejects_traversal(self, ws):
        await ws.ensure_workspace("task_trav")
        with pytest.raises(ValueError, match="非法文件名"):
            await ws.save_upload("task_trav", filename="../evil.txt", content=b"x")

    @pytest.mark.asyncio
    async def test_save_upload_rejects_bad_ext(self, ws):
        await ws.ensure_workspace("task_ext")
        with pytest.raises(ValueError, match="不支持的扩展名"):
            await ws.save_upload("task_ext", filename="evil.exe", content=b"x")

    @pytest.mark.asyncio
    async def test_save_upload_rejects_oversize(self, ws):
        await ws.ensure_workspace("task_size")
        with pytest.raises(ValueError, match="文件过大"):
            await ws.save_upload(
                "task_size",
                filename="big.md",
                content=b"x" * 100,
                max_bytes=50,
            )

    @pytest.mark.asyncio
    async def test_save_upload_enforces_max_files(self, ws):
        from app.services.workspace import FileCountLimitExceeded

        await ws.ensure_workspace("task_max_files")
        meta1 = await ws.save_upload(
            "task_max_files",
            filename="one.md",
            content=b"# one",
            max_files=1,
        )
        assert meta1["name"] == "one.md"
        with pytest.raises(FileCountLimitExceeded, match="文件数已达上限"):
            await ws.save_upload(
                "task_max_files",
                filename="two.md",
                content=b"# two",
                max_files=1,
            )

    @pytest.mark.asyncio
    async def test_save_upload_max_files_unlimited_by_default(self, ws):
        await ws.ensure_workspace("task_max_unlimited")
        for name in ("a.md", "b.md", "c.md"):
            meta = await ws.save_upload(
                "task_max_unlimited",
                filename=name,
                content=b"# x",
            )
            assert meta["name"] == name
        assert len(ws.list_files("task_max_unlimited")) == 3

    @pytest.mark.parametrize(
        "bad_name",
        [
            "metadata.json",
            "task_name.txt",
            "rp_fake.md",
            "rp_notes.txt",
            "METADATA.JSON",
            "RP_FAKE.MD",
            "Rp_Notes.Txt",
        ],
    )
    @pytest.mark.asyncio
    async def test_save_upload_rejects_reserved_report_names(self, ws, bad_name):
        await ws.ensure_workspace("task_reserved")
        with pytest.raises(ValueError, match="保留命名"):
            await ws.save_upload("task_reserved", filename=bad_name, content=b"x")


class TestStateInjection:
    @pytest.mark.asyncio
    async def test_run_research_creates_workspace(self, tmp_path):
        settings.WORKSPACE_ROOT = str(tmp_path / "ws")
        from app.models.state import ResearchState

        state = ResearchState(task="测试课题", max_iterations=1)
        assert state.workspace_dir == ""
        assert state.workspace_files == []


class TestPythonToolWorkspaceParam:
    @pytest.mark.asyncio
    async def test_python_tool_accepts_workspace_dir(self, tmp_path):
        from app.tools.python_executor import PythonTool

        tool = PythonTool()
        ws = str(tmp_path / "ws")
        import os
        os.makedirs(ws, exist_ok=True)
        result = await tool.execute(
            code="print('hello')",
            timeout=10,
            workspace_dir=ws,
        )
        assert result.success is True
        assert "hello" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_python_tool_injects_workspace_dir(self, tmp_path):
        from app.tools.python_executor import PythonTool

        tool = PythonTool()
        ws = str(tmp_path / "ws")
        import os
        os.makedirs(ws, exist_ok=True)
        result = await tool.execute(
            code="print(WORKSPACE_DIR)",
            timeout=10,
            workspace_dir=ws,
        )
        assert result.success is True
        assert ws in result.data["stdout"]
