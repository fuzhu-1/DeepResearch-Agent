"""Tests for workspace prompt-context builder."""

from app.utils.workspace_context import build_workspace_instruction


class TestBuildWorkspaceInstruction:
    def test_empty_workspace(self):
        text = build_workspace_instruction(workspace_dir="/tmp/ws", files=[])
        assert "工作目录" in text
        assert "/tmp/ws" in text
        assert "没有参考文件" in text

    def test_with_files(self):
        text = build_workspace_instruction("/tmp/ws", ["a.pdf", "b.md"])
        assert "a.pdf" in text
        assert "b.md" in text
        assert "参考文件" in text

    def test_relative_path_hint(self):
        text = build_workspace_instruction("/tmp/ws", [], relative_hint="data/workspaces/ws")
        assert "data/workspaces/ws" in text
