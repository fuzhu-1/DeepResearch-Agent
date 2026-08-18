"""Tests for TaskManager: task CRUD and workspace lifecycle coupling."""

import os

import pytest

from app.services.task_manager import TaskManager
from app.services.workspace import WorkspaceManager


@pytest.mark.asyncio
async def test_delete_task_cleans_workspace(tmp_path, monkeypatch):
    """Deleting a task that IS in memory removes its workspace."""
    from app.config import settings

    ws_root = str(tmp_path / "ws")
    monkeypatch.setattr(settings, "WORKSPACE_ROOT", ws_root)

    manager = TaskManager()
    manager.create_task("test", task_id="task_mem")
    ws = WorkspaceManager(root_dir=ws_root)
    await ws.ensure_workspace("task_mem")
    assert os.path.isdir(ws.workspace_path("task_mem"))

    removed = manager.delete_task("task_mem")
    assert removed is True
    assert not os.path.exists(ws.workspace_path("task_mem"))
    assert manager.get_task("task_mem") is None


@pytest.mark.asyncio
async def test_delete_task_cleans_workspace_not_in_memory(tmp_path, monkeypatch):
    """Deleting a task that is NOT in memory (e.g. after restart) must still
    clean up its workspace directory on disk."""
    from app.config import settings

    ws_root = str(tmp_path / "ws")
    monkeypatch.setattr(settings, "WORKSPACE_ROOT", ws_root)

    # Simulate a historical task: workspace dir exists on disk but the task
    # was never loaded into this manager's memory.
    ws = WorkspaceManager(root_dir=ws_root)
    await ws.ensure_workspace("task_disk")
    assert os.path.isdir(ws.workspace_path("task_disk"))

    manager = TaskManager()
    removed = manager.delete_task("task_disk")
    assert removed is False
    assert not os.path.exists(ws.workspace_path("task_disk"))
