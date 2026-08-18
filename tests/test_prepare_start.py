"""Tests for the prepare / start / upload lifecycle of pre-created research tasks.

Covers POST /api/research/prepare, POST /api/research/{task_id}/start,
POST /api/research/{task_id}/upload, and GET /api/research/{task_id}/workspace,
plus the service-level start_prepared_task guard against double-start.

TaskManager._run is always monkeypatched to a no-op so no real LLM / API key
is required.
"""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app, _task_manager
from app.models.database import TaskRepository, close_db, init_db
from app.services.task_manager import TaskManager
from app.services.workspace import WorkspaceManager


@pytest.fixture
def transport():
    return ASGITransport(app=app)


@pytest.fixture
async def client(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def db():
    """Initialise an in-memory SQLite database (prepare/start persist rows)."""
    await init_db("sqlite+aiosqlite://")
    yield
    await close_db()


async def _stub_run_completes(
    self,
    task_id: str,
    task_text: str,
    max_iterations: int,
    fmt: str,
    use_rag: bool,
    profile_id: str | None = None,
) -> None:
    """Stand-in for TaskManager._run: mark the task completed without an LLM."""
    self._tasks[task_id].status = "completed"
    self._tasks[task_id].final_report = "# done"


class TestPrepare:
    """POST /api/research/prepare lifecycle."""

    @pytest.mark.asyncio
    async def test_prepare_creates_pending_task(self, db, client, tmp_path, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "WORKSPACE_ROOT", str(tmp_path / "ws"))

        resp = await client.post("/api/research/prepare", json={"task": "测试课题"})
        assert resp.status_code == 200
        data = resp.json()
        task_id = data["task_id"]
        assert data["status"] == "pending"

        # Status endpoint still reports pending; NO workflow was launched.
        status = await client.get(f"/api/research/{task_id}")
        assert status.status_code == 200
        assert status.json()["status"] == "pending"

        task_info = _task_manager.get_task(task_id)
        assert task_info is not None
        assert task_info.events == []

        # prepare must NOT persist a database row (start does that).
        from app.models.database import _async_session_maker

        async with _async_session_maker() as session:
            task = await TaskRepository(session).get(task_id)
        assert task is None

    @pytest.mark.asyncio
    async def test_prepare_then_upload_then_start(self, db, client, tmp_path, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "WORKSPACE_ROOT", str(tmp_path / "ws"))
        monkeypatch.setattr(TaskManager, "_run", _stub_run_completes)

        prep = await client.post("/api/research/prepare", json={"task": "上传后启动"})
        assert prep.status_code == 200
        task_id = prep.json()["task_id"]

        # Upload a reference file to the prepared task.
        up = await client.post(
            f"/api/research/{task_id}/upload",
            files={"file": ("notes.md", "# 参考笔记".encode("utf-8"), "text/markdown")},
        )
        assert up.status_code == 200
        assert up.json()["name"] == "notes.md"

        # The workspace lists the uploaded file.
        lst = await client.get(f"/api/research/{task_id}/workspace")
        assert lst.status_code == 200
        names = [f["name"] for f in lst.json()["files"]]
        assert "notes.md" in names

        # Starting the prepared task reuses the SAME task_id.
        start = await client.post(f"/api/research/{task_id}/start", json={})
        assert start.status_code == 200
        assert start.json()["task_id"] == task_id

        await asyncio.sleep(0.3)
        info = _task_manager.get_task(task_id)
        assert info is not None
        assert info.status == "completed"
        assert info.task_id == task_id

    @pytest.mark.asyncio
    async def test_start_missing_task_404(self, db, client, tmp_path, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "WORKSPACE_ROOT", str(tmp_path / "ws"))

        resp = await client.post("/api/research/no-such-task/start", json={})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_duplicate_start_409(self, db, client, tmp_path, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "WORKSPACE_ROOT", str(tmp_path / "ws"))

        async def fake_run_slow(
            self,
            task_id: str,
            task_text: str,
            max_iterations: int,
            fmt: str,
            use_rag: bool,
            profile_id: str | None = None,
        ) -> None:
            await asyncio.sleep(0.2)
            self._tasks[task_id].status = "completed"

        monkeypatch.setattr(TaskManager, "_run", fake_run_slow)

        prep = await client.post("/api/research/prepare", json={"task": "重复启动"})
        assert prep.status_code == 200
        task_id = prep.json()["task_id"]

        first = await client.post(f"/api/research/{task_id}/start", json={})
        assert first.status_code == 200

        # A second start while the task is already running must be rejected.
        second = await client.post(f"/api/research/{task_id}/start", json={})
        assert second.status_code == 409
        assert second.json()["detail"] == "任务已启动"

        # Let the stubbed runner finish so no pending task leaks into teardown.
        await asyncio.sleep(0.3)

    @pytest.mark.asyncio
    async def test_upload_to_prepared_task_security(self, db, client, tmp_path, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "WORKSPACE_ROOT", str(tmp_path / "ws"))

        prep = await client.post("/api/research/prepare", json={"task": "上传安全"})
        assert prep.status_code == 200
        task_id = prep.json()["task_id"]

        # Allowed extension succeeds.
        md = await client.post(
            f"/api/research/{task_id}/upload",
            files={"file": ("notes.md", b"# ok", "text/markdown")},
        )
        assert md.status_code == 200

        # Disallowed extension is rejected.
        exe = await client.post(
            f"/api/research/{task_id}/upload",
            files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
        )
        assert exe.status_code == 400

        # Reserved report filename is rejected.
        meta = await client.post(
            f"/api/research/{task_id}/upload",
            files={"file": ("metadata.json", b"{}", "application/json")},
        )
        assert meta.status_code == 400
        assert "保留命名" in meta.json()["detail"]

        # Only the valid upload actually landed in the workspace.
        lst = await client.get(f"/api/research/{task_id}/workspace")
        assert lst.status_code == 200
        names = [f["name"] for f in lst.json()["files"]]
        assert names == ["notes.md"]


class TestStartPreparedTaskService:
    """TaskManager.start_prepared_task exercised directly (no HTTP)."""

    @pytest.mark.asyncio
    async def test_start_prepared_task_reuses_workspace_and_syncs_files(self, db, tmp_path, monkeypatch):
        from app.config import settings

        ws_root = str(tmp_path / "ws")
        monkeypatch.setattr(settings, "WORKSPACE_ROOT", ws_root)
        monkeypatch.setattr(TaskManager, "_run", _stub_run_completes)

        manager = TaskManager()
        ws = WorkspaceManager(root_dir=ws_root)
        tid = manager.create_task("准备后启动")
        await ws.ensure_workspace(tid)
        await ws.save_upload(tid, "notes.md", b"# reference note")

        # Uploading alone does not touch task_info.workspace_files.
        assert manager.get_task(tid).workspace_files == []

        started = await manager.start_prepared_task(tid)
        # Same task_id is reused — no new task is created for the run.
        assert started == tid

        await asyncio.sleep(0.1)
        info = manager.get_task(tid)
        # start re-syncs the workspace listing to pick up uploaded files.
        assert info.workspace_files == ["notes.md"]
        assert info.status == "completed"

    @pytest.mark.asyncio
    async def test_concurrent_double_start_guard(self, db, tmp_path, monkeypatch):
        """Two start_prepared_task calls on a *pending* task must let exactly
        one through; the other hits the atomic guard (ValueError -> 409)."""
        from app.config import settings

        ws_root = str(tmp_path / "ws")
        monkeypatch.setattr(settings, "WORKSPACE_ROOT", ws_root)
        monkeypatch.setattr(TaskManager, "_run", _stub_run_completes)

        manager = TaskManager()
        tid = manager.create_task("并发启动")

        results = await asyncio.gather(
            manager.start_prepared_task(tid),
            manager.start_prepared_task(tid),
            return_exceptions=True,
        )
        task_ids = [r for r in results if r == tid]
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(task_ids) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert "任务已启动" in str(errors[0])
