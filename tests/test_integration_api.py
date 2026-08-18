"""End-to-end integration tests using httpx.AsyncClient.

Tests the full API lifecycle including auth, research, settings,
knowledge base, and health check endpoints.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.database import close_db, init_db


@pytest.fixture
def transport():
    return ASGITransport(app=app)


@pytest.fixture
async def client(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def db():
    """Initialise an in-memory SQLite database (research/profile APIs need it)."""
    await init_db("sqlite+aiosqlite://")
    yield
    await close_db()


class TestHealth:
    """Health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_health_has_request_id(self, client):
        resp = await client.get("/health")
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0


class TestAuth:
    """Registration, login, token refresh, and protected endpoints."""

    @pytest.mark.asyncio
    async def test_register_and_login(self, client):
        # Register
        reg = await client.post("/api/auth/register", json={
            "username": "testuser",
            "password": "testpass123",
        })
        assert reg.status_code == 200
        assert reg.json()["username"] == "testuser"

        # Login
        login = await client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "testpass123",
        })
        assert login.status_code == 200
        tokens = login.json()
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_register_duplicate(self, client):
        reg = await client.post("/api/auth/register", json={
            "username": "dupeuser",
            "password": "pass123",
        })
        assert reg.status_code == 200

        dup = await client.post("/api/auth/register", json={
            "username": "dupeuser",
            "password": "otherpass",
        })
        assert dup.status_code == 409

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client):
        await client.post("/api/auth/register", json={
            "username": "wrongpwd",
            "password": "correctpass",
        })
        login = await client.post("/api/auth/login", json={
            "username": "wrongpwd",
            "password": "wrongpass",
        })
        assert login.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token(self, client):
        await client.post("/api/auth/register", json={
            "username": "refreshtest",
            "password": "pass123",
        })
        login = await client.post("/api/auth/login", json={
            "username": "refreshtest",
            "password": "pass123",
        })
        tokens = login.json()

        refresh = await client.post("/api/auth/refresh", json={
            "refresh_token": tokens["refresh_token"],
        })
        assert refresh.status_code == 200
        new_tokens = refresh.json()
        assert "access_token" in new_tokens

    @pytest.mark.asyncio
    async def test_me_endpoint(self, client):
        await client.post("/api/auth/register", json={
            "username": "metest",
            "password": "pass123",
        })
        login = await client.post("/api/auth/login", json={
            "username": "metest",
            "password": "pass123",
        })
        token = login.json()["access_token"]

        me = await client.get("/api/auth/me", headers={
            "Authorization": f"Bearer {token}",
        })
        assert me.status_code == 200
        assert me.json()["username"] == "metest"

    @pytest.mark.asyncio
    async def test_me_unauthorized(self, client):
        me = await client.get("/api/auth/me")
        assert me.status_code == 401


class TestResearch:
    """Research task lifecycle via API."""

    @pytest.mark.asyncio
    async def test_start_research(self, db, client):
        resp = await client.post("/api/research", json={
            "task": "test research task",
            "max_iterations": 1,
            "format": "markdown",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_task_status(self, db, client):
        start = await client.post("/api/research", json={
            "task": "status test",
            "max_iterations": 1,
        })
        task_id = start.json()["task_id"]

        status = await client.get(f"/api/research/{task_id}")
        assert status.status_code == 200
        data = status.json()
        assert data["task_id"] == task_id

    @pytest.mark.asyncio
    async def test_nonexistent_task(self, client):
        resp = await client.get("/api/research/nonexistent-task-id")
        assert resp.status_code == 404


class TestSettings:
    """Settings API (non-destructive tests only)."""

    @pytest.mark.asyncio
    async def test_get_settings(self, client):
        resp = await client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        # Should have either configured=True or configured=False
        assert "configured" in data

    @pytest.mark.asyncio
    async def test_post_settings_empty_key(self, client):
        resp = await client.post("/api/settings", json={
            "provider": "openai",
            "api_key": "",
            "model": "gpt-4o",
        })
        assert resp.status_code == 422


class TestKnowledge:
    """Knowledge base API."""

    @pytest.mark.asyncio
    async def test_ingest_document(self, client):
        resp = await client.post("/api/knowledge/ingest", json={
            "content": "Test document content about AI agents.",
            "source": "test_integration",
            "doc_type": "text",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_ingest_empty_content(self, client):
        resp = await client.post("/api/knowledge/ingest", json={
            "content": "",
            "source": "test",
        })
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_knowledge_search(self, client):
        resp = await client.get("/api/knowledge/search?q=AI+agents&k=3")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, (dict, list))


class TestCORS:
    """CORS headers."""

    @pytest.mark.asyncio
    async def test_cors_header_present(self, client):
        resp = await client.get("/health", headers={
            "Origin": "http://localhost:5173",
        })
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


class TestHistoryPagination:
    @pytest.mark.asyncio
    async def test_history_pagination(self, client):
        from app.main import _task_manager

        for i in range(5):
            _task_manager.create_task(f"task {i}", task_id=f"t{i}")
        resp = await client.get("/api/history?page=2&per_page=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 5
        assert data["total_pages"] >= 3
        assert data["page"] == 2
        assert len(data["tasks"]) == 2


class TestPdfCache:
    @pytest.mark.asyncio
    async def test_pdf_download_caches_file(self, client, monkeypatch):
        from app.main import _task_manager

        calls = {"n": 0}

        async def fake_generate(content, path):
            calls["n"] += 1
            with open(path, "w", encoding="utf-8") as f:
                f.write("%PDF-1.4 fake")

        monkeypatch.setattr("app.main.generate_pdf", fake_generate)

        tid = _task_manager.create_task("t")
        _task_manager._tasks[tid].status = "completed"
        _task_manager._tasks[tid].final_report = "# report"

        await client.get(f"/api/reports/{tid}?format=pdf")
        await client.get(f"/api/reports/{tid}?format=pdf")
        assert calls["n"] == 1


@pytest.mark.asyncio
async def test_worker_job_runs_task(monkeypatch, tmp_path):
    from app.config import settings
    from app.services.task_manager import TaskManager
    from app.worker import run_research_job

    monkeypatch.setattr(settings, "WORKSPACE_ROOT", str(tmp_path / "ws"))

    async def fake_run(self, task_id, task_text, max_iterations, fmt, use_rag, profile_id):
        self._tasks[task_id].status = "completed"
        self._tasks[task_id].final_report = "# done"

    monkeypatch.setattr(TaskManager, "_run", fake_run)
    ctx = {"task_manager": TaskManager()}
    await run_research_job(ctx, "w1", "test", 1, "markdown", False, "default")
    assert ctx["task_manager"]._tasks["w1"].status == "completed"


@pytest.mark.asyncio
async def test_worker_job_provisions_workspace(monkeypatch, tmp_path):
    """ARQ worker must provision the task workspace before running."""
    from app.config import settings
    from app.services.task_manager import TaskManager
    from app.worker import run_research_job

    monkeypatch.setattr(settings, "WORKSPACE_ROOT", str(tmp_path / "ws"))

    async def fake_run(self, task_id, task_text, max_iterations, fmt, use_rag, profile_id):
        pass

    monkeypatch.setattr(TaskManager, "_run", fake_run)
    ctx = {"task_manager": TaskManager()}
    await run_research_job(ctx, "w_ws", "test", 1, "markdown", False, "default")
    info = ctx["task_manager"]._tasks["w_ws"]
    assert info.workspace_dir
    assert os.path.isdir(info.workspace_dir)


class TestWorkspaceAPI:
    """Upload and workspace browsing endpoints."""

    @pytest.mark.asyncio
    async def test_upload_and_list(self, client, db, tmp_path):
        # Override WORKSPACE_ROOT so tests don't pollute real data dir
        from app.config import settings
        settings.WORKSPACE_ROOT = str(tmp_path / "ws")

        # Create task
        resp = await client.post("/api/research", json={
            "task": "测试工作目录上传",
            "max_iterations": 1,
        })
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        # Upload attachment
        up = await client.post(
            f"/api/research/{task_id}/upload",
            files={"file": ("notes.md", "# 参考笔记\n\n内容".encode("utf-8"), "text/markdown")},
        )
        assert up.status_code == 200
        assert up.json()["name"] == "notes.md"

        # Browse workspace
        lst = await client.get(f"/api/research/{task_id}/workspace")
        assert lst.status_code == 200
        names = [f["name"] for f in lst.json()["files"]]
        assert "notes.md" in names

    @pytest.mark.asyncio
    async def test_upload_missing_task_404(self, client, tmp_path):
        from app.config import settings
        settings.WORKSPACE_ROOT = str(tmp_path / "ws")

        up = await client.post(
            "/api/research/no-such-task/upload",
            files={"file": ("notes.md", b"content", "text/markdown")},
        )
        assert up.status_code == 404

    @pytest.mark.asyncio
    async def test_workspace_missing_task_404(self, client, tmp_path):
        from app.config import settings
        settings.WORKSPACE_ROOT = str(tmp_path / "ws")

        lst = await client.get("/api/research/no-such-task/workspace")
        assert lst.status_code == 404

    @pytest.mark.asyncio
    async def test_upload_reserved_report_name_rejected(self, client, db, tmp_path):
        """metadata.json must not be uploadable (report namespace is reserved)."""
        from app.config import settings
        settings.WORKSPACE_ROOT = str(tmp_path / "ws")

        resp = await client.post("/api/research", json={
            "task": "保留命名上传",
            "max_iterations": 1,
        })
        task_id = resp.json()["task_id"]

        up = await client.post(
            f"/api/research/{task_id}/upload",
            files={"file": ("metadata.json", b'{"markdown_path": "/etc/passwd"}', "application/json")},
        )
        assert up.status_code == 400
        assert "保留命名" in up.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_respects_max_files_cap(self, client, db, tmp_path, monkeypatch):
        """Uploading beyond UPLOAD_MAX_FILES is rejected."""
        from app.config import settings
        settings.WORKSPACE_ROOT = str(tmp_path / "ws")
        monkeypatch.setattr(settings, "UPLOAD_MAX_FILES", 2)

        resp = await client.post("/api/research", json={
            "task": "上传数量上限",
            "max_iterations": 1,
        })
        task_id = resp.json()["task_id"]

        first = await client.post(
            f"/api/research/{task_id}/upload",
            files={"file": ("one.md", b"# one", "text/markdown")},
        )
        assert first.status_code == 200
        second = await client.post(
            f"/api/research/{task_id}/upload",
            files={"file": ("two.md", b"# two", "text/markdown")},
        )
        assert second.status_code == 200

        third = await client.post(
            f"/api/research/{task_id}/upload",
            files={"file": ("three.md", b"# three", "text/markdown")},
        )
        assert third.status_code == 413
