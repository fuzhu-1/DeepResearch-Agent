"""Tests for ReportService: saving, retrieving, listing reports."""

import os
import pytest
from app.services.report_service import ReportService


@pytest.fixture
def report_service(tmp_path):
    """Create a ReportService with a temp output directory."""
    output_dir = os.path.join(str(tmp_path), "reports")
    return ReportService(output_dir=output_dir)


class TestReportService:
    """Tests for ReportService."""

    @pytest.mark.asyncio
    async def test_save_report_markdown(self, report_service):
        """Should save a markdown report and return metadata."""
        result = await report_service.save_report(
            task_id="test_task_1",
            content="## Findings\n\nContent here.",
            fmt="markdown",
        )
        assert result["task_id"] == "test_task_1"
        assert "report_id" in result
        assert result["report_id"].startswith("rp_")
        assert "markdown_path" in result
        assert result["markdown_path"].endswith(".md")
        assert "pdf_path" not in result

        # Verify file exists
        assert os.path.exists(result["markdown_path"])

    @pytest.mark.asyncio
    async def test_save_report_pdf(self, report_service):
        """Should save a PDF report and return metadata."""
        result = await report_service.save_report(
            task_id="test_task_2",
            content="# PDF Report\n\nPDF content.",
            fmt="pdf",
        )
        assert result["task_id"] == "test_task_2"
        assert "pdf_path" in result
        assert result["pdf_path"].endswith(".pdf")
        assert "markdown_path" not in result

        # Verify file exists
        if result.get("pdf_error"):
            pytest.skip(f"PDF generation failed: {result['pdf_error']}")
        assert os.path.exists(result["pdf_path"])

    @pytest.mark.asyncio
    async def test_save_report_both(self, report_service):
        """Should save both markdown and PDF when fmt='both'."""
        result = await report_service.save_report(
            task_id="test_task_3",
            content="# Both Formats\n\nContent.",
            fmt="both",
        )
        assert "markdown_path" in result
        assert "pdf_path" in result or "pdf_error" in result

        assert os.path.exists(result["markdown_path"])
        if "pdf_path" in result:
            assert os.path.exists(result["pdf_path"])

    @pytest.mark.asyncio
    async def test_get_report_markdown(self, report_service):
        """Should retrieve markdown content."""
        content = "# Test\n\nRetrieval test."
        await report_service.save_report(
            task_id="get_test", content=content, fmt="markdown"
        )
        retrieved = await report_service.get_report("get_test", fmt="markdown")
        assert retrieved is not None
        assert "Test" in retrieved
        assert "Retrieval test." in retrieved

    @pytest.mark.asyncio
    async def test_get_report_nonexistent(self, report_service):
        """Should return None for missing report."""
        result = await report_service.get_report("nonexistent_task")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_reports_empty(self, report_service):
        """Empty service should return empty list."""
        reports = report_service.list_reports()
        assert reports == []

    @pytest.mark.asyncio
    async def test_list_reports(self, report_service):
        """Should list saved reports."""
        await report_service.save_report(
            task_id="list_test_1",
            content="# First\n\nFirst report.",
            fmt="markdown",
        )
        await report_service.save_report(
            task_id="list_test_2",
            content="# Second\n\nSecond report.",
            fmt="markdown",
        )
        reports = report_service.list_reports(limit=10)
        assert len(reports) >= 2

    @pytest.mark.asyncio
    async def test_generate_report_id(self, report_service):
        """Should generate unique IDs starting with rp_."""
        id1 = report_service._generate_report_id()
        id2 = report_service._generate_report_id()
        assert id1.startswith("rp_")
        assert id2.startswith("rp_")
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_save_report_with_sources(self, report_service):
        """Should include sources in saved report."""
        sources = [
            {"title": "Src1", "url": "https://ex.com/1"},
            {"title": "Src2", "url": "https://ex.com/2"},
        ]
        result = await report_service.save_report(
            task_id="src_test",
            content="# Sources\n\nContent",
            fmt="markdown",
            sources=sources,
        )
        assert os.path.exists(result["markdown_path"])
        with open(result["markdown_path"], "r", encoding="utf-8") as f:
            content = f.read()
        assert "## References" in content
        assert "Src1" in content
        assert "https://ex.com/1" in content

    @pytest.mark.asyncio
    async def test_save_report_into_workspace(self, tmp_path):
        ws = os.path.join(str(tmp_path), "workspace")
        os.makedirs(ws, exist_ok=True)
        svc = ReportService(output_dir=os.path.join(str(tmp_path), "reports"))
        result = await svc.save_report(
            task_id="task_ws",
            content="# Report\n\nBody",
            fmt="markdown",
            workspace_dir=ws,
        )
        assert os.path.dirname(os.path.abspath(result["markdown_path"])) == os.path.abspath(ws)

    @pytest.mark.asyncio
    async def test_get_report_falls_back_to_workspace(self, tmp_path):
        """A report saved into a task's workspace should still be retrievable."""
        from app.config import settings
        from app.services.workspace import WorkspaceManager

        # Align the workspace root with settings, as it is in production.
        ws_root = str(tmp_path / "ws")
        settings.WORKSPACE_ROOT = ws_root
        svc = ReportService(output_dir=os.path.join(str(tmp_path), "reports"))
        ws = WorkspaceManager(root_dir=ws_root)
        ws_dir = await ws.ensure_workspace("task_ws")
        await svc.save_report(
            task_id="task_ws",
            content="# Workspace Report\n\nBody",
            fmt="markdown",
            workspace_dir=ws_dir,
        )
        retrieved = await svc.get_report("task_ws", fmt="markdown")
        assert retrieved is not None
        assert "Workspace Report" in retrieved

    @pytest.mark.asyncio
    async def test_get_report_prefers_generated_report_over_upload(self, tmp_path):
        """An uploaded .md in the workspace must not shadow the generated report."""
        from app.config import settings
        from app.services.workspace import WorkspaceManager

        ws_root = str(tmp_path / "ws")
        settings.WORKSPACE_ROOT = ws_root
        svc = ReportService(output_dir=os.path.join(str(tmp_path), "reports"))
        ws = WorkspaceManager(root_dir=ws_root)
        ws_dir = await ws.ensure_workspace("task_mixed")
        # Upload a reference file that sorts before any rp_* report.
        await ws.save_upload("task_mixed", filename="aa_notes.md", content=b"# Uploaded note")
        # Generate the real report into the same workspace.
        await svc.save_report(
            task_id="task_mixed",
            content="# Real Report\n\nGenerated body",
            fmt="markdown",
            workspace_dir=ws_dir,
        )
        retrieved = await svc.get_report("task_mixed", fmt="markdown")
        assert retrieved is not None
        assert "Real Report" in retrieved
        assert "Uploaded note" not in retrieved

    @pytest.mark.asyncio
    async def test_list_reports_excludes_upload_only_workspace(self, tmp_path):
        """A workspace holding only uploaded files is not listed as a report."""
        from app.config import settings
        from app.services.workspace import WorkspaceManager

        ws_root = str(tmp_path / "ws")
        settings.WORKSPACE_ROOT = ws_root
        svc = ReportService(output_dir=os.path.join(str(tmp_path), "reports"))
        ws = WorkspaceManager(root_dir=ws_root)
        await ws.ensure_workspace("task_upload_only")
        await ws.save_upload("task_upload_only", filename="notes.md", content=b"# Note")
        reports = svc.list_reports(limit=50)
        assert "task_upload_only" not in [r.get("task_id") for r in reports]

    @pytest.mark.asyncio
    async def test_list_reports_excludes_rp_named_txt_upload(self, tmp_path):
        """A stray rp_*.txt file must not count as a report marker."""
        from app.config import settings
        from app.services.workspace import WorkspaceManager

        ws_root = str(tmp_path / "ws")
        settings.WORKSPACE_ROOT = ws_root
        svc = ReportService(output_dir=os.path.join(str(tmp_path), "reports"))
        ws = WorkspaceManager(root_dir=ws_root)
        # save_upload now rejects reserved rp_* names, so write the file
        # directly to disk to verify the listing marker is still robust.
        ws_dir = await ws.ensure_workspace("task_rp_txt")
        with open(os.path.join(ws_dir, "rp_notes.txt"), "w", encoding="utf-8") as fh:
            fh.write("# Note")
        reports = svc.list_reports(limit=50)
        assert "task_rp_txt" not in [r.get("task_id") for r in reports]

    @pytest.mark.asyncio
    async def test_get_report_ignores_metadata_path_outside_task_dir(self, tmp_path):
        """Forged metadata pointing outside the task dir must not be read."""
        from app.config import settings
        from app.services.workspace import WorkspaceManager

        ws_root = str(tmp_path / "ws")
        settings.WORKSPACE_ROOT = ws_root
        svc = ReportService(output_dir=os.path.join(str(tmp_path), "reports"))
        ws = WorkspaceManager(root_dir=ws_root)
        ws_dir = await ws.ensure_workspace("task_forged")
        # A real generated report in the workspace.
        await svc.save_report(
            task_id="task_forged",
            content="# Legit Report\n\nBody",
            fmt="markdown",
            workspace_dir=ws_dir,
        )
        # Forge metadata.json with a markdown_path pointing outside the task dir.
        secret = tmp_path / "secret.env"
        secret.write_text("API_KEY=super_secret")
        forged = {
            "report_id": "rp_forged",
            "task_id": "task_forged",
            "markdown_path": str(secret),
        }
        with open(os.path.join(ws_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            import json
            json.dump(forged, fh)
        retrieved = await svc.get_report("task_forged", fmt="markdown")
        assert retrieved is not None
        assert "Legit Report" in retrieved
        assert "super_secret" not in retrieved

    @pytest.mark.asyncio
    async def test_get_report_ignores_cross_drive_metadata_path(self, tmp_path):
        """Forged metadata with a different-drive markdown_path must not raise."""
        from app.config import settings
        from app.services.workspace import WorkspaceManager

        ws_root = str(tmp_path / "ws")
        settings.WORKSPACE_ROOT = ws_root
        svc = ReportService(output_dir=os.path.join(str(tmp_path), "reports"))
        ws = WorkspaceManager(root_dir=ws_root)
        ws_dir = await ws.ensure_workspace("task_crossdrive")
        await svc.save_report(
            task_id="task_crossdrive",
            content="# Cross Drive Report\n\nBody",
            fmt="markdown",
            workspace_dir=ws_dir,
        )
        # A markdown_path on a different drive (Windows) or an absolute path
        # outside the workspace (POSIX) — commonpath raises ValueError for
        # differing Windows drives, which must be contained.
        drive, _ = os.path.splitdrive(os.path.abspath(ws_dir))
        if drive:
            other = "D:" if drive[0].lower() != "d" else "C:"
            cross_path = other + r"\secret.env"
        else:
            cross_path = "/nonexistent/outside/secret.env"
        forged = {
            "report_id": "rp_forged",
            "task_id": "task_crossdrive",
            "markdown_path": cross_path,
        }
        with open(os.path.join(ws_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            import json
            json.dump(forged, fh)
        retrieved = await svc.get_report("task_crossdrive", fmt="markdown")
        assert retrieved is not None
        assert "Cross Drive Report" in retrieved

    @pytest.mark.asyncio
    async def test_get_report_ignores_metadata_path_inside_nonexistent(self, tmp_path):
        """Forged metadata pointing inside the task dir but to a non-existent
        file must not break retrieval of the real report."""
        from app.config import settings
        from app.services.workspace import WorkspaceManager

        ws_root = str(tmp_path / "ws")
        settings.WORKSPACE_ROOT = ws_root
        svc = ReportService(output_dir=os.path.join(str(tmp_path), "reports"))
        ws = WorkspaceManager(root_dir=ws_root)
        ws_dir = await ws.ensure_workspace("task_missing_meta")
        # A real generated report in the workspace.
        await svc.save_report(
            task_id="task_missing_meta",
            content="# Real Report\n\nGenerated body",
            fmt="markdown",
            workspace_dir=ws_dir,
        )
        # Forge metadata.json with a markdown_path inside the task dir that
        # resolves to a non-existent file (never written). This exercises the
        # false-inside case of _path_within (inside the dir, nothing there).
        forged = {
            "report_id": "rp_forged",
            "task_id": "task_missing_meta",
            "markdown_path": os.path.join(ws_dir, "not_written.md"),
        }
        with open(os.path.join(ws_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            import json
            json.dump(forged, fh)
        retrieved = await svc.get_report("task_missing_meta", fmt="markdown")
        assert retrieved is not None
        assert "Real Report" in retrieved
        assert "not_written" not in retrieved
