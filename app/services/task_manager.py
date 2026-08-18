"""Task management service — single source of truth for task lifecycle.

Consolidates task creation, execution, event streaming, and persistence
into one place. Replaces the duplicate logic that previously lived in
both main.py and the old research_service.py.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.models.database import TaskModel, TaskRepository
from app.services.report_service import ReportService
from app.workflow.events import set_event_callback
from app.workflow.graph import run_research as run_workflow

logger = logging.getLogger(__name__)


class TaskInfo:
    """In-memory state for a single research task.

    Kept in memory for fast access; also persisted to SQLite via
    TaskRepository for durability across restarts.
    """

    def __init__(
        self,
        task_id: str,
        task: str,
        status: str = "pending",
    ):
        self.task_id = task_id
        self.task = task
        self.status = status
        self.events: List[Dict[str, Any]] = []
        self.final_report: str = ""
        self.review_score: float = 0.0
        self.review_feedback: str = ""
        self.errors: List[str] = []
        self.token_usage: list = []
        self.created_at: str = datetime.now().isoformat()
        self.completed_at: Optional[str] = None
        self.event_queues: List[asyncio.Queue] = []
        self.current_step: int = 0
        self.iteration_count: int = 0
        self.workspace_dir: str = ""
        self.workspace_files: List[str] = []


class TaskManager:
    """Orchestrates research task lifecycle: create, run, stream, persist."""

    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._report_service = ReportService()

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------

    def create_task(
        self,
        task_text: str,
        task_id: Optional[str] = None,
    ) -> str:
        """Create a new task record and return its ID."""
        tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
        self._tasks[tid] = TaskInfo(task_id=tid, task=task_text, status="pending")
        return tid

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task info by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all in-memory tasks with summary info."""
        results = []
        for task_id, info in self._tasks.items():
            results.append({
                "task_id": task_id,
                "task": info.task,
                "status": info.status,
                "created_at": info.created_at,
                "completed_at": info.completed_at or "",
                "summary": info.final_report[:200] if info.final_report else "",
            })
        return results

    def delete_task(self, task_id: str) -> bool:
        """Remove a task from memory and clean up its workspace directory."""
        removed = self._tasks.pop(task_id, None) is not None
        try:
            from app.config import settings
            from app.services.workspace import WorkspaceManager

            WorkspaceManager(root_dir=settings.WORKSPACE_ROOT).cleanup(task_id)
        except Exception as exc:
            logger.warning("Workspace cleanup failed for %s: %s", task_id, exc)
        return removed

    # ------------------------------------------------------------------
    # Event / SSE helpers
    # ------------------------------------------------------------------

    def push_event(self, task_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Push an event to all SSE listeners for a task."""
        task_info = self._tasks.get(task_id)
        if not task_info:
            return

        event_data = {"type": event_type, **data}
        task_info.events.append(event_data)

        dead_queues: List[asyncio.Queue] = []
        for q in task_info.event_queues:
            try:
                q.put_nowait(event_data)
            except asyncio.QueueFull:
                dead_queues.append(q)
        for q in dead_queues:
            task_info.event_queues.remove(q)

    def register_sse_queue(self, task_id: str, queue: asyncio.Queue) -> None:
        """Register an SSE listener queue for a task."""
        task_info = self._tasks.get(task_id)
        if task_info:
            task_info.event_queues.append(queue)
            # Replay existing events
            for event in task_info.events:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    break

    def unregister_sse_queue(self, task_id: str, queue: asyncio.Queue) -> None:
        """Remove an SSE listener queue."""
        task_info = self._tasks.get(task_id)
        if task_info and queue in task_info.event_queues:
            task_info.event_queues.remove(queue)

    # ------------------------------------------------------------------
    # Checkpoint / recovery
    # ------------------------------------------------------------------

    async def save_checkpoint(self, task_id: str) -> None:
        """Persist current task state as a checkpoint (best-effort).

        Called after each completed workflow node so a service restart
        can resume from the last checkpoint.
        """
        task_info = self._tasks.get(task_id)
        if not task_info:
            return
        try:
            from app.models.database import _async_session_maker

            if _async_session_maker is None:
                return
            async with _async_session_maker() as session:
                repo = TaskRepository(session)
                task = await repo.get(task_id)
                if task:
                    task.status = task_info.status
                    task.report = task_info.final_report
                    task.review_score = task_info.review_score
                    task.review_feedback = task_info.review_feedback
                    task.errors = json.dumps(task_info.errors)
                    await repo.update(task)
                else:
                    # Create if not exists
                    from app.models.database import TaskModel
                    await repo.create(TaskModel(
                        id=task_id,
                        task_text=task_info.task,
                        status=task_info.status,
                        report=task_info.final_report,
                        review_score=task_info.review_score,
                        review_feedback=task_info.review_feedback,
                        errors=json.dumps(task_info.errors),
                    ))
        except Exception as exc:
            logger.warning("Failed to save checkpoint for %s: %s", task_id, exc)

    async def recover_interrupted_tasks(self) -> None:
        """Handle tasks left running after a restart.

        Default: mark them failed with a clear message. Auto-resume is
        opt-in via RESUME_INTERRUPTED_TASKS=true to avoid surprise runs
        and unexpected token usage at startup.
        """
        try:
            from app.config import settings
            from app.models.database import _async_session_maker

            if _async_session_maker is None:
                return
            async with _async_session_maker() as session:
                repo = TaskRepository(session)
                from sqlalchemy import select
                result = await session.execute(
                    select(TaskModel).where(TaskModel.status.in_(["pending", "running"]))
                )
                interrupted = list(result.scalars().all())
                for task in interrupted:
                    if not settings.RESUME_INTERRUPTED_TASKS:
                        logger.info("Marking interrupted task %s as failed (restart)", task.id)
                        task.status = "failed"
                        task.errors = json.dumps(
                            ["任务因服务重启而中断；如需自动续跑，请设置 RESUME_INTERRUPTED_TASKS=true"]
                        )
                        await repo.update(task)
                        continue
                    try:
                        from app.workflow.graph import run_research_resume

                        final_state = await run_research_resume(task.id)
                        task.status = final_state.status
                        task.report = final_state.final_report
                        task.review_score = final_state.review_score
                        task.review_feedback = final_state.review_feedback
                        task.errors = json.dumps(final_state.errors)
                        task.research_data = json.dumps(
                            final_state.research_data, ensure_ascii=False, default=str
                        )
                        task.sources = json.dumps(
                            final_state.sources, ensure_ascii=False, default=str
                        )
                        task.completed_at = (
                            datetime.now() if final_state.status == "completed" else None
                        )
                        logger.info("Resumed interrupted task %s (%s)", task.id, task.status)
                    except Exception as exc:
                        logger.warning(
                            "Resume failed for %s, marking failed: %s", task.id, exc
                        )
                        task.status = "failed"
                        task.errors = json.dumps([f"Resume failed: {exc}"])
                    await repo.update(task)
        except Exception as exc:
            logger.warning("Failed to recover interrupted tasks: %s", exc)

    # ------------------------------------------------------------------
    # Background execution
    # ------------------------------------------------------------------

    async def start_research(
        self,
        task_text: str,
        max_iterations: int = 3,
        fmt: str = "markdown",
        use_rag: bool = False,
        profile_id: Optional[str] = None,
    ) -> str:
        """Create and launch a background research task.

        Returns the task_id immediately while research runs asynchronously.
        """
        task_id = self.create_task(task_text)
        task_info = self._tasks[task_id]

        from app.config import settings
        from app.services.workspace import WorkspaceManager

        ws = WorkspaceManager(root_dir=settings.WORKSPACE_ROOT)
        task_info.workspace_dir = await ws.ensure_workspace(task_id)
        task_info.workspace_files = [f["name"] for f in ws.list_files(task_id)]

        # Persist to database
        try:
            from app.models.database import _async_session_maker

            if _async_session_maker is not None:
                async with _async_session_maker() as session:
                    repo = TaskRepository(session)
                    await repo.create(TaskModel(
                        id=task_id,
                        task_text=task_text,
                        status="pending",
                    ))
        except Exception as exc:
            logger.warning("Failed to persist task to database: %s", exc)

        # Launch background runner
        asyncio.create_task(
            self._run(task_id, task_text, max_iterations, fmt, use_rag, profile_id)
        )

        # Save initial checkpoint
        asyncio.create_task(self.save_checkpoint(task_id))

        return task_id

    async def start_prepared_task(
        self,
        task_id: str,
        max_iterations: int = 3,
        fmt: str = "markdown",
        use_rag: bool = False,
        profile_id: Optional[str] = None,
    ) -> str:
        """Start background research for an existing pre-created task.

        Same as start_research but reuses an existing task/workspace instead
        of creating a new one. Raises KeyError if the task does not exist.
        """
        task_info = self._tasks.get(task_id)
        if not task_info:
            raise KeyError(f"task not found: {task_id}")

        from app.config import settings
        from app.services.workspace import WorkspaceManager

        ws = WorkspaceManager(root_dir=settings.WORKSPACE_ROOT)
        task_info.workspace_dir = await ws.ensure_workspace(task_id)
        # Re-sync so any files uploaded after prepare are picked up.
        task_info.workspace_files = [f["name"] for f in ws.list_files(task_id)]

        # Persist to database (was intentionally NOT persisted at prepare time)
        try:
            from app.models.database import _async_session_maker

            if _async_session_maker is not None:
                async with _async_session_maker() as session:
                    repo = TaskRepository(session)
                    await repo.create(TaskModel(
                        id=task_id,
                        task_text=task_info.task,
                        status="pending",
                    ))
        except Exception as exc:
            logger.warning("Failed to persist task to database: %s", exc)

        asyncio.create_task(
            self._run(task_id, task_info.task, max_iterations, fmt, use_rag, profile_id)
        )
        asyncio.create_task(self.save_checkpoint(task_id))
        return task_id

    async def _run(
        self,
        task_id: str,
        task_text: str,
        max_iterations: int,
        fmt: str,
        use_rag: bool,
        profile_id: Optional[str] = None,
    ) -> None:
        """Background runner: executes the LangGraph workflow."""
        task_info = self._tasks[task_id]
        task_info.status = "running"

        # Update database status
        await self._persist_status(task_id, "running")

        # Wire up event callback so workflow nodes emit events
        def _event_callback(event_type: str, data: Dict[str, Any]) -> None:
            self.push_event(task_id, event_type, data)

        set_event_callback(_event_callback)
        from app.utils.llm import set_usage_meter

        usage_meter: list = []
        set_usage_meter(usage_meter)

        try:
            self.push_event(task_id, "agent_status", {
                "agent": "System",
                "status": "running",
                "detail": "Starting research pipeline",
            })

            self.push_event(task_id, "agent_status", {
                "agent": "Workflow",
                "status": "running",
                "detail": "Running LangGraph research pipeline",
            })

            final_state = await run_workflow(
                task_text,
                use_rag=use_rag,
                profile_id=profile_id,
                max_iterations=max_iterations,
                task_id=task_id,
            )

            # Extract results (supports both object and dict-like return)
            if hasattr(final_state, "status"):
                state_status = final_state.status
                state_errors = final_state.errors
                report_text = final_state.final_report
                review_score = final_state.review_score
                review_feedback = final_state.review_feedback
                research_data = getattr(final_state, "research_data", []) or []
                sources = getattr(final_state, "sources", []) or []
            else:
                state_status = final_state.get("status", "failed")
                state_errors = final_state.get("errors", [])
                report_text = final_state.get("final_report", "")
                review_score = final_state.get("review_score", 0.0)
                review_feedback = final_state.get("review_feedback", "")
                research_data = final_state.get("research_data", []) or []
                sources = final_state.get("sources", []) or []

            if state_status == "failed":
                raise ValueError(state_errors[-1] if state_errors else "Research workflow failed")

            # Validate citations and append a verification section
            try:
                from app.utils.citation_validator import (
                    extract_citations,
                    render_validation_section,
                    validate_citations,
                )

                checks = await validate_citations(extract_citations(report_text))
                report_text += render_validation_section(checks)
            except Exception as exc:
                logger.warning("Citation validation failed: %s", exc)

            # Claim-evidence grounding audit
            try:
                from app.utils.grounding import GroundingChecker, render_evidence_table

                grounding_checks = await GroundingChecker().check_report(report_text)
                report_text += render_evidence_table(grounding_checks)
            except Exception as exc:
                logger.warning("Grounding check failed: %s", exc)

            # Stream report chunks
            chunk_size = 500
            for i in range(0, len(report_text), chunk_size):
                chunk = report_text[i: i + chunk_size]
                self.push_event(task_id, "report_chunk", {"chunk": chunk})
                await asyncio.sleep(0.02)

            # Save report
            try:
                await self._report_service.save_report(
                    task_id=task_id,
                    content=report_text,
                    fmt=fmt,
                    task_display=task_text,
                    sources=[{"title": task_text, "url": ""}],
                    workspace_dir=task_info.workspace_dir,
                )
            except Exception as save_err:
                logger.warning("Failed to save report: %s", save_err)

            # Mark complete
            task_info.final_report = report_text
            task_info.review_score = review_score
            task_info.review_feedback = review_feedback
            task_info.token_usage = list(usage_meter)
            total_tokens = sum(u.get("total_tokens", 0) for u in usage_meter)
            task_info.status = "completed"
            task_info.completed_at = datetime.now().isoformat()

            await self._persist_completion(
                task_id,
                report_text,
                review_score,
                review_feedback,
                research_data=research_data,
                sources=sources,
                total_tokens=total_tokens,
            )

            self.push_event(task_id, "completed", {
                "summary": "Research complete",
                "score": review_score,
                "report": report_text,
            })

            # Trigger post-task evolution analysis (background, non-blocking)
            try:
                from app.services.evolution_service import analyze_task

                asyncio.create_task(analyze_task(task_id, final_state))
            except Exception as exc:
                logger.warning("Evolution analysis dispatch failed: %s", exc)

        except Exception as exc:
            logger.exception("Research task %s failed", task_id)
            task_info.status = "failed"
            task_info.errors.append(str(exc))
            task_info.completed_at = datetime.now().isoformat()

            await self._persist_status(task_id, "failed", error=str(exc))

            self.push_event(task_id, "error", {
                "message": str(exc),
                "detail": "An error occurred during research",
            })
        finally:
            set_event_callback(None)
            set_usage_meter(None)

    # ------------------------------------------------------------------
    # Database persistence helpers
    # ------------------------------------------------------------------

    async def _persist_status(self, task_id: str, status: str, error: str = "") -> None:
        """Update task status in database (best-effort)."""
        try:
            from app.models.database import _async_session_maker

            if _async_session_maker is None:
                return
            async with _async_session_maker() as session:
                repo = TaskRepository(session)
                task = await repo.get(task_id)
                if task:
                    task.status = status
                    if error:
                        task.errors = json.dumps([error])
                    await repo.update(task)
        except Exception as exc:
            logger.warning("Failed to persist status for %s: %s", task_id, exc)

    async def _persist_completion(
        self,
        task_id: str,
        report: str,
        score: float,
        feedback: str,
        research_data: Optional[list] = None,
        sources: Optional[list] = None,
        total_tokens: int = 0,
    ) -> None:
        """Persist completed task data (best-effort)."""
        try:
            from app.models.database import ReportModel, ReportRepository, _async_session_maker

            if _async_session_maker is None:
                return
            async with _async_session_maker() as session:
                # Update task
                repo = TaskRepository(session)
                task = await repo.get(task_id)
                if task:
                    task.status = "completed"
                    task.report = report
                    task.review_score = score
                    task.review_feedback = feedback
                    task.completed_at = datetime.now()
                    task.research_data = json.dumps(
                        research_data or [], ensure_ascii=False, default=str
                    )
                    task.sources = json.dumps(
                        sources or [], ensure_ascii=False, default=str
                    )
                    task.total_tokens = total_tokens
                    await repo.update(task)

                # Save report
                report_repo = ReportRepository(session)
                await report_repo.create(ReportModel(
                    id=f"rp_{uuid.uuid4().hex[:12]}",
                    task_id=task_id,
                    content=report,
                    format="markdown",
                ))
        except Exception as exc:
            logger.warning("Failed to persist completion for %s: %s", task_id, exc)
