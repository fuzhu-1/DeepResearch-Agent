"""LangGraph workflow graph definition for the research pipeline."""

import logging
import uuid
from typing import Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, END

from langgraph.graph import StateGraph, END

from app.models.state import ResearchState
from app.workflow.nodes import (
    planner_node,
    executor_node,
    writer_node,
    reviewer_node,
    formatter_node,
    router_decision,
)

logger = logging.getLogger(__name__)


def build_graph(checkpointer=None) -> StateGraph:
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", executor_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("formatter", formatter_node)

    # Entry point: conditional start
    workflow.set_conditional_entry_point(
        router_decision,
        {
            "planner": "planner",
            "researcher": "researcher",
            "writer": "writer",
            "formatter": "formatter",
            "END": END,
        },
    )

    # After planner, go to researcher
    workflow.add_edge("planner", "researcher")

    # After researcher, check if more subtasks remain
    workflow.add_conditional_edges(
        "researcher",
        router_decision,
        {
            "researcher": "researcher",
            "writer": "writer",
            "formatter": "formatter",
            "END": END,
        },
    )

    # After writer, review
    workflow.add_edge("writer", "reviewer")

    # After reviewer, loop back to writer or finalize
    workflow.add_conditional_edges(
        "reviewer",
        router_decision,
        {
            "writer": "writer",
            "formatter": "formatter",
            "END": END,
        },
    )

    # Done
    workflow.add_edge("formatter", END)

    return workflow.compile(checkpointer=checkpointer)


async def run_research(
    task: str,
    use_rag: bool = False,
    profile_id: Optional[str] = None,
    max_iterations: int = 3,
    task_id: Optional[str] = None,
) -> ResearchState:
    """
    Convenience function to run a research task end-to-end.

    Args:
        task: The research topic/question.
        use_rag: Whether to enable RAG knowledge base retrieval.
        max_iterations: Maximum reviewer iteration rounds.
        task_id: Stable thread id for checkpointing/resume.

    Returns:
        Final ResearchState after workflow completion.
    """
    from app.config import settings
    from app.services.workspace import WorkspaceManager

    resolved_task_id = task_id or f"task_{uuid.uuid4().hex[:12]}"

    async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DB_PATH) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {
            "configurable": {
                "thread_id": resolved_task_id
            }
        }
        ws = WorkspaceManager(root_dir=settings.WORKSPACE_ROOT)
        workspace_dir = await ws.ensure_workspace(resolved_task_id)
        workspace_files = [f["name"] for f in ws.list_files(resolved_task_id)]

        initial_state = ResearchState(
            task=task,
            use_rag=use_rag,
            profile_id=profile_id,
            max_iterations=max_iterations,
            workspace_dir=workspace_dir,
            workspace_files=workspace_files,
        )
        return await graph.ainvoke(initial_state, config=config)


async def run_research_resume(task_id: str) -> ResearchState:
    """Resume an interrupted workflow from its latest checkpoint."""
    from app.config import settings
    from app.services.workspace import WorkspaceManager

    await WorkspaceManager(root_dir=settings.WORKSPACE_ROOT).ensure_workspace(task_id)

    async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DB_PATH) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": task_id}}
        return await graph.ainvoke(None, config=config)
