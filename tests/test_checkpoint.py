"""Tests for LangGraph checkpointing and resume."""

from typing import TypedDict

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph


class S(TypedDict, total=False):
    count: int


@pytest.mark.asyncio
async def test_resume_from_checkpoint_skips_done_nodes(tmp_path):
    calls: list = []

    def node1(state):
        calls.append("node1")
        return {"count": 1}

    def node2(state):
        calls.append("node2")
        return {"count": state.get("count", 0) + 1}

    builder = StateGraph(S)
    builder.add_node("n1", node1)
    builder.add_node("n2", node2)
    builder.set_entry_point("n1")
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", END)

    path = tmp_path / "ck.db"
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        graph = builder.compile(checkpointer=saver)
        await graph.ainvoke({}, {"configurable": {"thread_id": "t1"}})
        assert calls == ["node1", "node2"]
        calls.clear()

        # Resume with None input and the same thread_id: nothing left to run.
        result = await graph.ainvoke(None, {"configurable": {"thread_id": "t1"}})
        assert calls == []
        assert result == {"count": 2}


@pytest.mark.asyncio
async def test_run_research_writes_checkpoint(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setattr(settings, "CHECKPOINT_DB_PATH", str(tmp_path / "ckpt.db"))

    from app.workflow import graph as graph_mod

    async def fake_planner_llm(system_prompt, user_prompt, config=None, tools=None):
        return '[{"id":"s1","description":"搜索测试","tool":"search"}]'

    async def fake_other_llm(system_prompt, user_prompt, config=None, tools=None):
        return "mock response"

    monkeypatch.setattr("app.agents.planner.llm_call", fake_planner_llm)
    monkeypatch.setattr("app.agents.researcher.llm_call", fake_other_llm)
    monkeypatch.setattr("app.agents.writer.llm_call", fake_other_llm)
    monkeypatch.setattr("app.agents.reviewer.llm_call", fake_other_llm)

    from app.models.database import close_db, init_db

    await init_db("sqlite+aiosqlite://")
    try:
        result = await graph_mod.run_research("test", max_iterations=1, task_id="ck_task")
    finally:
        await close_db()

    status = result.status if hasattr(result, "status") else result.get("status")
    assert status == "completed"
    assert (tmp_path / "ckpt.db").exists()
