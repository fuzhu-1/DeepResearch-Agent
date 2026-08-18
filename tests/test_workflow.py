"""Tests for the LangGraph workflow graph."""

import pytest

from app.models.state import ResearchState, SubTask
from app.workflow.events import set_event_callback
from app.workflow.graph import build_graph
from app.workflow.nodes import (
    executor_node,
    router_decision,
)


@pytest.fixture(autouse=True)
async def _init_db():
    """Hermetic in-memory DB so profile lookups work in every test."""
    from app.models.database import close_db, init_db

    await init_db("sqlite+aiosqlite://")
    yield
    await close_db()


class TestResearchState:
    """Test ResearchState model creation."""

    def test_create_state(self):
        state = ResearchState(task="Test task")
        assert state.task == "Test task"
        assert state.status == "pending"
        assert state.plan == []
        assert state.current_step == 0
        assert state.report_draft == ""

    def test_subtask_model(self):
        st = SubTask(id="step-1", description="Do something", tool="search")
        assert st.id == "step-1"
        assert st.status == "pending"
        assert st.result is None

    def test_state_has_workspace_fields(self):
        state = ResearchState(task="Test")
        assert state.workspace_dir == ""
        assert state.workspace_files == []


class TestRouterDecision:
    """Test router_decision logic (replaces old router_node)."""

    def test_router_no_plan(self):
        state = ResearchState(task="test")
        assert router_decision(state) == "planner"

    def test_router_has_plan(self):
        state = ResearchState(
            task="test",
            plan=[SubTask(id="s1", description="Step 1", tool="search")],
        )
        assert router_decision(state) == "researcher"

    def test_router_complete(self):
        state = ResearchState(
            task="test",
            plan=[SubTask(id="s1", description="Step 1", tool="search")],
            current_step=1,
        )
        assert router_decision(state) == "writer"

    def test_router_failed(self):
        state = ResearchState(task="test", status="failed")
        assert router_decision(state) == "END"

    def test_router_low_score_loops_back(self):
        state = ResearchState(
            task="test",
            plan=[SubTask(id="s1", description="Step 1", tool="search")],
            current_step=1,
            report_draft="draft",
            review_score=0.4,
            iteration_count=0,
        )
        assert router_decision(state) == "writer"

    def test_router_good_score_goes_to_formatter(self):
        state = ResearchState(
            task="test",
            plan=[SubTask(id="s1", description="Step 1", tool="search")],
            current_step=1,
            report_draft="draft",
            review_score=0.85,
        )
        assert router_decision(state) == "formatter"

    def test_router_uses_max_iterations_from_state(self):
        state = ResearchState(
            task="test",
            plan=[SubTask(id="s1", description="Step 1", tool="search")],
            current_step=1,
            report_draft="draft",
            review_score=0.2,
            iteration_count=2,
            max_iterations=2,
        )
        assert router_decision(state) == "formatter"

    def test_router_loops_when_iteration_budget_remains(self):
        state = ResearchState(
            task="test",
            plan=[SubTask(id="s1", description="Step 1", tool="search")],
            current_step=1,
            report_draft="draft",
            review_score=0.2,
            iteration_count=2,
            max_iterations=5,
        )
        assert router_decision(state) == "writer"


class TestGraphBuilding:
    """Test that the graph compiles and runs."""

    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    @pytest.mark.asyncio
    async def test_graph_runs_simple_flow(self):
        """Test the graph runs end-to-end with a simple flow.

        NOTE: This test requires a live LLM API key. It will be skipped
        if neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set.
        """
        import os
        has_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
        if not has_key:
            pytest.skip("No LLM API key configured — skipping end-to-end graph test")

        set_event_callback(None)

        graph = build_graph()
        state = ResearchState(task="What is Python?")
        result = await graph.ainvoke(state)

        assert result is not None
        assert isinstance(result, dict) or isinstance(result, ResearchState)

        if isinstance(result, dict):
            final_report = result.get("final_report", "")
        else:
            final_report = result.final_report
        # The graph should produce a final report (even if fallback)
        assert final_report != ""


class TestExecutorNode:
    """Test the executor node."""

    @pytest.mark.asyncio
    async def test_executor_executes_step(self):
        state = ResearchState(
            task="test",
            plan=[SubTask(id="s1", description="Step 1", tool="search")],
        )
        result = await executor_node(state)
        assert result["current_step"] == 1
        assert len(result["research_data"]) == 1

    @pytest.mark.asyncio
    async def test_executor_no_steps_left(self):
        state = ResearchState(
            task="test",
            plan=[SubTask(id="s1", description="Step 1", tool="search")],
            current_step=1,
        )
        result = await executor_node(state)
        assert result.get("current_step", 1) == 1

    @pytest.mark.asyncio
    async def test_executor_runs_steps_in_parallel(self, monkeypatch):
        from app.agents.researcher import ResearcherAgent
        from app.config import settings

        monkeypatch.setattr(settings, "RESEARCH_PARALLELISM", 3)
        seen: list = []

        async def fake_step(self, state, step_index, tools=None):
            seen.append(step_index)
            return (
                {
                    "step": step_index,
                    "task_id": f"t{step_index}",
                    "description": f"d{step_index}",
                    "tool": "search",
                    "raw_result": "x",
                    "summary": "s",
                },
                [],
            )

        monkeypatch.setattr(ResearcherAgent, "execute_step", fake_step)

        plan = [SubTask(id=f"t{i}", description=f"d{i}", tool="search") for i in range(4)]
        state = ResearchState(task="test", plan=plan)
        result = await executor_node(state)

        assert seen == [0, 1, 2]
        assert result["current_step"] == 3
        assert len(result["research_data"]) == 3


@pytest.mark.asyncio
async def test_planner_node_stores_perspectives(monkeypatch):
    from app.workflow import nodes

    async def fake_invoke(self, state):
        return {
            "plan": [SubTask(id="s1", description="d", tool="search")],
            "perspectives": ["技术", "市场"],
        }

    monkeypatch.setattr("app.agents.planner.PlannerAgent.invoke", fake_invoke)
    state = ResearchState(task="t")
    updates = await nodes.planner_node(state)
    assert updates["perspectives"] == ["技术", "市场"]
