"""Tests for the self-evolution module: drafts, analysis, accept/reject, and API."""

import json

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.models.database import (
    EvolutionDraftModel,
    EvolutionDraftRepository,
    init_db,
    close_db,
)
from app.models.state import ResearchState, SubTask
from app.services.evolution_service import (
    accept_draft,
    analyze_task,
    list_drafts,
    reject_draft,
)


@pytest.fixture
async def db():
    await init_db("sqlite+aiosqlite://")
    yield
    await close_db()


@pytest.fixture(autouse=True)
def clear_caches():
    from app.services.profile_service import invalidate_profiles_cache
    from app.services.skill_service import invalidate_cache as invalidate_skills_cache

    invalidate_profiles_cache()
    invalidate_skills_cache()
    yield
    invalidate_profiles_cache()
    invalidate_skills_cache()


@pytest.fixture
async def session(db):
    from app.models.database import _async_session_maker

    async with _async_session_maker() as s:
        yield s


def draft_payload(**overrides):
    payload = {
        "id": "draft_1",
        "task_id": "task_1",
        "profile_id": "profile_default",
        "review_score": 0.85,
        "review_feedback": "来源标注规范，结构完整。",
        "lesson": "保持严格的来源标注习惯。",
        "draft_name": "citation-discipline",
        "draft_description": "严格来源标注",
        "draft_trigger_keywords": '["引用", "来源"]',
        "draft_agents": '["writer"]',
        "draft_content": "每个关键事实必须标注来源。",
        "promote_global": False,
        "status": "pending",
    }
    payload.update(overrides)
    return payload


class TestEvolutionDraftRepository:
    async def test_create_and_get(self, session):
        repo = EvolutionDraftRepository(session)
        created = await repo.create(EvolutionDraftModel(**draft_payload()))
        assert created.id == "draft_1"
        fetched = await repo.get("draft_1")
        assert fetched is not None
        assert fetched.draft_name == "citation-discipline"

    async def test_get_by_task(self, session):
        repo = EvolutionDraftRepository(session)
        await repo.create(EvolutionDraftModel(**draft_payload()))
        assert (await repo.get_by_task("task_1")) is not None
        assert await repo.get_by_task("task_missing") is None

    async def test_list_for_profile_with_status(self, session):
        repo = EvolutionDraftRepository(session)
        await repo.create(EvolutionDraftModel(**draft_payload()))
        await repo.create(
            EvolutionDraftModel(
                **draft_payload(id="draft_2", task_id="task_2", status="rejected")
            )
        )
        pending = await repo.list_for_profile("profile_default", status="pending")
        assert [d.id for d in pending] == ["draft_1"]
        all_drafts = await repo.list_for_profile("profile_default", status=None)
        assert len(all_drafts) == 2
        other = await repo.list_for_profile("profile_other", status="pending")
        assert other == []


class TestAnalyzeTask:
    async def test_mid_score_skipped(self, db):
        state = ResearchState(
            task="测试任务",
            plan=[SubTask(id="t1", description="背景", tool="search")],
            review_score=0.6,
            review_feedback="结构尚可，引用需加强。",
        )
        assert await analyze_task("task_mid", state) is None

    async def test_short_feedback_skipped(self, db):
        state = ResearchState(task="测试任务", review_score=0.85, review_feedback="好")
        assert await analyze_task("task_short", state) is None

    async def test_high_score_creates_draft(self, db, monkeypatch):
        async def fake_llm_call(system_prompt, user_prompt, **kwargs):
            return json.dumps({
                "lesson": "保持严格来源标注。",
                "name": "citation-discipline",
                "description": "严格来源标注规范",
                "trigger_keywords": ["引用", "来源"],
                "agents": ["writer"],
                "content": "每个关键事实必须标注来源。",
            })

        monkeypatch.setattr("app.services.evolution_service.llm_call", fake_llm_call)
        state = ResearchState(
            task="AI 市场分析",
            plan=[SubTask(id="t1", description="背景", tool="search")],
            review_score=0.88,
            review_feedback="来源标注规范，结构完整，数据详实，值得继续保持。",
            profile_id="profile_default",
        )
        draft = await analyze_task("task_high", state)
        assert draft is not None
        assert draft["status"] == "pending"
        assert draft["draft_name"] == "citation-discipline"
        assert draft["profile_id"] == "profile_default"
        assert draft["draft_agents"] == ["writer"]

    async def test_duplicate_task_skipped(self, db, monkeypatch):
        async def fake_llm_call(system_prompt, user_prompt, **kwargs):
            return json.dumps({
                "lesson": "x",
                "name": "skill-a",
                "description": "d",
                "trigger_keywords": [],
                "agents": ["writer"],
                "content": "c",
            })

        monkeypatch.setattr("app.services.evolution_service.llm_call", fake_llm_call)
        state = ResearchState(
            task="任务",
            review_score=0.9,
            review_feedback="来源标注规范，结构完整，数据详实，继续保持。",
            profile_id="profile_default",
        )
        first = await analyze_task("task_dup", state)
        second = await analyze_task("task_dup", state)
        assert first is not None
        assert second is None

    async def test_llm_failure_skipped(self, db, monkeypatch):
        async def fake_llm_call(system_prompt, user_prompt, **kwargs):
            raise ValueError("api error")

        monkeypatch.setattr("app.services.evolution_service.llm_call", fake_llm_call)
        state = ResearchState(
            task="任务",
            review_score=0.8,
            review_feedback="来源标注规范，结构完整，数据详实，继续保持。",
            profile_id="profile_default",
        )
        assert await analyze_task("task_fail", state) is None


class TestDraftWorkflow:
    async def _seed(self, db, **overrides):
        from app.models.database import _async_session_maker

        async with _async_session_maker() as session:
            repo = EvolutionDraftRepository(session)
            return await repo.create(EvolutionDraftModel(**draft_payload(**overrides)))

    async def _get(self, draft_id):
        from app.models.database import _async_session_maker

        async with _async_session_maker() as session:
            return await EvolutionDraftRepository(session).get(draft_id)

    async def test_accept_creates_personal_skill(self, db):
        await self._seed(db)
        skill = await accept_draft("draft_1", "profile_default")
        assert skill is not None
        assert skill["name"] == "citation-discipline"
        assert skill["owner_id"] == "profile_default"
        updated = await self._get("draft_1")
        assert updated.status == "accepted"

    async def test_accept_promote_global(self, db):
        await self._seed(db)
        skill = await accept_draft("draft_1", "profile_default", promote_global=True)
        assert skill is not None
        assert skill["owner_id"] is None

    async def test_accept_with_edits(self, db):
        await self._seed(db)
        skill = await accept_draft(
            "draft_1",
            "profile_default",
            edits={"name": "renamed-skill", "agents": ["writer", "reviewer"]},
        )
        assert skill["name"] == "renamed-skill"
        assert skill["agents"] == ["writer", "reviewer"]

    async def test_accept_other_profile_returns_none(self, db):
        await self._seed(db)
        assert await accept_draft("draft_1", "profile_other") is None

    async def test_reject(self, db):
        await self._seed(db)
        assert await reject_draft("draft_1", "profile_default") is True
        updated = await self._get("draft_1")
        assert updated.status == "rejected"

    async def test_reject_other_profile_false(self, db):
        await self._seed(db)
        assert await reject_draft("draft_1", "profile_other") is False

    async def test_list_drafts(self, db):
        await self._seed(db)
        drafts = await list_drafts("profile_default", status="pending")
        assert len(drafts) == 1
        assert drafts[0]["id"] == "draft_1"
        assert await list_drafts("profile_other", status="pending") == []


class TestEvolutionAPI:
    @pytest.fixture
    async def client(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def _seed(self, db):
        from app.models.database import _async_session_maker

        async with _async_session_maker() as session:
            repo = EvolutionDraftRepository(session)
            await repo.create(EvolutionDraftModel(**draft_payload()))

    async def test_list_drafts(self, db, client):
        await self._seed(db)
        resp = await client.get("/api/evolution/drafts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["draft_name"] == "citation-discipline"

    async def test_accept_draft(self, db, client):
        await self._seed(db)
        resp = await client.post("/api/evolution/drafts/draft_1/accept", json={})
        assert resp.status_code == 200
        skill = resp.json()
        assert skill["name"] == "citation-discipline"
        assert skill["owner_id"] == "profile_default"
        # Draft now accepted -> pending list empty
        drafts = (await client.get("/api/evolution/drafts")).json()
        assert drafts == []

    async def test_accept_promote_global(self, db, client):
        await self._seed(db)
        resp = await client.post(
            "/api/evolution/drafts/draft_1/accept", json={"promote_global": True}
        )
        assert resp.status_code == 200
        assert resp.json()["owner_id"] is None

    async def test_reject_draft(self, db, client):
        await self._seed(db)
        resp = await client.post("/api/evolution/drafts/draft_1/reject")
        assert resp.status_code == 200
        assert resp.json()["rejected"] is True
        drafts = (await client.get("/api/evolution/drafts")).json()
        assert drafts == []

    async def test_accept_missing_404(self, db, client):
        resp = await client.post("/api/evolution/drafts/draft_nope/accept", json={})
        assert resp.status_code == 404

    async def test_reject_missing_404(self, db, client):
        resp = await client.post("/api/evolution/drafts/draft_nope/reject")
        assert resp.status_code == 404
