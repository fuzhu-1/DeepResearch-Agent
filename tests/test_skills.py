"""Tests for the custom Skills system: model, service, matching, seeding, API, and agent injection."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.database import (
    SkillModel,
    SkillRepository,
    close_db,
    init_db,
)
from app.models.state import ResearchState, SubTask
from app.services.profile_service import (
    DEFAULT_PROFILE_ID,
    get_default_profile,
    get_or_create_profile,
)
from app.services.skill_service import (
    BUILTIN_SKILLS,
    DuplicateSkillNameError,
    InvalidSkillError,
    create_skill,
    delete_skill,
    enrich_prompt,
    get_skill,
    invalidate_cache,
    list_skills,
    list_skills_for_profile,
    match_skills,
    seed_builtin_skills,
    set_skill_pref,
    update_skill,
)


@pytest.fixture
async def db():
    await init_db("sqlite+aiosqlite://")
    yield
    await close_db()


@pytest.fixture
async def session(db):
    from app.models.database import _async_session_maker

    async with _async_session_maker() as s:
        yield s


@pytest.fixture(autouse=True)
def clear_cache():
    from app.services.profile_service import invalidate_profiles_cache

    invalidate_cache()
    invalidate_profiles_cache()
    yield
    invalidate_cache()
    invalidate_profiles_cache()


def skill_payload(**overrides):
    payload = {
        "name": "deep-tech-analysis",
        "description": "深度技术分析",
        "trigger_keywords": ["AI", "大模型"],
        "agents": ["planner", "researcher"],
        "content": "请从架构、实现细节、性能角度分析技术方案。",
        "enabled": True,
    }
    payload.update(overrides)
    return payload


class TestSkillRepository:
    async def test_create_and_get(self, session):
        repo = SkillRepository(session)
        skill = SkillModel(
            id="skill_1",
            name="alpha",
            description="d",
            trigger_keywords='["AI"]',
            agents='["planner"]',
            content="content",
            enabled=True,
        )
        await repo.create(skill)
        fetched = await repo.get("skill_1")
        assert fetched is not None
        assert fetched.name == "alpha"

    async def test_get_by_name(self, session):
        repo = SkillRepository(session)
        await repo.create(
            SkillModel(id="skill_2", name="beta", content="c", trigger_keywords="[]", agents="[]")
        )
        assert (await repo.get_by_name("beta")) is not None
        assert await repo.get_by_name("missing") is None

    async def test_list_all_and_count(self, session):
        repo = SkillRepository(session)
        await repo.create(
            SkillModel(id="s1", name="a", content="c", trigger_keywords="[]", agents="[]")
        )
        await repo.create(
            SkillModel(id="s2", name="b", content="c", trigger_keywords="[]", agents="[]")
        )
        assert len(await repo.list_all()) == 2
        assert await repo.count() == 2

    async def test_delete(self, session):
        repo = SkillRepository(session)
        await repo.create(
            SkillModel(id="s3", name="c", content="c", trigger_keywords="[]", agents="[]")
        )
        assert await repo.delete("s3") is True
        assert await repo.get("s3") is None
        assert await repo.delete("missing") is False


class TestSkillServiceCrud:
    async def test_create_returns_dict(self, db):
        created = await create_skill(skill_payload())
        assert created["id"].startswith("skill_")
        assert created["name"] == "deep-tech-analysis"
        assert created["trigger_keywords"] == ["AI", "大模型"]
        assert created["agents"] == ["planner", "researcher"]
        assert created["enabled"] is True

    async def test_create_duplicate_name_raises(self, db):
        await create_skill(skill_payload())
        with pytest.raises(DuplicateSkillNameError):
            await create_skill(skill_payload(name="deep-tech-analysis"))

    async def test_create_invalid_agent_raises(self, db):
        with pytest.raises(InvalidSkillError):
            await create_skill(skill_payload(agents=["bogus"]))

    async def test_get_returns_none_for_missing(self, db):
        assert await get_skill("skill_nope") is None

    async def test_update_fields(self, db):
        created = await create_skill(skill_payload())
        updated = await update_skill(created["id"], {"enabled": False, "agents": ["writer"]})
        assert updated["enabled"] is False
        assert updated["agents"] == ["writer"]
        fetched = await get_skill(created["id"])
        assert fetched["enabled"] is False

    async def test_update_missing_returns_none(self, db):
        assert await update_skill("skill_nope", {"enabled": False}) is None

    async def test_update_duplicate_name_raises(self, db):
        await create_skill(skill_payload())
        second = await create_skill(skill_payload(name="other-skill"))
        with pytest.raises(DuplicateSkillNameError):
            await update_skill(second["id"], {"name": "deep-tech-analysis"})

    async def test_delete(self, db):
        created = await create_skill(skill_payload())
        assert await delete_skill(created["id"]) is True
        assert await delete_skill(created["id"]) is False


class TestSkillMatching:
    async def test_keyword_match_case_insensitive(self, db):
        await create_skill(skill_payload())
        matched = await match_skills("最新 ai 大模型进展", "planner")
        assert len(matched) == 1

    async def test_keyword_no_match(self, db):
        await create_skill(skill_payload())
        assert await match_skills("量子计算进展", "planner") == []

    async def test_empty_keywords_always_match(self, db):
        await create_skill(skill_payload(trigger_keywords=[]))
        assert len(await match_skills("任意任务", "planner")) == 1

    async def test_agent_scope_filter(self, db):
        await create_skill(skill_payload())
        assert await match_skills("AI 大模型", "writer") == []

    async def test_disabled_skill_excluded(self, db):
        await create_skill(skill_payload(enabled=False))
        assert await match_skills("AI 大模型", "planner") == []


class TestEnrichPrompt:
    async def test_matched_appends_section(self, db):
        await create_skill(skill_payload())
        prompt = await enrich_prompt("基础提示", "planner", "AI 大模型研究")
        assert prompt.startswith("基础提示")
        assert "## 用户技能：deep-tech-analysis" in prompt
        assert "请从架构、实现细节" in prompt

    async def test_no_match_returns_base(self, db):
        await create_skill(skill_payload())
        base = "基础提示"
        prompt = await enrich_prompt(base, "writer", "AI 大模型研究")
        # 无技能命中：不追加技能章节，但默认档案仍追加用户偏好章节
        assert prompt.startswith(base)
        assert "## 用户技能" not in prompt
        assert "## 用户偏好" in prompt


class TestEnrichWithProfile:
    async def test_preferences_section_before_skills(self, db):
        from app.services.profile_service import update_profile

        default = await get_default_profile()
        await update_profile(default["id"], {"writing_style": "business", "domain_focus": "AI"})
        await create_skill(skill_payload(agents=["planner"]))  # global
        prompt = await enrich_prompt(
            "基础提示", "planner", "AI 大模型研究", profile_id=default["id"]
        )
        assert "## 用户偏好" in prompt
        assert "写作风格：商业务实" in prompt
        assert prompt.index("## 用户偏好") < prompt.index("## 用户技能：deep-tech-analysis")

    async def test_no_profile_uses_default(self, db):
        await create_skill(skill_payload(agents=["planner"]))
        prompt = await enrich_prompt("基础提示", "planner", "AI 大模型研究")
        assert "## 用户偏好" in prompt
        assert "## 用户技能：deep-tech-analysis" in prompt


class TestSeeding:
    async def test_seeds_when_empty(self, db):
        count = await seed_builtin_skills()
        assert count == len(BUILTIN_SKILLS)
        assert len(await list_skills()) == len(BUILTIN_SKILLS)

    async def test_skips_when_not_empty(self, db):
        await create_skill(skill_payload())
        assert await seed_builtin_skills() == 0
        assert len(await list_skills()) == 1


class TestAgentInjection:
    async def test_planner_prompt_includes_skill(self, db, monkeypatch):
        await create_skill(skill_payload(agents=["planner"]))
        captured = {}

        async def fake_llm_call(system_prompt, user_prompt, **kwargs):
            captured["system_prompt"] = system_prompt
            return json.dumps([{"id": "s1", "description": "研究背景", "tool": "search"}])

        monkeypatch.setattr("app.agents.planner.llm_call", fake_llm_call)
        from app.agents.planner import PlannerAgent

        state = ResearchState(task="AI 大模型最新进展")
        await PlannerAgent().invoke(state)
        assert "## 用户技能：deep-tech-analysis" in captured["system_prompt"]

    async def test_writer_prompt_includes_skill(self, db, monkeypatch):
        await create_skill(skill_payload(agents=["writer"]))
        captured = {}

        async def fake_llm_call(system_prompt, user_prompt, **kwargs):
            captured["system_prompt"] = system_prompt
            return "# 报告\n\n## 摘要\n足够长的正文内容。\n" * 60

        monkeypatch.setattr("app.agents.writer.llm_call", fake_llm_call)
        from app.agents.writer import WriterAgent

        state = ResearchState(
            task="AI 大模型市场分析", plan=[SubTask(id="t1", description="d", tool="search")]
        )
        await WriterAgent().invoke(state)
        assert "## 用户技能：deep-tech-analysis" in captured["system_prompt"]

    async def test_reviewer_prompt_includes_skill(self, db, monkeypatch):
        await create_skill(skill_payload(agents=["reviewer"]))
        captured = {}

        async def fake_llm_call(system_prompt, user_prompt, **kwargs):
            captured["system_prompt"] = system_prompt
            return json.dumps({"score": 0.9, "feedback": "很好", "passed": True})

        monkeypatch.setattr("app.agents.reviewer.llm_call", fake_llm_call)
        from app.agents.reviewer import ReviewerAgent

        state = ResearchState(task="AI 大模型研究报告", report_draft="# 标题\n内容")
        await ReviewerAgent().invoke(state)
        assert "## 用户技能：deep-tech-analysis" in captured["system_prompt"]

    async def test_researcher_prompt_includes_skill(self, db, monkeypatch):
        await create_skill(skill_payload(agents=["researcher"]))
        captured = {}

        async def fake_llm_call(system_prompt, user_prompt, **kwargs):
            captured["system_prompt"] = system_prompt
            return "要点：1) 关键发现 [来源: 网页](https://example.com)"

        monkeypatch.setattr("app.agents.researcher.llm_call", fake_llm_call)
        from app.agents.researcher import ResearcherAgent

        task = SubTask(id="t1", description="调研 AI 大模型", tool="search")
        await ResearcherAgent()._summarize_result(
            task, "raw data", "search", task_text="AI 大模型最新进展"
        )
        assert "## 用户技能：deep-tech-analysis" in captured["system_prompt"]


class TestSkillsAPI:
    @pytest.fixture
    async def seeded_db(self, db):
        await seed_builtin_skills()

    @pytest.fixture
    async def client(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_list_returns_builtins(self, seeded_db, client):
        resp = await client.get("/api/skills")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == len(BUILTIN_SKILLS)
        names = {s["name"] for s in data}
        assert "executive-summary" in names

    async def test_create_and_get(self, db, client):
        resp = await client.post("/api/skills", json=skill_payload())
        assert resp.status_code == 200
        created = resp.json()
        assert created["name"] == "deep-tech-analysis"

    async def test_create_duplicate_returns_422(self, db, client):
        await client.post("/api/skills", json=skill_payload())
        resp = await client.post("/api/skills", json=skill_payload())
        assert resp.status_code == 422

    async def test_update_missing_returns_404(self, db, client):
        resp = await client.put("/api/skills/skill_nope", json={"enabled": False})
        assert resp.status_code == 404

    async def test_update(self, db, client):
        created = (await client.post("/api/skills", json=skill_payload())).json()
        resp = await client.put(f"/api/skills/{created['id']}", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    async def test_delete(self, db, client):
        created = (await client.post("/api/skills", json=skill_payload())).json()
        assert (await client.delete(f"/api/skills/{created['id']}")).status_code == 200
        assert (await client.delete(f"/api/skills/{created['id']}")).status_code == 404

    async def test_match_endpoint(self, seeded_db, client):
        resp = await client.post(
            "/api/skills/match", json={"task": "AI 大模型市场分析", "agent": "planner"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["skills"], list)


class TestSkillIsolation:
    async def test_private_skill_only_matches_owner(self, db):
        default = await get_default_profile()
        other = await get_or_create_profile("user_99")
        await create_skill(skill_payload(name="my-skill", agents=["planner"], owner_id=default["id"]))
        assert len(await match_skills("AI 大模型", "planner", profile_id=default["id"])) == 1
        assert len(await match_skills("AI 大模型", "planner", profile_id=other["id"])) == 0

    async def test_global_disabled_via_pref_excluded(self, db):
        default = await get_default_profile()
        other = await get_or_create_profile("user_88")
        created = await create_skill(skill_payload(agents=["planner"]))  # owner None -> global
        assert await set_skill_pref(other["id"], created["id"], enabled=False) is True
        assert len(await match_skills("AI 大模型", "planner", profile_id=default["id"])) == 1
        assert len(await match_skills("AI 大模型", "planner", profile_id=other["id"])) == 0

    async def test_set_pref_on_private_skill_returns_false(self, db):
        default = await get_default_profile()
        created = await create_skill(
            skill_payload(agents=["planner"], owner_id=default["id"])
        )
        assert await set_skill_pref(default["id"], created["id"], enabled=False) is False

    async def test_list_flags_owner_and_enabled_for_me(self, db):
        default = await get_default_profile()
        other = await get_or_create_profile("user_77")
        global_skill = await create_skill(skill_payload(name="g-skill", agents=["planner"]))
        private_skill = await create_skill(
            skill_payload(name="p-skill", agents=["planner"], owner_id=other["id"])
        )
        await set_skill_pref(other["id"], global_skill["id"], enabled=False)

        default_list = await list_skills_for_profile(default["id"])
        default_flags = {s["name"]: s for s in default_list}
        assert default_flags["g-skill"]["owner_id"] is None
        assert default_flags["g-skill"]["enabled_for_me"] is True
        assert "p-skill" not in default_flags

        other_list = await list_skills_for_profile(other["id"])
        other_flags = {s["name"]: s for s in other_list}
        assert other_flags["g-skill"]["enabled_for_me"] is False
        assert other_flags["p-skill"]["enabled_for_me"] is True
        assert other_flags["p-skill"]["owner_id"] == other["id"]

    async def test_create_skill_with_owner(self, db):
        default = await get_default_profile()
        created = await create_skill(skill_payload(owner_id=default["id"]))
        assert created["owner_id"] == default["id"]
