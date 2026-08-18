"""Tests for user profiles, per-profile skill prefs, and schema migration."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.database import (
    SkillModel,
    SkillRepository,
    UserProfileModel,
    UserProfileRepository,
    UserSkillPrefRepository,
    close_db,
    init_db,
)
from app.services.profile_service import (
    DEFAULT_PROFILE_ID,
    build_preferences_section,
    get_default_profile,
    get_effective_profile,
    get_or_create_profile,
    get_profile,
    get_profile_model,
    update_profile,
)
from tests.test_skills import skill_payload


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
def clear_caches():
    from app.services.profile_service import invalidate_profiles_cache
    from app.services.skill_service import invalidate_cache as invalidate_skills_cache

    invalidate_profiles_cache()
    invalidate_skills_cache()
    yield
    invalidate_profiles_cache()
    invalidate_skills_cache()


class TestUserProfileRepository:
    async def test_create_and_get(self, session):
        repo = UserProfileRepository(session)
        created = await repo.create(
            UserProfileModel(id="profile_default", user_id=None, writing_style="business")
        )
        assert created.id == "profile_default"
        fetched = await repo.get("profile_default")
        assert fetched is not None
        assert fetched.writing_style == "business"

    async def test_get_by_user(self, session):
        repo = UserProfileRepository(session)
        await repo.create(UserProfileModel(id="profile_1", user_id="user_1"))
        found = await repo.get_by_user("user_1")
        assert found is not None
        assert found.id == "profile_1"
        assert await repo.get_by_user("missing") is None


class TestUserSkillPrefRepository:
    async def test_set_disabled_and_list(self, session):
        repo = UserSkillPrefRepository(session)
        await repo.set("profile_default", "skill_a", enabled=False)
        assert await repo.list_disabled("profile_default") == ["skill_a"]
        await repo.set("profile_default", "skill_a", enabled=True)
        assert await repo.list_disabled("profile_default") == []

    async def test_set_enabled_when_no_row_is_noop(self, session):
        repo = UserSkillPrefRepository(session)
        await repo.set("profile_x", "skill_b", enabled=True)
        assert await repo.list_disabled("profile_x") == []


class TestSkillOwnerColumn:
    async def test_new_skill_model_has_owner(self, session):
        repo = SkillRepository(session)
        await repo.create(
            SkillModel(
                id="skill_priv",
                name="private-skill",
                content="c",
                trigger_keywords="[]",
                agents="[]",
                owner_id="profile_default",
            )
        )
        fetched = await repo.get("skill_priv")
        assert fetched is not None
        assert fetched.owner_id == "profile_default"

    async def test_migration_adds_owner_column_preserving_data(self, tmp_path):
        """Simulate a phase-1 database without owner_id and verify migration."""
        from sqlalchemy import text

        db_url = f"sqlite+aiosqlite:///{tmp_path / 'migrate_test.db'}"
        await init_db(db_url)
        from app.models.database import _async_session_maker

        # Replace the modern skills table with a legacy schema (no owner_id) and a row
        async with _async_session_maker() as session:
            await session.execute(text("DROP TABLE skills"))
            await session.execute(
                text(
                    "CREATE TABLE skills ("
                    "id VARCHAR(64) PRIMARY KEY, name VARCHAR(128), description TEXT, "
                    "trigger_keywords TEXT, agents TEXT, content TEXT, "
                    "enabled BOOLEAN, created_at DATETIME, updated_at DATETIME)"
                )
            )
            await session.execute(
                text("INSERT INTO skills (id, name, content) VALUES ('old_skill', 'old', 'content')")
            )
            await session.commit()
            await close_db()

        # Re-run init_db: create_all + migration should add owner_id and keep data
        await init_db(db_url)
        async with _async_session_maker() as session:
            cols = (await session.execute(text("PRAGMA table_info(skills)"))).fetchall()
            names = [c[1] for c in cols]
            assert "owner_id" in names
            row = (
                await session.execute(text("SELECT id, name FROM skills WHERE id = 'old_skill'"))
            ).fetchone()
            assert row is not None
            assert row[1] == "old"
        await close_db()


class TestProfileService:
    async def test_default_profile_created(self, db):
        profile = await get_default_profile()
        assert profile["id"] == DEFAULT_PROFILE_ID
        assert profile["user_id"] is None
        assert profile["writing_style"] == "balanced"
        # idempotent
        again = await get_default_profile()
        assert again["id"] == DEFAULT_PROFILE_ID

    async def test_get_or_create_by_user(self, db):
        profile = await get_or_create_profile("user_42")
        assert profile["user_id"] == "user_42"
        same = await get_or_create_profile("user_42")
        assert same["id"] == profile["id"]

    async def test_get_effective_profile_anonymous(self, db):
        profile = await get_effective_profile(None)
        assert profile["id"] == DEFAULT_PROFILE_ID

    async def test_get_effective_profile_logged_in(self, db):
        profile = await get_effective_profile({"id": "user_7", "username": "alice"})
        assert profile["user_id"] == "user_7"

    async def test_update_profile(self, db):
        profile = await get_default_profile()
        updated = await update_profile(
            profile["id"], {"writing_style": "business", "domain_focus": "AI, 金融"}
        )
        assert updated["writing_style"] == "business"
        fetched = await get_profile(profile["id"])
        assert fetched["domain_focus"] == "AI, 金融"

    async def test_update_missing_returns_none(self, db):
        assert await update_profile("profile_nope", {"writing_style": "business"}) is None

    async def test_get_profile_model(self, db):
        profile = await get_default_profile()
        assert await get_profile_model(profile["id"]) is None
        await update_profile(profile["id"], {"preferred_model": "gpt-4o-mini"})
        assert await get_profile_model(profile["id"]) == "gpt-4o-mini"


class TestPreferencesSection:
    async def test_build_section(self, db):
        profile = await get_default_profile()
        await update_profile(
            profile["id"],
            {
                "writing_style": "academic",
                "domain_focus": "AI",
                "extra_instructions": "请引用权威来源。",
            },
        )
        section = build_preferences_section(await get_profile(profile["id"]))
        assert section.startswith("## 用户偏好")
        assert "学术严谨" in section
        assert "领域聚焦：AI" in section
        assert "请引用权威来源。" in section

    async def test_empty_fields_skipped(self, db):
        profile = await get_default_profile()
        section = build_preferences_section(await get_profile(profile["id"]))
        assert "领域聚焦" not in section


class TestModelOverride:
    class FakeGraph:
        def __init__(self):
            self.received = None

        async def ainvoke(self, state, config=None):
            self.received = state
            return state

    async def test_planner_uses_profile_model(self, db, monkeypatch):
        from app.services.profile_service import get_default_profile, update_profile

        profile = await get_default_profile()
        await update_profile(profile["id"], {"preferred_model": "custom-model-1"})
        captured = {}

        async def fake_llm_call(system_prompt, user_prompt, **kwargs):
            captured["config"] = kwargs.get("config")
            return "[]"

        monkeypatch.setattr("app.agents.planner.llm_call", fake_llm_call)
        from app.models.state import ResearchState
        from app.workflow.nodes import planner_node

        state = ResearchState(task="AI 研究", profile_id=profile["id"])
        await planner_node(state)
        assert captured["config"] is not None
        assert captured["config"].model == "custom-model-1"

    async def test_run_research_passes_profile_id(self, db, monkeypatch):
        from app.workflow import graph as graph_mod

        profile = await get_default_profile()
        fake = self.FakeGraph()
        monkeypatch.setattr(graph_mod, "build_graph", lambda: fake)
        state = await graph_mod.run_research("测试任务", use_rag=False, profile_id=profile["id"])
        assert state.profile_id == profile["id"]


class TestProfileAPI:
    @pytest.fixture
    async def client(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_get_profile_returns_default(self, db, client):
        resp = await client.get("/api/profile")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == DEFAULT_PROFILE_ID
        assert data["writing_style"] == "balanced"

    async def test_update_profile(self, db, client):
        resp = await client.put(
            "/api/profile",
            json={
                "writing_style": "business",
                "domain_focus": "AI",
                "extra_instructions": "简练",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["writing_style"] == "business"
        assert data["domain_focus"] == "AI"

    async def test_update_profile_invalid_style(self, db, client):
        resp = await client.put("/api/profile", json={"writing_style": "weird"})
        assert resp.status_code == 422

    async def test_set_skill_pref(self, db, client):
        from app.services.skill_service import seed_builtin_skills

        await seed_builtin_skills()
        skills = (await client.get("/api/skills")).json()
        target = next(s for s in skills if s["owner_id"] is None)
        resp = await client.put(f"/api/skills/{target['id']}/pref", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False
        # Now disabled for default profile
        skills2 = (await client.get("/api/skills")).json()
        flags = {s["id"]: s["enabled_for_me"] for s in skills2}
        assert flags[target["id"]] is False

    async def test_set_pref_missing_skill_404(self, db, client):
        resp = await client.put("/api/skills/skill_nope/pref", json={"enabled": False})
        assert resp.status_code == 404

    async def test_skills_list_has_owner_flags(self, db, client):
        created = (await client.post("/api/skills", json=skill_payload())).json()
        skills = (await client.get("/api/skills")).json()
        by_name = {s["name"]: s for s in skills}
        assert "owner_id" in by_name[created["name"]]
        assert by_name[created["name"]]["enabled_for_me"] is True


@pytest.mark.asyncio
async def test_get_agent_model_prefers_profile(monkeypatch):
    from app.services.profile_service import get_agent_model

    async def fake_profile_model(profile_id):
        return "profile-model"

    monkeypatch.setattr("app.services.profile_service.get_profile_model", fake_profile_model)
    assert await get_agent_model("writer", "p1") == "profile-model"


@pytest.mark.asyncio
async def test_get_agent_model_falls_back_to_role_setting(monkeypatch):
    from app.config import settings
    from app.services.profile_service import get_agent_model

    async def fake_profile_model(pid):
        return None

    monkeypatch.setattr("app.services.profile_service.get_profile_model", fake_profile_model)
    monkeypatch.setattr(settings, "LLM_MODEL_WRITER", "writer-mini")
    assert await get_agent_model("writer", None) == "writer-mini"
