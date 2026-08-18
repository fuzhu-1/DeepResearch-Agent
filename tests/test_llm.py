"""Tests for LLM client reuse, usage metering, and optional tracing."""

import pytest


def test_openai_client_is_cached():
    from app.utils import llm as llm_mod

    c1 = llm_mod._get_openai_client("k", "https://api.example.com")
    c2 = llm_mod._get_openai_client("k", "https://api.example.com")
    assert c1 is c2
    llm_mod._openai_clients.clear()


@pytest.mark.asyncio
async def test_usage_meter_records_tokens(monkeypatch):
    from app.utils import llm as llm_mod
    from app.utils.llm import LLMConfig, set_usage_meter

    meter: list = []
    set_usage_meter(meter)

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5

    class FakeMessage:
        content = "hi"

    class FakeChoice:
        message = FakeMessage()

    class FakeResp:
        usage = FakeUsage()
        choices = [FakeChoice()]

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    class FakeRT:
        api_key = "k"
        base_url = ""

    monkeypatch.setattr("app.services.config_service.get_active_config", lambda: FakeRT())
    monkeypatch.setattr(llm_mod, "_get_openai_client", lambda k, b: FakeClient())

    text = await llm_mod._call_openai("s", "u", LLMConfig(model="m"), None)
    assert text == "hi"
    assert meter[0]["total_tokens"] == 15
    assert meter[0]["model"] == "m"
    set_usage_meter(None)


@pytest.mark.asyncio
async def test_llm_call_uses_tracer_when_available(monkeypatch):
    from app.utils import llm as llm_mod

    calls: list = []

    class FakeSpan:
        def end(self, **kwargs):
            calls.append(("end", kwargs))

    class FakeTracer:
        def start_span(self, **kwargs):
            calls.append(("start", kwargs))
            return FakeSpan()

    async def fake_openai(s, u, c, t):
        return "ok"

    monkeypatch.setattr(llm_mod, "get_tracer", lambda: FakeTracer())
    monkeypatch.setattr(llm_mod, "_call_openai", fake_openai)

    text = await llm_mod.llm_call("s", "u")
    assert text == "ok"
    assert calls[0][0] == "start"
    assert calls[0][1]["name"] == "llm_call"
    assert calls[1][0] == "end"
