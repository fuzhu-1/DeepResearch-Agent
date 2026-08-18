"""Tests for the eval harness (judge parsing and scoring)."""

import pytest

from tests.eval.judge import parse_judge


def test_parse_judge_plain_json():
    text = (
        '{"completeness": 7, "citation_quality": 6, '
        '"coherence": 8, "depth": 5, "feedback": "ok"}'
    )
    data = parse_judge(text)
    assert data["completeness"] == 7.0
    assert data["depth"] == 5.0


def test_parse_judge_fenced_json():
    text = (
        '```json\n{"completeness": 9, "citation_quality": 8, '
        '"coherence": 9, "depth": 8, "feedback": "good"}\n```'
    )
    data = parse_judge(text)
    assert data["completeness"] == 9.0


@pytest.mark.asyncio
async def test_judge_report(monkeypatch):
    from tests.eval import judge as judge_mod

    async def fake_llm(system_prompt, user_prompt, config=None, tools=None):
        return '{"completeness": 8, "citation_quality": 7, "coherence": 6, "depth": 5, "feedback": "ok"}'

    monkeypatch.setattr("app.utils.llm.llm_call", fake_llm)
    scores = await judge_mod.judge_report("t", "# report")
    assert scores["completeness"] == 8.0
    assert scores["citation_quality"] == 7.0
