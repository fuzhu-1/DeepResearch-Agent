"""Tests for the research context budget truncator."""

from app.utils.context import truncate_research_context


def test_truncate_keeps_summaries_within_budget():
    items = [
        {"description": "a", "summary": "x" * 100, "raw_result": "y" * 5000},
        {"description": "b", "summary": "z" * 100, "raw_result": "w" * 5000},
    ]
    out = truncate_research_context(items, max_chars=500)
    total = sum(len(i["summary"]) + len(i["raw_result"]) for i in out)
    assert total <= 500
    assert len(out[0]["raw_result"]) <= 2000


def test_truncate_drops_overflow_items():
    items = [
        {"description": "a", "summary": "s" * 100, "raw_result": ""},
        {"description": "b", "summary": "t" * 100, "raw_result": ""},
    ]
    out = truncate_research_context(items, max_chars=100)
    assert len(out) == 1
    assert len(out[0]["summary"]) == 100
