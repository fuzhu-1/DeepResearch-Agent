"""Tests for citation extraction and validation."""

import pytest

from app.utils.citation_validator import (
    Citation,
    extract_citations,
    render_validation_section,
    validate_citations,
)


def test_extract_citations():
    report = (
        "见 [来源: 维基百科](https://a.com/page) 与 "
        "[来源: 某博客](https://b.com/x)。"
    )
    cites = extract_citations(report)
    assert len(cites) == 2
    assert cites[0] == Citation(title="维基百科", url="https://a.com/page")
    assert cites[1].url == "https://b.com/x"


def test_extract_citations_ignores_plain_urls():
    report = "没有来源标注的 https://c.com 不应被提取"
    assert extract_citations(report) == []


@pytest.mark.asyncio
async def test_validate_citations(monkeypatch):
    from app.utils import citation_validator

    async def fake_check(url, timeout=10.0):
        return (200, "") if "good" in url else (404, "HTTP 404")

    monkeypatch.setattr(citation_validator, "_check_one", fake_check)
    checks = await validate_citations(
        [
            Citation(title="a", url="https://good.com"),
            Citation(title="b", url="https://bad.com/x"),
        ]
    )
    assert checks[0].valid is True
    assert checks[0].status == 200
    assert checks[1].valid is False
    assert checks[1].status == 404


def test_render_validation_section():
    from app.utils.citation_validator import CitationCheck

    section = render_validation_section(
        [
            CitationCheck(title="a", url="https://good.com", valid=True, status=200),
            CitationCheck(title="b", url="https://bad.com", valid=False, status=404, error="HTTP 404"),
        ]
    )
    assert "1 条可达" in section
    assert "1 条无法访问" in section
    assert "https://bad.com" in section
