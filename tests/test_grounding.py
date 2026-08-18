"""Tests for claim-evidence grounding checks."""

import pytest

from app.utils.grounding import (
    ClaimCheck,
    GroundingChecker,
    extract_claims_with_citations,
    render_evidence_table,
)


def test_extract_claims_with_citations():
    report = (
        "- 大模型在医疗领域应用增长 [来源: 报告](https://a.com)\n"
        "- 无引用的一句话\n"
        "- 第二点 [来源: 博客](https://b.com)"
    )
    checks = extract_claims_with_citations(report)
    assert len(checks) == 2
    assert "大模型在医疗领域应用增长" in checks[0].claim
    assert checks[0].url == "https://a.com"
    assert checks[1].url == "https://b.com"


@pytest.mark.asyncio
async def test_check_report_marks_unsupported():
    async def fake_fetch(url):
        return "完全无关的页面内容 xyz"

    checker = GroundingChecker(fetch=fake_fetch, threshold=0.35)
    checks = await checker.check_report("- 量子计算突破 [来源: 某站](https://a.com)")
    assert checks[0].supported is False


@pytest.mark.asyncio
async def test_check_report_marks_supported():
    async def fake_fetch(url):
        return "量子 计算 突破 2026 年 实现"

    checker = GroundingChecker(fetch=fake_fetch, threshold=0.35)
    checks = await checker.check_report("- 量子计算突破 [来源: 某站](https://a.com)")
    assert checks[0].supported is True


def test_render_evidence_table():
    section = render_evidence_table(
        [ClaimCheck(claim="x", url="https://a.com", supported=False, reason="页面获取失败")]
    )
    assert "论断-证据核验" in section
    assert "无支撑" in section
