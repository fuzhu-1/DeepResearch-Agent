"""LLM-as-judge rubric for eval runs."""

import json
import re
from typing import Any, Dict

JUDGE_SYSTEM_PROMPT = """你是一名严谨的研究报告评审。请按以下四个维度打分（每项 0-10 分）：
- completeness: 是否覆盖研究问题的所有核心方面，有无明显信息缺口
- citation_quality: 每个关键事实是否都有来源标注，引用格式是否规范
- coherence: 结构是否清晰、论证是否连贯
- depth: 是否提供超越表面信息的分析、数据支撑与洞察
仅输出 JSON：{"completeness": 0-10, "citation_quality": 0-10, "coherence": 0-10, "depth": 0-10, "feedback": "..."}"""


def parse_judge(text: str) -> Dict[str, Any]:
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    raw = match.group(1).strip() if match else text.strip()
    data = json.loads(raw)
    return {
        "completeness": float(data.get("completeness", 0)),
        "citation_quality": float(data.get("citation_quality", 0)),
        "coherence": float(data.get("coherence", 0)),
        "depth": float(data.get("depth", 0)),
        "feedback": str(data.get("feedback", "")),
    }


async def judge_report(task: str, report: str, model: str = "gpt-4o") -> Dict[str, Any]:
    from app.utils.llm import LLMConfig, llm_call

    config = LLMConfig(model=model, temperature=0.0, max_tokens=500)
    text = await llm_call(
        JUDGE_SYSTEM_PROMPT,
        f"研究任务：{task}\n\n报告：\n{report[:12000]}",
        config,
    )
    return parse_judge(text)
