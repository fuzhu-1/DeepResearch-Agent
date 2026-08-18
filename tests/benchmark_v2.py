"""DeepResearch-Agent 性能测试 — 运行 5 个课题并收集全部指标。"""
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.workflow.graph import build_graph
from app.models.state import ResearchState

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = [
    "量子计算在药物研发中的应用现状与前景",
    "全球可再生能源发展对比：太阳能vs风能2024-2025装机增长",
    "大语言模型在2026年的最新进展：推理能力与多模态",
    "日本人口老龄化对经济和社会保障体系的影响",
    "基因编辑CRISPR 2.0技术在遗传病治疗中的临床进展",
]

async def run_topic(topic: str, idx: int) -> dict:
    task_id = f"bench_{uuid.uuid4().hex[:12]}"
    t0 = time.time()
    record = {
        "topic": topic, "task_id": task_id, "index": idx + 1,
        "start_time": datetime.now().isoformat(),
    }

    try:
        graph = build_graph()
        state = ResearchState(task=topic, use_rag=False)
        result = await graph.ainvoke(state)

        t1 = time.time()
        record["wall_time_seconds"] = round(t1 - t0, 1)
        status = getattr(result, "status", "unknown")
        record["status"] = status
        record["report_length"] = len(getattr(result, "final_report", "") or "")
        record["review_score"] = getattr(result, "review_score", 0.0)
        record["iteration_count"] = getattr(result, "iteration_count", 0)
        record["plan_size"] = len(getattr(result, "plan", []) or [])
        record["research_data_count"] = len(getattr(result, "research_data", []) or [])
        record["error_count"] = len(getattr(result, "errors", []) or [])
        record["review_feedback"] = (getattr(result, "review_feedback", "") or "")[:300]

        # Save report
        report = getattr(result, "final_report", "") or ""
        (OUTPUT_DIR / f"{task_id}.md").write_text(report, encoding="utf-8")
        record["report_file"] = f"{task_id}.md"

        print(f"  [{idx+1}/5] ✅ {topic[:30]}... | {record['wall_time_seconds']:.0f}s | "
              f"plan={record['plan_size']} | research_items={record['research_data_count']} | "
              f"report={record['report_length']}chars | score={record['review_score']:.2f}")

    except Exception as exc:
        t1 = time.time()
        record["wall_time_seconds"] = round(t1 - t0, 1)
        record["status"] = "error"
        record["error"] = str(exc)[:500]
        print(f"  [{idx+1}/5] ❌ {topic[:30]}... | {record['wall_time_seconds']:.0f}s | ERROR: {exc}")

    record["end_time"] = datetime.now().isoformat()
    return record

async def judge_reports(records: list) -> list:
    """LLM-as-Judge scoring."""
    from app.utils.llm import LLMConfig, llm_call
    import re

    for rec in records:
        if rec.get("status") != "completed":
            rec["judge_factual"] = 0.0
            rec["judge_structure"] = 0.0
            rec["judge_citations"] = 0.0
            continue

        report_file = rec.get("report_file")
        if not report_file:
            rec["judge_factual"] = rec["judge_structure"] = rec["judge_citations"] = 0.0
            continue

        report_path = OUTPUT_DIR / report_file
        report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

        if len(report_text) < 200:
            rec["judge_factual"] = rec["judge_structure"] = rec["judge_citations"] = 0.0
            continue

        prompt = f"""你是一位严格的研究报告评审专家。请对以下研究报告从三个维度打分(1-10):

【研究课题】{rec['topic']}

【报告内容】
{report_text[:8000]}

请按JSON格式输出:
{{"factual_accuracy": <float>, "factual_reason": "<理由>",
  "structure_completeness": <float>, "structure_reason": "<理由>",
  "citation_quality": <float>, "citation_reason": "<理由>"}}
"""
        try:
            config = LLMConfig(model="deepseek-chat", temperature=0.1, max_tokens=800)
            result = await llm_call(
                system_prompt="你是严格的研究报告评审专家，只输出JSON评分。",
                user_prompt=prompt, config=config
            )
            json_match = re.search(r'\{[\s\S]*\}', result)
            scores = json.loads(json_match.group(0)) if json_match else {}

            rec["judge_factual"] = float(scores.get("factual_accuracy", 0))
            rec["judge_structure"] = float(scores.get("structure_completeness", 0))
            rec["judge_citations"] = float(scores.get("citation_quality", 0))

            print(f"  📋 Judge #{rec['index']}: factual={rec['judge_factual']:.1f} "
                  f"structure={rec['judge_structure']:.1f} citations={rec['judge_citations']:.1f}")

        except Exception as exc:
            rec["judge_factual"] = rec["judge_structure"] = rec["judge_citations"] = 0.0
            print(f"  ⚠ Judge #{rec['index']} failed: {exc}")

        await asyncio.sleep(0.5)

    return records

def generate_report(records: list) -> str:
    n = len(records)
    completed = [r for r in records if r.get("status") == "completed"]

    lines = [
        "# DeepResearch-Agent 性能测试报告",
        f"\n**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**样本数**: {n} | **成功**: {len(completed)} | **失败**: {n - len(completed)}",
        "",
        "## 一、核心性能指标",
        "",
        "| # | 课题 | 状态 | 生成时间(s) | 报告长度 | Plan | Research | 迭代 | ReviewScore |",
        "|---|------|------|------------|---------|------|----------|------|-------------|",
    ]

    for r in records:
        lines.append(
            f"| {r['index']} | {r['topic'][:25]} | {r.get('status','?')} | "
            f"{r.get('wall_time_seconds',0):.0f} | {r.get('report_length',0)} | "
            f"{r.get('plan_size',0)} | {r.get('research_data_count',0)} | "
            f"{r.get('iteration_count',0)} | {r.get('review_score',0):.2f} |"
        )

    if completed:
        avg_time = sum(r["wall_time_seconds"] for r in completed) / len(completed)
        avg_len = sum(r["report_length"] for r in completed) / len(completed)
        avg_plan = sum(r["plan_size"] for r in completed) / len(completed)
        avg_iter = sum(r.get("iteration_count", 0) for r in completed) / len(completed)
        lines.append(f"| **平均** | - | - | **{avg_time:.0f}** | **{avg_len:.0f}** | **{avg_plan:.1f}** | - | **{avg_iter:.1f}** | - |")

    lines += [
        "",
        "## 二、LLM-as-Judge 质量评估 (1-10)",
        "",
        "| # | 课题 | 事实准确率 | 结构完整度 | 引用质量 | 综合分 |",
        "|---|------|----------|----------|---------|--------|",
    ]
    for r in records:
        avg = (r.get("judge_factual",0) + r.get("judge_structure",0) + r.get("judge_citations",0)) / 3
        lines.append(
            f"| {r['index']} | {r['topic'][:25]} | {r.get('judge_factual',0):.1f} | "
            f"{r.get('judge_structure',0):.1f} | {r.get('judge_citations',0):.1f} | {avg:.1f} |"
        )

    if completed:
        avg_f = sum(r.get("judge_factual",0) for r in completed) / len(completed)
        avg_s = sum(r.get("judge_structure",0) for r in completed) / len(completed)
        avg_c = sum(r.get("judge_citations",0) for r in completed) / len(completed)
        lines.append(f"| **平均** | - | **{avg_f:.1f}** | **{avg_s:.1f}** | **{avg_c:.1f}** | **{(avg_f+avg_s+avg_c)/3:.1f}** |")

    lines += [
        "",
        "## 三、综合汇总",
        "",
        "| 指标 | 样本数 | 均值 | 范围 |",
        "|------|--------|------|------|",
    ]

    metrics = [
        ("生成时间(s)", lambda r: r.get("wall_time_seconds", 0)),
        ("报告长度(chars)", lambda r: r.get("report_length", 0)),
        ("Plan子任务数", lambda r: r.get("plan_size", 0)),
        ("ReviewScore", lambda r: r.get("review_score", 0)),
        ("事实准确率(Judge)", lambda r: r.get("judge_factual", 0)),
        ("结构完整度(Judge)", lambda r: r.get("judge_structure", 0)),
        ("引用质量(Judge)", lambda r: r.get("judge_citations", 0)),
    ]

    for name, fn in metrics:
        vals = [fn(r) for r in completed] if completed else [fn(r) for r in records]
        if vals:
            lines.append(f"| {name} | {len(vals)} | {sum(vals)/len(vals):.1f} | {min(vals):.1f}–{max(vals):.1f} |")

    lines += ["", "---", "*报告由 benchmark_v2.py 自动生成*"]
    return "\n".join(lines)

async def main():
    print("=" * 60)
    print(f"DeepResearch-Agent Benchmark — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Topics: {len(TOPICS)} | Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Phase 1: Run all topics
    print("\n▶ Phase 1: 运行研究任务\n")
    phase1_start = time.time()
    records = []
    for i, topic in enumerate(TOPICS):
        rec = await run_topic(topic, i)
        records.append(rec)
        await asyncio.sleep(2)  # rate limit
    phase1_elapsed = time.time() - phase1_start
    print(f"\n  Phase 1 完成: {phase1_elapsed:.0f}s total, {phase1_elapsed/len(TOPICS):.0f}s avg/topic")

    # Save intermediate results
    (OUTPUT_DIR / "raw_results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # Phase 2: LLM-as-Judge
    print("\n▶ Phase 2: LLM-as-Judge 评估\n")
    records = await judge_reports(records)
    (OUTPUT_DIR / "raw_results_with_judge.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # Phase 3: Generate report
    print("\n▶ Phase 3: 生成报告\n")
    report = generate_report(records)
    report_path = OUTPUT_DIR / "benchmark_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n✅ 报告: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
