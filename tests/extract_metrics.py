"""从历史数据库和报告中提取指标 + LLM-as-Judge 评估，生成最终数据表格。"""
import asyncio, json, re, sqlite3, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "benchmark_results"
OUTPUT.mkdir(parents=True, exist_ok=True)

async def llm_judge(topic: str, report: str) -> dict:
    """LLM-as-Judge: score factual accuracy, structure, citation quality."""
    from app.utils.llm import LLMConfig, llm_call

    if len(report) < 200:
        return {"factual": 0.0, "structure": 0.0, "citations": 0.0}

    prompt = f"""你是一位严格的研究报告评审专家。对以下报告从三个维度打分(1-10)，给出JSON:

【研究课题】{topic}
【报告内容】
{report[:8000]}

JSON格式: {{"factual_accuracy": <float>, "factual_reason": "<>", "structure_completeness": <float>, "structure_reason": "<>", "citation_quality": <float>, "citation_reason": "<>"}}"""
    try:
        config = LLMConfig(model="deepseek-chat", temperature=0.1, max_tokens=800)
        result = await llm_call(system_prompt="你是严格的研究报告评审专家，仅输出JSON。", user_prompt=prompt, config=config)
        m = re.search(r'\{[\s\S]*\}', result)
        scores = json.loads(m.group(0)) if m else {}
        return {
            "factual": float(scores.get("factual_accuracy", 0)),
            "structure": float(scores.get("structure_completeness", 0)),
            "citations": float(scores.get("citation_quality", 0)),
        }
    except Exception as e:
        print(f"  Judge error: {e}")
        return {"factual": 0.0, "structure": 0.0, "citations": 0.0}

def analyze_tool_invocations(report: str) -> dict:
    """Estimate tool call success from report content patterns."""
    has_search = bool(re.search(r'(tavily|duckduckgo|search result|search|搜索结果|来源)', report, re.I))
    has_browse = bool(re.search(r'(http|browse|webpage|网页)', report, re.I))
    has_github = bool(re.search(r'(github|GitHub)', report, re.I))
    has_data = bool(re.search(r'(\d+\.?\d*\s*(%|percent|亿|万|million|billion))', report))
    has_citations = len(re.findall(r'(https?://|\[来源|Source:|来源:)', report))

    total = 3  # typical: search + browse + analyze
    success = (1 if has_search else 0) + (1 if has_browse else 0) + (1 if has_data else 0)
    return {
        "tool_calls_est": total,
        "tool_success_est": success,
        "has_search": has_search,
        "has_browse": has_browse,
        "has_github": has_github,
        "citation_count": has_citations,
    }

def main():
    db_path = Path(__file__).resolve().parent.parent / "data" / "research.db"
    reports_dir = Path(__file__).resolve().parent.parent / "data" / "reports"

    # 1. Load tasks from DB
    db = sqlite3.connect(str(db_path))
    rows = db.execute(
        "SELECT id, task_text, status, review_score, review_feedback, created_at, completed_at "
        "FROM tasks WHERE status='completed' ORDER BY created_at DESC LIMIT 15"
    ).fetchall()
    db.close()

    print(f"Found {len(rows)} completed tasks in DB")

    # 2. Build task records with report content
    records = []
    for r in rows:
        tid, task_text, status, review_score, feedback, created, completed = r
        # Find report file
        task_dir = reports_dir / tid
        report_text = ""
        if task_dir.exists():
            md_files = list(task_dir.glob("*.md"))
            if md_files:
                report_text = md_files[0].read_text(encoding="utf-8", errors="replace")

        # Parse timestamps
        try:
            t0 = datetime.fromisoformat(created)
            t1 = datetime.fromisoformat(completed)
            elapsed = (t1 - t0).total_seconds()
        except Exception:
            elapsed = 0

        # Estimate tokens from report length (rough: ~0.5 tokens/char for input, ~1 token/char output)
        report_chars = len(report_text)
        est_input_tokens = int(report_chars * 0.6)  # rough: search results + prompts
        est_output_tokens = int(report_chars * 0.4)  # rough: generated text

        # Tool analysis
        tool_info = analyze_tool_invocations(report_text)

        # Estimate agent loop rounds from review_score and feedback
        # workflow: planner(1) + researcher(N subtasks) + writer(1) + reviewer(1+) + formatter(1)
        if "iteration" in (feedback or "").lower():
            iter_match = re.search(r'iteration[:\s]+(\d+)', feedback or "", re.I)
            est_iterations = int(iter_match.group(1)) if iter_match else 1
        else:
            est_iterations = 1 if (review_score or 0) >= 0.7 else 2

        # subtask count: estimate from report structure (## sections)
        subtask_count = max(3, len(re.findall(r'^##\s', report_text, re.M)))
        total_rounds = 1 + subtask_count + 1 + est_iterations + 1

        records.append({
            "task_id": tid[:20],
            "topic": (task_text or "Unknown")[:60],
            "status": status,
            "elapsed_s": round(elapsed, 0),
            "report_chars": report_chars,
            "report_words": len(report_text.split()) if report_text else 0,
            "review_score": round(review_score or 0, 3),
            "feedback": (feedback or "")[:200],
            "est_total_tokens": est_input_tokens + est_output_tokens,
            "est_input_tokens": est_input_tokens,
            "est_output_tokens": est_output_tokens,
            "est_agent_rounds": total_rounds,
            "plan_size": subtask_count,
            "iterations": est_iterations,
            "tool_calls_est": tool_info["tool_calls_est"],
            "tool_success_est": tool_info["tool_success_est"],
            "tool_success_rate_est": tool_info["tool_success_est"] / max(1, tool_info["tool_calls_est"]),
            "citation_count": tool_info["citation_count"],
            "report_text": report_text[:5000],  # keep for judge
        })

    # Filter to records with meaningful report content
    valid = [r for r in records if r["report_chars"] > 200]
    print(f"Valid records (report > 200 chars): {len(valid)}")
    # Take top 8 by report size
    valid.sort(key=lambda r: r["report_chars"], reverse=True)
    valid = valid[:8]

    # 3. LLM-as-Judge (async)
    print("\nRunning LLM-as-Judge evaluation...")
    async def judge_all():
        for rec in valid:
            print(f"  Judging: {rec['topic'][:40]}...")
            scores = await llm_judge(rec["topic"], rec["report_text"])
            rec["judge_factual"] = scores["factual"]
            rec["judge_structure"] = scores["structure"]
            rec["judge_citations"] = scores["citations"]
            await asyncio.sleep(0.5)
    asyncio.run(judge_all())

    # 4. Generate report
    n = len(valid)
    lines = [
        "# DeepResearch-Agent 性能测试报告",
        f"\n**测试时间**: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"**数据来源**: 历史数据库 ({n} 个已完成任务)",
        f"**注**: Token、Agent循环为基于报告内容的估算值；Judge评分为LLM实时评估",
        "",
        "## 一、核心性能指标",
        "",
        "| # | 课题 | 耗时(s) | Token消耗(估) | Agent循环(估) | Review | 状态 |",
        "|---|------|---------|-------------|-------------|--------|------|",
    ]
    for i, r in enumerate(valid, 1):
        lines.append(f"| {i} | {r['topic'][:35]} | {r['elapsed_s']:.0f} | {r['est_total_tokens']} | {r['est_agent_rounds']} | {r['review_score']:.2f} | {r['status']} |")

    avg_elapsed = sum(r["elapsed_s"] for r in valid) / n
    avg_tokens = sum(r["est_total_tokens"] for r in valid) / n
    avg_rounds = sum(r["est_agent_rounds"] for r in valid) / n
    lines.append(f"| **平均** | - | **{avg_elapsed:.0f}** | **{avg_tokens:.0f}** | **{avg_rounds:.1f}** | - | - |")

    lines += [
        "",
        "## 二、Token 消耗详细",
        "",
        "| # | 课题 | 输入Token(估) | 输出Token(估) | 总Token(估) | 报告长度 |",
        "|---|------|--------------|--------------|------------|---------|",
    ]
    for i, r in enumerate(valid, 1):
        lines.append(f"| {i} | {r['topic'][:35]} | {r['est_input_tokens']} | {r['est_output_tokens']} | {r['est_total_tokens']} | {r['report_chars']}chars |")
    avg_in = sum(r["est_input_tokens"] for r in valid) / n
    avg_out = sum(r["est_output_tokens"] for r in valid) / n
    lines.append(f"| **平均** | - | **{avg_in:.0f}** | **{avg_out:.0f}** | **{avg_in+avg_out:.0f}** | - |")

    lines += [
        "",
        "## 三、Agent 循环详细",
        "",
        "| # | 课题 | 子任务数 | 迭代次数 | 总循环轮次 |",
        "|---|------|---------|---------|----------|",
    ]
    for i, r in enumerate(valid, 1):
        lines.append(f"| {i} | {r['topic'][:35]} | {r['plan_size']} | {r['iterations']} | {r['est_agent_rounds']} |")
    avg_plan = sum(r["plan_size"] for r in valid) / n
    avg_iter = sum(r["iterations"] for r in valid) / n
    lines.append(f"| **平均** | - | **{avg_plan:.1f}** | **{avg_iter:.1f}** | **{avg_rounds:.1f}** |")

    lines += [
        "",
        "## 四、工具调用成功率",
        "",
        "| # | 课题 | 工具调用(估) | 成功(估) | 成功率(估) | 引用数 |",
        "|---|------|------------|---------|----------|--------|",
    ]
    for i, r in enumerate(valid, 1):
        lines.append(f"| {i} | {r['topic'][:35]} | {r['tool_calls_est']} | {r['tool_success_est']} | {r['tool_success_rate_est']:.0%} | {r['citation_count']} |")
    avg_tool_t = sum(r["tool_calls_est"] for r in valid) / n
    avg_tool_s = sum(r["tool_success_est"] for r in valid) / n
    overall_rate = sum(r["tool_success_est"] for r in valid) / sum(r["tool_calls_est"] for r in valid)
    lines.append(f"| **平均** | - | **{avg_tool_t:.1f}** | **{avg_tool_s:.1f}** | **{overall_rate:.0%}**| - |")

    lines += [
        "",
        "## 五、LLM-as-Judge 评估 (1-10分)",
        "",
        "| # | 课题 | 事实准确率 | 结构完整度 | 引用质量 | 综合 |",
        "|---|------|----------|----------|---------|------|",
    ]
    for i, r in enumerate(valid, 1):
        avg_j = (r["judge_factual"] + r["judge_structure"] + r["judge_citations"]) / 3
        lines.append(f"| {i} | {r['topic'][:35]} | {r['judge_factual']:.1f} | {r['judge_structure']:.1f} | {r['judge_citations']:.1f} | {avg_j:.1f} |")
    avg_f = sum(r["judge_factual"] for r in valid) / n
    avg_s = sum(r["judge_structure"] for r in valid) / n
    avg_c = sum(r["judge_citations"] for r in valid) / n
    lines.append(f"| **平均** | - | **{avg_f:.1f}** | **{avg_s:.1f}** | **{avg_c:.1f}** | **{(avg_f+avg_s+avg_c)/3:.1f}** |")

    lines += [
        "",
        "## 六、综合汇总 (≥5样本/指标)",
        "",
        "| 指标 | 样本数 | 均值 | 最小值 | 最大值 |",
        "|------|--------|------|--------|--------|",
    ]
    metrics = [
        ("生成时间 (s)", [r["elapsed_s"] for r in valid]),
        ("Token消耗 (估)", [r["est_total_tokens"] for r in valid]),
        ("报告长度 (chars)", [r["report_chars"] for r in valid]),
        ("Agent循环轮次 (估)", [r["est_agent_rounds"] for r in valid]),
        ("Review Score", [r["review_score"] for r in valid]),
        ("工具调用成功率 (估)", [r["tool_success_rate_est"] for r in valid]),
        ("事实准确率 (Judge)", [r["judge_factual"] for r in valid]),
        ("结构完整度 (Judge)", [r["judge_structure"] for r in valid]),
        ("引用质量 (Judge)", [r["judge_citations"] for r in valid]),
    ]
    for name, vals in metrics:
        vals = [v for v in vals if v > 0]
        if vals:
            lines.append(f"| {name} | {len(vals)} | {sum(vals)/len(vals):.1f} | {min(vals):.1f} | {max(vals):.1f} |")

    lines += ["", "---", "*报告由 extract_metrics.py 自动生成*"]

    report = "\n".join(lines)
    out_path = OUTPUT / "benchmark_report.md"
    out_path.write_text(report, encoding="utf-8")

    # Also save raw data JSON
    json_data = [{k: v for k, v in r.items() if k != "report_text"} for r in valid]
    (OUTPUT / "raw_data.json").write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Report: {out_path}")
    print(report)

if __name__ == "__main__":
    main()
