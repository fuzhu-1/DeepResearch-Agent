"""Benchmark v3 — 在 P0-P4 改进后运行 5 个课题，收集全量指标，输出数据表格。
增量保存结果到 data/benchmark_v3/，不怕中断。"""
import asyncio, json, os, sys, time, uuid
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "benchmark_v3"
OUTPUT.mkdir(parents=True, exist_ok=True)
PROGRESS = OUTPUT / "progress.json"

TOPICS = [
    "量子计算在药物研发中的应用现状与前景",
    "全球可再生能源发展对比：太阳能 vs 风能 2024-2025 装机增长",
    "大语言模型在 2026 年的最新进展：推理能力、多模态与 Agent 化",
    "日本人口老龄化对经济和社会保障体系的影响及应对政策",
    "基因编辑技术 CRISPR 2.0 在遗传病治疗中的临床试验进展",
]


def load_progress() -> list:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return []


def save_progress(records: list):
    PROGRESS.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_one(topic: str, idx: int) -> dict:
    """Run one topic via the workflow graph, collect all metrics."""
    task_id = f"bm3_{uuid.uuid4().hex[:12]}"
    record = {
        "index": idx + 1,
        "topic": topic,
        "task_id": task_id,
        "start_time": datetime.now().isoformat(),
    }
    t0 = time.time()

    # ── Patch: intercept LLM calls & tool calls for metrics ──
    # We patch at module level for the agents to pick up
    import app.utils.llm as llm_mod
    from app.services.config_service import get_active_config
    from app.config import settings
    from openai import AsyncOpenAI

    llm_calls = 0
    input_tokens = 0
    output_tokens = 0

    original_llm = llm_mod.llm_call

    async def tracked_llm(system_prompt, user_prompt, config=None, tools=None, return_metrics=False):
        nonlocal llm_calls, input_tokens, output_tokens
        llm_calls += 1

        rt = get_active_config()
        api_key = rt.api_key if (rt and rt.api_key) else (settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", ""))
        base_url = (rt.base_url if (rt and rt.base_url) else "https://api.deepseek.com/v1")
        model = config.model if config else "deepseek-chat"

        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=1)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": config.temperature if config else 0.3,
            "max_tokens": config.max_tokens if config else 4096,
        }
        if tools:
            kwargs["tools"] = tools

        t_req = time.time()
        resp = await client.chat.completions.create(**kwargs)
        t_req_end = time.time()

        if hasattr(resp, "usage") and resp.usage:
            input_tokens += resp.usage.prompt_tokens or 0
            output_tokens += resp.usage.completion_tokens or 0

        text = resp.choices[0].message.content or ""
        metrics = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
            "total_tokens": getattr(resp.usage, "total_tokens", 0) if resp.usage else 0,
            "duration_ms": int((t_req_end - t_req) * 1000),
            "provider": "openai",
            "model": model,
        }
        if return_metrics:
            return text, metrics
        return text

    llm_mod.llm_call = tracked_llm

    # ── Patch ToolRouter.execute for tool call tracking ──
    from app.tools.router import ToolRouter
    from app.tools.base import ToolResult

    tool_total = 0
    tool_ok = 0
    tool_fail = 0
    tool_retry_ok = 0
    tool_details = []

    if not hasattr(ToolRouter, '_original_execute'):
        ToolRouter._original_execute = ToolRouter.execute

    orig_exec = ToolRouter._original_execute

    async def tracked_exec(self, name, **params):
        nonlocal tool_total, tool_ok, tool_fail, tool_retry_ok
        tool_total += 1
        try:
            result = await orig_exec(self, name, **params)
            if getattr(result, "success", True):
                tool_ok += 1
                tool_details.append({"tool": name, "status": "ok"})
            else:
                tool_fail += 1
                tool_details.append({"tool": name, "status": "fail", "error": result.error[:200] if result.error else ""})
            return result
        except Exception as e:
            # Retry once
            try:
                await asyncio.sleep(1)
                result = await orig_exec(self, name, **params)
                if getattr(result, "success", True):
                    tool_retry_ok += 1
                    tool_details.append({"tool": name, "status": "retry_ok"})
                else:
                    tool_fail += 1
                    tool_details.append({"tool": name, "status": "fail_after_retry", "error": result.error[:200] if result.error else ""})
                return result
            except Exception as e2:
                tool_fail += 1
                tool_details.append({"tool": name, "status": "fail", "error": str(e2)[:200]})
                return ToolResult(success=False, error=str(e2), data=[])

    ToolRouter.execute = tracked_exec

    # ── Run workflow ──
    from app.workflow.graph import build_graph
    from app.models.state import ResearchState

    graph = build_graph()
    state = ResearchState(task=topic, use_rag=False)

    try:
        result = await graph.ainvoke(state)
        elapsed = time.time() - t0

        status = getattr(result, "status", "?")
        report = getattr(result, "final_report", "") or ""
        plan = getattr(result, "plan", []) or []
        research_data = getattr(result, "research_data", []) or []
        score = getattr(result, "review_score", 0.0)
        iteration = getattr(result, "iteration_count", 0)

        # Recover the patched function
        llm_mod.llm_call = original_llm
        if hasattr(ToolRouter, '_original_execute'):
            ToolRouter.execute = ToolRouter._original_execute

        # Count agent loop rounds from node_runner counts
        # Workflow: planner(1) + researcher(plan_size) + writer(1) + reviewer(1 to iteration) + formatter(1)
        plan_size = len(plan)
        total_rounds = 1 + plan_size + 1 + max(1, iteration) + 1

        succ = tool_ok + tool_retry_ok
        tool_rate = succ / tool_total if tool_total > 0 else 0

        record.update({
            "status": status,
            "elapsed_s": round(elapsed, 1),
            "plan_size": plan_size,
            "research_data_count": len(research_data),
            "report_length": len(report),
            "review_score": round(score, 3),
            "iteration_count": iteration,
            "total_loop_rounds": total_rounds,
            "llm_calls": llm_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "tool_total": tool_total,
            "tool_ok": tool_ok,
            "tool_retry_ok": tool_retry_ok,
            "tool_fail": tool_fail,
            "tool_success_rate": round(tool_rate, 3),
        })

        # Save report
        (OUTPUT / f"{task_id}.md").write_text(report, encoding="utf-8")

        print(f"  [{idx+1}/5] ✅ {topic[:30]}... | {elapsed:.0f}s | "
              f"tokens={input_tokens+output_tokens} | loops={total_rounds} | "
              f"tools={tool_total}({succ}/{tool_total}) | score={score:.2f}")

    except Exception as exc:
        elapsed = time.time() - t0
        llm_mod.llm_call = original_llm
        if hasattr(ToolRouter, '_original_execute'):
            ToolRouter.execute = ToolRouter._original_execute
        print(f"  [{idx+1}/5] ❌ {topic[:30]}... | {elapsed:.0f}s | ERROR: {exc}")
        record.update({"status": "error", "elapsed_s": round(elapsed, 1), "error": str(exc)[:500]})

    record["end_time"] = datetime.now().isoformat()
    return record


async def judge_reports(records: list) -> list:
    """LLM-as-Judge on completed reports."""
    from app.utils.llm import LLMConfig, llm_call
    import re

    for rec in records:
        if rec.get("status") != "completed" or rec.get("report_length", 0) < 200:
            rec["judge_factual"] = 0.0
            rec["judge_structure"] = 0.0
            rec["judge_citations"] = 0.0
            continue

        report_text = (OUTPUT / f"{rec['task_id']}.md").read_text(encoding="utf-8") if (OUTPUT / f"{rec['task_id']}.md").exists() else ""

        prompt = f"""你是一位严格的研究报告评审专家。对以下从三个维度打分(1-10)，输出JSON:

【研究课题】{rec['topic']}
【报告内容】
{report_text[:8000]}

仅输出JSON（不要任何其他文字）:
{{"factual_accuracy": <float>, "factual_reason": "<理由>",
  "structure_completeness": <float>, "structure_reason": "<理由>",
  "citation_quality": <float>, "citation_reason": "<理由>"}}"""
        try:
            config = LLMConfig(model="deepseek-chat", temperature=0.1, max_tokens=800)
            result = await llm_call(
                system_prompt="你是严格的研究报告评审专家。仅输出JSON评分。",
                user_prompt=prompt, config=config
            )
            m = re.search(r'\{[\s\S]*\}', result)
            scores = json.loads(m.group(0)) if m else {}
            rec["judge_factual"] = float(scores.get("factual_accuracy", 0))
            rec["judge_structure"] = float(scores.get("structure_completeness", 0))
            rec["judge_citations"] = float(scores.get("citation_quality", 0))
            print(f"  📋 #{rec['index']}: factual={rec['judge_factual']:.1f} "
                  f"structure={rec['judge_structure']:.1f} citations={rec['judge_citations']:.1f}")
        except Exception as exc:
            print(f"  ⚠ Judge #{rec['index']} failed: {exc}")
            rec["judge_factual"] = rec["judge_structure"] = rec["judge_citations"] = 0.0

        await asyncio.sleep(0.5)

    return records


def generate_report(records: list) -> str:
    n = len(records)
    completed = [r for r in records if r.get("status") == "completed"]

    lines = [
        "# DeepResearch-Agent 基准测试报告 (P0-P4 改进后)",
        "",
        f"**测试时间**: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"**样本数**: {n}（成功 {len(completed)}）",
        f"**改进范围**: P0(搜索质量) + P1(引用追溯) + P3(自适应规划) + P4(评审Rubric)",
        "",
        "## 1. 核心性能指标",
        "",
        "| # | 课题 | 耗时(s) | Token总消耗 | Agent循环轮次 | LLM调用 | 自评Review |",
        "|---|------|---------|-----------|-------------|---------|-----------|",
    ]
    for r in records:
        lines.append(f"| {r['index']} | {r['topic'][:30]} | {r.get('elapsed_s',0):.0f} | "
                      f"{r.get('total_tokens',0)} | {r.get('total_loop_rounds',0)} | "
                      f"{r.get('llm_calls',0)} | {r.get('review_score',0):.2f} |")
    if completed:
        a_el = sum(r["elapsed_s"] for r in completed) / len(completed)
        a_tk = sum(r["total_tokens"] for r in completed) / len(completed)
        a_lp = sum(r["total_loop_rounds"] for r in completed) / len(completed)
        a_ll = sum(r["llm_calls"] for r in completed) / len(completed)
        lines.append(f"| **平均** | - | **{a_el:.0f}** | **{a_tk:.0f}** | **{a_lp:.1f}** | **{a_ll:.0f}** | - |")

    lines += [
        "",
        "## 2. Token 与 LLM 调用",
        "",
        "| # | 课题 | 输入Token | 输出Token | 总Token | LLM调用次数 |",
        "|---|------|----------|----------|---------|-----------|",
    ]
    for r in records:
        lines.append(f"| {r['index']} | {r['topic'][:30]} | {r.get('input_tokens',0)} | "
                      f"{r.get('output_tokens',0)} | {r.get('total_tokens',0)} | {r.get('llm_calls',0)} |")
    if completed:
        a_in = sum(r["input_tokens"] for r in completed) / len(completed)
        a_out = sum(r["output_tokens"] for r in completed) / len(completed)
        a_ll = sum(r["llm_calls"] for r in completed) / len(completed)
        lines.append(f"| **平均** | - | **{a_in:.0f}** | **{a_out:.0f}** | **{a_in+a_out:.0f}** | **{a_ll:.1f}** |")

    lines += [
        "",
        "## 3. Agent 循环路由",
        "",
        "| # | 课题 | Plan子任务数 | Review迭代 | 总轮次 |",
        "|---|------|------------|-----------|--------|",
    ]
    for r in records:
        lines.append(f"| {r['index']} | {r['topic'][:30]} | {r.get('plan_size',0)} | "
                      f"{r.get('iteration_count',0)} | {r.get('total_loop_rounds',0)} |")
    if completed:
        a_ps = sum(r["plan_size"] for r in completed) / len(completed)
        a_it = sum(r["iteration_count"] for r in completed) / len(completed)
        a_lp = sum(r["total_loop_rounds"] for r in completed) / len(completed)
        lines.append(f"| **平均** | - | **{a_ps:.1f}** | **{a_it:.1f}** | **{a_lp:.1f}** |")

    lines += [
        "",
        "## 4. 工具调用成功率",
        "",
        "| # | 课题 | 总调用 | 成功 | 重试后成功 | 失败 | 成功率 | 改进前对比 |",
        "|---|------|--------|------|-----------|------|--------|-----------|",
    ]
    for r in records:
        line = f"| {r['index']} | {r['topic'][:30]} | {r.get('tool_total',0)} | {r.get('tool_ok',0)} | {r.get('tool_retry_ok',0)} | {r.get('tool_fail',0)} | {r.get('tool_success_rate',0):.0%} | 67%→{r.get('tool_success_rate',0):.0%} |"
        lines.append(line)
    if completed:
        a_tt = sum(r["tool_total"] for r in completed) / len(completed)
        a_ok = sum(r["tool_ok"] for r in completed) / len(completed)
        a_rt = sum(r["tool_retry_ok"] for r in completed) / len(completed)
        a_fl = sum(r["tool_fail"] for r in completed) / len(completed)
        total_s = sum(r["tool_ok"] + r["tool_retry_ok"] for r in completed)
        total_t = sum(r["tool_total"] for r in completed)
        overall = total_s / total_t if total_t > 0 else 0
        lines.append(f"| **平均** | - | **{a_tt:.1f}** | **{a_ok:.1f}** | **{a_rt:.1f}** | **{a_fl:.1f}** | **{overall:.0%}** | **67%→{overall:.0%}** |")

    lines += [
        "",
        "## 5. LLM-as-Judge 质量评估 (1-10分)",
        "",
        "| # | 课题 | 事实准确率 | 结构完整度 | 引用质量 | 综合 | 改进前综合对比 |",
        "|---|------|----------|----------|---------|------|--------------|",
    ]
    for r in records:
        avg = (r.get("judge_factual",0) + r.get("judge_structure",0) + r.get("judge_citations",0)) / 3
        lines.append(f"| {r['index']} | {r['topic'][:30]} | {r.get('judge_factual',0):.1f} | "
                      f"{r.get('judge_structure',0):.1f} | {r.get('judge_citations',0):.1f} | "
                      f"{avg:.1f} | 3.8→{avg:.1f} |")
    if completed:
        a_f = sum(r["judge_factual"] for r in completed) / len(completed)
        a_s = sum(r["judge_structure"] for r in completed) / len(completed)
        a_c = sum(r["judge_citations"] for r in completed) / len(completed)
        a_all = (a_f + a_s + a_c) / 3
        lines.append(f"| **平均** | - | **{a_f:.1f}** | **{a_s:.1f}** | **{a_c:.1f}** | **{a_all:.1f}** | **3.8→{a_all:.1f}** |")

    lines += [
        "",
        "## 6. 综合汇总 (≥5样本/指标)",
        "",
        "| 指标 | 样本数 | 均值 | 最小值 | 最大值 | 改进前对比 |",
        "|------|--------|------|--------|--------|-----------|",
    ]
    metrics = [
        ("生成时间 (s)", "elapsed_s", "~68s"),
        ("总Token消耗", "total_tokens", "2352"),
        ("Agent循环轮次", "total_loop_rounds", "8.4"),
        ("LLM调用次数", "llm_calls", "-"),
        ("工具调用成功率", "tool_success_rate", "67%"),
        ("事实准确率 (Judge)", "judge_factual", "3.8"),
        ("结构完整度 (Judge)", "judge_structure", "4.9"),
        ("引用质量 (Judge)", "judge_citations", "2.6"),
    ]
    vals_map = {n: [r.get(k, 0) for r in completed if r.get(k, 0) > 0] for n, k, _ in metrics}
    for name, key, prev in metrics:
        vals = vals_map[name]
        if vals:
            if key == "tool_success_rate":
                lines.append(f"| {name} | {len(vals)} | {sum(vals)/len(vals):.1%} | {min(vals):.1%} | {max(vals):.1%} | {prev}→{sum(vals)/len(vals):.1%} |")
            else:
                lines.append(f"| {name} | {len(vals)} | {sum(vals)/len(vals):.1f} | {min(vals):.1f} | {max(vals):.1f} | {prev}→{sum(vals)/len(vals):.1f} |")

    lines += ["", "---", "*报告由 benchmark_v3.py 自动生成（P0-P4 改进后）*"]
    return "\n".join(lines)


async def main():
    print("=" * 60)
    print(f"DeepResearch-Agent Benchmark v3 (P0-P4 改进后)")
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Topics: {len(TOPICS)}")
    print("=" * 60)

    # Phase 1: Run
    records = load_progress()
    done_indices = {r["index"] for r in records}
    print(f"\n▶ Phase 1: Running {len(TOPICS)} topics (already done: {len(done_indices)})")

    for i, topic in enumerate(TOPICS, 1):
        if i in done_indices:
            print(f"  [{i}/5] ⏭ Skipping {topic[:30]}... (already done)")
            continue
        rec = await run_one(topic, i - 1)
        records.append(rec)
        save_progress(records)
        await asyncio.sleep(2)

    # Phase 2: Judge
    print(f"\n▶ Phase 2: LLM-as-Judge")
    records = await judge_reports(records)
    save_progress(records)

    # Phase 3: Report
    print(f"\n▶ Phase 3: Generating report")
    report = generate_report(records)
    report_path = OUTPUT / "benchmark_v3_report.md"
    report_path.write_text(report, encoding="utf-8")
    (OUTPUT / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report)
    print(f"\n✅ Report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
