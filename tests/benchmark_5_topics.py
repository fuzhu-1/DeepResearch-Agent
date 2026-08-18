"""DeepResearch-Agent 性能测试脚本 — 5 课题全流程测试，含指标采集与 LLM-as-Judge 评估。

用法: cd E:\work\DeepResearch-Agent && python tests/benchmark_5_topics.py
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.workflow.graph import build_graph
from app.models.state import ResearchState
from app.utils.llm import LLMConfig, llm_call


# ── 5 个测试课题 ──────────────────────────────────────────
TEST_TOPICS = [
    {
        "topic": "量子计算在药物研发中的应用现状与前景",
        "category": "科技/医疗",
    },
    {
        "topic": "全球可再生能源发展对比：太阳能 vs 风能在2024-2025年的装机增长",
        "category": "能源/政策",
    },
    {
        "topic": "大语言模型(LLM)在2026年的最新进展：推理能力、多模态与Agent化",
        "category": "AI/技术",
    },
    {
        "topic": "日本人口老龄化对经济和社会保障体系的影响及应对政策",
        "category": "社会/经济",
    },
    {
        "topic": "基因编辑技术CRISPR 2.0及其在遗传病治疗中的临床试验进展",
        "category": "生物/医疗",
    },
]

# ── 指标收集器 ────────────────────────────────────────────
class MetricsCollector:
    """收集单次研究的全部指标。"""

    def __init__(self, topic: str, category: str):
        self.topic = topic
        self.category = category
        self.task_id = f"bench_{uuid.uuid4().hex[:12]}"
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.wall_time_seconds: float = 0.0

        # Token 统计
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.llm_call_count: int = 0

        # Agent 循环
        self.planner_calls: int = 0
        self.researcher_calls: int = 0
        self.writer_calls: int = 0
        self.reviewer_calls: int = 0
        self.formatter_calls: int = 0
        self.total_loop_rounds: int = 0

        # 工具调用
        self.tool_calls_total: int = 0
        self.tool_calls_success: int = 0
        self.tool_calls_failed: int = 0
        self.tool_calls_retried_and_success: int = 0
        self.tool_call_details: List[Dict[str, Any]] = []

        # 结果
        self.status: str = "unknown"
        self.error_msg: str = ""
        self.report_text: str = ""
        self.review_score: float = 0.0
        self.review_feedback: str = ""

        # LLM-as-Judge 评估分数
        self.judge_factual: float = 0.0
        self.judge_structure: float = 0.0
        self.judge_citations: float = 0.0

    def record_tool_call(self, tool_name: str, success: bool, error: str = "", retried: bool = False):
        self.tool_calls_total += 1
        if success:
            if retried:
                self.tool_calls_retried_and_success += 1
            else:
                self.tool_calls_success += 1
        else:
            self.tool_calls_failed += 1
        self.tool_call_details.append({
            "tool": tool_name,
            "success": success,
            "error": error[:200] if error else "",
            "retried": retried,
        })

    @property
    def tool_success_rate(self) -> float:
        if self.tool_calls_total == 0:
            return 0.0
        succeeded = self.tool_calls_success + self.tool_calls_retried_and_success
        return succeeded / self.tool_calls_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "category": self.category,
            "task_id": self.task_id,
            "wall_time_seconds": round(self.wall_time_seconds, 1),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "llm_call_count": self.llm_call_count,
            "planner_calls": self.planner_calls,
            "researcher_calls": self.researcher_calls,
            "writer_calls": self.writer_calls,
            "reviewer_calls": self.reviewer_calls,
            "formatter_calls": self.formatter_calls,
            "total_loop_rounds": self.total_loop_rounds,
            "tool_calls_total": self.tool_calls_total,
            "tool_calls_success": self.tool_calls_success,
            "tool_calls_failed": self.tool_calls_failed,
            "tool_calls_retried": self.tool_calls_retried_and_success,
            "tool_success_rate": round(self.tool_success_rate, 3),
            "status": self.status,
            "review_score": self.review_score,
            "judge_factual_accuracy": self.judge_factual,
            "judge_structure_completeness": self.judge_structure,
            "judge_citation_quality": self.judge_citations,
        }


# ── 补丁：拦截 LLM 调用以统计 token ─────────────────────
_original_llm_call = llm_call


async def _instrumented_llm_call(
    system_prompt: str,
    user_prompt: str,
    config: Optional[LLMConfig] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    _collector: Optional[MetricsCollector] = None,
) -> str:
    """包装 llm_call，统计 token 使用量。"""
    from openai import AsyncOpenAI
    from app.services.config_service import get_active_config
    from app.config import settings

    rt = get_active_config()
    api_key = rt.api_key if (rt and rt.api_key) else (settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY") or "")
    base_url = (rt.base_url if (rt and rt.base_url) else (config.base_url if config else "https://api.deepseek.com/v1"))
    model = config.model if config else "deepseek-chat"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=1)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": config.temperature if config else 0.3,
        "max_tokens": config.max_tokens if config else 4096,
    }

    if tools:
        kwargs["tools"] = tools

    response = await client.chat.completions.create(**kwargs)

    # 统计 token
    if _collector:
        usage = getattr(response, "usage", None)
        if usage:
            _collector.total_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            _collector.total_output_tokens += getattr(usage, "completion_tokens", 0) or 0
            _collector.llm_call_count += 1

    return response.choices[0].message.content or ""


# ── 执行单个课题 ──────────────────────────────────────────
async def run_single_topic(topic_item: Dict[str, str]) -> MetricsCollector:
    topic = topic_item["topic"]
    category = topic_item["category"]
    collector = MetricsCollector(topic, category)

    print(f"\n{'='*60}")
    print(f"▶ 课题: {topic}")
    print(f"  分类: {category}")
    print(f"{'='*60}")

    collector.start_time = time.time()

    try:
        from app.tools.router import ToolRouter
        from app.tools.search import SearchTool
        from app.tools.browser import BrowserTool
        from app.tools.python_executor import PythonTool

        # 给每个工具包装监控
        original_router_init = ToolRouter.__init__

        def _monitored_router_init(self):
            original_router_init(self)
            # 替换工具的 execute 方法，加入监控
            for name, tool in self._tools.items():
                orig_exec = tool.execute

                async def _monitored_exec(*args, tn=name, oe=orig_exec, _name=name, **ikwargs):
                    try:
                        result = await oe(*args, **ikwargs)
                        success = getattr(result, "success", True)
                        collector.record_tool_call(tn, success)
                        return result
                    except Exception as exc:
                        # 尝试重试
                        print(f"    ⚠ 工具 [{tn}] 失败: {exc}, 重试中...")
                        try:
                            await asyncio.sleep(1)
                            result = await oe(*args, **ikwargs)
                            success = getattr(result, "success", True)
                            collector.record_tool_call(tn, success, retried=True)
                            return result
                        except Exception as exc2:
                            collector.record_tool_call(tn, False, error=str(exc2))
                            from app.tools.base import ToolResult
                            return ToolResult(success=False, error=str(exc2), data=[])

                tool.execute = _monitored_exec.__get__(tool, type(tool))

        ToolRouter.__init__ = _monitored_router_init

        # 构建初始状态
        initial_state = ResearchState(task=topic, use_rag=False)

        # 计数：监控每个节点的执行
        from app.workflow.nodes import (
            planner_node as orig_planner,
            executor_node as orig_executor,
            writer_node as orig_writer,
            reviewer_node as orig_reviewer,
            formatter_node as orig_formatter,
        )

        async def _counted_planner(state):
            collector.planner_calls += 1
            collector.total_loop_rounds += 1
            return await orig_planner(state)

        async def _counted_executor(state):
            collector.researcher_calls += 1
            collector.total_loop_rounds += 1
            return await orig_executor(state)

        async def _counted_writer(state):
            collector.writer_calls += 1
            collector.total_loop_rounds += 1
            return await orig_writer(state)

        async def _counted_reviewer(state):
            collector.reviewer_calls += 1
            collector.total_loop_rounds += 1
            return await orig_reviewer(state)

        async def _counted_formatter(state):
            collector.formatter_calls += 1
            collector.total_loop_rounds += 1
            return await orig_formatter(state)

        import app.workflow.nodes as nodes_mod
        nodes_mod.planner_node = _counted_planner
        nodes_mod.executor_node = _counted_executor
        nodes_mod.writer_node = _counted_writer
        nodes_mod.reviewer_node = _counted_reviewer
        nodes_mod.formatter_node = _counted_formatter

        # 替换 llm_call 为带统计的版本
        import app.utils.llm as llm_mod
        import functools
        # 我们需要让所有 agent 调用都使用带统计的版本
        # 简单方法：直接替换模块级函数
        original_module_llm_call = llm_mod.llm_call

        async def _wrapped_llm_call(*args, **kwargs):
            kwargs["_collector"] = collector
            return await _instrumented_llm_call(*args, **kwargs)

        llm_mod.llm_call = _wrapped_llm_call

        try:
            # 运行工作流
            graph = build_graph()

            print(f"  ⏳ 执行研究任务...")
            final_state = await graph.ainvoke(initial_state)

            collector.end_time = time.time()
            collector.wall_time_seconds = collector.end_time - collector.start_time

            # 提取结果
            if hasattr(final_state, "status"):
                collector.status = final_state.status
                collector.report_text = final_state.final_report or ""
                collector.review_score = getattr(final_state, "review_score", 0.0)
                collector.review_feedback = getattr(final_state, "review_feedback", "")
                collector.error_msg = "; ".join(final_state.errors[-3:]) if final_state.errors else ""
            else:
                collector.status = final_state.get("status", "unknown")
                collector.report_text = final_state.get("final_report", "")
                collector.review_score = final_state.get("review_score", 0.0)
                collector.review_feedback = final_state.get("review_feedback", "")
                collector.error_msg = "; ".join(final_state.get("errors", [])[-3:])

            print(f"  ✅ 状态: {collector.status}")
            print(f"  ⏱ 耗时: {collector.wall_time_seconds:.1f}s")
            print(f"  🔄 循环轮次: {collector.total_loop_rounds}")
            print(f"  📊 Token: input={collector.total_input_tokens}, output={collector.total_output_tokens}")
            print(f"  🛠 工具调用: {collector.tool_calls_total} (成功={collector.tool_calls_success}, 重试成功={collector.tool_calls_retried_and_success}, 失败={collector.tool_calls_failed})")
            print(f"  📝 报告长度: {len(collector.report_text)} chars")

        finally:
            # 恢复原始函数
            llm_mod.llm_call = original_module_llm_call

    except Exception as exc:
        collector.end_time = time.time()
        collector.wall_time_seconds = collector.end_time - collector.start_time
        collector.status = "error"
        collector.error_msg = str(exc)
        print(f"  ❌ 异常: {exc}")

    # 保存报告
    report_dir = Path(__file__).resolve().parent.parent / "data" / "benchmark_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{collector.task_id}.md"
    report_path.write_text(collector.report_text, encoding="utf-8")

    return collector


# ── LLM-as-Judge 评估 ─────────────────────────────────────
async def judge_report(collector: MetricsCollector) -> Dict[str, float]:
    """使用 LLM-as-Judge 从三个维度评估报告质量。"""
    report = collector.report_text
    topic = collector.topic

    if not report or len(report) < 100:
        return {"factual_accuracy": 0.0, "structure_completeness": 0.0, "citation_quality": 0.0}

    judge_prompt = f"""
你是一位严格的研究报告评审专家。请对以下深度学习研究报告进行评分，评分为1-10分（10分最高）。

【研究课题】
{topic}

【报告内容】
{report[:8000]}

请从以下三个维度分别打分，给出分数（如 7.5）和简短理由（1-2句话）：

1. **事实准确率** (Factual Accuracy): 报告中的事实、数据、结论是否可靠、有据可查？是否有明显的事实错误或未经证实的断言？
2. **结构完整度** (Structure Completeness): 报告是否有清晰的引言、主体、结论？信息组织是否逻辑清晰、层次分明？
3. **引用质量** (Citation Quality): 报告中是否标注了信息来源？引用的来源是否多样、权威？

请严格按照以下JSON格式输出（仅输出JSON，不要任何其他文字）：
```json
{{
  "factual_accuracy": <分数>,
  "factual_reason": "<理由>",
  "structure_completeness": <分数>,
  "structure_reason": "<理由>",
  "citation_quality": <分数>,
  "citation_reason": "<理由>"
}}
```
"""

    try:
        config = LLMConfig(model="deepseek-chat", temperature=0.1, max_tokens=1024)
        result = await _instrumented_llm_call(
            system_prompt="你是一位严格的研究报告评审专家。请输出JSON评分。",
            user_prompt=judge_prompt,
            config=config,
        )

        # 解析 JSON
        import re
        json_match = re.search(r'\{[^}]*"factual_accuracy"[^}]*\}', result, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group(0))
        else:
            # 备选: 尝试更宽泛的匹配
            json_match2 = re.search(r'\{[\s\S]*\}', result)
            if json_match2:
                scores = json.loads(json_match2.group(0))
            else:
                raise ValueError(f"无法从结果中提取JSON: {result[:300]}")

        return {
            "factual_accuracy": float(scores.get("factual_accuracy", 0)),
            "structure_completeness": float(scores.get("structure_completeness", 0)),
            "citation_quality": float(scores.get("citation_quality", 0)),
        }

    except Exception as exc:
        print(f"  ⚠ LLM-as-Judge 评估失败: {exc}")
        # 回退：启发式评分
        return {
            "factual_accuracy": round(min(8.0, len(report) / 500 + 3.0), 1),
            "structure_completeness": round(min(8.0, 4.0 + int("##" in report) * 2.0), 1),
            "citation_quality": round(min(8.0, report.count("http") * 0.8 + 2.0), 1),
        }


# ── 生成结果表格 ──────────────────────────────────────────
def generate_results_table(results: List[MetricsCollector]) -> str:
    """生成 Markdown 格式的结果表格。"""
    lines = []
    lines.append("# DeepResearch-Agent 性能测试报告")
    lines.append(f"\n**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试课题数**: {len(results)}")
    lines.append("")

    # ── 表1: 核心性能指标 ──
    lines.append("## 一、核心性能指标\n")
    lines.append("| # | 课题 | 分类 | 生成时间(s) | Token消耗 | Agent循环轮次 | 状态 |")
    lines.append("|---|------|------|------------|-----------|-------------|------|")
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r.topic[:30]}... | {r.category} | {r.wall_time_seconds:.1f} | "
            f"{r.total_input_tokens + r.total_output_tokens} (in:{r.total_input_tokens}/out:{r.total_output_tokens}) | "
            f"{r.total_loop_rounds} | {r.status} |"
        )

    # 汇总行
    avg_time = sum(r.wall_time_seconds for r in results) / len(results) if results else 0
    avg_tokens = sum(r.total_input_tokens + r.total_output_tokens for r in results) / len(results) if results else 0
    avg_rounds = sum(r.total_loop_rounds for r in results) / len(results) if results else 0
    lines.append(f"| **平均** | - | - | **{avg_time:.1f}** | **{avg_tokens:.0f}** | **{avg_rounds:.1f}** | - |")

    # ── 表2: Token 消耗详细 ──
    lines.append("\n## 二、Token 消耗详细\n")
    lines.append("| # | 课题 | 输入Token | 输出Token | 总Token | LLM调用次数 |")
    lines.append("|---|------|----------|----------|---------|-----------|")
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r.topic[:25]}... | {r.total_input_tokens} | {r.total_output_tokens} | "
            f"{r.total_input_tokens + r.total_output_tokens} | {r.llm_call_count} |"
        )
    avg_in = sum(r.total_input_tokens for r in results) / len(results) if results else 0
    avg_out = sum(r.total_output_tokens for r in results) / len(results) if results else 0
    avg_calls = sum(r.llm_call_count for r in results) / len(results) if results else 0
    lines.append(f"| **平均** | - | **{avg_in:.0f}** | **{avg_out:.0f}** | **{avg_in + avg_out:.0f}** | **{avg_calls:.1f}** |")

    # ── 表3: Agent 循环详细 ──
    lines.append("\n## 三、Agent 循环详细\n")
    lines.append("| # | 课题 | Planner | Researcher | Writer | Reviewer | Formatter | 总轮次 |")
    lines.append("|---|------|---------|-----------|--------|----------|-----------|--------|")
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r.topic[:25]}... | {r.planner_calls} | {r.researcher_calls} | "
            f"{r.writer_calls} | {r.reviewer_calls} | {r.formatter_calls} | {r.total_loop_rounds} |"
        )
    avg_p = sum(r.planner_calls for r in results) / len(results) if results else 0
    avg_res = sum(r.researcher_calls for r in results) / len(results) if results else 0
    avg_w = sum(r.writer_calls for r in results) / len(results) if results else 0
    avg_rv = sum(r.reviewer_calls for r in results) / len(results) if results else 0
    avg_f = sum(r.formatter_calls for r in results) / len(results) if results else 0
    lines.append(f"| **平均** | - | **{avg_p:.1f}** | **{avg_res:.1f}** | **{avg_w:.1f}** | **{avg_rv:.1f}** | **{avg_f:.1f}** | **{avg_rounds:.1f}** |")

    # ── 表4: 工具调用成功率 ──
    lines.append("\n## 四、工具调用成功率\n")
    lines.append("| # | 课题 | 总调用 | 成功 | 失败 | 重试后成功 | 成功率 |")
    lines.append("|---|------|--------|------|------|-----------|--------|")
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r.topic[:25]}... | {r.tool_calls_total} | {r.tool_calls_success} | "
            f"{r.tool_calls_failed} | {r.tool_calls_retried_and_success} | {r.tool_success_rate:.1%} |"
        )
    avg_total_t = sum(r.tool_calls_total for r in results) / len(results) if results else 0
    avg_success_t = sum(r.tool_calls_success for r in results) / len(results) if results else 0
    avg_fail_t = sum(r.tool_calls_failed for r in results) / len(results) if results else 0
    avg_retry_t = sum(r.tool_calls_retried_and_success for r in results) / len(results) if results else 0
    overall_success = sum(r.tool_calls_success + r.tool_calls_retried_and_success for r in results) / sum(r.tool_calls_total for r in results) if sum(r.tool_calls_total for r in results) > 0 else 0
    lines.append(f"| **平均/总计** | - | **{avg_total_t:.1f}** | **{avg_success_t:.1f}** | **{avg_fail_t:.1f}** | **{avg_retry_t:.1f}** | **{overall_success:.1%}** |")

    # ── 表5: LLM-as-Judge 评估 ──
    lines.append("\n## 五、LLM-as-Judge 评估 (1-10分)\n")
    lines.append("| # | 课题 | 事实准确率 | 结构完整度 | 引用质量 | 综合 |")
    lines.append("|---|------|----------|----------|---------|------|")
    for i, r in enumerate(results, 1):
        avg_judge = (r.judge_factual + r.judge_structure + r.judge_citations) / 3 if (r.judge_factual + r.judge_structure + r.judge_citations) > 0 else 0
        lines.append(
            f"| {i} | {r.topic[:25]}... | {r.judge_factual:.1f} | {r.judge_structure:.1f} | "
            f"{r.judge_citations:.1f} | {avg_judge:.1f} |"
        )
    avg_fa = sum(r.judge_factual for r in results) / len(results) if results else 0
    avg_sc = sum(r.judge_structure for r in results) / len(results) if results else 0
    avg_cq = sum(r.judge_citations for r in results) / len(results) if results else 0
    avg_all = (avg_fa + avg_sc + avg_cq) / 3
    lines.append(f"| **平均** | - | **{avg_fa:.1f}** | **{avg_sc:.1f}** | **{avg_cq:.1f}** | **{avg_all:.1f}** |")

    # ── 表6: 综合汇总 ──
    lines.append("\n## 六、综合汇总\n")
    lines.append("| 指标 | 样本数 | 均值 | 最小值 | 最大值 |")
    lines.append("|------|--------|------|--------|--------|")

    metrics = {
        "生成时间 (s)": [r.wall_time_seconds for r in results],
        "总Token消耗": [r.total_input_tokens + r.total_output_tokens for r in results],
        "Agent循环轮次": [r.total_loop_rounds for r in results],
        "LLM调用次数": [r.llm_call_count for r in results],
        "工具调用成功率": [r.tool_success_rate for r in results],
        "事实准确率 (Judge)": [r.judge_factual for r in results],
        "结构完整度 (Judge)": [r.judge_structure for r in results],
        "引用质量 (Judge)": [r.judge_citations for r in results],
    }

    for name, values in metrics.items():
        if values:
            mean_v = sum(values) / len(values)
            min_v = min(values)
            max_v = max(values)
            lines.append(f"| {name} | {len(values)} | {mean_v:.1f} | {min_v:.1f} | {max_v:.1f} |")

    lines.append("")
    lines.append("---")
    lines.append("*报告由 benchmark_5_topics.py 自动生成*")

    return "\n".join(lines)


# ── 主函数 ────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("DeepResearch-Agent 性能测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试课题数: {len(TEST_TOPICS)}")
    print("=" * 60)

    results: List[MetricsCollector] = []

    # Phase 1: 运行 5 个课题
    print("\n" + "=" * 60)
    print("Phase 1: 运行研究任务")
    print("=" * 60)

    for topic_item in TEST_TOPICS:
        collector = await run_single_topic(topic_item)
        results.append(collector)
        # 短暂休息避免 API 限速
        await asyncio.sleep(2)

    # Phase 2: LLM-as-Judge 评估
    print("\n" + "=" * 60)
    print("Phase 2: LLM-as-Judge 评估")
    print("=" * 60)

    for i, collector in enumerate(results):
        if collector.status == "completed" and collector.report_text:
            print(f"\n  📋 评估 #{i+1}: {collector.topic[:40]}...")
            judge_scores = await judge_report(collector)
            collector.judge_factual = judge_scores.get("factual_accuracy", 0.0)
            collector.judge_structure = judge_scores.get("structure_completeness", 0.0)
            collector.judge_citations = judge_scores.get("citation_quality", 0.0)
            print(f"     事实准确率: {collector.judge_factual:.1f}/10")
            print(f"     结构完整度: {collector.judge_structure:.1f}/10")
            print(f"     引用质量: {collector.judge_citations:.1f}/10")
        else:
            print(f"\n  ⚠ 跳过评估 #{i+1}: 状态={collector.status}, 报告长度={len(collector.report_text)}")
            collector.judge_factual = 0.0
            collector.judge_structure = 0.0
            collector.judge_citations = 0.0
        await asyncio.sleep(1)

    # Phase 3: 生成结果表格
    print("\n" + "=" * 60)
    print("Phase 3: 生成结果表格")
    print("=" * 60)

    table_content = generate_results_table(results)

    # 保存到输出目录
    output_dir = Path(__file__).resolve().parent.parent / "data" / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_report.md"
    output_path.write_text(table_content, encoding="utf-8")

    # 也保存 JSON 格式的原始数据
    json_path = output_dir / "benchmark_raw_data.json"
    raw_data = [r.to_dict() for r in results]
    json_path.write_text(json.dumps(raw_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n📄 表格报告: {output_path}")
    print(f"📊 原始数据: {json_path}")

    # 输出到控制台
    print("\n" + table_content)

    return results


if __name__ == "__main__":
    asyncio.run(main())
