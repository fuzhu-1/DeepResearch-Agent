"""Benchmark: run ONE research topic through the workflow and collect detailed metrics.

Topic: "量子计算在药物研发中的应用现状与前景"
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "benchmark_results"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Global metrics counters
METRICS = {
    "llm_calls": 0,
    "tool_calls_total": 0,
    "tool_success": 0,
    "tool_fail": 0,
    "tool_retry_ok": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "node_runs": {"planner": 0, "researcher": 0, "writer": 0, "reviewer": 0, "formatter": 0},
}


def patch_all():
    """Monkey-patch LLM calls, tool execution, and workflow nodes for metrics collection."""

    # 1. Patch LLM call to count tokens and calls (via the OpenAI client directly)
    import app.utils.llm as llm_mod
    from app.services.config_service import get_active_config
    from app.config import settings

    async def tracked_llm_call(*args, **kwargs):
        METRICS["llm_calls"] += 1
        rt = get_active_config()
        api_key = rt.api_key if (rt and rt.api_key) else (
            settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        )
        base_url = rt.base_url if (rt and rt.base_url) else "https://api.deepseek.com/v1"
        model = (kwargs.get("config") and kwargs["config"].model) or "deepseek-chat"

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=90.0, max_retries=1)

        sys_p = kwargs.get("system_prompt", "") or (args[0] if args else "")
        usr_p = kwargs.get("user_prompt", "") or (args[1] if len(args) > 1 else "")
        cfg = kwargs.get("config")

        messages = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
        call_kwargs = {
            "model": cfg.model if cfg else model,
            "messages": messages,
            "temperature": cfg.temperature if cfg else 0.3,
            "max_tokens": cfg.max_tokens if cfg else 4096,
        }
        tools = kwargs.get("tools")
        if tools:
            call_kwargs["tools"] = tools

        resp = await client.chat.completions.create(**call_kwargs)
        if hasattr(resp, "usage") and resp.usage:
            METRICS["input_tokens"] += resp.usage.prompt_tokens or 0
            METRICS["output_tokens"] += resp.usage.completion_tokens or 0
        return resp.choices[0].message.content or ""

    llm_mod.llm_call = tracked_llm_call

    # 2. Patch ToolRouter.execute to count tool calls (success/fail/retry)
    from app.tools.router import ToolRouter
    from app.tools.base import ToolResult

    _orig_exec = ToolRouter.execute

    async def tracked_exec(self, name, **params):
        METRICS["tool_calls_total"] += 1
        try:
            result = await _orig_exec(self, name, **params)
            if getattr(result, "success", True):
                METRICS["tool_success"] += 1
            else:
                METRICS["tool_fail"] += 1
            return result
        except Exception as e:
            # Retry once
            try:
                await asyncio.sleep(1)
                result = await _orig_exec(self, name, **params)
                if getattr(result, "success", True):
                    METRICS["tool_retry_ok"] += 1
                else:
                    METRICS["tool_fail"] += 1
                return result
            except Exception:
                METRICS["tool_fail"] += 1
                return ToolResult(success=False, error=str(e), data=[])

    ToolRouter.execute = tracked_exec

    # 3. Patch workflow nodes to count invocations
    import app.workflow.nodes as n

    # (func_name_in_modules, display_name_in_metrics)
    node_mappings = [
        ("planner_node", "planner"),
        ("executor_node", "researcher"),
        ("writer_node", "writer"),
        ("reviewer_node", "reviewer"),
        ("formatter_node", "formatter"),
    ]

    for func_name, display_name in node_mappings:
        original_fn = getattr(n, func_name)

        def _create_wrapper(orig, dname):
            async def wrapper(state):
                METRICS["node_runs"][dname] += 1
                return await orig(state)
            return wrapper

        setattr(n, func_name, _create_wrapper(original_fn, display_name))


async def main():
    topic = "量子计算在药物研发中的应用现状与前景"
    task_id = f"bench_{uuid.uuid4().hex[:12]}"

    print(f"Topic: {topic}", flush=True)
    print(f"TaskID: {task_id}", flush=True)
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

    patch_all()

    from app.workflow.graph import build_graph
    from app.models.state import ResearchState

    graph = build_graph()
    state = ResearchState(task=topic, use_rag=False)

    print("Graph built, starting ainvoke...", flush=True)

    t0 = time.time()
    try:
        result = await graph.ainvoke(state)
    except Exception as exc:
        print(f"ERROR during ainvoke: {type(exc).__name__}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    elapsed = time.time() - t0

    status = getattr(result, "status", "?")
    report = getattr(result, "final_report", "") or ""
    plan = getattr(result, "plan", []) or []
    research = getattr(result, "research_data", []) or []
    score = getattr(result, "review_score", 0.0)
    feedback = (getattr(result, "review_feedback", "") or "")[:300]
    iteration = getattr(result, "iteration_count", 0)
    errors = getattr(result, "errors", []) or []

    print(f"\n{'='*50}", flush=True)
    print(f"Status: {status}", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)
    print(f"Plan size: {len(plan)}", flush=True)
    print(f"Research items: {len(research)}", flush=True)
    print(f"Report length: {len(report)} chars", flush=True)
    print(f"Review score: {score:.3f}", flush=True)
    print(f"Iteration count: {iteration}", flush=True)
    print(f"Errors: {len(errors)}", flush=True)
    print(f"\n--- Metrics ---", flush=True)
    print(f"LLM calls: {METRICS['llm_calls']}", flush=True)
    print(f"Input tokens: {METRICS['input_tokens']}", flush=True)
    print(f"Output tokens: {METRICS['output_tokens']}", flush=True)
    print(
        f"Total tokens: {METRICS['input_tokens'] + METRICS['output_tokens']}", flush=True
    )
    print(
        f"Tool calls: total={METRICS['tool_calls_total']} "
        f"success={METRICS['tool_success']} retry_ok={METRICS['tool_retry_ok']} "
        f"fail={METRICS['tool_fail']}",
        flush=True,
    )
    print(f"Node runs: {METRICS['node_runs']}", flush=True)
    total_rounds = sum(METRICS['node_runs'].values())
    print(f"Total loop rounds: {total_rounds}", flush=True)

    # Tool success rate
    total_t = METRICS['tool_calls_total']
    succeeded = METRICS['tool_success'] + METRICS['tool_retry_ok']
    rate = succeeded / total_t if total_t > 0 else 0
    print(f"Tool success rate: {rate:.1%} ({succeeded}/{total_t})", flush=True)
    if feedback:
        print(f"Review feedback: {feedback}", flush=True)

    # Save report to topic_1_report.md
    report_path = OUTPUT / "topic_1_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {report_path} ({len(report)} chars)", flush=True)

    # Save metrics to topic_1_metrics.json
    metrics_data = {
        "topic": topic,
        "task_id": task_id,
        "start_time": datetime.now().isoformat(),
        "elapsed_s": round(elapsed, 1),
        "status": status,
        "plan_size": len(plan),
        "plan": [
            {"id": p.id, "description": p.description, "tool": p.tool} for p in plan
        ],
        "research_items": len(research),
        "report_length": len(report),
        "review_score": round(score, 3),
        "review_feedback": feedback,
        "iteration_count": iteration,
        "error_count": len(errors),
        "llm_calls": METRICS["llm_calls"],
        "input_tokens": METRICS["input_tokens"],
        "output_tokens": METRICS["output_tokens"],
        "total_tokens": METRICS["input_tokens"] + METRICS["output_tokens"],
        "tool_calls_total": METRICS["tool_calls_total"],
        "tool_success": METRICS["tool_success"],
        "tool_retry_ok": METRICS["tool_retry_ok"],
        "tool_fail": METRICS["tool_fail"],
        "tool_success_rate": round(rate, 3),
        "node_runs": dict(METRICS["node_runs"]),
        "total_rounds": total_rounds,
    }
    metrics_path = OUTPUT / "topic_1_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Metrics saved: {metrics_path}", flush=True)

    # Also dump full metrics to stdout
    print(f"\n{'='*50}", flush=True)
    print("FULL METRICS JSON:", flush=True)
    print(json.dumps(metrics_data, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
