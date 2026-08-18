"""快速基准测试 — 单课题运行，收集详细指标。"""
import asyncio, json, sys, time, uuid
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "benchmark_results"
OUTPUT.mkdir(parents=True, exist_ok=True)

# 全局统计
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
    """Monkey-patch LLM and tool calls for metrics."""
    import app.utils.llm as llm_mod
    from app.services.config_service import get_active_config
    from app.config import settings
    import os

    original = llm_mod.llm_call

    async def tracked_call(*args, **kwargs):
        METRICS["llm_calls"] += 1
        rt = get_active_config()
        api_key = rt.api_key if (rt and rt.api_key) else (settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", ""))
        base_url = (rt.base_url if (rt and rt.base_url) else "https://api.deepseek.com/v1")
        model = kwargs.get("config") and kwargs["config"].model or "deepseek-chat"

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

    llm_mod.llm_call = tracked_call

    # Patch tool execution
    from app.tools.router import ToolRouter
    orig_exec = ToolRouter.execute

    async def tracked_exec(self, name, **params):
        METRICS["tool_calls_total"] += 1
        try:
            result = await orig_exec(self, name, **params)
            if getattr(result, "success", True):
                METRICS["tool_success"] += 1
            else:
                METRICS["tool_fail"] += 1
            return result
        except Exception as e:
            # retry once
            try:
                await asyncio.sleep(1)
                result = await orig_exec(self, name, **params)
                if getattr(result, "success", True):
                    METRICS["tool_retry_ok"] += 1
                else:
                    METRICS["tool_fail"] += 1
                return result
            except Exception:
                METRICS["tool_fail"] += 1
                from app.tools.base import ToolResult
                return ToolResult(success=False, error=str(e), data=[])
    ToolRouter.execute = tracked_exec

    # Patch nodes
    import app.workflow.nodes as n
    for node_name in ["planner", "researcher", "writer", "reviewer", "formatter"]:
        key = node_name if node_name != "researcher" else "researcher"
        node_key = node_name

        async def make_wrapper(name):
            orig = getattr(n, f"{name}_node")

            async def wrapper(state):
                METRICS["node_runs"][name] += 1
                return await orig(state)
            return wrapper

        setattr(n, f"{node_key}_node", make_wrapper(node_key))

async def main():
    topic = "量子计算在药物研发中的应用现状与前景"
    task_id = f"bench_{uuid.uuid4().hex[:12]}"

    print(f"Topic: {topic}")
    print(f"TaskID: {task_id}")
    print(f"Start: {datetime.now():%H:%M:%S}")

    patch_all()

    from app.workflow.graph import build_graph
    from app.models.state import ResearchState

    graph = build_graph()
    state = ResearchState(task=topic, use_rag=False)

    t0 = time.time()
    result = await graph.ainvoke(state)
    elapsed = time.time() - t0

    status = getattr(result, "status", "?")
    report = getattr(result, "final_report", "") or ""
    plan = getattr(result, "plan", []) or []
    research = getattr(result, "research_data", []) or []
    score = getattr(result, "review_score", 0.0)
    feedback = (getattr(result, "review_feedback", "") or "")[:200]
    iteration = getattr(result, "iteration_count", 0)

    print(f"\n{'='*50}")
    print(f"Status: {status}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Plan size: {len(plan)}")
    print(f"Research items: {len(research)}")
    print(f"Report length: {len(report)} chars")
    print(f"Review score: {score:.3f}")
    print(f"Iteration count: {iteration}")
    print(f"\n--- Metrics ---")
    print(f"LLM calls: {METRICS['llm_calls']}")
    print(f"Input tokens: {METRICS['input_tokens']}")
    print(f"Output tokens: {METRICS['output_tokens']}")
    print(f"Total tokens: {METRICS['input_tokens'] + METRICS['output_tokens']}")
    print(f"Tool calls: total={METRICS['tool_calls_total']} "
          f"success={METRICS['tool_success']} retry_ok={METRICS['tool_retry_ok']} "
          f"fail={METRICS['tool_fail']}")
    print(f"Node runs: {METRICS['node_runs']}")
    total_rounds = sum(METRICS['node_runs'].values())
    print(f"Total loop rounds: {total_rounds}")

    # Tool success rate
    total_t = METRICS['tool_calls_total']
    succeeded = METRICS['tool_success'] + METRICS['tool_retry_ok']
    rate = succeeded / total_t if total_t > 0 else 0
    print(f"Tool success rate: {rate:.1%} ({succeeded}/{total_t})")
    print(f"Feedback: {feedback}")

    # Save
    (OUTPUT / f"{task_id}.md").write_text(report, encoding="utf-8")

    result_data = {
        "topic": topic, "task_id": task_id, "elapsed_s": round(elapsed, 1),
        "status": status, "plan_size": len(plan), "research_items": len(research),
        "report_length": len(report), "review_score": round(score, 3),
        "iteration_count": iteration,
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
        "feedback": feedback,
    }
    (OUTPUT / f"{task_id}_metrics.json").write_text(
        json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved: {OUTPUT}/{task_id}_metrics.json")

if __name__ == "__main__":
    asyncio.run(main())
