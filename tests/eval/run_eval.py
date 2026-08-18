"""Run the workflow over a fixed topic set and score reports with LLM-as-judge."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "eval" / "fixtures" / "topics.json"
OUTPUT = ROOT / "data" / "eval"
OUTPUT.mkdir(parents=True, exist_ok=True)


async def run_one(topic: str, max_iterations: int = 2) -> dict:
    from app.workflow.graph import run_research
    from tests.eval.judge import judge_report

    start = datetime.now()
    state = await run_research(topic, max_iterations=max_iterations)
    report = state.final_report if hasattr(state, "final_report") else state.get("final_report", "")
    scores = await judge_report(topic, report)
    dims = ("completeness", "citation_quality", "coherence", "depth")
    return {
        "topic": topic,
        "status": state.status if hasattr(state, "status") else state.get("status"),
        "report_length": len(report),
        "elapsed_s": round((datetime.now() - start).total_seconds(), 1),
        "scores": scores,
        "average": round(sum(scores[k] for k in dims) / len(dims), 2),
    }


async def main():
    topics = json.loads(FIXTURES.read_text(encoding="utf-8"))
    results = []
    for topic in topics:
        print(f"== {topic[:40]} ...")
        results.append(await run_one(topic))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUTPUT / f"results-{stamp}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    for r in results:
        print(f"{r['average']:.2f}  {r['topic'][:50]}")


if __name__ == "__main__":
    asyncio.run(main())
