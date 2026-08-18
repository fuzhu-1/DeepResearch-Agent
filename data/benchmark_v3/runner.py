"""Minimal runner — run benchmark topics one at a time, checkpointing to progress.json."""
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path("/sessions/trusting-festive-edison/mnt/DeepResearch-Agent")
sys.path.insert(0, str(PROJECT))

from tests.benchmark_v3 import (
    TOPICS, OUTPUT, load_progress, save_progress,
    run_one, judge_reports, generate_report
)


async def main():
    records = load_progress()
    done_indices = {r["index"] for r in records}
    started = time.time()

    for i, topic in enumerate(TOPICS, 1):
        if i in done_indices:
            continue

        t0 = time.time()
        try:
            rec = await run_one(topic, i - 1)
            elapsed = time.time() - t0
            records.append(rec)
            save_progress(records)
            status = rec.get("status", "?")
            score = rec.get("review_score", 0)
            msg = f"TOPIC_{i}_DONE | {elapsed:.0f}s | status={status} | score={score:.2f}"
        except Exception as e:
            elapsed = time.time() - t0
            msg = f"TOPIC_{i}_FAILED | {elapsed:.0f}s | {e}"
            # Append a minimal error record
            records.append({
                "index": i,
                "topic": topic,
                "status": "error",
                "elapsed_s": round(elapsed, 1),
                "error": str(e)[:500],
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
            })
            save_progress(records)

        print(msg, flush=True)
        await asyncio.sleep(2)

    total_elapsed = time.time() - started
    print(f"\nALL_TOPICS_DONE in {total_elapsed:.0f}s | {len(records)} records", flush=True)

    # Phase 2: Judge
    completed = [r for r in records if r.get("status") == "completed"]
    if completed:
        print(f"\nPHASE 2: LLM-as-Judge on {len(completed)} reports", flush=True)
        records = await judge_reports(records)
        save_progress(records)

    # Phase 3: Report
    print(f"\nPHASE 3: Generating report", flush=True)
    report = generate_report(records)
    report_path = OUTPUT / "benchmark_v3_report.md"
    report_path.write_text(report, encoding="utf-8")
    (OUTPUT / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report, flush=True)
    print(f"\nREPORT_SAVED: {report_path}", flush=True)
    print(f"RESULTS_SAVED: {OUTPUT / 'results.json'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
