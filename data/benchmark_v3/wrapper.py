"""Robust wrapper for benchmark_v3 — keeps running topics until bash kills the process.
On timeout, does NOT save error records so topics are retried next run.
Checkpoints every completed topic."""
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

TOPIC_TIMEOUT = 42  # seconds per topic — as long as possible within bash limit


async def main():
    records = load_progress()
    done_indices = {r["index"] for r in records}
    started = time.time()

    print(f"Loaded {len(records)} records. Done indices: {done_indices}", flush=True)
    all_completed = all(
        r.get("status") == "completed" for r in records
        if r.get("index") in done_indices
    )
    to_run = [idx for idx in range(1, 6) if idx not in done_indices]
    print(f"Topics to run: {to_run}", flush=True)

    # If all 5 are already completed, skip to judge + report
    if len(done_indices) >= 5 and all(
        r.get("status") == "completed"
        for r in records if r["index"] >= 1 and r["index"] <= 5
    ):
        print("All 5 topics already completed. Generating report.", flush=True)
        # Already did judge? Check if judge done
        needs_judge = any(r.get("judge_factual") is None for r in records)
        if needs_judge:
            records = await judge_reports(records)
            save_progress(records)
        report = generate_report(records)
        report_path = OUTPUT / "benchmark_v3_report.md"
        report_path.write_text(report, encoding="utf-8")
        (OUTPUT / "results.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(report, flush=True)
        print(f"REPORT_SAVED: {report_path}", flush=True)
        return

    topics_attempted_this_run = 0
    for i, topic in enumerate(TOPICS, 1):
        if i in done_indices:
            continue

        if topics_attempted_this_run >= 1:
            # Don't run more than 1 topic per invocation to avoid busting timeout
            break

        elapsed_from_start = time.time() - started
        remaining = TOPIC_TIMEOUT - elapsed_from_start - 3  # 3s buffer
        if remaining < 10:
            print(f"Not enough time remaining ({remaining:.0f}s), exiting", flush=True)
            break

        print(f"[{i}/5] RUNNING {topic[:50]}... (timeout={remaining:.0f}s)", flush=True)
        t0 = time.time()
        try:
            rec = await asyncio.wait_for(run_one(topic, i - 1), timeout=remaining)
            elapsed = time.time() - t0
            records.append(rec)
            save_progress(records)
            status = rec.get("status", "?")
            score = rec.get("review_score", 0)
            print(f"[{i}/5] DONE in {elapsed:.0f}s | status={status} | score={score:.2f}", flush=True)
            topics_attempted_this_run += 1
        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            print(f"[{i}/5] TIMEOUT after {elapsed:.0f}s — will retry next run", flush=True)
            # DO NOT save error record — let it be retried
            topics_attempted_this_run += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{i}/5] ERROR: {e}", flush=True)
            # Save error for non-timeout errors
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
            topics_attempted_this_run += 1

    # After running what we can, check if all done
    records = load_progress()  # Reload to get latest
    completed = [r for r in records if r.get("status") == "completed"]
    print(f"\nStatus after this run: {len(completed)}/{len(TOPICS)} completed, {len(records)} total records", flush=True)

    # If all 5 topics processed (some completed, some error), run judge + report
    processed = [r for r in records if r.get("index") in [1,2,3,4,5]]
    if len(processed) >= 5:
        print("\n=== ALL TOPICS PROCESSED ===", flush=True)
        completed_records = [r for r in records if r.get("status") == "completed"]
        if completed_records:
            print(f"PHASE 2: LLM-as-Judge on {len(completed_records)} reports", flush=True)
            records = await judge_reports(records)
            save_progress(records)
        print(f"PHASE 3: Generating report", flush=True)
        report = generate_report(records)
        report_path = OUTPUT / "benchmark_v3_report.md"
        report_path.write_text(report, encoding="utf-8")
        (OUTPUT / "results.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(report, flush=True)
        print(f"REPORT_SAVED: {report_path}", flush=True)

    print("WRAPPER_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
