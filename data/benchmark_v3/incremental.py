"""Incremental benchmark runner — runs one topic per invocation, checkpoints to progress.json."""
import asyncio
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path("/sessions/amazing-confident-pasteur/mnt/DeepResearch-Agent")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from tests.benchmark_v3 import (
    TOPICS, OUTPUT, load_progress, save_progress,
    run_one, judge_reports, generate_report
)

TOPIC_TIMEOUT = 120  # seconds per topic


async def main():
    records = load_progress()
    done_indices = {r["index"] for r in records}

    print(f"Loaded {len(records)} records. Done indices: {done_indices}", flush=True)
    print(f"Topics to run: {[i for i in range(1, 6) if i not in done_indices]}", flush=True)

    ran_one = False

    for i, topic in enumerate(TOPICS, 1):
        if i in done_indices:
            print(f"[{i}/5] SKIP {topic[:40]}... (already done)", flush=True)
            continue

        print(f"[{i}/5] RUNNING {topic[:40]}... (timeout={TOPIC_TIMEOUT}s)", flush=True)
        t0 = time.time()
        try:
            rec = await asyncio.wait_for(run_one(topic, i - 1), timeout=TOPIC_TIMEOUT)
            elapsed = time.time() - t0
            records.append(rec)
            save_progress(records)
            status = rec.get("status", "?")
            score = rec.get("review_score", 0)
            print(f"[{i}/5] DONE in {elapsed:.0f}s | status={status} | score={score:.2f}", flush=True)
        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            print(f"[{i}/5] TIMEOUT after {elapsed:.0f}s", flush=True)
            records.append({
                "index": i,
                "topic": topic,
                "status": "timeout",
                "elapsed_s": round(elapsed, 1),
                "error": f"Timeout after {TOPIC_TIMEOUT}s",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
            })
            save_progress(records)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{i}/5] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
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

        ran_one = True
        break  # Only run ONE topic per invocation

    if not ran_one:
        # All topics processed — run judge + report
        print("\n=== ALL TOPICS PROCESSED ===", flush=True)
        completed_records = [r for r in records if r.get("status") == "completed"]

        if completed_records:
            print(f"\nPHASE 2: LLM-as-Judge on {len(completed_records)} reports", flush=True)
            try:
                records = await judge_reports(records)
                save_progress(records)
            except Exception as e:
                print(f"Judge phase error: {e}", flush=True)

        print(f"\nPHASE 3: Generating report", flush=True)
        report = generate_report(records)
        report_path = OUTPUT / "benchmark_v3_report.md"
        report_path.write_text(report, encoding="utf-8")
        (OUTPUT / "results.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

        print(report, flush=True)
        print(f"\nREPORT_SAVED: {report_path}", flush=True)
        print("BENCHMARK_COMPLETE", flush=True)

    completed = [r for r in records if r.get("status") == "completed"]
    print(f"\nProgress: {len(completed)}/{len(TOPICS)} completed, {len(records)} total records", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
