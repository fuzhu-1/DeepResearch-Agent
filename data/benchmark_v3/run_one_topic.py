"""Run one topic from benchmark_v3.py — invoked repeatedly with progress checkpointing."""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path("/sessions/kind-festive-knuth/mnt/DeepResearch-Agent")))
from tests.benchmark_v3 import TOPICS, run_one, load_progress, save_progress, judge_reports, generate_report
import asyncio

async def main():
    records = load_progress()
    done_indices = {r["index"] for r in records}
    print(f"{len(records)}/{len(TOPICS)} topics already done. Done indices: {done_indices}", flush=True)

    for i, topic in enumerate(TOPICS, 1):
        if i in done_indices:
            continue

        print(f"[{i}/5] START {topic[:30]}...", flush=True)
        t0 = time.time()
        rec = await run_one(topic, i - 1)
        elapsed = time.time() - t0
        rec["_single_run_elapsed"] = round(elapsed, 1)
        records.append(rec)
        save_progress(records)
        print(f"[{i}/5] DONE in {elapsed:.0f}s status={rec.get('status')}", flush=True)

        # Check if all done
        if len(records) >= len(TOPICS):
            print("All topics done. Running LLM-as-Judge...", flush=True)
            records = await judge_reports(records)
            save_progress(records)
            report = generate_report(records)
            OUTPUT = Path("/sessions/kind-festive-knuth/mnt/DeepResearch-Agent/data/benchmark_v3")
            (OUTPUT / "benchmark_v3_report.md").write_text(report, encoding="utf-8")
            (OUTPUT / "results.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            print("\n" + report, flush=True)
            print(f"\nDONE — report generated", flush=True)
        break  # Always exit after one topic

    print("ONE_TOPIC_DONE", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
