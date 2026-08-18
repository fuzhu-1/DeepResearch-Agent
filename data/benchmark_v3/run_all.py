"""Run all 5 benchmark topics sequentially with a single python process."""
import json, sys, os, asyncio, time
sys.path.insert(0, "/sessions/kind-festive-knuth/mnt/DeepResearch-Agent")

from pathlib import Path
BASE = Path("/sessions/kind-festive-knuth/mnt/DeepResearch-Agent")

from tests.benchmark_v3 import TOPICS, run_one, load_progress, save_progress, judge_reports, generate_report

async def main():
    print("=" * 60, flush=True)
    print("DeepResearch-Agent Benchmark v3", flush=True)
    print("=" * 60, flush=True)

    records = load_progress()
    done_indices = {r["index"] for r in records}
    print(f"Loaded {len(records)} records. Already done: {done_indices}", flush=True)

    OUTPUT = BASE / "data" / "benchmark_v3"

    for i, topic in enumerate(TOPICS, 1):
        if i in done_indices:
            print(f"[{i}/5] SKIP {topic[:30]}... already done", flush=True)
            continue

        print(f"\n--- [{i}/5] RUNNING: {topic} ---", flush=True)
        t0 = time.time()
        try:
            rec = await run_one(topic, i - 1)
            elapsed = time.time() - t0
            records.append(rec)
            save_progress(records)
            print(f"[{i}/5] DONE in {elapsed:.0f}s | status={rec.get('status')} | score={rec.get('review_score', 0):.2f}", flush=True)
        except Exception as e:
            print(f"[{i}/5] FAILED: {e}", flush=True)
            save_progress(records)

        await asyncio.sleep(2)

    # Phase 2: Judge
    print("\n--- Phase 2: LLM-as-Judge ---", flush=True)
    completed = [r for r in records if r.get("status") == "completed"]
    if completed:
        records = await judge_reports(records)
        save_progress(records)

    # Phase 3: Report
    print("\n--- Phase 3: Generating report ---", flush=True)
    report = generate_report(records)
    report_path = OUTPUT / "benchmark_v3_report.md"
    report_path.write_text(report, encoding="utf-8")
    (OUTPUT / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report, flush=True)
    print(f"\nREPORT_GENERATED:{report_path}", flush=True)

    # Copy to resume folder
    import shutil
    dest = Path("/sessions/kind-festive-knuth/mnt/面试简历/DeepResearch-Agent_Benchmark_v3_Report.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(report_path), str(dest))
    print(f"COPIED_TO:{dest}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
