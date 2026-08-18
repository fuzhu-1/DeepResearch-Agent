import asyncio, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent) if '__file__' in dir() else str(Path.cwd()))

from tests.benchmark_v3 import run_one, TOPICS

async def main():
    topic = TOPICS[0]
    print(f"Running topic 1: {topic[:50]}...", flush=True)
    t0 = time.time()
    try:
        result = await asyncio.wait_for(run_one(topic, 0), timeout=120)
        print(f"Done in {time.time()-t0:.1f}s", flush=True)
        print(f"Status: {result.get('status')}", flush=True)
        print(f"Elapsed: {result.get('elapsed_s')}", flush=True)
        print(f"LLM calls: {result.get('llm_calls')}", flush=True)
        print(f"Tokens: {result.get('total_tokens')}", flush=True)
        print(f"Tool calls: {result.get('tool_total')}", flush=True)
        print(f"Report length: {result.get('report_length')}", flush=True)
        print("SUCCESS")
    except asyncio.TimeoutError:
        print(f"TIMEOUT after {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()

asyncio.run(main())
