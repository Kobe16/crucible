#!/usr/bin/env python3
"""
Lightweight load benchmark for the Crucible inference gateway.

Fires concurrent requests at POST /predict, verifies correctness,
and prints a latency/throughput summary. All requests are dispatched as
fast as possible, throttled only by the concurrency semaphore (no pacing
or randomized delays).

Prerequisites:
    1. Start the stack:  make up
    2. Wait a few seconds for the worker to load the model.
    3. pip install aiohttp  (only external dependency)

Usage:
    python tests/load/benchmark.py
    python tests/load/benchmark.py --requests 200 --concurrency 50

    # low traffic — small batches
    python tests/load/benchmark.py --requests 50 --concurrency 2

    # high traffic — batches fill up
    python tests/load/benchmark.py --requests 500 --concurrency 100

Flags:
    --requests      Total number of requests to send (default: 100)
    --concurrency   Max requests in-flight at once (default: 10)

Server-side knobs (env vars in docker-compose.yml):
    MAX_BATCH_SIZE    Requests per batch before flush (default: 8)
    BATCH_TIMEOUT_MS  Max wait for a batch to fill (default: 50ms)
    QUEUE_DEPTH       Max queued requests before 503 (default: 1000)
    MAX_WORKERS       gRPC thread pool size (default: 4)
"""

import argparse
import asyncio
import json
import statistics
import time

import aiohttp

SAMPLE_INPUTS = [
    "This movie was absolutely wonderful and I loved every minute of it",
    "Terrible film, complete waste of time",
    "The acting was okay but the plot was predictable",
    "A masterpiece of modern cinema",
    "I fell asleep halfway through",
    "Not bad, not great, just average",
    "One of the best experiences I have ever had",
    "Disappointing sequel that fails to capture the original magic",
    "Surprisingly good for a low budget production",
    "The worst movie I have seen this year by far",
]


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    text: str,
) -> dict:
    payload = {"input": text}
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            body = await resp.text()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "status": resp.status,
                "body": body,
                "latency_ms": elapsed_ms,
                "error": None,
            }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "status": 0,
            "body": "",
            "latency_ms": elapsed_ms,
            "error": str(e),
        }


async def worker(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    url: str,
    text: str,
) -> dict:
    async with sem:
        return await send_request(session, url, text)


def validate_response(result: dict) -> bool:
    if result["error"] is not None:
        return False
    if result["status"] != 200:
        return False
    try:
        data = json.loads(result["body"])
    except json.JSONDecodeError:
        return False
    output = data.get("output", "")
    return "POSITIVE" in output or "NEGATIVE" in output


def percentile(sorted_data: list[float], p: float) -> float:
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def print_summary(results: list[dict], wall_time_s: float) -> None:
    total = len(results)
    successes = sum(1 for r in results if validate_response(r))
    failures = total - successes
    latencies = sorted(r["latency_ms"] for r in results)
    error_details: dict[str, int] = {}
    for r in results:
        if not validate_response(r):
            key = r["error"] or f"HTTP {r['status']}"
            error_details[key] = error_details.get(key, 0) + 1

    print("\n" + "=" * 55)
    print("  BENCHMARK RESULTS")
    print("=" * 55)
    print(f"  Total requests:    {total}")
    print(f"  Successful:        {successes}")
    print(f"  Failed:            {failures}")
    if error_details:
        for reason, count in error_details.items():
            print(f"    - {reason}: {count}")
    print(f"  Wall time:         {wall_time_s:.2f}s")
    print(f"  Throughput:        {total / wall_time_s:.1f} req/s")
    print("-" * 55)
    print("  LATENCY (ms)")
    print("-" * 55)
    print(f"  Mean:              {statistics.mean(latencies):.1f}")
    print(f"  Median (p50):      {percentile(latencies, 50):.1f}")
    print(f"  p90:               {percentile(latencies, 90):.1f}")
    print(f"  p95:               {percentile(latencies, 95):.1f}")
    print(f"  p99:               {percentile(latencies, 99):.1f}")
    print(f"  Min:               {latencies[0]:.1f}")
    print(f"  Max:               {latencies[-1]:.1f}")
    print("=" * 55)


BASE_URL = "http://localhost:8080"


async def run(num_requests: int, concurrency: int) -> None:
    predict_url = BASE_URL + "/predict"
    sem = asyncio.Semaphore(concurrency)

    print(f"Benchmarking {predict_url}")
    print(f"  Requests: {num_requests}  |  Concurrency: {concurrency}")
    print()

    # Build request list, cycling through sample inputs
    inputs = [SAMPLE_INPUTS[i % len(SAMPLE_INPUTS)] for i in range(num_requests)]

    async with aiohttp.ClientSession() as session:
        # Quick health check
        try:
            async with session.get(BASE_URL + "/health") as resp:
                if resp.status != 200:
                    print(f"WARNING: /health returned {resp.status}, server may not be ready")
        except aiohttp.ClientError as e:
            print(f"ERROR: Cannot reach server at {BASE_URL} ({e})")
            return

        wall_start = time.perf_counter()
        tasks = [worker(sem, session, predict_url, text) for text in inputs]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - wall_start

    print_summary(results, wall_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crucible load benchmark")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests")
    args = parser.parse_args()

    asyncio.run(run(args.requests, args.concurrency))


if __name__ == "__main__":
    main()
