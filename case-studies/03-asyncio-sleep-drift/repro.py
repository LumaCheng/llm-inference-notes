"""
Minimal reproduction of asyncio.sleep cumulative drift in a load-generator pattern.

Demonstrates two pacing strategies:
  1. Naive: `await asyncio.sleep(1 / target_tps)` per request
  2. Wall-clock-aligned: sleep only the remaining gap to the next scheduled send

The naive version drifts because each `asyncio.sleep` call wakes up slightly
late (event loop timer resolution + context switching across worker tasks),
and the overshoots accumulate. Over 1000 requests at TPS=95, the drift can
exceed 1 second of wall-clock time, which makes the measured TPS materially
lower than the requested TPS.

Run:
    python repro.py

What you should see: at higher target TPS the naive measured TPS falls below
target by an increasing margin, while the wall-clock-aligned version tracks
target tightly until the simulated server actually saturates.
"""

import asyncio
import time
from contextlib import asynccontextmanager


# ----- a fake "server" that simulates a fixed per-request service time -----
# Real LLM endpoints have per-request latency on the order of ~30-100 ms;
# here we use 30 ms to keep the repro fast. The point of this script is
# load-generator-side drift, not server behavior.
SERVER_LATENCY_S = 0.030
SERVER_CONCURRENCY = 10  # how many requests the "server" handles in parallel


class FakeServer:
    """A trivially saturating server: fixed latency, fixed concurrency cap."""

    def __init__(self, latency_s: float, concurrency: int):
        self.latency_s = latency_s
        self.sem = asyncio.Semaphore(concurrency)

    async def handle(self) -> None:
        async with self.sem:
            await asyncio.sleep(self.latency_s)


# ----- Strategy A: naive per-request sleep -----
async def run_naive(server: FakeServer, n_requests: int, target_tps: float) -> dict:
    """Send n_requests at target_tps using `await asyncio.sleep(1/tps)` between sends."""
    interval = 1.0 / target_tps
    workers: list[asyncio.Task] = []
    queue: asyncio.Queue = asyncio.Queue()

    async def worker():
        while True:
            req = await queue.get()
            if req is None:
                queue.task_done()
                break
            await server.handle()
            queue.task_done()

    for _ in range(SERVER_CONCURRENCY):
        workers.append(asyncio.create_task(worker()))

    expected_send_times: list[float] = []
    actual_send_times: list[float] = []

    start = time.perf_counter()
    for i in range(n_requests):
        expected_send_times.append(i * interval)
        actual_send_times.append(time.perf_counter() - start)
        await queue.put(i)
        await asyncio.sleep(interval)

    # drain queue and stop workers
    for _ in workers:
        await queue.put(None)
    await queue.join()
    for w in workers:
        await w
    end = time.perf_counter()

    elapsed = end - start
    measured_tps = n_requests / elapsed
    drift_last = actual_send_times[-1] - expected_send_times[-1]
    return {
        "strategy": "naive",
        "target_tps": target_tps,
        "n_requests": n_requests,
        "elapsed_s": elapsed,
        "measured_tps": measured_tps,
        "drift_last_request_ms": drift_last * 1000,
    }


# ----- Strategy B: wall-clock-aligned sleep -----
async def run_aligned(server: FakeServer, n_requests: int, target_tps: float) -> dict:
    """Send n_requests at target_tps, sleeping only the remaining gap to the next slot."""
    workers: list[asyncio.Task] = []
    queue: asyncio.Queue = asyncio.Queue()

    async def worker():
        while True:
            req = await queue.get()
            if req is None:
                queue.task_done()
                break
            await server.handle()
            queue.task_done()

    for _ in range(SERVER_CONCURRENCY):
        workers.append(asyncio.create_task(worker()))

    expected_send_times: list[float] = []
    actual_send_times: list[float] = []

    start = time.perf_counter()
    for i in range(n_requests):
        expected = i / target_tps
        elapsed = time.perf_counter() - start
        gap = expected - elapsed
        if gap > 0:
            await asyncio.sleep(gap)
        expected_send_times.append(expected)
        actual_send_times.append(time.perf_counter() - start)
        await queue.put(i)

    for _ in workers:
        await queue.put(None)
    await queue.join()
    for w in workers:
        await w
    end = time.perf_counter()

    elapsed = end - start
    measured_tps = n_requests / elapsed
    drift_last = actual_send_times[-1] - expected_send_times[-1]
    return {
        "strategy": "aligned",
        "target_tps": target_tps,
        "n_requests": n_requests,
        "elapsed_s": elapsed,
        "measured_tps": measured_tps,
        "drift_last_request_ms": drift_last * 1000,
    }


async def main() -> None:
    server = FakeServer(SERVER_LATENCY_S, SERVER_CONCURRENCY)
    n_requests = 1000

    # theoretical server ceiling = concurrency / latency
    server_ceiling = SERVER_CONCURRENCY / SERVER_LATENCY_S

    print(f"server: latency={SERVER_LATENCY_S*1000:.0f}ms, concurrency={SERVER_CONCURRENCY}")
    print(f"server theoretical max TPS: {server_ceiling:.0f}")
    print(f"requests per run: {n_requests}")
    print()
    print(f"{'strategy':<10}{'target':>8}{'measured':>12}{'gap':>10}{'drift(ms)':>14}")
    print("-" * 54)

    for target_tps in [50, 100, 150, 200, 250, 300]:
        if target_tps > server_ceiling * 1.1:
            # past saturation, both strategies will be capped by server
            continue
        for strategy in (run_naive, run_aligned):
            result = await strategy(server, n_requests, target_tps)
            gap = result["target_tps"] - result["measured_tps"]
            print(
                f"{result['strategy']:<10}"
                f"{result['target_tps']:>8.0f}"
                f"{result['measured_tps']:>12.2f}"
                f"{gap:>10.2f}"
                f"{result['drift_last_request_ms']:>14.1f}"
            )
        print()


if __name__ == "__main__":
    asyncio.run(main())
