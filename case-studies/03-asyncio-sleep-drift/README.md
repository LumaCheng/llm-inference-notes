# How a known Python async pitfall is inflating GPU inference fleets

`await asyncio.sleep(1/tps)` accumulating drift over many calls is a well-documented pattern in the Python async community. What's less written-up is what this specific Python pitfall costs when the load tester it's hiding inside is being used to size an LLM inference fleet. With GPU supply tight enough that capacity sitting unused in one team's fleet is capacity another team can't get, a quiet load-generator bug that systematically under-reports server throughput translates directly into over-provisioned GPU spend.

This case study walks through how the bug surfaces during real LLM benchmarking, a minimal Python reproduction anyone can run in a few seconds, the underlying mechanism, the five-line fix, and four independent cross-validation methods.

## TL;DR

**The pitfall** — `await asyncio.sleep(1/tps)` does not sleep for exactly `1/tps` seconds. Each call wakes up roughly 1ms late (event-loop timer resolution + context switching across worker tasks), and the overshoots compound. By request 1000 the wall-clock send time is roughly a second behind schedule, and the reported "measured TPS" comes in well below target. Python async docs and Stack Overflow have covered this pattern in general terms for years; the specifics here are what it does in an LLM benchmarking context.

**Why this matters in an LLM context** — when the drifting load tester is being used to find an LLM serving endpoint's saturation point, the under-reported TPS gets used for capacity planning. The fleet ends up sized to a load-generator artifact, not a real server saturation point. At AWS list price for g6e.2xlarge (~$2.24/hr on-demand), each unnecessary instance is ~$1,600/month; at typical inference fleet sizes (10-100 instances) that's $16k-$160k/month of GPU spend that wasn't actually needed. In a tight GPU market, that over-provisioned capacity isn't sitting somewhere harmless — it's blocking another team from getting the GPUs they need.

**The fix** — replace per-request fixed-interval sleep with wall-clock-aligned sleep that compensates for accumulated drift. Five lines.

## How the bug shows up

The pattern that caught my attention in production benchmarking was a sweep across configs that should have produced different throughput numbers (different quantization, different batch sizes) and watching them all converge on roughly the same TPS. The first instinct is "we've found the server's saturation point." But if every config saturates at the same number, regardless of how cheap the per-request work is, the bottleneck probably isn't the server.

Same config, multiple runs would also vary by ±10-15%, which was inconsistent with the deterministic nature of the workload. And single-curl latency × concurrency gave a theoretical ceiling far above the measured number, with no obvious explanation for the gap.

Three soft anomalies, each individually rationalizable. Together they point at the measurement infrastructure, not the system being measured.

## Reproducing it

Below is a minimal standalone Python script that demonstrates the drift on any machine in a few seconds. No GPU, no network, no external service: just an asyncio "fake server" with fixed latency and concurrency, and two pacing strategies sending 1000 requests at varying target TPS.

The full script is in this directory ([`repro.py`](repro.py)). Run:

```bash
python3 repro.py
```

Output (sample, on a quiet laptop):

```
server: latency=30ms, concurrency=10
server theoretical max TPS: 333
requests per run: 1000

strategy    target    measured       gap     drift(ms)
------------------------------------------------------
naive           50       47.48      2.52        1048.7
aligned         50       49.97      0.03           1.1

naive          100       94.99      5.01         505.8
aligned        100       99.78      0.22           0.3

naive          150      132.62     17.38         849.3
aligned        150      149.46      0.54           0.2

naive          200      187.74     12.26         300.8
aligned        200      198.97      1.03           0.2

naive          250      223.80     26.20         441.4
aligned        250      248.34      1.66           0.2

naive          300      257.95     42.05         516.2
aligned        300      297.52      2.48           0.5
```

A few things to notice:

- The naive strategy reports under target at every TPS, with the gap growing as TPS increases.
- The drift column shows the gap between expected and actual wall-clock time at the last request: hundreds of milliseconds to a full second of cumulative slip.
- The aligned strategy stays within a couple of TPS of target at every level.
- At target=300, naive reports 258, well below the server's 333 ceiling. A capacity-planning team using this number would conclude the server tops out at ~260 and provision accordingly. The actual ceiling is 333.

## Why the drift accumulates

`await asyncio.sleep(X)` does not sleep for exactly X seconds. The asyncio event loop checks expired timers once per loop iteration, and the loop's tick rate depends on what else it's doing. With ten concurrent HTTP worker tasks plus I/O scheduling, the next "ready" check after a sleep typically fires 1-2 ms after the timer's nominal expiry.

Each sleep overshoots by a small amount, and because each sleep's wakeup time is the basis for the next sleep's start, **the overshoots compound additively**. After N requests at target T tps, the cumulative drift is roughly N × per-sleep overshoot, regardless of the request interval itself.

In the repro script with N=1000 and 10 worker tasks, drift comes in around 0.5-1.0 seconds. In production benchmarking with similar request volumes, it pushes the measured-vs-target gap large enough to trip the script's stop condition (which compares "measured TPS" to "requested TPS" and exits when the gap exceeds a threshold).

The script then concludes "server saturated at this TPS" and reports the last-known-good measured value. The server, meanwhile, was happily handling all the requests; the rate just wasn't getting through to it.

## The fix

Replace fixed-interval sleep with wall-clock-aligned sleep:

```python
# Before: drift accumulates
await asyncio.sleep(1 / tps)

# After: only sleep the remaining gap to the next scheduled send
expected_send_time = req_count / tps
elapsed = time.perf_counter() - start_time
gap = expected_send_time - elapsed
if gap > 0:
    await asyncio.sleep(gap)
```

The intuition: instead of saying "wait this long between requests," say "the next request should go out at time T; sleep until then." If previous sleeps overshot, the next sleep is shorter and the schedule self-corrects. The repro script's `aligned` strategy uses exactly this pattern.

The same small change is the difference between a load tester that lies and one that tells the truth.

## Cross-validation: trust no single measurement

The fix produced a much higher measured TPS, but a large change in a benchmark number is exactly the kind of result you want to verify before relying on it. So I cross-checked against three independent measurement strategies, each with different failure modes:

| Method | Mechanism | Notes |
|---|---|---|
| Connection-pool synchronous | `requests.Session()` + `ThreadPoolExecutor`, thread-based concurrency | No asyncio loop, no `asyncio.sleep`. Different failure modes (thread contention, GIL) but no shared bug with async pacing. |
| Single curl × concurrency (theoretical ceiling) | `time curl <endpoint>` once, divide concurrency by latency | Establishes a hard upper bound from first principles. Doesn't account for queueing or saturation effects, but the measurement should never *exceed* this. |
| Parallel curl loops | bash for-loop spawning N parallel curl loops, OS-level process scheduling | No async runtime at all. Failure modes around process startup and OS scheduling, which don't correlate with async timing bugs. |
| Fixed async script (`aligned`) | Same async stack as the buggy version, but with wall-clock-aligned pacing | Confirms the fix actually addresses the cause, on the same code path. |

All four methods converged on roughly the same answer in the production setup, materially above what the broken async tool was reporting. Multiple independent mechanisms agreeing is the strongest evidence available that the higher number is the real one and the original measurement was wrong.

The general principle: any single benchmarking tool can have a bug. Two tools with similar internals (e.g., two async-based tools) can have the same bug. But four tools with structurally different internals can't all be wrong in the same direction by the same magnitude. That would require a coincidence, not a bug.

## Why this is worth writing up

A systematic error in a benchmarking number, hidden inside a load tester nobody is auditing, is the kind of thing that gets used for months without being caught. Capacity-planning meetings cite the number, fleet sizes get fixed against it, dashboards visualize it, downstream teams trust it. Every consumer of that number inherits the error.

The cost shows up in two places:

1. **Direct GPU spend.** Each unnecessary g6e.2xlarge instance is ~$1,600/month at on-demand pricing. The over-provisioning multiplier depends on how badly the load tester underreported, but even modest underreporting on a 50-100 instance fleet adds up to tens of thousands of dollars per month.
2. **Opportunity cost in a tight market.** Inference GPUs (H100s, L40Ses, A100s) have been supply-constrained for the past two years. Capacity sitting idle in one team's over-provisioned fleet is capacity another team isn't getting. The dollar number understates the real impact when GPUs are the gating constraint, not money.

The bug took some digging to find, not because it was complex but because nobody had a reason to question the load tester. Most benchmark bugs don't crash; they lie quietly. The lesson worth carrying forward is to treat benchmarking infrastructure as load-bearing code: when its output drives expensive decisions, it deserves the same scrutiny as production code, including cross-validation by independent methods before its numbers get trusted.

## Reproducing

```bash
git clone <this repo>
cd case-studies/03-asyncio-sleep-drift/
python3 repro.py
```

The script is self-contained: standard library only, no GPU, no network, no external dependencies. It runs in a few seconds on any machine. The pattern reproduces deterministically (with small per-run variance from system scheduling) across macOS and Linux on the laptops I've tested.

For the full repro, see [`repro.py`](repro.py) in this directory.
