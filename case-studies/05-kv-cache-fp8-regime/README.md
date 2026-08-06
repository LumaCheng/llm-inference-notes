# When KV-cache FP8 costs more than it saves: a roofline argument against a default

KV-cache FP8 is usually pitched as a free throughput win: halve the KV bytes, read half as much from HBM, go faster. On a small model serving short sequences, I measured the opposite. Switching the KV cache from FP16 to FP8 (weight precision held fixed) lowered throughput, raised tail latency, *and* cost ranking quality at the same time. Not a trade-off — worse on every axis I cared about. The roofline model explains why, and points to the regime where the tradeoff reverses.

This is a mechanism note, not a benchmark dump. The measurements behind it were taken on a production workload whose absolute numbers I can't publish, so this reports directions and the reasoning, and gives a public-model recipe so you can reproduce the effect in your own regime.

## TL;DR

**Claim** — KV-cache FP8 is a memory-bandwidth optimization. Its benefit scales with how many KV bytes you read per decode step. On a small model with short input/output, the KV cache is already tiny, so the benefit shrinks toward zero while the costs (dequant on the hot path, a less-mature kernel path, precision loss on stored K/V) stay fixed. In that regime it is strictly worse.

**Evidence** — holding weight precision constant and flipping only the KV-cache dtype from FP16 to FP8, I observed **lower max throughput, higher p99 latency, and lower retrieval quality (recall and MRR both down)**. The direction held across every weight precision I paired it against; the strongest configurations quantized weights but kept the KV cache in FP16.

**Boundary** — the roofline predicts a crossover: at long context and large batch the KV cache grows until the bandwidth saving outweighs the overhead. I did not measure that crossover, so I state it as a prediction from the model, not a result.

> KV cache size scales with `layers × kv_heads × head_dim × sequence_length × batch`. "Memory-bandwidth bound" means the decode step spends most of its time moving bytes from HBM, not doing math — the regime where reading fewer KV bytes actually buys wall-clock time.

## The intuition, and where it's anchored

The reasoning behind "quantize the KV cache" is sound in the regime it was formed in: at long context and high batch, the KV cache is the largest thing decode touches, attention is squarely memory-bound, and cutting the KV footprint in half is close to cutting decode time in half. The advice is correct there.

The failure is applying it as a blanket default. The benefit is not a property of FP8; it's a property of *how big your KV cache is*. Change the regime and the same knob does nothing useful.

## Why it inverts for a small model on short sequences

Two forces work against KV FP8 when the KV cache is small.

**The benefit shrinks with the cache.** KV bytes per token scale with the model's layer/head geometry and the sequence length. A small model on short input/output has few layers, few KV heads, and a short sequence, so the absolute bytes saved by halving the cache are small. You're optimizing a term that isn't the bottleneck; by the roofline, if the decode step wasn't bandwidth-bound to begin with, spending fewer bytes on KV reads doesn't move wall-clock time.

**The costs don't shrink.** They're regime-independent:

1. **Dequant on the hot path.** FP8 K/V has to be converted back to a higher precision before the attention matmul. That conversion runs every decode step regardless of how small the cache is.
2. **A less-mature kernel path.** FP8 KV attention kernels are newer and more hardware-specialized than the well-worn FP16 path. Selecting the FP8 KV dtype can route you onto a slower or fallback kernel, which is exactly the backend-swap failure mode I traced in [case study 02](../02-vllm-fp8-mixed-precision-l40s/README.md).
3. **Precision loss on stored K/V.** Every key and value is now stored with fewer bits. During autoregressive decoding those imprecise values are read back many times and propagate into the output distribution, which is where the ranking-quality drop comes from.

Near-zero benefit plus fixed overhead plus a quality hit sums to a strict loss. There's nothing to trade.

## What I observed (directions only)

On a small-model, short-sequence serving workload, changing only the KV-cache dtype from FP16 to FP8 while holding weight precision fixed:

| Metric | KV FP16 → FP8 | Effect |
|---|---|---|
| Max throughput | ↓ lower | worse |
| p99 latency | ↑ higher | worse |
| Recall / MRR | ↓ lower | worse |

All three axes move the wrong way — KV FP8 is strictly worse in this regime, not a tradeoff.

The pattern was consistent, not incidental: across multiple weight precisions, every FP16-KV configuration beat its FP8-KV counterpart on quality and tail latency. The configurations I'd actually ship kept the KV cache in FP16 and, if anything, quantized the weights instead — weight quantization shrinks a footprint that *is* on the critical path for this model without touching the values that get re-read during decode.

> Absolute metrics are omitted on purpose. This note reports directions and the mechanism behind them, not numbers from any specific deployment. The recipe below reproduces the same directional effect on a public model.

## Why weight quantization behaves differently from KV quantization

It's worth separating the two, because "quantize to FP8" gets used as if it were one decision. Weight FP8 and KV FP8 optimize different bytes:

- **Weight FP8** shrinks the parameter footprint that's streamed once per forward pass. Whether it helps depends on whether weight loading is on your critical path, but it doesn't touch the KV cache and doesn't degrade values that get re-read across every decode step.
- **KV FP8** shrinks the cache that's read once per token per layer during decode. Its payoff is entirely a function of cache size.

So "should I quantize?" has two independent answers on the same model. On the small-model regime here, weight quantization was defensible and KV quantization was not. Collapsing them into a single toggle is how the wrong default gets shipped.

## Where KV FP8 does pay off (a prediction, not a result)

The roofline says there's a crossover. As sequence length and batch size grow, KV bytes per step grow with them, decode moves firmly into the bandwidth-bound region, and the halved KV footprint starts buying real wall-clock time — enough to outweigh the fixed dequant overhead, and enough that the quality cost may be an acceptable price for the throughput. Long-context, high-batch serving is the regime KV FP8 was designed for.

I have not measured that crossover in this workload, so this is a hypothesis from the model, not a finding. It's also the single most useful thing to measure before enabling KV FP8 anywhere: find the context length and batch size where the lines cross for *your* model, and set the default from that, not from folklore.

## Reproducing the mechanism — and the one trap that invalidates it

You don't need my workload to see the effect; you need a small model and short sequences, holding weight precision fixed and varying only `kv_cache_dtype`. But there's a hardware trap that will confound the measurement.

**Do not run this on vLLM below SM 9.0 (e.g. L40S, A100).** On those GPUs, flipping `kv_cache_dtype=fp8` doesn't just change the KV precision — it silently swaps the whole backend (V1→V0 engine, FlashAttention→XFormers) and runs FP8 attention with uncalibrated scales. What you'd measure there is the *fallback*, not the roofline effect this note is about. The two are separate phenomena that happen to share a config flag, and mixing them up is exactly the mistake to avoid.

To isolate the roofline effect, you need a stack where flipping the KV dtype leaves the attention backend untouched:

- **vLLM on H100 (SM 9.0)** — FA3 + FP8 KV is natively supported, so switching the dtype doesn't force a backend swap. This is the clean vLLM path.
- **TRT-LLM on any supported SM** — it ships per-SM FP8 attention kernels (separate paths for SM 8.0 / 8.6 / 8.9 / 9.0), so there's no fallback to confound the result. This is the stack the production measurements behind this note were taken on, calibrated, which is why they were clean on Ada-class hardware.

Clean vLLM recipe (H100), comparing on a generation-quality task — not perplexity; see [case study 02](../02-vllm-fp8-mixed-precision-l40s/README.md) for why PPL misses KV-cache regressions — plus a throughput bench:

```bash
# Requires SM 9.0 (H100/H200). On SM < 9.0 this measures the case-study-02 fallback, not this effect.
pip install "vllm==0.10.2" "transformers<5.0" lm-eval

# Quality: KV FP16 vs KV FP8, weights held at FP8, small model
lm-eval --model vllm \
  --model_args "pretrained=Qwen/Qwen2.5-1.5B,dtype=float16,quantization=fp8,kv_cache_dtype=auto,gpu_memory_utilization=0.7" \
  --tasks gsm8k --num_fewshot 5 --batch_size 4

lm-eval --model vllm \
  --model_args "pretrained=Qwen/Qwen2.5-1.5B,dtype=float16,quantization=fp8,kv_cache_dtype=fp8,gpu_memory_utilization=0.7" \
  --tasks gsm8k --num_fewshot 5 --batch_size 4

# Throughput: same two configs, short input/output to stay out of the bandwidth-bound regime
vllm bench throughput --model Qwen/Qwen2.5-1.5B --dtype float16 \
  --quantization fp8 --kv-cache-dtype auto \
  --num-prompts 200 --input-len 128 --output-len 64 --gpu-memory-utilization 0.7

vllm bench throughput --model Qwen/Qwen2.5-1.5B --dtype float16 \
  --quantization fp8 --kv-cache-dtype fp8 \
  --num-prompts 200 --input-len 128 --output-len 64 --gpu-memory-utilization 0.7
```

To watch the crossover appear, sweep `--input-len` / `--output-len` upward and re-run the throughput bench: the FP8-KV config should close the gap and eventually overtake FP16-KV as the KV cache grows into the bandwidth-bound regime. That sweep is the measurement this note deliberately leaves as future work.

## Takeaway

"Can I quantize this?" and "should I quantize this *here*?" are different questions with different answers on the same model. KV-cache quantization is a bandwidth optimization; if attention isn't bandwidth-bound in your regime, it buys nothing and can cost throughput, tail latency, and quality at once. Decide it per workload from the roofline, and measure your own crossover point before making it a default.

## Related

- [Case study 02 — FP8 KV cache backend fallback on L40S](../02-vllm-fp8-mixed-precision-l40s/README.md): the "less-mature kernel path" cost made concrete, plus why generation-quality evals catch KV regressions that perplexity misses.
