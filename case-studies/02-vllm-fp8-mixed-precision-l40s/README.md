# Why FP8 KV cache silently breaks vLLM on L40S: a backend-fallback story

Enabling FP8 KV cache in vLLM 0.10.2 on an L40S (SM 8.9) GPU drops generation accuracy on Qwen2.5-1.5B GSM8K from **61% to 2%** and halves output throughput. The same model with weight-only FP8 (KV stays in FP16) is essentially indistinguishable from the FP16 baseline on both axes. The user-facing config doesn't surface any of this. Three things change silently when you flip `kv_cache_dtype=fp8`: the V1 engine drops to V0, FlashAttention drops to XFormers, and FP8 attention runs with uncalibrated scales. Tracing through vLLM's source explains why.

## TL;DR

**Recipe** — on AWS g6e (L40S, SM 8.9) or p4d/p4de (A100, SM 8.0) running vLLM 0.10.2: quantize weights to FP8 but keep KV cache in FP16. Full FP8 collapses GSM8K accuracy from 61% to 2% and halves throughput. The quality cost is severe enough to be a non-starter for production.

**Why** — flipping `kv_cache_dtype=fp8` triggers three silent changes on SM < 9.0 GPUs: the V1 engine drops to V0, FlashAttention drops to XFormers, and FP8 attention runs with uncalibrated `q_scale=1.0` / `prob_scale=1.0`. None are visible in the user config; they only show up in the engine startup log and in vLLM's source.

> SM = NVIDIA compute capability. SM 9.0 = Hopper (H100/H200); SM 8.9 = Ada Lovelace (L40S, RTX 4090); SM 8.0 = Ampere (A100). L40S has FP8 tensor cores. The constraint is vLLM's attention-backend choice, not hardware (see "Why this isn't a hardware ceiling").

## Setup

| Component | Value |
|---|---|
| Hardware | AWS g6e.xlarge (1× L40S 48GB, **SM 8.9**) |
| Inference engine | vLLM 0.10.2 |
| Eval framework | lm-evaluation-harness 0.4.12 |
| Model | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |
| Quality eval | GSM8K (5-shot exact match accuracy) |
| Throughput eval | `vllm bench throughput`, 200 prompts × 256 input / 256 output tokens |
| Quantization | vLLM dynamic FP8 (per-tensor abs-max, on-the-fly, no calibration data) |

Three configs:

| Config | Weight | KV cache |
|---|---|---|
| FP16 baseline | FP16 | FP16 |
| Mixed | FP8 | FP16 |
| Full FP8 | FP8 | FP8 |

## Numbers

**Quality.** GSM8K accuracy (5-shot, exact match, higher better). GSM8K requires multi-step chain-of-thought generation, so unlike single-forward-pass perplexity, it actually exercises the KV cache read/write loop that's affected by KV quantization.

| Config | flexible-extract | strict-match | vs FP16 |
|---|---|---|---|
| FP16 | 0.6096 | 0.6050 | — |
| Mixed (W FP8 + KV FP16) | 0.6042 | 0.5997 | **−0.9% (lossless)** |
| Full FP8 (W FP8 + KV FP8) | **0.0205** | **0.0114** | **−96.6%** |

**Throughput.** Output tokens/sec on `vllm bench throughput`, 200 prompts × 256 input / 256 output tokens (higher better).

| Config | output tok/s | vs FP16 |
|---|---|---|
| FP16 | 10,562 | — |
| Mixed | 11,150 | +5.6% |
| Full FP8 | 5,092 | **−51.8%** |

Two stories in these numbers:

1. **Mixed precision is a strict win.** Lossless quality, +5.6% throughput. Nothing to think about. If you're already running vLLM on this hardware and considering FP8, this is the recipe.
2. **Full FP8 is broken on this hardware/stack combination.** Not "slow." *Broken*. A 60-percentage-point GSM8K accuracy drop is catastrophic; the model effectively can't do multi-step reasoning anymore. The 50% throughput cost is the secondary concern. This is a "don't use this at all" config until the underlying issues are addressed.

The accuracy collapse is the kind of result that doesn't show up in routine benchmarks. PPL (the more common quantization metric) on the same three configs is essentially flat between Mixed and Full FP8. See the methodology note below.

## Where the regression comes from

Re-launching the full-FP8 config with logging on:

```bash
python -c "from vllm import LLM; LLM(
    model='Qwen/Qwen2.5-1.5B', dtype='float16',
    quantization='fp8', kv_cache_dtype='fp8',
    gpu_memory_utilization=0.5, max_model_len=2048,
    enable_prefix_caching=False)" 2>&1 | grep -iE "engine|backend|attention|warning"
```

Surfaces three lines that the mixed config doesn't print:

```
INFO ... [llm_engine.py:221] Initializing a V0 LLM engine (v0.10.2) ...
INFO ... [cuda.py:441]       Cannot use FlashAttention backend for FP8 KV cache.
INFO ... [cuda.py:453]       Using XFormers backend.
WARNING ... [kv_cache.py:130] Using uncalibrated q_scale 1.0 and/or prob_scale 1.0
                              with fp8 attention. This may cause accuracy issues.
```

So `kv_cache_dtype=fp8` does three things at once:

1. Engine downgrade V1 → V0
2. Attention backend FlashAttention → XFormers
3. Uncalibrated FP8 attention scales (q_scale=1.0, prob_scale=1.0)

The user-facing config doesn't say any of this; you only see it in the startup log. And critically: vLLM's `WARNING` about accuracy is mild, but the actual measured accuracy impact is severe (96% relative drop on GSM8K).

## Why the fallback happens

Tracing the chain through vLLM's source.

**V1 engine support gate.** `vllm/engine/arg_utils.py::_is_v1_supported_oracle`:

```python
if self.kv_cache_dtype != "auto":
    supported = current_platform.is_kv_cache_dtype_supported(
        self.kv_cache_dtype, model_config)
    if not supported:
        _raise_or_fallback(feature_name="--kv-cache-dtype",
                           recommend_to_remove=False)
        return False
```

V1 only handles non-`auto` KV dtypes if the platform layer signs off.

**Platform check.** `vllm/platforms/cuda.py::is_kv_cache_dtype_supported` routes to FlashAttention's own predicate:

```python
# vllm/attention/utils/fa_utils.py
def flash_attn_supports_fp8() -> bool:
    return get_flash_attn_version() == 3 and \
        current_platform.get_device_capability().major == 9
```

FA3 + FP8 KV cache requires compute capability major == 9 (Hopper).

L40S is SM **8.9**: major 8, minor 9. The major check fails. FlashAttention is out for this combination, and vLLM cascades down: V1 → V0 → XFormers attention backend → uncalibrated FP8 attention path.

The 50% throughput drop comes from leaving the optimized FlashAttention path. The 96% accuracy drop is harder to attribute precisely. It's the combined effect of the XFormers backend running FP8 attention with default scales, the KV dtype change itself, and however those interact during multi-step generation. The vLLM warning ("uncalibrated `q_scale=1.0` / `prob_scale=1.0`... may cause accuracy issues") is the most direct flag, but the size of the actual measured impact (catastrophic, not "may cause") suggests the combined effect is much worse than any single source would predict.

## Why this isn't a hardware ceiling

The SM 9.0 requirement is specific to vLLM 0.10.2's FlashAttention v3 path, not a hardware capability gap. L40S has FP8 tensor cores and is fully capable of FP8 KV-cache attention. FA3 just happens to be written against Hopper-only instructions (wgmma, TMA) and didn't ship a Lovelace fallback in this version.

NVIDIA's TensorRT-LLM ships its own FP8 attention kernels with per-SM specializations (separate code paths for SM 8.0, 8.6, 8.9, 9.0), so on the same L40S a TRT-LLM-based deployment can run FP8 KV cache without dropping out of an optimized path. Same hardware, same precision, different attention backend, different outcome.

So the precise framing of this finding is: **vLLM's choice of attention backend matters as much as the GPU's quantization capability for the FP8 KV path**. A future vLLM version that ships an SM 8.9-aware FP8 attention kernel (via Triton or FlashInfer, for instance) would change this picture without any hardware change.

## Methodology note: why GSM8K and not perplexity

Perplexity on wikitext is the default quantization-quality metric in most papers and tutorials, so it's the obvious first thing to reach for. But on these three configs the two metrics tell very different stories:

| Config | wikitext word_PPL | GSM8K accuracy |
|---|---|---|
| FP16 | 12.10 | 0.6096 |
| Mixed | 12.22 (+1.0%) | 0.6042 (−0.9%) |
| Full FP8 | 15.99 (+32.2%) | 0.0205 (**−96.6%**) |

Wikitext PPL puts the full-FP8 regression at +32%. Sounds bad but survivable. GSM8K shows the model is catastrophically broken. Two-orders-of-magnitude difference in apparent severity, on the same configuration.

The reason is structural: perplexity is computed from a single forward pass on the prompt. The K and V tensors are calculated and consumed within that pass; they aren't persisted to and re-read from the KV cache the way they are during autoregressive generation. So perplexity barely exercises the code path that KV-cache quantization actually changes. (Cross-checked on Qwen2.5-7B: Mixed and Full FP8 PPL came out 9.7854 and 9.7858, indistinguishable to four decimal places, despite the configs differing in exactly the variable PPL is supposed to measure. 7B GSM8K and throughput weren't run in this study, just the PPL cross-check.)

For evaluating KV-cache-related quantization changes, the metrics that actually measure the affected code path are the ones that exercise generation: throughput benchmarks (KV read cost directly) and generation-quality tasks like GSM8K, HumanEval, or open-ended generation evals (the propagated quality cost of imprecise KV reads). PPL on a static prompt set is the wrong tool for this specific question.

## Is +5.6% throughput worth bothering with?

On 1.5B, fair question. The model is small enough that L40S has plenty of memory either way, so halving the weight footprint with FP8 doesn't buy much.

But the same recipe on bigger models is a different story. A 7B FP16 model uses ~14 GB of weight; FP8 cuts that to ~7 GB. Those 7 GB don't disappear; vLLM reallocates them to KV cache, which means larger batches or longer contexts at the same memory budget. On 14B+ models, FP8 weights can be the difference between fitting on a single L40S and not fitting at all. I didn't measure these directly here, so I can't put numbers on the speedup, but the memory math is concrete.

What I can say with confidence is the other half: **Full FP8 is unsafe to use on this hardware/stack regardless of model size**. The accuracy collapse comes from the backend fallback chain, not from anything specific to a 1.5B model. So even if Mixed only buys a small speedup at 1.5B, ruling out Full FP8 is what makes Mixed the only viable FP8 path on g6e/p4d/p4de with vLLM.

The natural follow-up, comparing KV FP16 vs KV FP8 with everything else held constant, would need H100 (where FA3 + FP8 KV is supported and switching the KV dtype doesn't force a backend swap) and a pre-calibrated FP8 checkpoint (which avoids the uncalibrated `q_scale=1.0` issue separately). That's outside this study's scope.

## Reproducing

```bash
# Setup (CUDA 12.x driver assumed)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "vllm==0.10.2" "transformers<5.0" lm-eval

# 1.5B GSM8K, three configs
lm-eval --model vllm \
  --model_args "pretrained=Qwen/Qwen2.5-1.5B,dtype=float16,gpu_memory_utilization=0.7" \
  --tasks gsm8k --batch_size 4 --num_fewshot 5

lm-eval --model vllm \
  --model_args "pretrained=Qwen/Qwen2.5-1.5B,dtype=float16,quantization=fp8,kv_cache_dtype=auto,gpu_memory_utilization=0.7" \
  --tasks gsm8k --batch_size 4 --num_fewshot 5

lm-eval --model vllm \
  --model_args "pretrained=Qwen/Qwen2.5-1.5B,dtype=float16,quantization=fp8,kv_cache_dtype=fp8,gpu_memory_utilization=0.7" \
  --tasks gsm8k --batch_size 4 --num_fewshot 5

# 1.5B throughput, three configs
vllm bench throughput --model Qwen/Qwen2.5-1.5B --dtype float16 \
  --num-prompts 200 --input-len 256 --output-len 256 --gpu-memory-utilization 0.7

vllm bench throughput --model Qwen/Qwen2.5-1.5B --dtype float16 \
  --quantization fp8 --kv-cache-dtype auto \
  --num-prompts 200 --input-len 256 --output-len 256 --gpu-memory-utilization 0.7

vllm bench throughput --model Qwen/Qwen2.5-1.5B --dtype float16 \
  --quantization fp8 --kv-cache-dtype fp8 \
  --num-prompts 200 --input-len 256 --output-len 256 --gpu-memory-utilization 0.7

# Surface the fallback in the startup log
python -c "from vllm import LLM; LLM(
  model='Qwen/Qwen2.5-1.5B', dtype='float16',
  quantization='fp8', kv_cache_dtype='fp8',
  gpu_memory_utilization=0.5, max_model_len=2048,
  enable_prefix_caching=False)" 2>&1 | grep -iE "engine|backend|attention|warning"
```

## Source references

- `_is_v1_supported_oracle`, `_raise_or_fallback`: [`vllm/engine/arg_utils.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/engine/arg_utils.py)
- `is_kv_cache_dtype_supported`: [`vllm/platforms/cuda.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/platforms/cuda.py)
- `flash_attn_supports_fp8`: [`vllm/attention/utils/fa_utils.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/attention/utils/fa_utils.py)
- vLLM V1 engine announcement: [blog.vllm.ai/2025/01/27/v1-alpha-release.html](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
