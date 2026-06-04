# Why TRT-LLM is the only LLM serving framework with online beam search

When a TRT-LLM beam search bug forced a decision between fixing it and migrating to a different framework, I checked whether any other LLM serving framework would solve the problem. The answer is that none of them do. vLLM, TGI, and SGLang have either removed beam search from their online serving paths or never supported it. As of 2026, TRT-LLM is the only mainstream framework that supports beam search in online serving mode, which means anyone whose use case actually needs beam search has no alternative.

This was a survey, not a fix. But the conclusion mattered: it ruled out one of the two viable migration paths and forced commitment to the other.

## TL;DR

**Context** — production deployment was hitting bugs in TRT-LLM's C++ beam search path (separately documented in [case study 01](../01-trtllm-beam-search-abi/README.md)). Two options for moving forward: fix the C++ bugs upstream, or migrate the deployment to a different serving framework. This case study covers the framework-migration option.

**Finding** — none of the obvious vLLM-based migration paths work, and no other mainstream framework supports online beam search either:

| Framework | Online beam search support |
|---|---|
| vLLM (v0.11.x, current) | Removed in v0.6.0 ([PR #8763](https://github.com/vllm-project/vllm/pull/8763), 2024-09) |
| vLLM (v0.5.x, pre-removal) | Has beam search, but doesn't load Qwen2.5 (PyTorch / `rope_scaling` incompatibility) |
| vLLM standalone API | `LLM.beam_search()` exists in offline batch mode only ([RFC #8306](https://github.com/vllm-project/vllm/issues/8306)) |
| HuggingFace TGI | No beam search; `best_of` is sampling-based, not beam expansion. Repo is now archived. |
| SGLang | No beam search; `SamplingParams` has no beam-related fields |
| **TensorRT-LLM** | **Supported (`beam_width` in Executor API)** |

**Outcome** — migration ruled out, and the C++ bug-fix path became the only option.

## Why this question came up

The deployment in question used beam search at moderate width to generate a small set of diverse but precise candidates per request. Beam search is the right algorithm for this class of workload because it gives you both axes at once: candidates are decoded under a quality-ordered heuristic (precision) but the beam-width parameter directly controls how many alternative continuations come out (diversity). Sampling-based decoding with `n>1` doesn't substitute for it. Either you sample at low temperature and get near-duplicate outputs, or you raise the temperature and lose precision. The tradeoff is a knob you can't avoid with sampling, and beam search bypasses it.

So when the production stack was hitting bugs in TRT-LLM's C++ beam search code path, "use a different decoding strategy" wasn't on the table. The question was strictly: is there another *framework* that runs beam search in serving mode?

I tried two vLLM paths.

## Option 1: Triton + vLLM backend

This was the obvious migration target. Keeps the Triton frontend (so existing API contracts and health checks stay the same) but swaps the engine underneath from TRT-LLM to vLLM.

### Attempt with current vLLM (v0.11.x via NGC 25.12 image)

Using a vLLM-compatible Triton image (NVIDIA's official `nvcr.io/nvidia/tritonserver:25.12-vllm-python-py3`):

```bash
docker run --rm -it --gpus all --net host --shm-size=8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/model_repo:/model_repo \
  nvcr.io/nvidia/tritonserver:25.12-vllm-python-py3 \
  tritonserver --model-repository=/model_repo
```

Greedy decoding works:

```bash
curl -X POST http://localhost:8000/v2/models/vllm_model/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text_input": "...",
    "stream": false,
    "sampling_parameters": "{\"max_tokens\": 20, \"temperature\": 0}"
  }'
# returns valid greedy output
```

Beam search fails:

```bash
curl -X POST http://localhost:8000/v2/models/vllm_model/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text_input": "...",
    "sampling_parameters": "{\"use_beam_search\": true, \"best_of\": 10, \"n\": 10, \"temperature\": 0}"
  }'
# Error: Unexpected keyword argument 'use_beam_search'
```

The error isn't subtle. The Triton vLLM backend's `request.py` doesn't contain any beam-search-related code at all. This isn't a bug; it's a feature that was never wired through, because it's not present in the underlying vLLM version anymore. vLLM removed beam search from `SamplingParams` in v0.6.0 ([PR #8763](https://github.com/vllm-project/vllm/pull/8763)), which means anything built on top of v0.6+ inherits the absence.

I tried sampling alternatives (`n=10, temperature=0.3` through `0.7`) to see if it could substitute. It can't. At `temperature=0.3` all 10 outputs were near-identical. At `temperature=0.7` only 3 unique outputs in 10 (the rest were duplicates). Sampling has a precision-diversity trade-off; beam search doesn't.

### Attempt with older vLLM (v0.5.x via NGC 24.08 image)

Before v0.6, vLLM still had beam search. The Triton vLLM backend at v24.08 used a vLLM version (~0.5.3) that pre-dates the removal:

```bash
docker pull nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3
docker run --rm -it --gpus all -v /path/to/model_repo:/model_repo \
  nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 bash
```

But this version of vLLM can't load Qwen2.5. Two layered incompatibilities:

1. The bundled `transformers` is too old to support Qwen2.5's tokenizer config. Upgrading `transformers` triggers `Disabling PyTorch because PyTorch >= 2.4 is required`, since vLLM 0.5.3 ships with PyTorch 2.3.1.
2. Even after working around (1), Qwen2.5's `rope_scaling` field uses the new `yarn` format that vLLM 0.5.3's config parser doesn't recognize: `assert "factor" in rope_scaling` fails.

This isn't specific to Qwen2.5. Most modern models (Llama-3.x, Qwen2/2.5/3, Gemma 2) use the same updated `rope_scaling` format and the post-2024 PyTorch features that vLLM 0.5.x doesn't support. So the older image has beam search but loads almost no current models; the current image loads them but no longer has beam search. There's no version of the Triton vLLM backend where both work.

## Option 2: vLLM standalone (no Triton)

The fallback was running vLLM directly via its OpenAI-compatible API server, removing Triton from the picture. This is a step backward in terms of API compatibility, but worth checking.

It also doesn't work. vLLM's online API server only exposes sampling-based decoding. The `LLM.beam_search()` method does exist in vLLM, but it's offline-only: it lives in the batch-inference Python class, not the serving stack.

This isn't an oversight. It's an explicit design decision documented in [RFC #8306](https://github.com/vllm-project/vllm/issues/8306): the vLLM team scoped beam search as an offline batch inference feature and chose not to expose it through the serving path. Even if I went back through git history and reconstructed an old version of the API server, future updates would be against the current scope, and the gap would only widen.

## Closing the migration question

The whole evaluation took about a day: two vLLM image attempts, reading PR #8763 and RFC #8306, and confirming the cross-framework picture against source code. The result was simple: there is no viable migration target.

Before doing this, the team's plan B was "if the C++ bugs in TRT-LLM are bad enough, we'll switch to vLLM." That assumption is the kind of thing that goes unchallenged in planning meetings until someone tries it. After this evaluation, plan B is gone, and the C++ bug-fix path (case study 01) is the only option. Knowing that upfront saved weeks of fallback work that would have ended in the same place.

This isn't necessarily permanent. RFC #8306's design decision was debated in its own thread, and any of the OSS frameworks could add beam search back in 2026. But for any deployment decision being made today, TRT-LLM is the only choice for online beam search.

## References

- vLLM beam search removal: [PR #8763](https://github.com/vllm-project/vllm/pull/8763) (2024-09-24)
- vLLM beam search scoping decision: [RFC #8306](https://github.com/vllm-project/vllm/issues/8306)
- Triton vLLM backend releases: [GitHub releases](https://github.com/triton-inference-server/vllm_backend/releases)
- TRT-LLM Executor API (`beam_width` parameter): [tensorrt_llm/executor/executor.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/executor/executor.py)
- Related case study: [TRT-LLM beam search params silently ignored](../01-trtllm-beam-search-abi/README.md)
