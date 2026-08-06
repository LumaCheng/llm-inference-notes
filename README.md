# LLM Inference Notes

Notes from working on production LLM serving — debug investigations and benchmarks across the Python, C++, and CUDA layers of TRT-LLM, Triton, and vLLM.

## Case studies

| # | Title | Stack | Outcome |
|---|---|---|---|
| 01 | [TRT-LLM beam search params silently ignored: a `std::optional` ABI trap](case-studies/01-trtllm-beam-search-abi/README.md) | TensorRT-LLM, Triton C++ backend, FP8 | Diagnosis drove 2 NVIDIA upstream fixes: [#13633](https://github.com/NVIDIA/TensorRT-LLM/pull/13633), [#13692](https://github.com/NVIDIA/TensorRT-LLM/pull/13692) |
| 02 | [Why FP8 KV cache silently breaks vLLM on L40S: a backend-fallback story](case-studies/02-vllm-fp8-mixed-precision-l40s/README.md) | vLLM, FlashAttention, FP8 quantization | Source-traced finding: FP8 KV cache on SM < 9.0 collapses GSM8K accuracy 61%→2% and halves throughput. Practical recipe: weight FP8 + FP16 KV. |
| 03 | [How a known Python async pitfall is inflating GPU inference fleets](case-studies/03-asyncio-sleep-drift/README.md) | Python asyncio, load testing, capacity planning | An `asyncio.sleep` pacing pattern that's well-documented in the Python community accumulates drift that under-reports LLM server TPS, leading capacity planning to over-provision GPUs. Five-line fix, includes minimal repro. |
| 04 | [Why TRT-LLM is the only LLM serving framework with online beam search](case-studies/04-vllm-beam-search-feasibility/README.md) | vLLM, TGI, SGLang, TRT-LLM, framework selection | A 2026 cross-framework survey: vLLM removed beam search from online serving in v0.6.0, TGI and SGLang never supported it, leaving TRT-LLM as the only option for use cases that need it. |
| 05 | [When KV-cache FP8 costs more than it saves: a roofline argument against a default](case-studies/05-kv-cache-fp8-regime/README.md) | KV-cache quantization, roofline, FP8, capacity planning | KV FP8's benefit scales with KV-cache size; on a small model with short sequences it lowers throughput, raises p99, and cuts quality all at once. Mechanism + public-model recipe to find your own crossover point. |
