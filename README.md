# LLM Inference Notes

Notes from working on production LLM serving — debug investigations and benchmarks across the Python, C++, and CUDA layers of TRT-LLM, Triton, and vLLM.

## Case studies

| # | Title | Stack | Outcome |
|---|---|---|---|
| 01 | [TRT-LLM beam search params silently ignored: a `std::optional` ABI trap](case-studies/01-trtllm-beam-search-abi/README.md) | TensorRT-LLM, Triton C++ backend, FP8 | Diagnosis drove 2 NVIDIA upstream fixes: [#13633](https://github.com/NVIDIA/TensorRT-LLM/pull/13633), [#13692](https://github.com/NVIDIA/TensorRT-LLM/pull/13692) |
| 02 | [Why FP8 KV cache silently breaks vLLM on L40S: a backend-fallback story](case-studies/02-vllm-fp8-mixed-precision-l40s/README.md) | vLLM, FlashAttention, FP8 quantization | Source-traced finding: FP8 KV cache on SM < 9.0 collapses GSM8K accuracy 61%→2% and halves throughput. Practical recipe: weight FP8 + FP16 KV. |
