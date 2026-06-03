# TRT-LLM beam search params silently ignored: a `std::optional` ABI trap

Productionizing a model on TRT-LLM behind Triton, the online recall came in roughly 12% below the vLLM-based offline reference. The cause turned out to be an off-by-one in the Triton C++ backend's call to `SamplingConfig` — the call site passed 17 args while the constructor signature had grown to 20. Because every parameter is `std::optional<...>` and the inner types implicitly convert across `std::optional<float>` / `std::optional<int32>`, the misaligned call compiled silently and ran without warnings, with `length_penalty`, `early_stopping`, and the parameters after them all bound to the wrong slots.

NVIDIA shipped two upstream fixes from the diagnosis: [TensorRT-LLM #13633](https://github.com/NVIDIA/TensorRT-LLM/pull/13633), [#13692](https://github.com/NVIDIA/TensorRT-LLM/pull/13692).

## My role

I owned the diagnosis end-to-end: built the bypass-with-Python-backend isolation setup, found the root cause by diffing the open-source `SamplingConfig` constructor against the Triton backend's call site, and patched both issues locally to confirm. I shared the analysis with NVIDIA engineers; they wrote the production-grade fixes (proper plumbing across BLS/ensemble configs, unit tests) and merged them upstream. I verified the official fixes end-to-end and confirmed the recall gap closed to within noise of the offline reference.

## Setup

| Component | Version |
|---|---|
| Hardware | AWS `g6e.xlarge` (1× L40S, 48 GB) |
| Inference engine | TensorRT-LLM v1.2.0rc4 / v1.2.0 GA |
| Serving frontend | Triton with `inflight_batcher_llm` backend |
| Public repro model | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) base + FP8 engine (used to demonstrate the bug to NVIDIA on a publicly-available model) |
| Decoding | Beam search at `beam_width >= 8` (V2 beam search path), with `length_penalty` and `early_stopping` configured |

## The anomaly

The model's offline evaluation ran on vLLM (chosen for its Python-friendly inference loop), and was the recall reference the team had been measuring against. The production serving path used Triton with the TRT-LLM C++ backend (chosen for online throughput). When we brought up the production path on the same model weights and equivalent sampling parameters, recall came in about 12% below the offline reference. Nothing on the config side explained the gap.

Beam search was returning ~10 nearly-identical candidates. Changing `length_penalty` or `early_stopping` had no observable effect on the output. The parameters looked like they were being ignored.

## Isolating the failure

The recall gap could come from a few places:

1. The TRT-LLM engine itself produced lower-quality beams than vLLM
2. Triton's request translation layer was mangling the sampling params
3. The shipped C++ backend (`libtriton_tensorrtllm.so`) was passing wrong values to the engine

To narrow it down I wrote a parallel Python backend for Triton that called the TRT-LLM `Executor` directly via Python bindings, bypassing the shipped `libtriton_tensorrtllm.so`. Same Triton frontend, same engine, same model — but a Python backend instead of the C++ one.

At `length_penalty=2.0`, the C++ backend returned 10 identical short beams. The Python backend with the same engine and same parameters returned 10 diverse longer completions. That ruled out (1) and (2) and pointed at the C++ backend.

## First bug: type mismatch on `early_stopping`

The first parameter I dug into was `early_stopping`. Adding debug logs in the BLS `model.py` confirmed it reached Triton as a boolean tensor — but on the C++ side, the extraction call was looking for an `int32_t`. A wire-protocol type mismatch:

- Triton config (`config.pbtxt`) declared `early_stopping` as `TYPE_BOOL` (1 byte)
- C++ extracted it via `extractOptionalSingleton<int32_t>` (4 bytes)

When the type doesn't match, the extraction silently returns `std::nullopt`. The parameter is just dropped, no error.

The TRT-LLM executor defines three values for `early_stopping`:

- `0`: heuristic stopping
- `1`: stop as soon as `beam_width` finished beams have emitted EOS
- `2`: stop only when all beams have emitted EOS

For 0 and 1, a bool's byte pattern happens to round-trip through `int32_t` extraction without breaking the low byte's meaning. For 2 it doesn't. So the parameter looked like it worked for the common cases but couldn't express the third documented value, and strictly speaking the whole thing was undefined behavior. On top of that, `early_stopping` was missing entirely from five other config files (ensemble, BLS, multimodal/ensemble, gpt/ensemble, gpt/tensorrt_llm) — even with the type fixed, clients routing through those wrappers couldn't set it.

I patched the type locally and confirmed `early_stopping` started having an effect. But `length_penalty` was still being ignored, and the beams were still collapsing into near-duplicates. There was a second bug.

## Second bug: positional-argument shift in `SamplingConfig`

The `early_stopping` fix wasn't the whole story. So I went one layer up — straight to the `SamplingConfig` call site. I cloned TRT-LLM at the matching release tag and lined up:

- The constructor declaration of `executor::SamplingConfig` in `cpp/include/tensorrt_llm/executor/executor.h`
- The call site in `triton_backend/inflight_batcher_llm/src/utils.cc::getSamplingConfigFromTensors()`

The constructor had 20 positional `std::optional<...>` parameters. The call site was passing 17.

The reason this compiled silently: every parameter is `std::optional<...>` of various inner types (`int32`, `float`, `bool`), and `std::optional<float>` will implicitly construct from `std::optional<int32>` because the contained types are convertible. So you can pass arguments whose inner types don't match the declared inner types and the call still resolves.

An earlier upstream PR had inserted a new param, `promptIgnoreLength` (an optional `int32`), at position 14 of the constructor, between `frequencyPenalty` and `lengthPenalty`. The Triton call site was never updated, so every argument from position 14 onward was bound to the wrong slot:

| What `utils.cc` thought it was passing | Slot it landed in |
|---|---|
| `lengthPenalty` (float) | `promptIgnoreLength` (int32, value silently truncated) |
| `earlyStopping` | `lengthPenalty` |
| `noRepeatNgramSize` | `earlyStopping` |
| `numReturnSequences` | `noRepeatNgramSize` |

`numReturnSequences`, `minP`, and `beamWidthArray` defaulted to `std::nullopt`.

That single shift explained the rest. `length_penalty` did nothing because it was never read as `length_penalty`. The beam outputs collapsed because the diversity-controlling parameter never reached the engine. And in retrospect, `early_stopping` was failing for two layered reasons — both the type mismatch and being shifted into the `lengthPenalty` slot. Fixing only the type would not have fully restored its behavior; the argument shift had to go too.

My local patch was a one-line argument-list fix to confirm the diagnosis.

## Reporting upstream

The two bugs were structurally independent — different layers, different failure modes — but both produced the same kind of silent parameter drop. I reported them to NVIDIA with the side-by-side source vs. call-site analysis. The fixes shipped as two separate PRs:

- [PR #13633](https://github.com/NVIDIA/TensorRT-LLM/pull/13633) — the deeper `SamplingConfig` argument shift. NVIDIA's fix plumbs `prompt_ignore_length` end-to-end as a real input: extraction in `utils.cc`, declarations in the ensemble/BLS/main `config.pbtxt` files, BLS Python decode-layer forwarding, and a unit test in `utilsTest.cpp` that pushes `prompt_ignore_length=7` and asserts the resulting `SamplingConfig` reads it back.
- [PR #13692](https://github.com/NVIDIA/TensorRT-LLM/pull/13692) — the `early_stopping` type mismatch and missing config plumbing. Type changed to `TYPE_INT32`, missing input declarations and `input_map` forwarding added across all five config files, plus a `utilsTest.cpp` test for `early_stopping=2`. End-to-end verification on TinyLlama-1.1B with `beam_width=4` confirmed correct behavior across all three Triton entry points (`tensorrt_llm`, `ensemble`, `tensorrt_llm_bls`).

## Verification

Two passes. First, with my local patch, to confirm the diagnosis: rebuilt TRT-LLM from source at `v1.2.0rc4`, replaced `libtriton_tensorrtllm.so` in the Triton container at `/opt/tritonserver/backends/tensorrtllm/`, reran the benchmark.

Building the patched backend wasn't quite a one-shot make. The container's `libtensorrt_llm.so` was built with the CXX11 ABI, so I had to configure cmake with `-DUSE_CXX11_ABI=ON` (the default produces symbols that won't bind against the shipped library). The link stage also failed under `--no-undefined` against the in-container libs, so I dropped that flag and linked manually against the `tensorrt_llm` and CUDA/TensorRT shared objects living in the container.

Once NVIDIA's PRs merged, I pulled the official fixes, rebuilt, redeployed, and ran a parameter sweep over `length_penalty` and `early_stopping`. With the right configuration the online recall closed the gap to within noise of the offline reference — the parameters now had real effect, and tuning them recovered the lost ~12% and slightly more.

Both PRs were authored by NVIDIA after I shared the diagnosis (#13633 merged May 1, 2026; #13692 merged May 2, 2026).

## Notes

A few things I'll remember from this:

The `std::optional<T>` everywhere pattern in this constructor was what made the bug silent. Without that uniformity the type checker would have caught the misalignment immediately. Worth being suspicious of any C++ glue layer that uses a long sequence of optionals positionally.

Building a parallel caller (the Python Triton backend) was the highest-ROI move in the whole investigation. Without it I'd have been reading kernels.

There were actually two independent silent drops in this stack — a config-vs-C++ type mismatch on `early_stopping`, and the deeper `SamplingConfig` argument shift. Either one alone could be mistaken for "the model just behaves that way" if you only checked one parameter. The two failure modes are unrelated mechanically, which is itself a reminder: finding one silent drop in a glue layer doesn't mean the others are clean.
