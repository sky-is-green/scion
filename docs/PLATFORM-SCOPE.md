# Platform scope and report triage

Which runtime paths this project can verify locally, and what a report needs to
carry to get a fix. Measurements below are from the reference box (2x RX 7900 XT
gfx1100, ROCm 7.2.4, Ryzen 7 7800X3D) unless stated otherwise.

## Scope

| path | status |
|---|---|
| CPU (all platforms) | verified: PQ2_0 kernels, reference quality runs |
| Linux x64 + AMD ROCm (gfx1100/gfx11) | **primary target**: every release number was measured here |
| Linux x64 + NVIDIA CUDA | prebuilt binaries published; runtime-unverified |
| Windows x64 (CPU, CUDA) | prebuilt binaries published; runtime-unverified |
| macOS Metal | kernels present; untested |
| Vulkan | unsupported: no PQ2_0 kernels |

AMD/Linux and CPU are the paths we can measure and fix locally. CUDA, Windows
and Metal are community paths that rely on volunteers who can run iterations on
the affected hardware.

## Triage of the 2026-10-06 Windows CUDA report

A report from a Windows x64 / CUDA 13.3 user (build `v0.4.1`) claimed four
issues. Each was re-tested on the primary target with the same fork source
(v0.4.1, `c45a49e21`, HIP build) against a mainline baseline (upstream
llama.cpp at `dc178a7cf`, 2026-09-28, HIP build). **None of the four
reproduced.**

### 1. MTP fails to load

ROCm: the drafter loads and drafts. Server log:
`speculative decoding enabled: draft-mtp-sidecar` followed by per-slot
`draft acceptance = ...` on generation. The Windows CUDA zip does contain the
sidecar code (checked `llama-common.dll` for the `draft-mtp-sidecar` symbols),
so the report is a load-time failure in the CUDA allocation/scheduler path of
`llama_mtp_sidecar_load`, not a missing feature.

### 2. Decode 13-15% slower than mainline, benchmarks 2-3x faster

`llama-bench`, 5 repetitions, same box:

| model / build | pp512 t/s | tg128 t/s |
|---|---:|---:|
| fork + IQ2_M | 2009 +/- 164 | 97.68 +/- 1.07 |
| mainline + IQ2_M | 2020 +/- 85 | 95.23 +/- 0.33 |
| fork + Scion PQ2_0 | 2471 +/- 224 | 98.68 +/- 0.43 |
| fork + Scion, tg at depth 4096 | - | 92.69 +/- 5.35 |
| mainline + IQ2_M, tg at depth 4096 | - | 92.87 +/- 0.72 |

The fork is not slower than upstream on this platform, and Scion is the
fastest configuration measured. No 2-3x pp claim shows either (Scion pp is
+22% over the mainline reference). Treat the reported regression as
CUDA-specific until it is reproduced on CUDA.

### 3. q4_0 context quantization poor, q8_0 good

`llama-perplexity`, wikitext-2 raw:

| KV type | Scion c512 (64 chunks) | Scion c4096 (24 chunks) | IQ2_M c4096 (control) |
|---|---:|---:|---:|
| f16 | 7.8724 | 6.9092 | 6.9738 |
| q8_0 | 7.8711 | 6.9121 | - |
| q4_0 | 7.8927 | 6.9294 | 7.0039 |

q4_0 KV costs about 0.3% PPL at 4k context, and the mainline reference quant is
slightly more sensitive to it than Scion. No cliff on ROCm.

### 4. Weak math and instruction following

Not reproduced on the primary target. The pre-fix run scored Scion and the
IQ2_M reference at 17/20 each with an identical category breakdown (factual
5/5, math 5/5, instruction 3/4, code 1/3, logic 3/3); every failure was the
checker scoring the raw `<think>` trace instead of the answer. `strip_reasoning`
now removes closed and truncated reasoning blocks before scoring, with unit
coverage in `moe/tests/test_cap_eval.py`. Post-fix rerun at a 1024-token
budget: **Scion 20/20, IQ2_M 20/20**, identical category breakdown (factual
5/5, math 5/5, instruction 4/4, code 3/3, logic 3/3).

## Reporting a CUDA / Windows / Metal issue

Attach:

1. the exact command line;
2. the full console output; for MTP the decisive line is
   `llama_context::mtp_sidecar_load: failed to load sidecar '<path>': <reason>`;
3. release tag and artifact (CUDA 12.4 vs 13.3, CPU vs CUDA);
4. GPU model and driver version;
5. model and drafter file SHA256;
6. for speed claims, `llama-bench` numbers for both sides of the comparison.

Fixes for these paths are best-effort: the project has no CUDA/Windows hardware,
so a fix requires a volunteer who can test iterations on the affected platform.

## Known packaging note (AMD)

The prebuilt `linux-x64-hip.tar.gz` from `v0.4.1` links ROCm 6.x sonames
(`libhipblas.so.2`, `librocblas.so.4`, `libamdhip64.so.6`) and does not load
models on ROCm 7.x. Use the source build (`-DGGML_HIP=ON`) or rebuild the
artifact against the ROCm you run.
