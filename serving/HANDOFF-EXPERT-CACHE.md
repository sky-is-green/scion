# HAND-OFF — AMD tiered MoE serving / expert cache (session 2026-09-27)

Read this top to bottom before touching anything. Everything below is verified
on this machine unless marked otherwise.

## TL;DR

We built a hot-expert cache for MoE models in the TAARDIS llama.cpp fork
(sidecar + graph split), proved it numerically safe (greedy/PPL identical on a
healthy model), then measured its payoff: **no throughput gain, by design** —
the dense `mul_mat_id` shape keeps the CPU computing k experts per token, so
the split adds GPU work and sync without reducing the CPU tail. See
`placement-sweep-20260927/THROUGHPUT.md`. The effective levers remain the ones
already measured: ternary container, layer-level `-ncmoe` placement,
single-GPU-when-it-fits, threads = physical cores.

## Repos, branches, commits

| What | Where |
|---|---|
| Engine fork | `~/Desktop/work/ternary-serve/taardis-llama.cpp` (remote: `CodeMasterCody3D/prism-ml-llama.cpp`) |
| Branch | **`expert-cache`** — `53388d74a` (base `moe-corr-runtime2`) → `4f38fbd02` (adapter raw sidecar loading) → `2b521be01` (graph split + tool). **Not pushed anywhere.** |
| Active build | `~/Desktop/work/ternary-serve/build-hip` (HIP, ROCm 7.2.4, gfx1100). `build-hip-vmm` deleted. |
| Sidecar tool | `taardis-llama.cpp/tools/make_hot_sidecar.py` (build + `--check`) |
| Server planner | `~/Desktop/work/hivebench/tools/moe_tier/` (**uncommitted**, 12 tests) |
| Artifacts/logs | `~/Desktop/work/ternary-serve/placement-sweep-20260927/` (`SUMMARY.md`, `EQUIVALENCE.md`, `THROUGHPUT.md`, `plans/`, logs) |
| 125B smoke test | `~/Desktop/work/ternary-serve/qwen125-smoke-20260927/` (`RESULTS.md`, logs, `bench2.sh`, `gdb-run.sh`) |
| Status doc | `~/Desktop/work/ternary-serve/SERVING-EXTRAS-PLAN.md` |

## The feature as implemented

- Sidecar GGUF (adapter-style, `adapter.type = taardis-lora`): per layer
  `blk.N.ffn_{gate,up,down,gate_up}_exps.hot` (hot expert blocks, raw bytes,
  **plus `n_expert_used` dummy blocks**), `blk.N.ffn_hot_map` (I32), and
  `ffn_hot_mask` / `ffn_cold_mask` (F32).
- Adapter loader accepts those raw tensors and anchors them to the layer's
  router (`ffn_gate_inp`), so they land on the compute device.
- `build_moe_ffn` gained `moe_split_t`: selection + normalized weights computed
  once, then hot (mapped ids, masked weights) and cold (original ids, complement
  mask) `mul_mat_id` passes summed. Dummy experts give out-of-set slots unique
  ids (the kernel counts one id per expert per token).
- Constraints (asserted): SILU only, no biases, no per-expert scales; only
  **OLMoE** is wired (`models/olmoe.cpp`). `qwen4exp` is NOT wired.
- Graph node budget gets `64 × n_layer` headroom when a sidecar is present
  (`llama-context.cpp`).

## Verified results

Equivalence on corrected model (PPL 16.2): hot64 control PPL-identical,
hot16 within 0.03% and **greedy token-identical** (`EQUIVALENCE.md`).
Payoff sweep: baseline `-ncmoe 16` = 93.3 t/s; hot16 = 89.4; hot32 = 76.9;
hot64 = 89.4 (`THROUGHPUT.md`) — i.e. no gain, explained above.

125B `qwen4exp` smoke test (2026-09-27 evening): single-GPU tiering is
RAM-bound (2–6 t/s decode, 4–28 t/s prefill under page-cache thrash), but the
dual-GPU all-resident config now works after fixing an HSA upload stall with a
64 MiB bounce buffer in `ggml_backend_cuda_buffer_set_tensor`
(`ggml-cuda.cu`): 57 s load, **~25 t/s decode / ~385 t/s prefill**, stable.
Patch is uncommitted; diff at
`qwen125-smoke-20260927/hsa-repro/backend-set-tensor-bounce.patch`. Full
diagnosis + numbers: `qwen125-smoke-20260927/RESULTS.md`.

## Machine facts (for planning)

- GPUs: 2× RX 7900 XT 20 GB (gfx1100), 40.9 GB total VRAM. `ROCm0` = display
  card (`/sys/class/drm/card1`, desktop holds ~1.1–1.3 GiB); `ROCm1` = headless
  (`card0`).
- RAM: 30 GB (Strata's 64 GB guidance does NOT apply — design for single GPU +
  CPU tail).
- CPU: Ryzen 7 7800X3D, 8C/16T → **`-t 8`**; `-t 16` collapses MoE th (95 → 35 t/s).
- ROCm 7.2.4; `-sm row/tensor` unavailable (SYCL-only backend feature).
- Disk: 178 GB free on `/home`.

## 125B model download (ready to test)

`Qwen3.8-Flash-Next GSQ-RCO Q2_0` at
`~/Desktop/work/models/qwen38-q2_0/Q2_0/`:
- shard 1 (weights, must be resident): 36 G
- shard 2 (n-gram lookup, keep on disk): 27 G
- Arch `qwen4exp`; fork supports it + `-lm mmap --lazy-mode on` + PLE n-gram.
- Q2_0 chosen over IQ2_XS (same size class, ~3.4× prompt t/s, stable decode).

## Next steps (recommended order)

1. **125B single/dual-GPU smoke test — DONE 2026-09-27 evening**: single-GPU
   tiering is RAM-bound (2–6 t/s decode); dual all-resident now loads (57 s)
   and runs at ~25 t/s decode / ~385 t/s prefill with the HSA bounce-buffer
   patch. Next actions: commit/upstream the patch (consider `set_tensor_2d`
   too) and file the repro with ROCm. See
   `qwen125-smoke-20260927/RESULTS.md`.
2. **Only if pursuing the cache further**: sparse per-token dispatch (GPU and
   CPU take variable shares of the 8 selected experts). That is the real
   engine project; current evidence says don't, unless a target workload shows
   the CPU tail dominating with a steep hot set. If done, also wire `qwen4exp`
   (fused `gate_up` supported; `_s` scales and biases would need work) and add
   router-stat instrumentation to profile hot sets (no HF weights on disk).
3. **KV**: use `--cache-type-k/v q8_0`; RAM↔VRAM windowing deferred.
4. **Commit `moe_tier`** to hivebench when happy with it.

## Gotchas (learned the hard way today)

- `pkill -f <pattern>` matches your own shell command line — use exact PIDs.
- Parallel tool calls race when one creates a file the other uses.
- GUI-file writer: tensor type IDs are ggml types (`F32=0`, `I32=26`), NOT
  GGUF metadata types (those are `4/5/6/8`).
- `ggml_get_rows`: source must be shaped so `ne[2]/ne[3]` match the index
  dims; `mul_mat_id`'s fallback requires unique ids per token (release builds
  skip its bounds `assert`, so out-of-range ids fail later with a size assert).
- `llama-cli` loops in interactive mode when redirected; use `-st` and
  `< /dev/null`, or prefer `/completion` via `llama-server`.
- Graph object pool: `graph_max_nodes` needed headroom for the split.
- 125B loads used to hang in `ggml_backend_cuda_buffer_set_tensor` →
  `libhsa-runtime64` (userspace spin, `stime` frozen) after a few GiB of
  **distinct unpinned host source ranges**. Root cause repro:
  `qwen125-smoke-20260927/hsa-repro/`. Workaround in place: stage copies
  >64 MiB through one reused pageable buffer (patch:
  `hsa-repro/backend-set-tensor-bounce.patch`, uncommitted). `HSA_ENABLE_SDMA=0`
  makes it worse. Repeated SIGKILLs of hung loads degrade driver state (hang
  threshold moved 19 → 15.5 GiB); prefer the repro over reloading the model.

## Exact commands to rebuild / reproduce

```bash
cd ~/Desktop/work/ternary-serve
cmake --build build-hip -j 12 --target llama-server llama-perplexity llama-cli

# sidecar (hot fraction from the profile)
python3 taardis-llama.cpp/tools/make_hot_sidecar.py \
  --model olmoe-q1exp-q8rest.gguf \
  --profile placement-sweep-20260927/expert-hits.json \
  --hot 16 --out /tmp/opencode/hot16.sidecar.gguf

# serve (corrected model + sidecar)
./build-hip/bin/llama-server -m olmoe-q1exp-q8rest.gguf \
  --lora branches-r1024-attnoutlloyd-step8000-q1g128-routers.lora.gguf \
  --lora /tmp/opencode/hot16.sidecar.gguf \
  -c 512 -ncmoe 16 -ngl 99 -t 8 --split-mode none --main-gpu 0 --port 8083
```
