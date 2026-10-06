# Serving extras — plan and status (2026-09-27)

The four Strata/FreeToken-style extras, what each really is in this stack, and
what is being built. Measured evidence lives in `placement-sweep-20260927/`.

## Status

| Extra | Kind | Status |
|---|---|---|
| Adaptive per-expert cache | engine (fork) | **built + validated, shelved for performance**: exact (hot16 greedy-identical; hot64 PPL-identical), but the payoff sweep shows no gain — dense `mul_mat_id` keeps the CPU at k experts/token. Real gains need sparse per-token dispatch. Details: `placement-sweep-20260927/{EQUIVALENCE,THROUGHPUT}.md` |
| GPU-GPU tensor parallelism (`-sm tensor`) | build/backend | **closed**: only the SYCL backend registers `ggml_backend_split_buffer_type` in this revision; CUDA/HIP never did. VMM was a red herring (VMM works, but split buffers are still unsupported). |
| KV streaming (RAM + GPU window) | engine (fork) | deferred; substitutes first: KV quant (`--cache-type-k/v`), `--no-kv-offload`. KV split across GPUs is also gated on split buffers → unavailable on ROCm. |
| Semantic KV anchors | serving layer | **measured**: append turns already reuse ~everything (10/1806 tokens); mid-context edits re-evaluate the suffix (899/1806). Only worth engine work if mid-edit loops matter |

Test target for the first real-regime run: **Qwen3.8-Flash-Next GSQ-RCO Q2_0**
(36G + 27G shards). Correctness first on OLMoE, then single-GPU Q2_0 with the
cache. First smoke run (2026-09-27): single-GPU tiering is RAM-bound; dual-GPU
full residency loads (57 s) and serves at ~25 t/s decode / ~385 t/s prefill
after an HSA upload-stall fix (bounce buffer in `ggml-cuda.cu`).

## Adaptive per-expert cache — design (sharpened)

Why not config: llama.cpp stores a whole expert bank as ONE tensor
(`blk.N.ffn_*_exps.weight`), so `-ncmoe`/`-ot` move all experts of a layer
together, and `-sm tensor` (the other per-axis splitter) does not exist for
CUDA/HIP.

Why `build_moe_ffn` cannot be reused twice as-is: feeding `probs_in` /
`selected_experts_in` still runs the internal softmax and `norm_w`
renormalization, which re-normalizes per bank and makes hot+cold ≠ full. An
exact split needs the normalized weights computed once and applied per bank.

Chosen shape — a **hot-expert sidecar** plus one new graph helper:

1. **Profile** (exists): `placement-sweep-20260927/profile_experts.py` gives
   per-layer expert hit counts; hot set = top-K experts per layer.
2. **Sidecar GGUF** (`taardis-llama.cpp/tools/make_hot_sidecar.py`, done):
   - `blk.N.ffn_{gate,up,down,gate_up}_exps.hot` — the gathered hot expert
     blocks (raw bytes, same ggml type as the base);
   - `blk.N.ffn_hot_map` I32[n_expert] — original expert id → local hot index
     (out-of-set ids map to 0; their weight is zeroed by the mask);
   - `blk.N.ffn_hot_mask` / `blk.N.ffn_cold_mask` F32[n_expert] — 1.0/0.0.
   Loaded through the fork's string-keyed adapter path, which now accepts
   raw tensors and anchors them to the layer's router (GPU), so **base models
   stay stock GGUF**.
3. **Graph** (`src/llama-graph.cpp`, new `build_moe_ffn_split`): compute
   router logits/probs/top-k and the normalized weights ONCE (same math as
   `build_moe_ffn`), scatter them back to `[n_expert, n_tokens]`, then:
   - hot pass: `mul_mat_id` over the sidecar with mapped ids and masked weights;
   - cold pass: `mul_mat_id` over the base bank with original ids and the
     complement mask (zero-weight hot ids still index valid experts);
   - sum, then the shared down-projection/combine tail.
   `mul_mat_id` is linear in the selected experts, so hot+cold equals the full
   bank exactly; placement cannot change outputs.
4. **Per-model wiring**: OLMoE first (then `qwen35moe`), used only when the
   sidecar tensors are present; otherwise the existing path is untouched.
5. **Validation**: `llama-perplexity` equality (split vs full bank at several
   hot fractions), then `llama-bench` vs the `-ncmoe 16` baseline (94 t/s) at
   hot fractions 25/50/75% (hit coverage 44/71/90% from the profile).

Roadmap: static hot set first. Dynamic promotion/eviction at runtime only if a
real workload's hot set drifts and the static version proves the win.

## Maintenance

- One concern per commit series (adapter raw-tensor loading / graph split math /
  sidecar tooling), so any piece can be rebased or dropped.
- The sidecar is an adapter file — no base-model format changes; models stay
  stock GGUF and serve without the sidecar unchanged.

