# Porting notes: Xing4.0-29B-A4B (`qwen`-family custom code)

Status: assessment only, 2026-09-27.  Source: `XingChen-AGI/Xing4.0-29B-A4B`
(Apache-2.0, 62.4 GB bf16, 41 safetensors shards, `trust_remote_code=True`;
also FP8 and GGUF releases).

## What it is

| property | value |
|---|---|
| params | 29B total / 4B active |
| layers | 40, hidden 3584 |
| attention | MLA (`q_a/q_b`, `kv_a_with_mqa`, `kv_b`, `o_proj`) |
| MoE | 64 routed experts, top-4, **per-expert `nn.ModuleList`** of MLPs (expert intermediate 1024) + 1 shared expert |
| extra | **mHC hyper-connections** (`attn_hc` / `ffn_hc` in every decoder layer), MTP |
| serving | llama.cpp support is an **unmerged PR** (ggml-org/llama.cpp#29012); vLLM/SGLang PRs pending, prebuilt Docker images exist |

## Why it is not the first target

1. **Custom modeling code.** The model is not in transformers; the harness patches
   HF classes at runtime (fused-bank STE, router hooks). Here the MoE is a
   per-expert `nn.ModuleList` with a custom `Xing4_0TopkRouter`, so the STE patch
   is the old per-expert shape (cf. `olmoe_experts.py`), not the fused-bank path.
2. **Hyper-connections (mHC).** The residual stream is multi-stream:
   `post * attn_output + comb @ streams` (and the same around the FFN). The
   `moe_out` placement (block input -> block output) has no single stream to
   anchor on; the branch placement rule needs rethinking for mHC (the MLP sees
   the *collapsed* stream, so an `attn_out`-style branch is the natural first cut).
3. **Serving.** llama.cpp needs the pending arch PR, and the
   `moe-corr-runtime` virtual target would need hooking into that arch's graph.

## Porting checklist (when the recipe is proven)

- [ ] `transformers` integration or vendored modeling file; confirm forward hooks
      on `Xing4_0TopkRouter` and the expert MLPs.
- [ ] STE/ternary patch for per-expert `nn.ModuleList` (reuse the per-expert arm).
- [ ] Placement study under mHC: `attn_out` on `o_proj` first (LoRA-mappable),
      then a readout-style branch (see the fork's `ssm_readout` convention) if the
      collapsed stream is not enough.
- [ ] Ternary build + role map (norms, MLA projections, shared expert).
- [ ] Serving: rebase the TAARDIS fork onto the merged llama.cpp arch support,
      add the `ffn_moe_out`-equivalent hook for the new graph.
- [ ] Benchmarks: compare against the published table (SWE-bench 75.0 etc.) as
      the retention yardstick.

## What can be used right now

- Published benchmarks as a **modern-MoE retention yardstick**.
- Its architecture as the **generality test** for the blueprint: MLA + mHC +
  fine-grained experts is the next class after qwen3_5_moe.
- Its GGUF/FP8 releases for external tooling comparisons (not this container).
