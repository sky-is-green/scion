# v2 pipeline smoke - 4-layer student prefix, 2026-11-11

Purpose: validate the new v2 training path before renting a 96 GB pod.  This
is a **pipeline test**, not a capability result: the cache is synthetic and the
prefix is 4 layers.

## Setup

- Student prefix: `Qwen3_5MoeTextModel`, 4 layers, built from the first three
  empero HF shards (71 tensors; one later shard absent and skipped).
- Expert banks: seeded from `Scion-35B-A3B-PQ2_0-corr.gguf` with
  `moe/seed_banks_from_gguf.py` (layers 0-3, real deployed codes).
- Cache: `moe/build_trace_cache.py` format, two 512-token records with `ids`,
  top-64 `idx`, `val`, `w` (random values, deliberately).
- GPU: one RX 7900 XT, 20 GB; 4.72M trainable branch+router params.

## Result

```
FP vs ternary: hidden drift 0.3066, router top-8 agreement 0.8260
seeded 4 expert banks from .../Scion-35B-A3B-PQ2_0-corr.gguf
trainable 4.72M (branches + routers) target both
cache carries ids for all 2 records; using cached windows
step 1 lm 14.0865 kd 1.3961 tail 0.0000 total 16.8786
step 2 lm 14.1403 kd 1.4477 tail 0.0000 total 17.0356
saved .../qwen35-corr-r64-g128-step2-prefixsmoke.pt
```

The LM/KD numbers are meaningless with a random cache; the point is that the
seeded banks, `ids`-carrying cache, `both` branch placement, checkpointing and
adapter export all run.

## Bugs found and fixed

1. `attach_branches` used `layer.layer_type`, which transformers 5.5 does not
   set on `Qwen3_5MoeDecoderLayer`.  It now falls back to `hasattr(layer,
   "linear_attn")`; without this, `--branch-target both` crashed on every
   linear-attention layer.
2. A teacher-trace cache stores `ids` as int32; `F.cross_entropy` needs int64.
   The trainer now casts `ids.long()`.
3. `save()` did not create `$MOE_ARTIFACTS/qwen35`, so the first checkpoint
   crashed on a fresh pod.  It now `mkdir(parents=True)`.

## Export finding

`export_branches_lora.py --dtype q1_0_g128` requires the GGUF group axis to be
a multiple of 128.  For `lora_b` that axis is the rank, so **rank 64 cannot be
packed as Q1_0_g128** (the v1 rank-512 path is fine).  A rank-64 sweep must use
`--dtype f16` (10 MB for the 4-layer prefix) or a new g64 container.  The
rank-64 F16 export path was exercised successfully.

## Files

- `moe/qwen35_moe_proxy.py` (fixes), `moe/seed_banks_from_gguf.py`,
  `moe/build_trace_cache.py`.
- `v2/scripts/make_ptq1_0_parity_gguf.py` and `v2/PTQ1_0.md` for the separate
  size-track check.
