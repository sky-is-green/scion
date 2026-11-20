# 35B hot/cold body — local prep and pod runbook

Status: **local type/codec/body pipeline implemented; 4-layer calibration done;
pod not started.**  The remaining paid step is the 4096-step correction run
against the hot/cold body.

## What changed vs the accepted plan

GGUF stores **one type per tensor**, and `MUL_MAT_ID` has no per-expert type
dispatch.  A single `blk.L.ffn_gate_exps.weight` therefore cannot hold 41 Q4
experts and 215 TQ1 experts.  The drop-in interpretation is **per expert
tensor**: `bits=4` on selected layers/projections, `bits=2` (TQ1) on the rest.
That is exactly what the trainer's existing `--alloc-file` does
(`blk.L.ffn_{gate_up|down}_exps`).

Per-expert hot/cold would need either:

- splitting each projection into `_hot`/`_cold` tensors, permuting the router
  rows and running two `MUL_MAT_ID`s with masks (graph + loader change), or
- a new variable-row container (fixed `ggml_row_size` is a core assumption).

Neither is implemented.  The local 4-layer calibration also shows why the
original "16% of experts" number was optimistic: on 350/360 rollout prompts,
the top 41/256 experts per layer cover **46.6%** of routed token-expert
assignments (range 37–54% by layer).  Even 50% hot covers only 76.0%.  A
16%-hot body would leave more than half of the active routing on TQ1.  The
per-tensor allocations below spend the bit budget on whole projections/layers
instead.

## Type and codec

- `Q4_0_G128` = GGUF type **147**, 128 weights / **66 bytes** = 4.125 bpw.
  - fp16 scale per 128; symmetric `amax/7`, codes −7..7 stored as 1..15.
  - standard Q4_0 nibble order: low = element j, high = element j+64.
  - C++ `block_q4_0_g128`, `quantize_row_q4_0_g128_ref`,
    `dequantize_row_q4_0_g128`, CPU `ggml_vec_dot_q4_0_g128_q8_0`.
  - HIP/CUDA dequant + MMVQ `vec_dot_q4_0_g128_q8_1`; convert/get_rows/
    supports_op registrations.
  - Python `moe/q4_0_g128.py`; byte-identical to `rco_alloc.quantize_grouped(
    bits=4)` and to the C++ reference (verified by ctypes on a fixed vector).
- `TQ1_0_G128` = type **143**, 128 weights / 28 bytes = 1.75 bpw.  The
  exporter packs the same Lloyd trits/scales as the deployed v1 body.
- `test-quantize-fns` passes Q4_0_G128 on CPU (quantisation, reference, dot,
  nrc=2).  The HIP test harness is unreliable on this box (standard Q4_0
  GET_ROWS also segfaults in `ggml_backend_compare_graph_backend`); validate
  the kernel with the real body/server instead.

## Size

Expert params: 32.21B; non-expert/file overhead: ~2.59 GiB.  Current v1 body
is 10.558 GiB (all experts 2.125 bpw).

| policy | expert bits | expert GiB | total GiB |
|---|---|---:|---:|
| all TQ1 (v1-like) | 1.75 | 7.99 | 10.58 |
| down4 (down Q4, gate/up TQ1) | 2.54 | 9.52 | **12.02** (measured) |
| half layers all-Q4 | 2.94 | 11.02 | 13.61 |
| gateup4 | 3.41 | 12.79 | 15.38 |
| all Q4 | 4.125 | 15.48 | 18.07 |

`down4` is the first build: it protects the expert output projection (1/3 of
expert weights) and stays at 12.1 GiB, leaving 6 GiB of the 18 GiB display
budget for context/KV.

## Local body build

Source: `models-p310/bf16/Qwen3.8-35B-A3B-Distill-BF16.gguf` (65 GB, local)
for the expert rows, and the v1 deployed body for the non-expert tensors (so
the v1 router/shared expert/Q8 dense quantization is preserved and the file
does not carry 5 GiB of BF16 non-experts).  The exporter streams one tensor at
a time, so 30 GB RAM is enough.

```bash
cd .../work/scion
OMP_NUM_THREADS=8 .../.envs/scion-rocm/bin/python moe/build_hotcold_body.py \
  --src .../models-p310/bf16/Qwen3.8-35B-A3B-Distill-BF16.gguf \
  --dense-src .../models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf \
  --out v2/hotcold/Scion-35B-A3B-v2-down4.gguf \
  --policy down4
```

Without `--dense-src` the non-expert tensors are copied from the bf16 base and
the file is **14.14 GiB**; with it the expected size is ~12.1 GiB.  The first
(dense=bf16) build is a valid model but not the one to ship.

`--policy gateup4|half|third|all4|all2` and `--alloc-file` are supported.
Single-tensor checks (byte layout + dequant against the Q4/TQ1 reference):

```bash
... build_hotcold_body.py --src ... --out /tmp/one.gguf \
  --policy all4 --only-tensor blk.0.ffn_down_exps.weight
```

## 4-layer calibration (done)

`moe/calibrate_hot_experts.py` loads the first N layers of `empero-hf`, counts
the routed top-8 per expert on the 360 rollout prompts, and writes
`v2/hotcold/hotmap-4l.json`.  The local PyTorch/ROCm prefix forward faults
after ~10–25 variable-length prompts, so the run is sharded:

```bash
# shards of 10-20 prompts; 350/360 covered (shard 100:110 faults)
HIP_VISIBLE_DEVICES=0 .../python moe/calibrate_hot_experts.py \
  --model-dir .../models-p310/empero-prefix/empero-hf --layers 4 \
  --start 0 --end 20 --seq-len 512 \
  --counts-out /tmp/opencode/hotcounts/shard-0.npz
# merge
.../python moe/calibrate_hot_experts.py --model-dir .../empero-hf --layers 4 \
  --merge '/tmp/opencode/hotcounts/shard-*.npz' --hot-fraction 0.16 \
  --out v2/hotcold/hotmap-4l.json
```

The 40-layer map must be regenerated on the full checkpoint (`--layers 40`) on
the pod; the 4-layer map is a ranking/codec check only.

## 4-layer training/export smoke

Synthetic 2-record cache (same format as `build_trace_cache.py`):
`ids [512] int32`, `idx [511,64] int32`, `val/w [511,64] fp16`.
Alloc file for the prefix (Q4 on layers 0–1, TQ1 on 2–3):

```json
{
 "blk.0.ffn_gate_up_exps": 4, "blk.0.ffn_down_exps": 4,
 "blk.1.ffn_gate_up_exps": 4, "blk.1.ffn_down_exps": 4,
 "blk.2.ffn_gate_up_exps": 2, "blk.2.ffn_down_exps": 2,
 "blk.3.ffn_gate_up_exps": 2, "blk.3.ffn_down_exps": 2
}
```

```bash
MOE_ARTIFACTS=.../models-p310/empero-prefix \
  python moe/qwen35_moe_proxy.py train --prefix-layers 4 --device cuda:0 \
  --alloc-file v2/hotcold/alloc-4l.json \
  --cache-file /tmp/opencode/qwen35-hotcold-cache.pt \
  --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
  --steps 2 --tag hotcold4l

PYTHONPATH=.../scion-llama.cpp/gguf-py python moe/export_branches_lora.py \
  --load .../qwen35-corr-r512-g128-step2-hotcold4l.pt --arch qwen35moe \
  --target both --dtype q1_0_g128 --out /tmp/hotcold4l.gguf
PYTHONPATH=.../scion-llama.cpp/gguf-py python moe/merge_adapter_into_body.py \
  --body .../Scion-35B-A3B-PQ2_0-corr.gguf --adapter /tmp/hotcold4l.gguf \
  --out /tmp/hotcold4l-merged.gguf --replace
PYTHONPATH=.../scion-llama.cpp/gguf-py python v2/scripts/export_audit.py \
  --deployed /tmp/hotcold4l-merged.gguf --branches-only
```

The 4-layer adapter only has tensors for layers 0–3; merging into the full v1
body replaces those branches and leaves the rest unchanged.  This proves the
training-time `bits=4` path, the exporter and the merge/audit, not capability.

Local result (1 step, synthetic cache): `10.59 GiB` peak, 8 exported branch
tensors, 16 adapter tensors merged, and the audit still reports **160 tensors
/ 209,715,200 weights / 0 code3 / 0 bad scales**.  The training log shows the
four `alloc-file` entries at 4-bit and four at 2-bit as intended.

## Compact-bank support (pod path)

`read_bank_blocks(..., with_bits=True)` reads type 147 as Q4 nibbles/scales,
type 143 as the equivalent 2-bit ternary codes, and type 142 as before.
`CompactBank(bits=4)` decodes Q4; `compact_banks_from_blocks` takes separate
`gu_bits`/`dn_bits`.  The pod's `--bank-gguf --compact-banks` path therefore
seeds each tensor with its deployed type and no longer assumes all-2-bit.

Run the compact tests before the pod:

```bash
.../.envs/scion-rocm/bin/python -m pytest -q moe/tests/test_qwen35_compact.py
```

## Pod runbook (stop before this)

1. Top up to ~$25 (A100 SXM 80 GB, $2.19/h, EUR-IS-1, 150 GB volume).
2. Upload `Scion-35B-A3B-v2-down4.gguf` (12.1 GiB) and the synthetic/real
   cache if available (`runpodctl send`), or rebuild the cache on the pod.
3. Setup: the v2 full-run `v2/scripts/box-run-v2.sh` equivalents; install
   the fork, `torch`, `transformers`, `safetensors`, `datasets`; put the full
   HF checkpoint on the volume.
4. Calibration (optional if the body is already built): 40-layer
   `calibrate_hot_experts.py` on the 360 prompts.
5. Curriculum/cache/ref: same commands as `v2/RUN-PLAN.md` (4096 windows,
   cache ~10 GB, ref 2 wikitext windows).
6. Train 4096 steps:
   ```bash
   python moe/qwen35_moe_proxy.py train --device cuda:0 \
     --bank-gguf /workspace/artifacts/Scion-35B-A3B-v2-down4.gguf \
     --seed-routers --compact-banks --grad-checkpoint --checkpoint-mode group \
     --expert-group-size 64 --quant lloyd --branch-quant g128 \
     --branch-target both --rank 512 --freeze-router \
     --cache-file .../cache.pt --ref-file .../ref.pt \
     --steps 4096 --tag hotcold-down4
   ```
   `--seed-routers` copies the v1 F32 routers/shared-expert gates from the
   body.  The v2 run trained against the base router while deploying a
   different one; this removes that confound.
   Expected ~6.4–6.9 s/step (same 14 GiB compact footprint; Q4 banks add
   ~2 GiB), 7.5–8 h, **~$18–20** plus setup/cache (~$4–6).
7. Export with `--device cuda` (the CPU Lloyd export was the 4 h incident):
   ```bash
   python moe/export_branches_lora.py --load .../step4096.pt --arch qwen35moe \
     --target both --dtype q1_0_g128 --device cuda --out .../adapter.gguf
   ```
8. Pull the adapter via `runpodctl send`; merge into the **down4 body** with
   `--replace`; zero nothing (the router was frozen and the body already has
   the BF16 router the branches saw); run `v2/scripts/export_audit.py
   --branches-only`; then local subset30.  The 4-layer smoke merges into the
   v1 body only because it has tensors for every layer; a real adapter must
   not be merged onto the v1 expert weights.
9. Terminate the pod/volume after the pull.

## Papers / next levers

- **Hy4-flash** (Tencent): 220B total / 12B active, 512 experts top-16,
  LatentMoE (experts in a 1536-d latent), Engram n-gram tables, 4-bit serving.
  The useful ideas are LatentMoE and cheap n-gram capacity, not a drop-in for
  the 35B.  Its GGUF expert tensors are homogeneous (Q4_K_M or IQ2_XXS with
  Q4_K down), confirming the per-tensor granularity.
- **Chorus (arXiv 2611.23386)**: shared n-gram tables + per-layer projections,
  87% smaller tables.  A capacity add-on for a later phase; no effect on
  ternary damage now.
- **Leech lattice VQ (arXiv 2603.11021)**: the most relevant cold-tier idea —
  better rate-distortion than scalar/group quant at ~2 bpw, with indexable
  bitstrings and a parallel dequant.  The 24-d lattice does not divide the
  2048/512 expert rows (pad to 2064/528, ~0.8–3%), and it needs a new GGUF
  type/kernel.  Candidate only if Q4/TQ1 still misses the gate; Spherical GPTQ
  is also a possible body-side improvement.
- **AngelSlim**: INT4 GPTQ/AWQ/GPTAQ and a MIX-STQ1_0 ternary build; useful
  reference for mixed-precision recipes, but its GGUF path is per-tensor too.

## Known confounds

- **Per-tensor, not per-expert**: see the format note at the top.  The hot
  budget is spent on whole `down` tensors, not the top-load experts.
- **Dense train/deploy mismatch**: the trainer loads the full HF checkpoint
  in bf16; the body ships the v1 Q8_0/F32 dense tensors.  This is the same
  mismatch the v2 run had.  `--seed-routers` fixes the router half; seeding
  the dense tensors (or shipping a bf16-dense body, 14.14 GiB) is the next
  consistency fix if the expert change shows a signal.
- **4-layer map, not 40-layer**: the load calibration only covers layers 0–3
  and 350/360 prompts.  The full map must be regenerated with `--layers 40`.
- **Coverage is modest**: top 41/256 experts cover 46.6% of routed events at
  4 layers; the per-tensor body does not depend on that number, but a future
  per-expert split should not assume 16% is enough.

## PPL / coherence

`llama-perplexity` on `/tmp/opencode/ppl.txt` (11 KB of rollout prompts,
4 x 512-token chunks, batch 128) with the ROCm 7.2.4 runtime:

- v1 reference (`Scion-35B-A3B-Q2_0_G128-corr-v1.gguf`): **PPL 2.4445**.
- v2 cur05 merged model: **PPL 2.5243**.
- first `down4` build: **NaN** (the source-offset bug above; 2176 Inf TQ1
  scales).  Rebuild pending.
- v2 merged model loads/generates on the same build (91.6 tok/s prompt,
  49.3 tok/s generation with `-ngl 99` on one 7900 XT).

## Current state

- `Q4_0_G128` type/kernel/GGUF/Python codec: done; CPU tests pass; HIP builds.
- 4-layer hot map: done (350/360 prompts).
- Full `down4` body: **12.02 GiB**, 733 tensors (40 Q4_0_G128 down, 80
  TQ1_0_G128 gate/up, 312 Q8_0 + 301 F32 dense/router), no embedded adapter.
  **The first 12.02 GiB file is invalid**: the exporter read the bf16 source at
  `GGUFReader.data_offset + t.data_offset`, but `t.data_offset` is already
  absolute, so every expert read was shifted by ~11 MB.  That produced Inf TQ1
  scales and NaN PPL.  The read is now `f.seek(t.data_offset)`; `down4` and an
  `all2` control are being rebuilt.  `v2/hotcold/validate_body.py` checks
  finite scales and source-equality on sample tensors.
- 4-layer train/export/merge/audit: **passed** (1 step, 10.59 GiB peak,
  160-tensor branch audit clean).
- HIP binaries: a clean `--clean-first` rebuild is done.  The earlier crashes
  were **not** the build: running without
  `LD_LIBRARY_PATH=.../work/rocm-7.2.4/opt/rocm/lib:.../work/rocm-7.2.4/usr/lib`
  loads the wrong ROCm runtime and dies inside `ggml_backend_cuda_synchronize`.
  With the 7.2.4 path, v2 loads at 91.6 tok/s prompt / 49.3 tok/s generation.
- The older `Scion-35B-A3B-PQ2_0-corr.gguf` is rejected by the current
  llama.cpp GGUF reader (`ffn_moe_out.weight.lora_b` offset mismatch); use the
  v2 merged model or the new contiguous hot/cold body for llama.cpp checks.
- Pod: **not started**; balance **$9.50**.
