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
| gateup4 | 3.41 | 12.79 | **14.99** (measured) |
| all Q4 | 4.125 | 15.48 | 18.07 |

`gateup4` is the recommended body: it protects the gate/up feature
computation (2/3 of expert weights) and measures **14.99 GiB**, with the
display card at **12.37 GB** and the spare at 6.52 GB under the 0.75/0.25
split.  `down4` (12.02 GiB) is the size-safe fallback.

## Local body build

Source: `models-p310/bf16/Qwen3.8-35B-A3B-Distill-BF16.gguf` (65 GB, local)
for the expert rows, and the v1 deployed body for the non-expert tensors (so
the v1 router/shared expert/Q8 dense quantization is preserved and the file
does not carry 5 GiB of BF16 non-experts).  The exporter streams one tensor at
a time, so 30 GB RAM is enough.

```bash
cd .../work/scion
HIP_VISIBLE_DEVICES=1 .../.envs/scion-rocm/bin/python moe/build_hotcold_body.py \
  --src .../models-p310/bf16/Qwen3.8-35B-A3B-Distill-BF16.gguf \
  --dense-src .../models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf \
  --out v2/hotcold/Scion-35B-A3B-v2-gateup4.gguf \
  --policy gateup4 --device cuda:0
```

`--device cuda:0` runs the Q4 packer and the 8-iteration Lloyd rule on the
GPU and copies only the packed bytes back; the CPU path is the reference.
One-tensor and small-array checks are byte-identical between the two.  Running
two CPU exporters at once swaps the 30 GB box and is much slower.

Without `--dense-src` the non-expert tensors are copied from the bf16 base and
the file is **14.14 GiB**; with it `down4` is 12.02 GiB and `gateup4` is
14.99 GiB.  The first (dense=bf16) build is a valid model but not the one to
ship.

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
Alloc file for the prefix (exact gateup4: Q4 gate_up on layers 0–1, TQ1
down; TQ1 on 2–3):

```json
{
 "blk.0.ffn_gate_up_exps": 4, "blk.0.ffn_down_exps": 2,
 "blk.1.ffn_gate_up_exps": 4, "blk.1.ffn_down_exps": 2,
 "blk.2.ffn_gate_up_exps": 2, "blk.2.ffn_down_exps": 2,
 "blk.3.ffn_gate_up_exps": 2, "blk.3.ffn_down_exps": 2
}
```

```bash
MOE_ARTIFACTS=.../models-p310/empero-prefix \
  python moe/qwen35_moe_proxy.py train --prefix-layers 4 --device cuda:0 \
  --alloc-file v2/hotcold/alloc-4l-gateup4.json \
  --cache-file /tmp/opencode/qwen35-hotcold-cache.pt \
  --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
  --steps 2 --tag hotcold4l-gateup4

PYTHONPATH=.../scion-llama.cpp/gguf-py python moe/export_branches_lora.py \
  --load .../qwen35-corr-r512-g128-step2-hotcold4l-gateup4.pt --arch qwen35moe \
  --target both --dtype q1_0_g128 --recipe "hotcold 4-layer smoke" \
  --out /tmp/hotcold4l.gguf
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
four `alloc-file` entries at 4-bit and four at 2-bit as intended.  The exact
`gateup4` alloc (Q4 gate_up, TQ1 down) also completed a 1-step run at
10.59 GiB peak.

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
2. Upload `Scion-35B-A3B-v2-gateup4.gguf` (14.99 GiB) and the synthetic/real
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
     --bank-gguf /workspace/artifacts/Scion-35B-A3B-v2-gateup4.gguf \
     --seed-routers --compact-banks --grad-checkpoint --checkpoint-mode group \
     --expert-group-size 64 --quant lloyd --branch-quant g128 \
     --branch-target both --rank 512 --freeze-router \
     --cache-file .../cache.pt --ref-file .../ref.pt \
     --steps 4096 --tag hotcold-gateup4
   ```
   `--seed-routers` copies the v1 F32 routers/shared-expert gates from the
   body.  The v2 run trained against the base router while deploying a
   different one; this removes that confound.
   Expected ~6.5–7.0 s/step (gateup4 compact banks ~16–18 GiB), 7.5–8.5 h,
   **~$18–22** plus setup/cache (~$4–6).  Balance is $9.50; top up ~$12–15.
7. Export with `--device cuda` (the CPU Lloyd export was the 4 h incident):
   ```bash
   python moe/export_branches_lora.py --load .../step4096.pt --arch qwen35moe \
     --target both --dtype q1_0_g128 --device cuda --out .../adapter.gguf
   ```
8. Pull the adapter via `runpodctl send`; merge into the **gateup4 body** with
   `--replace`; zero nothing (the router was seeded from the same body); run
   `v2/scripts/export_audit.py --branches-only`; then local subset30.  The
   4-layer smoke merges into the v1 body only because it has tensors for every
   layer; a real adapter must not be merged onto the v1 expert weights.
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

## subset30 result: 12/30 (gate passed, no pod training)

`gateup4 + v1 corrections`, frozen protocol (T=0.6, seed 1001, 32k budget,
recovery sentence):

| family | correct |
|---|---:|
| reasoning | **3/10** (R21, R29, R11) |
| coding | **0/10** (all fail; 9 truncate at 32k) |
| tools | **9/10** (T04, T09, T01, T29, T02, T25, T30, T18, T13; T03 fails) |
| **total** | **12/30** |

Reference: v1 **5/30**, v2 **1/10**, gate **8/30**.  This is well above the
gate and the first configuration that clears it.  The v1 corrections transfer
to the Q4 gate/up body; no new training was needed.

The remaining weakness is coding: **0/10**, nine of them truncated.  A fresh
correction run could target coding, but it is now an optional improvement,
not a gate risk.

## Hy4 AMA answers (2026-11-20)

- **Down is more sensitive in general** because of activation outliers; they
  allocate bits with Hessian-based sensitivity.  They prefer **per-tensor**
  allocation over per-expert mixed precision because inference engines stack
  experts as one matrix; per-tensor is also more portable.  This confirms our
  per-tensor choice and closes the per-expert split branch.
- **QAT trains the router** with the rest of the model and keeps it stable.
  They prefer **full-parameter QAT**; they have not run LoRA, so no adapter
  advice.
- They expect **reasoning depth, long-context tracking and tool-call
  reliability** to degrade first; **2-bit PTQ breaks well before ternary QAT**.
- **Engram** is a separate n-gram table per layer, not counted in the 12B
  active parameters; they maximize capability within the memory budget.

Our direct check of the down-sensitivity rule on the 35B:
`down4 + v1 corrections` PPL **2.2797** vs `gateup4 + v1 corrections` **2.0402**.
Gate/up is the more sensitive projection here, so the general rule is
architecture-dependent (their down lives in the LatentMoE space; ours is
2048->512).

An all-Q4 body (17.96 GiB) validates and generates coherently on CPU, but its
GPU PPL is garbage (1e7) on this fork, single- and two-GPU.  The all-Q4 expert
path is not usable without a kernel fix, so it is not a candidate; `gateup4`
stands.

The 12/30 pass is a PTQ body plus transferred LoRA corrections, not their QAT
recipe.  If coding (0/10) is the goal, full-parameter QAT with router training
is the lever the team points to; it is optional now that the gate is passed.

## Body additions and adaptive tests (2026-11-21)

The `down4` test was a **substitution**: Q4 down instead of Q4 gate/up.  It
lost (first-10 1/10, PPL 2.2797).  The **addition** test is different: Q4
down on top of Q4 gate/up.

- Partial down (last 20 layers), custom Q4_0_G128 down: **CPU PPL 1.9270**,
  better than gateup4's 2.0402.  The fork's GPU path produces garbage when
  Q4 gate/up and Q4 down are both type 147, even at 20/40 layers.
- Standard Q4_0 down (type 2) on the same 20 layers: **GPU PPL 1.9307**,
  16.81 GiB, display 13.4 GB at 0.75/0.25.  The mixed types avoid the broken
  same-type path and the standard Q4_0 kernel works.  This is the deployable
  form of the down addition; subset10 is running.
- v2 corrections on gateup4 + base router: **PPL 2.3035**, worse than v1
  2.0402.  The v2 branches are the problem, not the body.
- Top-k override on the release: 8 = 2.0402, 16 = 2.0996, 24 = 2.2084.
  Zero-shot extra experts hurt; adaptive active capacity needs a trained
  budget router, not a metadata override.

The release remains `gateup4 + v1 corrections` (12/30).  The standard-Q4_0
partial is the candidate if its subset10 holds up.

## Coding repair profile (2026-11-21)

`v2/scripts/coding_repair.py` on the release (T=0.3, 8k thinking budget,
one failing-assertion feedback turn):

- **1/10** correct.  C11 passes on the **first** turn (it was false in the
  earlier 8k probe, so this is profile variance, not a repair win).
- **0/9 repaired**: every failed item was sent its assertion and rewritten
  code, and every one still failed.  C05's repair turn looped to 32k with no
  code block.  Typical remaining errors are edge-case assertions (C09),
  queue ordering (C30), route matching (C03), DSU counts (C24), atomic
  rollback (C27), ferry scheduling (C22), recursive-descent syntax (C01),
  and a None comparison (C26).
- More feedback turns are not promising: the rewrites often reach the
  thinking budget and still miss the same edge case.  Coding remains the
  release's weak family (0/10 frozen, 1/10 with the coding profile); tools
  (9/10) and reasoning (3/10) carry the 12/30.

## HF upload (2026-11-22)

- Uploaded the release to `SkyIsNotGreen/Scion-35B-A3B` as
  `Scion-35B-A3B-Q4_0-GateUp-TQ1Down-v1corr.gguf` (renamed from
  `Q4GateUp-...` so the Hub's filename-derived quant selector groups it
  as a 4-bit variant; the selector now shows Q2_0 and Q4_0):
  16,202,300,928 bytes, SHA256
  `83256e60ecd7bf8a8745f6222d27816d602a5e1be29fdf5e816bc47ad85502e5`.
  The Hub LFS OID matches the local hash; GGUF magic verified by range read.
- Card commit `76682e7`: added the Q4 gate/up variant section, the
  SHA256SUMS entry, and the `Q4_0_g128` (147) row to CONTAINER-NAMES.
  The 11.3 GB Q2 file remains the main release.
- The card states plainly: no new training, every trained tensor is v1,
  and types 147/143 require `scion-llama.cpp` `scion-base` at
  `2089cd1d3`; stock llama.cpp and the HF GGUF viewer cannot load it.

## Full-TQ1 tests (2026-11-21)

- The v1 reference's Q2_0_G128 expert banks and TQ1_0_G128 encode the same
  ternary weights, so Q2 -> TQ1 is a **lossless repack** (34 -> 28 bytes per
  128).  `v2/scripts/transcode_experts_ptq1_0.py` does the repack; the script
  needed the fork's current enum name `TQ1_0_G128` (the old `PTQ1_0` alias is
  gone).
- `Scion-35B-A3B-v1-tq1.gguf`: **9.15 GiB**, 120 type-143 banks and the 160
  v1 correction tensors kept at 142.  Sample parity on
  `blk.0.ffn_gate_exps.weight`: trits equal, scales equal, dequantized values
  bit-equal.  **GPU PPL 2.4468** vs v1 2.4445 (0.1% kernel accumulation, not
  a weight difference); batch-1 PPL 2.4472 vs 2.4495.
  **Behavioral result: 4/30** (tools 4/10: T29, T03, T02, T13; reasoning
  0/10; coding 0/10; 15 truncated) vs v1's 5/30 (T29, T03, T25, T18, R11).
  TQ1 loses T25/T18/R11 and gains T02/T13, i.e. parity within sampling
  variance, not a broken path.  Full TQ1 is **not** a release candidate:
  4/30 is far below the 8/30 floor and the 12/30 release.  The 9.15 GiB
  saving costs ~8 capability points, so the Q4 gate/up is worth its 6 GiB.
  **Baseline recheck** on the current binary: v1 gets **4/5** on the five
  items it passed in the old 5/30 run (T29, T03, T25, R11 pass; T18 now
  fails).  So the old 5/30 was slightly stale, and TQ1's 4/30 is genuine
  near-parity with today's v1, not a regression from the container.

Item-level comparison, gateup4+v1corr (12/30) vs v1 (5/30) and TQ1 (4/30):

- v1 passes: T29, T03, T25, T18, R11.
- gateup4 passes: R21, R29, T04, T09, T01, T29, T02, T25, T30, T18, T13, R11.
  It **gains 8** (R21, R29, T04, T09, T01, T02, T30, T13) and loses 1 (T03):
  net +7, all in tools plus two reasoning items.
- TQ1 passes: T29, T03, T02, T13.  It gains T02/T13 but loses T25/T18/R11:
  net -1.  Coding is 0/10 in every variant.
- `Scion-35B-A3B-v2-all2.gguf`: 9.06 GiB, all experts TQ1, no corrections,
  PPL 2.9251.  On v1's five pass items it gets **2/5** (T29, T25; loses T03,
  T18, R11), the same count as v1-TQ1 but a different mix and below v1's
  current 4/5.  Fresh PTQ without the v1 QAT weights/corrections does not
  recover the lost passes; the full 30 was not run.

The partial standard-Q4_0 down body (16.81 GiB, PPL 1.9307) finished its
first-10 at **3/10** (R21, R29, T04; coding 0/5), identical to gateup4's
3/10.  The PPL gain did not turn into capability, and the body is 1.7 GiB
larger, so it is not promoted; `gateup4 + v1 corrections` stays the release.

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

| model | experts | corrections | size | PPL |
|---|---|---|---:|---:|
| v1 reference | Q2_0_G128 | v1 embedded | 10.56 GiB | 2.4445 |
| v2 cur05 | Q2_0_G128 | v2 embedded (zero router) | 10.56 GiB | 2.5243 |
| down4 | Q4 down + TQ1 gate/up | none | 12.02 GiB | 2.6837 |
| all2 | TQ1 all | none | 9.06 GiB | 2.9251 |
| **gateup4** | **Q4 gate/up + TQ1 down** | none | **14.99 GiB** | **2.3413** |
| gateup4 + v1 corr | Q4 gate/up + TQ1 down | v1 embedded | 15.09 GiB | **2.0402** |
| down4 + v1 corr | Q4 down + TQ1 gate/up | v1 embedded | 12.12 GiB | 2.2797 |

`gateup4` is the winner: **2.3413 PPL**, better than the v1 model that has
correction branches (2.4445), with **no corrections at all**.  Protecting the
gate/up projections is worth much more than protecting `down`: `gateup4`
beats `down4` by 0.342 PPL for +2.97 GiB, and beats all-TQ1 by 0.584.  The
Q4-down-only body is still 0.241 better than all-TQ1, but the gate/up result
is the one to train.

VRAM check with `-ts 0.75,0.25`, 40k context, `gateup4`:
**display card 12.37 GB, spare 6.52 GB** (both 20.5 GB cards); server healthy.
The display-card limit is ~18 GB, so the 15 GiB body has ~5.6 GB of headroom.

The **v1 corrections transfer** to gateup4: merging the extracted v1 adapter
(`v2/hotcold/extract_adapter.py`) gives **PPL 2.0402**, better than either the
v1 model (2.4445) or gateup4 without corrections (2.3413).  subset30 on this
merged model is running; it is the free capability pre-check before the pod.

Generation: gateup4 109.3 tok/s prompt / 81.1 tok/s generation; down4 151.3 /
89.2; v2 91.6 / 49.3 on the same short prompt.

The display-card budget is per card: at `-ts 0.75,0.25` even an all-Q4 body
(18.07 GiB total) would put ~13.6 GiB on the display card, under the ~18 GiB
cap.  All-Q4 is the next local body if gateup4 + v1 corrections lands short of
the gate.

## Free capability pre-check

Before paying for 4096 corrections, the v1 adapter is a free probe of whether
the body is compatible with the correction site:

```bash
PYTHONPATH=.../gguf-py python v2/hotcold/extract_adapter.py \
  --src .../Scion-35B-A3B-Q2_0_G128-corr-v1.gguf \
  --out /tmp/opencode/v1-adapter.gguf

PYTHONPATH=.../gguf-py python moe/merge_adapter_into_body.py \
  --body v2/hotcold/Scion-35B-A3B-v2-gateup4.gguf \
  --adapter /tmp/opencode/v1-adapter.gguf \
  --out v2/hotcold/Scion-35B-A3B-v2-gateup4-v1corr.gguf --replace
```

Serve it with `-ts 0.75,0.25`, 40k ctx (display card 12.45 GB, spare 6.55 GB)
and run `v2/scripts/hard_subset.py --port 8083`.  The v1 reference is 5/30;
the first ten items are **not** a reliable early signal: v1 itself scores
**0/10** on them, and all five v1 passes (T29, T03, T25, T18, R11) are in
items 11–30.  gateup4's 3/10 (R21, R29, T04) is a real gain over v1 on the
same ten, but a candidate that matches 0/10 has not yet lost anything v1 had.

## Current state

- `Q4_0_G128` type/kernel/GGUF/Python codec: done; CPU tests pass; HIP builds.
- 4-layer hot map: done (350/360 prompts).
- **`gateup4` body: 14.99 GiB, PPL 2.3413** — the recommended pod body.
  733 tensors (80 Q4_0_G128 gate/up, 40 TQ1_0_G128 down, 312 Q8_0 + 301 F32
  dense/router), no embedded adapter.  Validated on layers 0/20/39: finite
  scales and **0.0** source-encode difference.
- `down4` body: **12.02 GiB**, PPL 2.6837, **first-10 1/10** (R21 only;
  T04 and R29 lost) versus gateup4 3/10.  The substitution test is clear:
  Q4 gate/up is the valuable projection; Q4 down instead of it is worse.
  It is not a release candidate.
- `all2` control: **9.06 GiB**, all experts TQ1, PPL 2.9251, same dense/router.
- The first 12.02 GiB file was invalid: the exporter read the bf16 source at
  `GGUFReader.data_offset + t.data_offset`, but `t.data_offset` is already
  absolute, so every expert read was shifted by ~11 MB.  That produced Inf TQ1
  scales and NaN PPL.  Fixed; the GPU rebuild is the valid file above.
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
- **Release candidate: `Scion-35B-A3B-v2-gateup4-v1corr.gguf`** — 15.09 GiB,
  PPL 2.0402, subset30 **12/30**.  The v1 corrections are embedded and the
  body is 12.37 GB on the display card at 0.75/0.25 with 40k context.
- **The pod is not needed for the gate.**  A fresh correction run is optional
  and would be aimed at coding (0/10), not at the 8/30 gate.
- Pod: **not started**; balance **$9.50**.
