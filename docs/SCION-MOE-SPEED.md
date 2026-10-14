# Scion MoE speed lane — record (opened 2026-10-14)

Lane goal (from `NEXT-SESSION-SCION-MOE-SPEED.md`): apply the closed Bonsai-2 dense lane's speed methodology to
the two Scion MoE models in Strata — **Scion-FlashNext-176B-A6B** (live on the two cards) and
**Scion-35B-A3B** (no Strata lane yet).  Engine tree `work/strata`, branch `scion-0.1.44`; fixed protocol
`serving/bench_strata.sh`; `--vram-reserve-mib 3072`; never `git push`; model-load tests only in serve-down
windows.

## Step 1 — the verify-window instrument and the warmed budget (commit `58aa4712`)

### The instrument (opt-in; both unset = no kernel, node or stream change)

| env | what it adds |
| --- | --- |
| `STRATA_VERIFY_GRAPH_TIME=1` | per replay: the GPU span (events around the graph launch), per-window-size T averages, the launch-call host time, and the captured node count per T.  One line per stage per request with `STRATA_DECODE_TIMING=1`: `strata decode replay, stage N: replay ... \| T=.. \| nodes..` |
| `STRATA_VERIFY_ABLATE=1` | bench-only: the routed expert GEMMs are skipped, so the node-count delta and (with GRAPH_TIME) the replay split are measurable.  The output is garbage. |

The ablation's T collapses to 1 (garbage drafts end the spec chain), so its replay span is not a clean per-T
delta; the expert-GEMM share is read from the existing profile stamps instead (19->20 "VRAM hits" brackets the
grouped expert launches on the GPU clock).

### The warmed budget of the live serve (2026-10-14; Q6K + corrections, `--pipeline-windows 2`, reserve 3072)

Warmed 128-token greedy request, 2.33-2.37 tokens/window over 54-56 windows.  Fixed protocol gen: 68.3-77.0
tok/s (desktop-load spread; single runs up to 77.9).  Envs off except where said.

| item | ms/window | source |
| --- | ---: | --- |
| window wall | **30.0-30.9** (74-77 tok/s) | `STRATA_DECODE_TIMING` |
| GPU-reach wait | **22.1-22.7** | `STRATA_DECODE_TIMING` (the dominant item) |
| per-layer host | 1.1-2.3 (plan 0.09-0.13, actq 0.01, jobs ~0, CPU 0.6-0.8) | `STRATA_DECODE_TIMING` |
| host staging + launch call | 0.01 + 0.04-0.05 | GRAPH_TIME |
| unattributed pipeline host | ~6.3 | wall minus the parts above |
| replay, stage 0 (layers 0-25) | 16.0-17.0 over 40 windows/request | GRAPH_TIME |
| replay, stage 1 (layers 26-47) | 15.5-16.5 over 28 windows/request | GRAPH_TIME |
| nodes | stage0 T=1..6: 1307 / 1320 / 1334 / 1348 / 1362 / 1376; stage1: 1118 / 1129 / 1140 / 1151 / 1162 / 1173 | GRAPH_TIME |

Per-T replay spans (warmed averages; the served mix is mostly T=4 with T=1/2/3 mixed in):

| T | stage 0 | stage 1 |
| --: | ---: | ---: |
| 1 | 12.2 | 11.3 |
| 2 | 14.3 | 13.0 |
| 3 | 16.5 | 15.5 |
| 4 | 18.8 | 18.9 |

Profile arm (`STRATA_VERIFY_PROFILE=1`; fusions off, so ~2 ms/window pessimistic — wall 32.5-33.0, wait
24.2-24.5), GPU stage times per window, stage 0:

| item | ms/window |
| --- | ---: |
| expert GEMM ("VRAM hits", stamps 19->20) | 4.06 (GDN 3.15 + QSA 0.91) |
| hyper-connection hc0 + hc1 (norm/down/up) | ~3.0 |
| out-proj (corrections included) | 1.58 |
| q8+qkv / q-idx gemv | 1.08 |
| shared expert + quant | 1.16 |
| router + plan + doorbell | 0.93 |
| waits A/B/CPU | ~1.2 |
| copy+combine (corrections included) | 1.07 |
| total | 17.8 |

Stage 1 mirrors this and adds the head (1.21 ms/window).

Ablation node delta (clean): skipping the routed expert GEMMs removes **6 nodes per layer** (stage 0 T=1
1307 -> 1151 = 156 = 6 x 26 layers; stage 1 132 = 6 x 22).  In the profiled listing the expert path is
`native_gu_multi<143>`, `swiglu_q8_1_entries`, `native_down_multi<143>` plus the quantize/hit launches around
them.

**The wait decomposition:** a window's critical path is stage 0's graph (16.5) + hand-off + stage 1's graph
(16.2); consecutive windows overlap only ~3 ms, and the host's ring-to-ring wait (22.4) exceeds stage 0's replay
span by ~6 ms.  That excess is the hand-off/doorbell exposure, and it is step 2's first target.

## The identity finding: the shipped serve does not reproduce across restarts

Same binary, same config, same fixed protocol, cold restarts on 2026-10-14 produced **six distinct stable warmed
streams** (each stable over 7-16 consecutive requests inside its process; the bodies differ from token ~90 on, a
near-tie list flip):

| run | notes | warmed stream |
| --- | --- | --- |
| base / new2 / off3 | shipped config | C (565 chars) |
| off4 | shipped config | Y (572) |
| gt1 / gt2 / gt3 | instrumented | D (571) / A (571) / Y |
| instr-off | shipped config, no bench first | B (548) |

The startup expert-cache fill varies with desktop VRAM (auto: 11729 / 11733 / 11745 experts observed) and the
adaptive tier's warm-up swaps land on a timing-dependent schedule; both move experts between the VRAM cache and
the CPU pool, whose last bits differ.  The pre-change engine fails the cross-restart byte gate the same way, so
the gate as written is not defined on the shipped serve.

**Identity protocol for this lane:** the identity arm is the serve config + `--adapt-every 0` (adaptive tier
off; `work/strata/strata-q6k-176b-pin.json`), same fixed protocol, full-body compare of the warmed requests.  It
reproduces across restarts: three runs of the instrumented binary (auto fill 11745, and fixed 11000 slots) and
one run of the pre-change binary (exe-swap config `strata-q6k-176b-pin-ref.json` + `/tmp/opencode/scion-moe-ref`)
all produced state E (558 chars), all 8 bodies byte-identical.  Every subsequent change is checked there; the
shipped config is used for speed arms and reports its own (lottery-selected) warmed stream.

## Lane instruments

- `serving/moe-exact.sh <tag>` — the warmed full-body capture (8 requests of the fixed prompt; compares g4-g8 to
  the stored reference in `/tmp/opencode/scion-moe-exact`).
- `serving/moe-cycle.sh <tag>` — cold-restart + bench + exact; the standard shipped-config cycle.
- `serving/moe-pin-cycle.sh <tag>` — the identity arm in a serve-down window (pinned config on 8081; restores the
  service afterwards).

## Step 2 — the GPU-reach wait, first pass (2026-10-14; probes only, no engine change)

The wait (22.1-22.7 ms/window) minus stage 0's replay span (16.0-17.0) leaves ~6 ms/window of
hand-off/doorbell exposure inside the serial stage chain.  The pipeline classes line shows where the time
sits: fresh windows ~39.5 ms, speculative ~20.6 ms, and 43-48 of 55 windows run speculatively with 22-33
rolled back.

- **Gate probe, dead end.** `STRATA_PIPELINE_THETA=0.1` (needs `STRATA_PIPELINE_DEBUG=1`; 0.2 is the
  default) moves ~4 more windows to speculative and ~4 more to rollback; the warmed wall is flat (medians
  30.8 vs 31.0 ms/window over 5-6 warmed requests each).  The config stays at 0.2.
- **ISA lens on the biggest in-graph item** — the expert GEMM, 4.06 ms/window by the profile stamps.  The
  `native_down_multi_kernel<143>` inner loop issues 4 x `global_load_b128` back to back with pipelined
  `s_waitcnt vmcnt(8->7->6...)`, not the per-load serialization the Bonsai lane fixed.  Its effective rate
  is ~190 GB/s on the scattered expert blobs read per window, so the next measurement is a grouped
  micro-bench at the served shapes before any rewrite (scatter/shape vs load latency).
- **A real, small lens win exists on gfx11:** the register-staged `gr_norm_split_fast_kernel` ("loads issued
  together, write once from registers", the vendor's gfx12 default) is compiled on gfx1100 and selectable
  with `STRATA_GR_NORM_FAST=1`.  Measured with `STRATA_VERIFY_PROFILE=1`: hc0 norm 0.41 -> 0.37, hc1 norm
  0.20 -> 0.16 (stage 0 GDN), 0.17/0.17 -> 0.13/0.13 (stage 1), QSA norms 0.06 -> 0.05 — about 0.2-0.3
  ms/window across the stages.  Below the wall's run-to-run spread; recorded as an opt-in candidate, not a
  default change (the same pattern in `gr_up_multi_kernel` has no gfx11 fast variant yet).
- **`STRATA_VERIFY_DEVICE_PLAN` is unavailable under `--pipeline-windows`** by design: the stage verifiers
  are set to `always_publish`, which turns the device plan off (the device residency table may lag the
  host's while windows are in flight).

## Step 3 — Scion-35B-A3B bring-up (commit `3d34f904`): it runs

The released file is **v1** — `Scion-35B-A3B-PQ2_0-corr.gguf` (11.337 GB, sha `4e9e3292…`), now fetched
from `SkyIsNotGreen/Scion-35B-A3B` and verified, at
`/home/penis/models/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf`.  Its census: `qwen35moe`, 40 layers,
256 experts top-8, hidden 2048; **expert banks are ggml type 142** (PQ2_0, 34-byte/128 blocks); **the
embedded corrections are type 43** (the same block) plus F16 router pairs; no MTP block in the file.

What landed:

1. **The type-142 expert path** (`iq_kernels.cu`): `block_pq2_0`, `vec_dot_pq2_0_q8_1` (the fork's
   per-chunk order), `Fmt<142>`, `Split<142>` (the whole 128-block per call, as 143), the GU/DOWN
   format lists, `dq_pq2_0`, `is_iq`/`iq_row_bytes`, and `block_geometry` for 142.  The architecture
   guard also stopped requiring `qwen35moe.nextn_predict_layers` (a file without an in-file MTP block
   now simply has none).
2. **Type 43** as the same 34-byte block (sizes/names only; the correct branch decode still needs the
   fork's `g128` rule before the corrections can be applied).
3. **The layout check** learned the corrections as optional tensors (base qwen35moe files do not carry
   them).
4. **A warm-up overrun fix**: the load-time warm-up processed a full prompt chunk (512) into per-layer
   K/V caches sized by `--max-context` and then two more passes; at ctx 256 that walks past the caches
   (found by bisecting the GPU page fault down to `k_kv_f16` at P=512 into a 256-cell cache).  The
   three passes now fit `ctx` together.

Verified: the release loads (973 tensors, 4.32 GiB dense + 7.97 GiB experts on one card) and
benchmarks **pp8 346 tok/s / tg8 102 tok/s** at ctx 256/512/1024 (single card, uncorrected, no MTP).
`tests/pq2_0_expert_parity` (new, real blobs from the file) checks the GPU dot against a q8_1-rounded
float reference: mean |gpu-ref| / mean |ref| **0.2%**, max 2% (intermediate-rounding boundary cases).

Still open for the 35B lane: (a) applying the type-43/F16 corrections in q36 (the release's quality
is the corrected one); (b) the MTP drafter — the released v1 file has no `nextn` block, so the dev
sidecars are not part of it; (c) the two-card arm; (d) then steps 1-2's instruments and speed work on
this model.

## Step 4 — the 35B corrections (engine `dd673041`, 2026-10-14)

**What the container is.**  The export (`scion/moe/export_branches_lora.py`) packs the deployed factors
as rank-512 LoRA pairs on the three output projections plus rank-256 pairs on `ffn_gate_inp`
(`adapter.lora.alpha = 0.0` -> the fork applies scale 1.0); `--branch-quant g128` packs them as
`Q1_0_g128`: per 128 weights one fp16 scale and 32 bytes of 2-bit codes, value = `d * (code - 1)`.
The fork imports that legacy type 43 **as PQ2_0** (commit `96222cf9a`: "byte-identical ... codes
{0,1,2} -> {-1,0,+1}") and runs the factors as ordinary `mul_mat`s: g128 factors therefore multiply a
**q8_1** activation (MMVQ/MMQ), the f16 router pairs a **f16** rounding (cuBLAS path).  The apply
places (fork `moe-corr-runtime`): `build_lora_mm` adds `B (A x)` to `ssm_out.weight` / `attn_output.weight`
from the gated pre-projection input, `build_lora_branch` adds it to the MoE block output from the
normalized FFN input, and the router delta rides the ordinary `mul_mat(gate_inp, cur)` into the
logits **before** top-k.

**What landed** (single change to `src/qwen35/q36_engine.cu` + one MMVQ type registration): the loader
takes the optional `.lora_` tensors (type 43 remapped to 142, the fork's own rule); `corr_branch` runs
A on the q8_1 image of x through `iq_mmvq(142)` and B on the A pass's own q8_1 image, `k_corr_add`
adds it; the router pairs run through the repo's f16 round-trip rule (`lora_corr_apply`, the 176B's
validated kernel); `k_moe_out` takes the branch as one more addend; base files run exactly as before.
`STRATA_Q36_NO_CORR=1` is the base-model arm.

**Parity** (`g128_corr_parity`, real factors from the release): the block decode against a host
reference on the q8_1 two-pass chain, A / B mean |gpu - ref| / mean |ref| = **1.5e-7 / 7.0e-8**
(ffn_moe_out), **2.0e-7 / 9.3e-8** (ssm_out), **1.9e-7 / 5.4e-8** (attn_output); max 2.8e-6 (f32
summation order only).

**Oracle vs the fork** (warmed greedy, 10 parity prompts x 64 tokens, same ids):

| arm | identical streams | tokens equal |
| --- | ---: | ---: |
| corrected fork vs corrected q36 | 6/10 | 512/640 |
| uncorrected fork vs uncorrected q36 (the floor) | 6/10 | 470/640 |

Every first divergence of the corrected pair sits inside the engines' own logit noise: the fork's
margin is 0.02-0.26 with both candidates' logits agreeing to < 0.1 on three of four; the fourth
(p04) is inherited — the *uncorrected* pair already diverges 10 tokens earlier on that prompt, the
corrections move both engines *further along* the same path.  With a fixed teacher-forced history the
corrections' logit deltas correlate **0.93-0.99** per step between the engines.  The two engines are
not bitwise siblings anywhere (q36's base-vs-fork floor is 6/10 streams on their own), so this is the
ceiling; the corrections land on the release's distribution.

**Cost** (one card, ctx 1024): corrected **pp8 293 / tg8 81.6 tok/s** against the part-1 uncorrected
**346 / 102** -> the corrections are ~**2.5 ms/token**, the step-6 target (the 176B's reshape is the
starting point, not the naive branch).

**The 176B gate:** the pinned identity arm (`--adapt-every 0`) on the new binary is byte-identical to
the part-1 state E on all 8 warmed bodies (558 chars, md5 `c1aa2407cb`), so `iq_kernels`' one-line
type-list change and the q36 additions move nothing in the Flash-Next serve.

**Tool gotcha (reusable).**  `tools/qwen35/q36_llama_dump` (built against a llama.cpp checkout) does
**not** apply a release's embedded adapter by itself: the raw API only attaches `model.internal_lora`
when `llama_set_adapters_lora(ctx, nullptr, 0, nullptr)` is called (the common CLI does it inside
`common_set_adapter_lora`).  Without it the fork silently runs the uncorrected model — the first
oracle run compared corrected q36 against *uncorrected* fork and "failed" 0/3.  The tool now calls it;
`Q36_DUMP_NO_ADAPTER=1` is the base arm.

## Step 5 — the 35B serve lane (2026-10-14)

- **Config** `strata-35b-a3b.json` (gitignored, like the 176B configs): `build-hip/strata-q36`
  `--model Scion-35B-A3B-PQ2_0-corr.gguf --max-context 4096 --chunk 512 --vram-reserve-mib 3072`,
  `gpu [0]` (single card; q36 ignores the reserve flag - it is the policy's global value, and with the
  display on card1 the 35B touches no display card), tokenizer extracted from the GGUF into
  `pack-35b/tokenizer` (`tools/strata_tokenizer.py --check`: 16/16 round trips).
- **Unit** `scion-35b.service` (port **8082**); ready ~9 s after start.  The two services cannot run
  at once (the 176B uses both cards); every 35B window stops `scion-flashnext` first.
- **Protocol**: `serving/bench_strata.sh <label> [port]` gained the port argument (the `moe-exact.sh`
  pattern); `moe-exact.sh` already took it.  Fixed protocol, warmed: **prompt 328-329 tok/s, gen
  81.8-82.2 tok/s** (runs 2-4), no drafter (v1 has no MTP block).
- **Identity stash** `/tmp/opencode/scion-moe-exact/35b-base-g1..8.json`: all 8 warmed bodies
  byte-identical (735 chars, md5 `9cce9e259b`), and a cold restart reproduces them exactly — the 35B
  lane has a *deterministic* warmed stream where the 176B's shipped config has the cold-start
  lottery.  Every later 35B change is checked with `moe-exact.sh <tag> 8082 35b-base`.

## Step 6 — the 35B decode budget and the corrections' first lens (engine `e582e92f`, 2026-10-14)

**Instrument.** `STRATA_Q36_DECODE_TIME=1`: events around each warmed T=1 replay (the headline) plus, in
the eager arm (`--no-graph`), a per-layer class split - mixer / out-projection correction / router+topk /
routed experts / shared+combine - a 64-step window average on stderr.  Opt-in; unset = no kernel, node or
stream change (the identity arm confirms).

**Budget** (one card, ctx 4096, 192 steps):

| item | corrected | uncorrected | corrections |
| --- | ---: | ---: | ---: |
| replay, T=1 (graph on) | **12.22** | 9.83 | **2.39** |
| eager total | 14.78 | 12.07 | 2.71 |
| mixer | 3.62 | 3.60 | 0.02 |
| out-corr | 1.12 | 0.16 | 0.96 |
| router+topk | 2.07 | 1.19 | 0.88 |
| experts | 3.71 | 3.69 | 0.02 |
| shared+combine | 2.71 | 1.90 | 0.81 |

The eager arm pays ~2.6 ms/step of launch overhead the graphs hide (the class split needs eager, the same
trade as the 176B's profile arm).  The corrections are 120 branches/token (40 out + 40 router + 40 MoE) x
~5 nodes for ~5 us of math each: **node-bound, not work-bound**.  The two uncorrected-heavy classes
(mixer 3.62, experts 3.71) are the next lens targets after the corrections.

**The lens.**  Shipped in `e582e92f`:

- the f16 router pair at T=1: the rounding and the A pass in one launch (a shared x16 row rounded once per
  block, then `lora_a_kernel`'s exact quarter split) and B in one, replacing round+A+B+memset+add;
- the g128 branches fold the trailing f32 add into B (`iq_corr_g128_b`), one node fewer on the out paths;
- both bitwise the staged chains (the parity test now checks the fused B against `iq_mmvq(142)` + the add).

Measured: eager router+topk 2.07 -> 1.74, out-corr 1.12 -> 0.96, total 14.78 -> 14.32; replay
12.22 -> **11.92** (-0.30 ms, -2.5%); fixed protocol 81.8-82.2 -> **84.3-84.4 tok/s**; warmed bodies
8/8 byte-identical (md5 `9cce9e259b`), cross-restart.

**Dead end, recorded.**  Fusing the q8_1 *quantizer* into the A/B dot kernels (a shared image built per
block) loses: every block re-quantizes the whole activation (128 blocks for the ssm A), about 3x the dot
work per row - out-corr 1.12 -> 1.37, replay 12.22 -> 12.45.  Reverted; the staged `native_quantize_q8_1`
launches stay.

**Identity note (176B).**  The pinned arm drew 550 chars once on the new binary - the same state as part 1's
`pin2`; fresh reruns of *both* the new binary and the pre-change ref binary are byte-identical to state E
(558, md5 `c1aa2407cb`).  The pinned arm has a rare second attractor; a single draw is not evidence.

## Step 7 — single-entry expert groups + the MoE branch's staged quantize (engine `5ea747fe`, 2026-10-14)

**Finer budget** (class split now 9 wide; eager, 192 steps, corrected / uncorrected, ms/step):

| item | corrected | uncorrected |
| --- | ---: | ---: |
| mix-qkv | 1.89 | 1.88 |
| mix-core | 1.14 | 1.13 |
| mix-out | 0.92 | 0.91 |
| out-corr | 0.96 | 0.16 |
| router+topk | 1.72 | 1.18 |
| exp-setup | 0.87 | 0.86 |
| experts (before the fix: 3.20) | 2.32 | 3.12 |
| shared | 0.98 | 0.98 |
| moe+combine | 1.77 | 1.08 |

The routed experts dominated the untouched work: 80 us/layer, ~82 GB/s on the ~6.6 MB/layer the 35B
reads - and `exp-setup` (two tiny nodes) another 22 us/layer.

**Win 1 — single-entry groups.**  The grouped kernels' multi path decodes each 128-value block into
registers once per row and reuses it across entries; a single token's top-8 of 256 are all distinct, so
every group has one entry and the decode is pure overhead.  `native_expert_grouped` gained
`single_entry` (default false, other callers untouched) and q36 passes `T == 1`, selecting the
per-entry kernels (bitwise the same values, `iq_multi_parity`'s contract).  Replay **11.91 -> 10.95**;
eager experts **3.20 -> 2.32**; fixed protocol **84.3-84.4 -> 91.9 tok/s**.

**Win 2 — the MoE branch re-quantized h.**  The native expert path had just staged `q8_1(h)` in `xq`
for the shared expert; the MoE correction branch read the same h and quantized it again.  It now reuses
those blocks (`corr_branch_staged`): replay **-> 10.83-10.90**, eager moe+combine **1.92 -> 1.77**,
fixed protocol **-> 92.9-93.0 tok/s**.

Identity after both: 8/8 warmed bodies byte-identical (md5 `9cce9e259b`); prompt tps unchanged (329).
176B: the pin arm drew the 550-char state once and state E the next run - both predate this session's
q36 changes (part 1's `pin2` drew 550; the pre-change ref binary draws E).  It is the arm's startup
cache lottery; two fresh runs on the new binary drew E.

## Step 8 — the corrections' A pass carries its own q8_1 image (engine `bc4c7002`, 2026-10-14)

Each g128 branch paid a separate quantize of the A pass's output between A and B (120 branches/token,
node-bound).  `corr_g128_aq` computes 32 rows per block (one warp each, mmvq's exact mapping), stages
the 32 outputs in shared, and one warp turns them into one q8_1 block with the standalone quantizer's
own XOR reduction (lane `l` holds row `32 b + l`), so the blocks are bitwise `quantize_q8_1_rows` of
A's output - `g128_corr_parity` checks it against the staged chain, bitwise.  One row now runs
**quant x, A+q, B+add** (three nodes); the batch path keeps the staged chain.

Replay **10.83-10.90 -> 10.69-10.76 ms/step**; eager out-corr **0.96 -> 0.88**, moe+combine
**1.77 -> 1.67**; fixed protocol **92.9-93.0 -> 94.1-94.2 tok/s**.  Warmed bodies 8/8 byte-identical
(md5 `9cce9e259b`), prompt unchanged (330).

The 35B decode has gone 12.22 ms/step (81.8 tok/s) to **10.72** (94.1) across steps 6-8: corrections
2.39 -> ~1.7 (their nodes: out 4->3, router 3->2, MoE 4->3 at one row), experts 3.2 -> 2.3, and the
setup/quantize nodes the wins left behind.

## Step 9 — the expert setup in one node + the mixer staged by k_moe_out (engine `beec6f23`, 2026-10-14)

Two node-bound fusions, both bitwise:

- **`k_group_quant`:** the expert setup's group-table build (one 256-thread block) and the q8_1 image of h
  (`native_quantize_q8_1_kernel`'s blocks) are independent, so they run as one launch - block 0 groups,
  blocks 1.. quantize with that kernel's exact mapping and warp reduction.  Eager exp-setup 0.87 -> 0.71.
- **`k_moe_out` stages the next layer's mixer activation:** it writes `hout` anyway; the q8_1 image of the
  same value is now staged in the same loop (lane `l` holds element `l` of each block, as the standalone
  quantizer's warp does).  `layer()` carries a `bool& xq_staged` through `forward()`: layer 0 and the
  first row after a prompt still prep, the other 39 layers' mixer quantize is gone (T <= 8 only).
  Eager mix-qkv 1.89 -> 1.75, the moved work shows in moe+combine.

Replay **10.69-10.76 -> 10.58-10.65 ms/step**; fixed protocol **94.1-94.2 -> 95.2 tok/s**; identity
8/8 (md5 `9cce9e259b`); prompt unchanged.  The 35B decode is now 81.8 -> 95.2 tok/s since step 6.

## Step 10 — the out-projection correction reuses mm()'s staged q8_1 (engine `0e253d11`, 2026-10-14)

`mm(L.ssm_out/wo, x, y, T)` preps `q8_1(x)` into `q8` for its own MMVQ, and the out-projection
correction reads that same `x`.  At T <= 8 it now takes the staged image instead of quantizing x
again: eager out-corr 0.89 -> 0.74, replay 10.58-10.65 -> 10.50-10.52, fixed protocol 95.2 -> 96.0.

## Step 11 — the router and MoE corrections in one launch each (engine `9244bf4a`, 2026-10-14)

The router delta (f16) and the MoE delta (g128) are independent at one row, so each phase is
now a single launch:

- the h quantize moved into `k_add_rms` (which writes that h; same warp-per-32-block mapping as
  `k_moe_out`, so the blocks are bitwise the standalone quantizer) - the expert setup is the group table
  alone;
- `iq_corr_mixed_a` runs the g128 A+q pass and the f16 router A pass in one kernel;
  `iq_corr_mixed_b` the g128 B pass and the f16 router B pass (`+=` into the logits row).  `f16_dot8`
  moved to a shared header.  `g128_corr_parity` checks the pair against the four standalone calls,
  bitwise, on the real f16 router factors (all three placements pass).

Replay **10.50-10.52 -> 10.17-10.24 ms/step**; eager moe+combine 1.77 -> 1.27, total 14.25 -> 13.91;
fixed protocol **96.0 -> 98.7 tok/s**; identity 8/8 (md5 `9cce9e259b`).  The 35B decode is 81.8 -> 98.7
tok/s across steps 6-11.

## Step 12 — the mixer pairs (engine `92e41681`, 2026-10-14)

The decode applies two Q8_0 matrices to the same staged activation per mixer: `qkv`+`zg` on the 30 GDN
layers, `wk`+`wv` on the 10 attention layers.  `native_mmvq_pair1` runs each pair in one launch - the
packed persistent kernel when both matrices are registered, the small one-per-row kernel otherwise
(mixed paths return false and the caller falls back) - each row the single call's own body, so the pair
is bitwise.  Eager mix-qkv 1.74 -> 1.61; replay 10.17-10.24 -> 10.08-10.12; fixed protocol
98.7 -> 99.7 tok/s.

## Step 13 — the shared expert's pair + the 12-class split (engine `34ce314c`, 2026-10-14)

`mm_pair` also covers the shared expert's gate/up.  The class split now separates the GDN core and the
shared tail (12 classes), and the verdict is that **no core kernel dominates**:

| item | ms/step (eager, 192 steps) |
| --- | ---: |
| mix-qkv | 1.60 |
| mix-ab / mix-conv / mix-rec (GDN core) | 0.50 / 0.51 / 0.48 |
| mix-out | 0.93 |
| out-corr / router+topk / moe+combine (corrections, with topk/combine) | 0.75 / 1.94 / 1.27 |
| exp-setup / experts | 0.71 / 2.32 |
| shared / sh-down | 0.46 / 0.56 |

Each core item is a couple of launches of a few microseconds: the part-1 serialized-load lens has no
target here worth its risk, and that is recorded rather than chased.

Replay **9.95-9.97 ms/step**; fixed protocol **101.1-101.2 tok/s**.  The uncorrected floor on this build
is **8.73 ms/step (~114 tok/s)**, so ~1.2 ms of correction cost remains (out-corr 0.75 the largest single
item).  Identity 8/8 (md5 `9cce9e259b`).  The 35B decode is 81.8 -> 101.2 tok/s across steps 6-13.

## Step 14 — the grouped experts' down phase: 32/nb rows per warp (engine `e348c540`, 2026-10-14)

The queued micro-bench answered "scatter vs latency" and named the wall: at the served 176B shapes
(G=28, real PTQ1_0 blobs) the **down phase ran at 55-90 GB/s** while gate/up ran at 205-305.  A down
row is only nb = n_ff/128 blocks (5 for the 176B, 4 for the 35B), so the one-row mapping's warp XOR
dot used nb of 32 lanes and re-decoded the block for every row and entry.

`native_down_rows_kernel`: **Q = 32/nb rows per warp** - lane (q, p) decodes row q's block p once,
applies it to every entry of the group, and the per-row sum reproduces the same 32-lane XOR tree with
only nb active lanes via masked shuffles (lane p adds lane p ^ off when p ^ off < nb, else +0).
Bitwise the one-row mapping by construction; selected for the qk-128 formats (142/143) where nb is
2..16, `STRATA_IQ_DOWN_ROWS=0` keeps the old path (the bench toggles it).

Bench (G=28, the 176B release's blobs): down **112 -> 19.8 us at T=1** (90 -> 512 GB/s), 131 -> 22.5
at T=2, 182 -> 29.5 at T=4; **outputs bitwise equal in every shape**.  Served, fixed protocol:

| arm | before | after step 14 |
| --- | ---: | ---: |
| 176B generation (two cards) | 76-78 tok/s | **80.5-80.8** |
| 35B generation (one card) | 101.1-101.2 tok/s | **108.4-108.6** |
| 35B prompt read | 330-331 tok/s | **415-417** |
| 35B replay | 9.96 ms/step | **9.29** (eager experts 2.31 -> 1.63) |

Identity: 176B pinned arm byte-identical (state E); 35B warmed bodies 8/8 (md5 `9cce9e259b`).  Both
model cards updated: 35B HF `a5e8e76b`, 176B HF `eb8f9768`.  Next expert target: the **gate/up phase**
(nb = 20/16 keeps Q at 1-2; the same warp-utilization analysis applies differently).

## Step 15 — the gate/up phase's rows kernel where Q >= 2 (engine `1e2b47f2`, 2026-10-14)

The down phase's rows mapping carried to the gate/up row space: `native_gu_rows_kernel` puts
Q = 32/nb rows per warp (the 35B's nb = 16 gives Q = 2; the 176B's nb = 20 gives Q = 1 and keeps its
multi path exactly - the branch is compile-time inert there).  Same masked-shuffle XOR-tree contract,
bitwise by construction.  The bench gained an expert-geometry argument and now builds blobs from
`native_expert_layout` instead of the CPU pool's `native_fmt` (no PQ2_0 entry; segfaulted on the 35B
file).

Bench at the 35B shapes: gu **30.9 -> 15.7 us** at G=8 T=1 (144 -> 283 GB/s), **492.7 -> 157.1** at
G=64 T=8; outputs bitwise equal everywhere.  Served:

| arm | step 14 | step 15 |
| --- | ---: | ---: |
| 35B generation | 108.4-108.6 tok/s | **109.7-109.8** |
| 35B prompt read | 415-417 tok/s | **480-484** |
| 35B replay | 9.29 ms/step | **9.17** (eager experts 1.63 -> 1.54) |

Identity 8/8 (md5 `9cce9e259b`); the 176B's gu branch is inert and its fixed protocol stays in band
(77-81 tok/s across the restart warm-up).  The 35B card was refreshed to these numbers.

## Steps 16-17 - the node-latency ablations and two launches fused (engines `5d76fd7f`, `0ecfc0ca`, 2026-10-15)

The 14-class eager split named the corrections and the router cluster, but the eager spans carry launch
gaps.  The engine gained measurement-only ablation flags (`STRATA_Q36_SKIP_XA / XB / MA / MB / TK / RT`)
that drop one node each - or swap top-k for a trivial ids fill - so the graph replay's difference to the
full arm is that node's own cost.  On the 35B (`--tokens`, 192 greedy steps):

| node dropped | step (ms) | its cost (ms) |
| --- | ---: | ---: |
| - (full) | 9.19 | - |
| out-corr A | 8.94 | 0.25 |
| out-corr B | 8.86 | 0.33 |
| mixed pair A | 8.90 | 0.29 |
| mixed pair B | 8.87 | 0.32 |
| top-k (vs trivial fill) | 8.95 | 0.24 |
| router f32 product | 8.89 | 0.30 |

Every node costs 6-8 us: the tiny kernels are node-latency bound, not compute bound, so the lever is
*fusion* - two independent kernels in one launch with their own grid regions (the pair idiom at the node
level).  Two landed:

- **`native_mmvq_corr_aq`** (step 16): the out projection's MMVQ rows and the correction's A pass in one
  launch.  Both read the same staged q8_1 image; the MMVQ part is one warp per row (lane 0 stores), the A
  part is `corr_g128_aq`'s body unchanged.  Step **9.19 -> 9.02 ms**; served 109.7-109.8 -> **110.9-111.4 tok/s**.
- **`native_gemv_corr_mixed_a`** (step 17): the router's f32 GEMV rows and the mixed A pass in one launch.
  Step **9.02 -> 8.87 ms**; served 110.9-111.4 -> **113.6-113.7 tok/s** (prompt 480.4-483.5).

Identity held through both (the five warmed 128-token bodies of `moe-exact.sh` byte-identical; step 17
re-verified after the revert below).

### Tried and reverted: the out-corr B inside the residual add + norm (step 18, negative)

The B pass's per-row value is `fadd(fadd(p0, p2), fadd(p1, p3))` (its `warp_sum` tree over four live
lanes), which one block can reproduce exactly, so `corr_b_add_rms` folded B into k_add_rms' add/norm/
staging node.  First mapping (eight rows per warp, four lanes a row): exactly break-even, 8.86 vs 8.87 -
the node saved (~6 us) equals the single-block work added, because the standalone B kernel's work is
spread over 512 blocks.  A rewrite to one lane per row (two rounds, no shuffles) regressed badly (23.78
ms/step, the eager class +0.75 ms): a single block on strided 34-byte weight loads is latency bound.
Reverted; the knot wants a cooperative launch or an order-free formulation.

## Steps 19-20 - two more launches fused (engines `0fa55459`, `25d621b1`, 2026-10-15)

**Step 19, alpha/beta + conv** (`0fa55459`): the GDN layer's `k_gdn_ab` (a row's float4 dot and XOR tree)
and `k_conv_seq` (unchanged body) read different inputs and write different buffers, so at one row they run
as one launch, `k_gdn_ab_conv` (128-thread blocks, four warps an alpha/beta block; T > 1 keeps the two
launches).  Graph 8.85 -> 8.84 ms; served 113.5-113.6 -> **115.2-115.3 tok/s**.

**Step 20, the router tail** (`25d621b1`): mixed B's two halves, top-k and the group tables are a chain of
three nodes (mixedB -> topk -> setup) with small works.  At one row they run as one launch: the MoE output
delta's rows first (the g128 B pass's own `warp_sum`, 32 rows a block), then the launch's last block runs the
router f16 delta's rows (`f16_dot8` over 32 lanes, a warp a row), top-k and the group tables in order with
barriers between.  The top-k body moved to `include/strata/kernels/router_topk_body.hpp`, shared by
`native_router.cu`'s `route_topk` kernel ((256,8) and (512,10) both call it) and the fused tail; the group
table kernel moved from q36 to `iq_kernels.cu` (`native_group_quant_kernel`, its block-0 body in
`group_tables`) next to the expert launches that read it.  Graph **8.84 -> 8.69 ms**; served 115.2-115.3 ->
**116.0 tok/s** (prompt 481-482).  The 176B's pin arm re-verified after the router refactor (state E, md5
`c1aa2407cb`); the 35B's warmed bodies 8/8 byte-identical.

The eager split shows the moves: corr-mix 0.47 -> 0.16, topk 0.52 -> 1.34 (the tail carries the deltas, the
picks and the tables), exp-setup 0.72 -> 0.16.

## Steps 21-22 - the delta rule stages its output; the down recomputes the SwiGLU (engines `73285ebd`, `3bd41a9a`, 2026-10-15)

**Step 21, `k_gdn_seq` stages its own q8_1 image** (`73285ebd`): the delta rule's output row (ZV = h_v * 128)
is written by its rg == 0 threads, one column a lane, so the q8_1 image the out projection's MMVQ and the
correction's A pass read is the same warps' reduction - the staging runs at the end of the kernel (the
standalone quantizer's bits) and `mm()`'s prep is skipped for that projection (`mm_fused_corr_a`'s `staged`).
8.69 -> 8.62-8.64 ms; served 116.0 -> **117.1-117.2 tok/s**.

**Step 22, the down kernel recomputes the entry's SwiGLU** (`3bd41a9a`): the expert chain's middle node
(`swiglu_q8_1_entries_kernel`) only fed the down kernel's rows, so the qk-128 rows kernel now computes the
entry's SwiGLU and its q8_1 image in shared itself - the product rounded on its own and `q8_1_store`'s warp
trees and finite rule, the same bits - and the separate pass is skipped.  The predicate is gated to
decode-shaped calls (<= 16 entries): a prompt's groups hold many entries and the recompute is per row-block,
where the separate pass is the cheaper shape (measured at T=29: prompt 59.2-59.7 ms fused-gated vs 60.7-61.1
fused-ungated vs 59.0-59.7 separate).  8.64 -> **8.54 ms**; served 117.1-117.2 -> **117.4-117.5 tok/s**
(prompt 479-482).  The 35B's greedy stream and warmed bodies are byte-identical and the 176B's pin arm
re-verified (state E, md5 `c1aa2407cb`) - its down kernel took the same change.

## Steps 23-24 - the attention output stages itself; the shared SwiGLU rides in its down (engines `37418f08`, `b4d04cb1`, 2026-10-15)

**Step 23** (`37418f08`): `k_attn_merge` writes each attention output row one value a lane, so it stages the
row's q8_1 image itself (the step-21 trick on the attention path) and the out projection's prep is skipped.
Same-build A/B (alternating 400-step runs): 117.3/117.0 vs 116.7/116.8 -> ~+0.35%.

**Step 24** (`b4d04cb1`): the shared expert ran three nodes at one row (the gate/up pair,
`native_swiglu_quantize_q8_1`, the down MMVQ); the down's small-K kernel now stages the SwiGLU and its q8_1
image in shared itself (`__fmul_rn(__fdividef(...))` and the warp trees, the standalone kernel's own bits),
so the middle node is gone.  `native_mmvq_swiglu_q8_0` covers type 8, one column, the small-K shape and
refuses the packed layout (the caller runs the two calls).  Same-build A/B: 117.5/117.7 vs 117.2/117.3 ->
~+0.3%.  Served 117.4-117.5 -> **118.0-118.2 tok/s** (prompt 480.4-482.8); the 176B's pin arm re-verified
after both (state E, md5 `c1aa2407cb`).

## Publication gate (updated 2026-10-14, both cards live)

Both models' results are on the live Hugging Face cards, as a **Performance** section near the top:

- `SkyIsNotGreen/Scion-35B-A3B`, revision `e5d120bb`: generation **118.0-118.2 tok/s** on one RX 7900 XT
  (warmed, greedy, 128 tokens per request), prompt read **480-482**, ~8.5 ms/token at ctx 4096, the
  determinism note (eight warmed requests byte-identical, cold restart the same).
- `SkyIsNotGreen/Scion-FlashNext-176B-A6B`, revision `eb8f9768`: generation **80.5-80.8 tok/s** on the
  two-card box (the same fixed protocol), ~12.4 ms/token at ctx 4096, 2.3-2.4 tokens per pipeline
  window (`--pipeline-windows 2`, +62-68% over the non-pipelined engine), corrections applied.

The local snapshots (`docs/RELEASE-35B-MODEL-CARD.md`, `docs/RELEASE-FLASHNEXT-176B-A6B-MODEL-CARD.md`)
are refreshed from the published revisions.  Both numbers rose once more with step 14's down-phase
kernel; the remaining engine work (the gate/up phase's warp utilization, topk/group merges, the core
kernels) can only firm them up.

## The two-card arm, scoped (2026-10-14; not implemented)

The 35B fits one card entirely (4.42 GiB dense + 7.97 GiB experts) and its decode is a serial
per-layer chain at T=1. What a second card could buy, and what each costs:

- **Layer split (the 176B's model).**  Layers 0..k on card 0, the rest on card 1, one 8 KB hidden
  hand-off per token.  A single conversation makes the cards run *alternately* (the 176B's measured
  story: layer parallelism is capacity, not throughput), and the 35B needs no capacity - a layer
  split would add hand-off latency for no work offload.  **A loss on this model**, and it would
  duplicate the 176B's split machinery (`--layer-split`, peer buffers, pipeline windows) in the q36
  driver.
- **Expert parallelism.**  The routed experts are 2.31 ms of the 9.99 ms step (~23%).  Splitting
  their rows needs per-expert dispatch (the router decides the experts per token), a PCIe round trip
  for each layer's 6.6 MB, and a cross-device combine; at T=1 that is latency-bound (~0.5-1 ms),
  more than the ~1.1 ms the ideal split could save.  A batched serve could amortize it - a different
  serving shape than this lane's fixed protocol.
- **Two independent contexts.**  Not one model on two cards: two `scion-35b` instances, one per
  card, serving two conversations.  No engine change; only useful with the 176B stopped (it owns
  both cards today).  The one configuration that pays today; out of scope here.

**Verdict:** for the 35B at T=1 the two-card arm is not worth its engine cost.  Revisit if the model
outgrows one card, if batched serving becomes the protocol, or if expert-parallel dispatch is wanted
for the 176B (where capacity *is* the constraint) and is built there first.

## Budget closes (2026-10-14)

`STRATA_Q36_DECODE_TIME` now also brackets `head_rows` (the output MMVQ + argmax): the eager budget
closes at **12 layer classes 12.03 + head+embed 0.71 ms**, the rest (1.87) being the eager arm's
per-boundary launch bubbles - what the graphs hide.  The head reads `output.weight` (248320 x 2048,
Q8_0, 540 MB) in 0.71 ms (~760 GB/s): the card's wall, nothing to win without changing the weights.
Every remaining item is engine work: the corrections' `out-corr` 0.75 (fusing A/B into the
out-projection kernel), `router+topk` 1.94 and `exp-setup` 0.71 (merging topk / the group table into
their neighbours), the expert kernel's ISA lens (2.31), the ~0.5 ms core kernels, and the two-card
arm above.

## Next steps (session queue)

The continuation for these was `NEXT-SESSION-SCION-MOE-SPEED-2.md` (state + the copy-paste prompt); all three
part-2 gaps, the 35B speed lens through step 13, and the 176B expert lens (step 14) are done.  The 35B is
**8.51 ms/step (118.1 tok/s)** and the 176B **80.5-80.8 tok/s**, both bitwise-verified.

1. ~~**35B corrections**~~ `dd673041`; ~~**35B serve lane**~~ Step 5; ~~**35B decode budget + first lens**~~
   `e582e92f`; ~~**experts + MoE-quantize**~~ `5ea747fe`; ~~**A+q fusion**~~ `bc4c7002`; ~~**setup + mixer
   staging**~~ `beec6f23`; ~~**q8_1 reuses**~~ `0e253d11`; ~~**mixed router+MoE pair**~~ `9244bf4a`;
   ~~**mixer pairs**~~ `92e41681`; ~~**shared pair + 12-class split**~~ `34ce314c`;
   ~~**grouped expert micro-bench + down-phase rows kernel**~~ `e348c540`.
2. ~~**The gate/up phase**~~ `1e2b47f2` (step 15: gu 30.9 -> 15.7 us at the 35B shapes, G=64/T=8
   492.7 -> 157.1; the 176B's nb = 20 keeps Q = 1 and its old path); ~~**the correction nodes' costs +
   the mmvq/A and router/mixed-A fusions**~~ `5d76fd7f`, `0ecfc0ca` (steps 16-17, 9.19 -> 8.87 ms).
3. ~~**35B, remaining**~~ `0fa55459`, `25d621b1` (steps 19-20: alpha/beta + conv at one row; the
   router tail - the deltas, top-k and the tables in one launch, 8.84 -> 8.69 ms, served 116.0 tok/s).
   What remains: the **out-corr B** (break-even fused into add+rms; wants a cooperative launch or an
   order-free formulation), the attention layers' `wq` + `wk/wv` pair (a triple launch, one node a layer),
   `k_moe_out`'s staging vs the next layer's first pair (grid-wide dependency), and the ISA lens on the core
   splits (ab/conv/rec ~0.5 ms each eager, no dominant kernel).
4. The two-card arm for q36: scoped and ruled out for the 35B at T=1 (see the scope section).
5. The register-staged lens on `gr_up_multi_kernel` (gfx11 has no fast variant) and on the
   `native_quantize_q8_1` / `lora_round` small kernels that appear 3x per layer in the served graph.
6. A per-window timeline split (launch -> first ring vs last ring -> finish) to bound the ~6 ms hand-off
   exposure without adding kernel time.

