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

## The k=1 MTP drafter in q36 (engine `ed3188d5`, 2026-10-15; opt-in, off in the serve)

The recipe's drafter lane (SCION-RECIPE.md 7) reaches the Strata 35B lane: the k=1 sidecar
(`mtp-sidecar-v2-cur05.gguf` in `work/hivebench/models/gguf`: `mtp.fc1`/`mtp.fc2` f16, hidden 2048, width
4096, source `qwen35-release-v2-cur05-soup`, acceptance 0.486 fineweb / 0.359 wikitext teacher-forced) now
loads with `--mtp-sidecar` and drafts inside `spec_step` (D = 1; the head's inputs are `hlast` and the decided
token's embedding, its output goes through the target's own `output_norm`/head; no redo rows - it recomputes
each step).  The verify/accept/commit machinery is the in-file MTP block's own, so the emitted text stays the
plain greedy text - measured byte-identical on the served fixed protocol.

**The verify windows surfaced a latent bug** (fixed in the same commit): `corr_branch_staged`'s T > 1 arm
ignored `accumulate` and *replaced* the mixer output with the correction for every 2 <= T <= 8 row.  The
prompt path (T > kGemmT) uses `corr_add` and the decode path the T == 1 kernels, so nothing had exercised
that range; the forced-draft arm showed it (a forced-correct draft rejected, the stream diverging).

**The economics, measured.**  Served: drafts **33/94 = 35%** accepted, generation **118.0-118.2 -> 109.3
tok/s** (prompt unchanged, bodies byte-identical).  A window costs a 2-row verify (+~3 ms for the second row -
the MoE's own experts; the recipe measured the same +2.3-3.1 ms/row) plus the draft's head call (~0.7 ms) and
yields 1 + p tokens, so at this engine's 8.5 ms plain step break-even sits at **p ~ 0.46**.  The fork's
1.135x was on a 12.2 ms/step baseline, where the same marginal row is a smaller fraction.  Kept opt-in and
documented; it pays when the acceptance rises (on-policy retrain) or the verify row gets cheaper.

## The drafter retrained on the engine's own hiddens (2026-10-15; the acceptance is the whole ballgame)

Two engine-side levers were priced first and both came out empty, so the acceptance was the only one
left:

- **Merge the drafter's head call into the verify's.**  The head is a 540 MB read whatever the row
  count, so a draft's head looks free next to a 2-row verify's.  It is not: the verify window's row 1
  **is the draft's token**, so the draft must exist before the window is built.  Circular - dropped
  without a build.
- **Defer row 1's head until the draft is accepted** - the head is one launch over both rows, so the
  second row's head costs its FLOPs only (~0).  Nothing to save.
- **An abstain gate on a low-confidence draft.**  The confidence *is* the draft's logit margin, and
  getting it needs the head call that produces the draft.  Gated-out steps would then pay the drafter
  (0.85 ms) *and* the plain step: 9.35 ms for one token against 8.5.  Only pays if the drafter itself
  gets cheaper; dropped.
- **A cheaper verify row.**  The second row is +3 ms of serial recurrence (GDN) plus its own routed
  experts, measured, not a node-count artifact.  Irreducible without changing the target.

So: **the acceptance.**  The shipped drafter was fitted to llama.cpp's hiddens of the release model.  The
engine computes the same model its own way (PQ2_0 + the corrections), so at runtime it was reading a
slightly different h.  The engine now dumps its own training pairs, so the head can be fitted to them:

- `STRATA_Q36_MTP_PROBE=<file>` (engine, opt-in): one record per decode step - the hidden at pos - 1,
  the token the step processed, the token the engine's own head then chose, and pos.  That is
  `moe/mtp_release_train.py`'s convention exactly, so its loader reads the dump as it reads a probe dir.
  A pos jump marks a prompt, so windows are cut at request boundaries.  The plain decode path only
  (T = 1, `process(n == 1)`), warm-up excluded.
- `mtp_release_train.py` now **keeps the best held-out checkpoint**, not the last one (the train windows
  saturate at acceptance 1.0 long before the held-out number peaks), and the report says which step was
  exported.

**Round 1 (on-policy, fork probes, 51 windows x 512 from 32 served prompts).**  Held-out acceptance
**0.389 -> 0.498** (`head-op.pt`, 300 steps, warm start).  Teacher-forced on engine-native pairs the two
heads score 0.4105 (shipped) and 0.6315 (this one), and the runtime says the same thing:

| prompt (CLI, 400 tokens) | plain | shipped head | this head |
| --- | --- | --- | --- |
| how a compiler works | 117.1 | 113.9 (43%) | **132.4** (66%) |
| merge two sorted lists | 117.4 | 116.7 (47%) | **141.8** (79%) |
| TCP vs UDP | 117.5 | 111.7 (40%) | **127.7** (60%) |

**Served, the same thing with the chat template** (400 tokens per request, this box today):

| prompt | plain | drafts | this head |
| --- | --- | --- | --- |
| how a compiler works | 116.7 | 156/244 | **130.3** |
| how a database index works | 116.8 | 140/259 | **122.1** |
| how a packet is routed | 116.8 | 149/250 | **127.0** |
| noise-cancelling headphones | 117.0 | 129/270 | **117.4** |
| how GPS finds a position | 116.7 | 178/221 | **144.1** |
| how a refrigerator stays cold | 117.0 | 157/243 | **130.3** |
| *the card's fixed prompt* | **117.9** | 35/92 | *111.0* |

So the drafter is **+10% averaged over the six** (116.8 -> 128.5) and **-5.9% on the card's fixed
prompt**, which is the break-even story from the previous section landing where it predicted: the
served acceptance there is 38%, and p ~ 0.47 is the line (48% measured 117.4, i.e. even).  What decides
it is the *text*: formulaic technical prose is easy to draft, a factual paragraph is not.

**Text can also go degenerate, and then the drafter is a 24% loss.**  The oracle's `p00` prompt (a
chat-formatted story prompt) has the model emitting the template's `user` control token in a loop after
~600 tokens; from there the acceptance is **0% for both heads** - 84 accepted, then 400 consecutive
rejections - and 800 tokens cost 88.4 tok/s against plain's 116.7.  Plain decode degenerates too (the
same text), so this is the model's behaviour, not the engine's, but it is the tail risk an always-on
drafter would carry.

**Exactness, re-proved with the new head.**  `moe-exact.sh mtp-op1-8082 8082 35b-base`: all five warmed
bodies byte-identical to the plain reference (735 chars each).

**Round 2 (running).**  Engine-native pairs this time, and a wider mix: 87 served prompts (factual
paragraphs, technical explanations, code, step-by-step maths, lists, stories, dialogue, reviews,
structured output), 69,210 records, 81 windows - 70 train, and 11 windows from **11 held-out prompts**
for the eval.  Warm start from round 1, 600 steps at lr 2e-4.  The round-1 head scores **0.5339** on that
held-out set.  The open question is whether the weak styles (the card's prompt at 38%, the index prompt
at 54%) clear ~47%; if they do, the drafter is a win on every text measured and can go on the card.

## The drafter's inputs at runtime are the verify window's, not a plain step's (the T=1 pairs were the wrong data)

Round 2 finished at held-out **0.5690** and every served set went positive - technical **142.9 tok/s
(+21.4%, 77.8%)**, held-out 123.1 (+4.6%), held-out factual prose 125.9 (+7.0%, 56.4%) - and the five
warmed identity bodies stayed byte-identical.  But the card's fixed prompt stayed at **35/92 = 38%** for
rounds 1 and 2 alike, while the round-2 head had *memorized that exact prompt's* training window
(teacher-forced 0.9961).  Chasing that gap:

- The engine dumps the drafter's pairs, and the round-2 data came from the plain T = 1 path.  A window
  advances a + 1 positions, so at runtime the drafter is seeded from **row a's hidden and picks[a]** - and
  the window's hiddens come from the 2-row kernels, which are not the same numbers as a 1-row step's
  (compared on the same prompt: 0.43216 vs 0.40014 in the first element of position 39's hidden).
- The probe's first verify-path mapping was itself off by one row (dumped the rejected row instead of the
  kept one) - found because the acceptance computed from the dump disagreed with the engine's own
  counter; the direct acceptance now matches it to the record (69/128 = 0.5391 against the engine's
  70 of 129 = 54%; the pair right after the prompt is the one the dump cannot see).
- So the correct dataset is the verify path's pairs, held across a window and written when the next
  window reports its picks (engine `2b5d28d0`, `62735d6f`).  On the card prompt the heads' acceptance on
  those pairs reads **0.46 (round 1) / 0.54 (round 2)** - the 38% runtime number was the T=1-trained head
  meeting the T=2 hiddens, and the same number is exactly the lesson: fit the head to the path it reads.

**Round 3 (running).**  The verify-path dump over the same 87 prompts: 46,536 records, 77 train + 12
held-out windows, warm start from round 2, 600 steps at lr 2e-4.  The round-2 head's held-out acceptance
on the verify-path data is **0.5039**, which matches the served runtime on those prompts (53.1%) - so the
trainer's number now predicts the runtime, which is the whole point.

## The tail, fixed engine-side: --mtp-backoff (and the near-tie finding that came with it)

Two things the worst-case work turned up, both measured:

**The backoff.**  `--mtp-backoff <streak>[:<cap>]` (engine `27d8c29f`) pauses speculation for 16 steps
after `<streak>` consecutive rejections, then 32, 64 .. up to `<cap>` (128), and any acceptance ends the
streak and the growth.  On the degenerate prompt (800 tokens, op3): **89.0 -> 119.9 tok/s** against
plain's 116.4 - from -24% to +3% - with the offered windows falling 707 -> 494 (213 steps paused).
A paused step is the plain one-row path and returns -1 to its callers, so no draft is counted as
offered; the sidecar path only, since the in-file MTP block chains rows through its own calls.  Off by
default.

**The drafter is not bit-exact, and that had to go on the card.**  Chasing the degenerate prompt's
divergence with drafts *forced to always reject* - no accept/commit path runs at all, only 2-row verify
batches against 1-row plain steps - an 800-token generation still diverges from plain at token 61
(9047 against 32635).  The plain logits at that position read 19.92149 against 19.89257: a top-two margin
of **0.029**, next candidate 1.4 behind - a genuine near-tie flipped by the batched kernels' own
rounding, which is what llama.cpp's speculative decoding documents too.  Six free-running 400-token
generations first diverge at tokens **32, 41, 49, 64, 72 and 112**, so it is the normal case for long
drafted text, not an edge: one early flip cascades (60-90% of later tokens differ).  The card said "the
generated text is unchanged" when first published and now says otherwise (README `130ede91`,
`a5998bb0`); the card's own 128-token protocol does pass byte-for-byte, and the plain path is untouched.
Any future claim about the drafter has to carry this.

## Round 3, measured: the drafter pays on every ordinary text, and the card's prompt went 38% -> 74%

Round 3 finished at held-out **0.5077** (peak 0.5126 at step 100 - the best-checkpoint condition was
inverted and exported the last step, half a point down; fixed for the next run, the numbers stand).
Served, 400-token requests, this box, byte-identical bodies throughout (`moe-exact.sh` all five warmed):

| set | plain | round 1 | round 2 | **round 3** |
| --- | --- | --- | --- | --- |
| six technical prompts *(in the training mix)* | 117.3 | 128.4 (+9.5%) | 142.9 (+21.4%) | **153.5 (+30.8%, 94.1%)** |
| eleven held-out prompts | 117.2 | 119.0 (+1.6%) | 123.1 (+4.6%) | **121.5 (+3.7%, 53.2%)** |
| six held-out factual-paragraph prompts | 116.9 | 115.6 (-1.1%) | 125.9 (+7.0%) | **129.6 (+10.9%, 63.1%)** |
| **the card's fixed prompt** (bench protocol, 128 tokens) | **117.9** | 111.0 (38%) | 111.9 (38%) | **139.1-139.3 (74%)** |

The card's prompt is the one that matters most here: it sat at 38% for two rounds - including for a head
that had memorized that prompt's own window on the T = 1 pairs - and the verify-path data takes it to
**74%**, 139.3 tok/s against plain's 117.9 (**+18.1%**, and +24.5% over round 2's 111.9).  Its acceptance
by generation length: 83% at 64 tokens, 74% at 128, 71% at 256 and 400, 68% at 800.

Over the 17 never-trained prompts the drafter averages **124.4 tok/s against plain's 117.1 (+6.2%)**.

**The tail is still a loss.**  The oracle's `p00` prompt (the model looping on the template's `user`
control token) accepts 92/707 = **13%** and costs 89.0 tok/s against plain's 116.4 (-23.5%) - the drafter
does not follow a degenerate loop, and this is why it stays opt-in (`--mtp 1 --mtp-sidecar <head>`) with
the serve config plain.  A prompt that degenerates *after* ~600 tokens is the price; the alternative
(an abstain gate) was priced empty earlier because the decision needs the draft, hence the head call.

## Round 4, measured: the structured-output worst case was never trained (op4 <- op3, 2026-10-15)

The weak styles were not weak in isolation - they were absent from the mix.  The round-2/3 training set had
exactly three structured prompts (a markdown table, a JSON object, a YAML config) and all three sat in its
*held-out* set, so structured output had never been trained: op3 accepted 44.6-48.9% there, around or below
the ~46-47% break-even, and the structured rows of the served table were a wash (110-121 tok/s against plain
117.2-117.3).

Data: 38 fresh prompts weighted to structured styles (JSON, YAML, TOML, XML, CSV, markdown tables) plus code,
math, dialogue and degenerate-prone long prose, 1800 tokens each; 26 train / 12 held out; 45 train + 20
held-out windows (`probe5`).  Warm start op3, lr 1.5e-4, 600 steps.  On the new prose held-out op3 scores
0.5283 and the run's best was 0.5254 at step 100 - the best-checkpoint logic kept it over the final 0.5145
(the fix from round 3 doing its job).

Served A/B, one service, same day, 400-token protocol (plain rows the same day: 117.2-117.3):

| set | op3 | op4 |
| :--- | ---: | ---: |
| old held-out structured: markdown / JSON / YAML | 48.9 / 48.3 / 46.2% @ 118.8 / 118.7 / 117.0 | **57.7 / 61.3 / 50.0%** @ 117.0 / 124.9 / 118.5 |
| five fresh structured prompts | 45.6 / 48.7 / 50.8 / 48.3 / 38.1% @ 110.4-120.6 | **59.6 / 55.6 / 61.7 / 66.3 / 47.8%** @ 117.2-129.7 |
| the card's fixed prompt (128 tok, card protocol) | 140.2-140.3, 74.0% | **140.9-142.7, 78.9%** |
| six technical prompts | ~153, 94.1% | ~154, 93.4% |
| held-out factual 6 | 917/1480 = 62.0% | 901/1496 = 60.2% |

So round 4 fixes exactly what it aimed at - structured output is now well above break-even - and lifts the
card's prompt, but the factual prose gives back ~2 points: a structured-heavy round replayed away some prose
acceptance.  The numbers come from single 400-token runs per prompt whose texts already differ between heads
(one early near-tie flip cascades), so a 2-point gap is at the edge of what this protocol resolves; still, it
was consistent across five of six prompts, so it is treated as real.  Round 5 is the fix: a mixed replay
(`probe6` = probe4's 77 prose windows + probe5's 45 structured windows, held-out = 12 + 20), warm from op4,
lr 1e-4, 400 steps.  A head ships only if it beats op3 on every set; otherwise op4 is kept with the prose
caveat stated, since structured was the worst case.

## The degenerate loop made a win: `--mtp-period` (engine `5ee873c4`, 2026-10-15)

The worst case was never an acceptance problem - it was a control problem.  In the 800-token degenerate run
the output is one **period-4 stretch, tokens 398-800 (402 of 800 tokens)**, so the continuation is trivially
predictable exactly where the head is confidently wrong (13%).  The engine now supplies the draft itself:
when the last 2p emitted tokens confirm period p <= N, draft `token[i - p]` and skip the head call
entirely.  The verify is untouched, so the model's own pick still decides every emitted token, and a
period-p loop is self-similar, so a near-tie flip does not throw the rule off.

| run (CLI, op5 head) | plain | drafter | + `--mtp-period 4` |
| :--- | ---: | ---: | ---: |
| degenerate prompt, 800 tokens | 114.8 | 88.2 (-23%) | **132.3 (+15%)** |
| card prompt, 128 | - | 133.6 | 133.2 (identical windows, 0 firings) |
| technical, 400 | - | 125.1 | 125.7 (0 firings) |
| markdown table, 400 | - | 131.1 | 130.5 (0 firings) |

So the tail went from -24% to +15%: the same text that motivated the backoff is now a speedup, and the
backoff's own premise weakens there - its pauses give up the rule's wins (both flags: 124.3 against 132.3),
so with the rule on it should be relaxed or dropped.  The rule is the crudest possible *jev-controller*:
deterministic, free, and it changes only which token is drafted.  The learned version (a readout over the
hidden state and recent tokens) generalizes what the rule cannot: near-periodic text rather than strict
periods, firing before the 2p-token confirmation, and the transition in and out of the loop.  That is the
next control step, not the next training round - and it is measured by the same protocol as the rule.

## Round 6, measured: long-form coverage, and the loop trained away (op6 <- op5, 2026-10-15)

The long-form regime is where acceptance actually goes: round 5's 26 prompts at 1500 tokens averaged 0.542
(31-94%) against 68-76% at 128-800 tokens on the card prompt.  Round 6 collected two more long-form sets
(56 fresh prompts, 1500-2000 tokens, acceptance 21-66%) plus a 3000-token degenerate run, and trained on a
merged 237-window set: probe6's 122 (rounds 4-5) + 50 from the new collections + the degenerate stretches
oversampled x4 - the round-4/5 replay kept in the mix so prose cannot be replayed away again.  Warm from
op5, lr 1e-4, 400 steps; best checkpoint step 350 at merged held-out 0.5446 (op5's plateau ~0.533).

Served A/B, one service, same day:

| set | op5 | op6 |
| :--- | ---: | ---: |
| card prompt (128 tok) | 142.0, 76.4% | 141.5-142.3, 76.4% (equal) |
| structured old held-out | 57.7 / 61.9 / 49.4% | **59.0 / 66.7 / 51.1%** |
| five fresh structured | 58.3 / 56.9 / 62.6 / 66.3 / 48.9% | 59.0 / 59.4 / 59.9 / **70.5** / 50.0% |
| structured 8-prompt aggregate | 124.5, 57.4% | **126.7, 59.1%** |
| held-out factual 6 | 61.5% | 61.2% (equal) |
| technical 6 | 94.2% | 94.0% (equal) |
| held-out 11 | 54.9% | **55.6%** |
| 17-prompt aggregate | 125.7 (+7.3%) | **125.9 (+7.5%)** |
| round-4 prose 12 | 54.1% | **55.1%** |
| long-form held-out 7 (1500 tok) | 50.5%, 117.2 tok/s | 51.2%, 117.9 tok/s |

Headline: **the degenerate loop was trained away.**  The same 800-token p00 run the previous head lost 24%
on (88.9 tok/s, 13% acceptance) now measures **155.6 tok/s and 96% acceptance with no control flag at
all** - the head itself drafts the period-4 loop.  `--mtp-period 4` matches it (160.8, 95%) and remains the
free backstop for unseen text, as does the backoff.  So the control-vs-training question is answered
empirically: on a frozen hidden state, a handful of oversampled worst-case windows in training beat
controlling around them, and control survives as the safety net.

The one regression to note: the card-prompt acceptance-by-length profile moved from 78/76/72/71/68% to
83/76/71/70/67% - flat at 128 and -1 point at 256/400/800 (deterministic reruns, so real but tiny), while
the 64-token point rose 5.

Publication: head file `78064ec7` (op6-350, lfs sha256 `a375a459`), README `b44a3e1b`, sums `de285106`,
drafter README `31c7b802`.  The card's table is now 141.5-142.3 (+20%, 76%), 155.1 (+32%), 126.7 (+8%),
125.9 (+7%), and the last row - repeated-token text - reads **+33% at 96%**, with the old -24% kept in the
bullet as the before/after.

## The controller readout, measured: the rule is the signal, a linear head is not (2026-10-15)

`STRATA_Q36_JEV_PROBE` records one line per verify window - the head's input hidden, the last 16 emitted
tokens, the head's draft, the model's pick, whether it was accepted and whether the rule would fire -
collected with the rule *off* so every window carries a head draft.  Served: 26 fresh long-form prompts
(dialogue, code in five languages, math, CSV/XML/TOML/JSON, factual and prose), 1500 tokens each, 20,909
windows.  CLI: the degenerate p00 prompt, 2500 tokens, 2,409 windows.

- **Separation.**  In the degenerate run the head accepts **0.000** on the 2,154 windows (89.3%) where the
  rule would fire, and 0.346 elsewhere.  In the served run the rule fires on only 0.9% of windows - and the
  head accepts **0.924** there, 0.539 elsewhere.  A free deterministic rule already isolates the regime.
- **A closed-form linear readout** (ridge over standardized h + rule + tail run-length, the decision-head
  method) overfits: train AUC 0.893, **cross-prompt held-out AUC 0.629**; leave-one-request-out (within
  prompt) 0.672 weighted, range 0.49-0.78.  Real but modest.
- **Policy simulation** on held-out requests (W = 12.45 ms per window, P = 8.5 ms per plain step):
  always-draft 8.50 ms/token; **oracle 7.06 (+17%)**; gate at p-hat > 0.35 8.23 (+3%), 0.46 8.28 (+3%),
  0.55 8.33 (+2%).  The gate captures about a fifth of the oracle - so for control, the deterministic rule
  is the paying gate, and the rest of the prize is acceptance.
- **Long-form is where acceptance goes.**  The 26 prompts at 1500 tokens average **0.542** (31-94%), against
  68-76% at 128-800 tokens on the card prompt.  Round 6's data is assembled
  (`/tmp/opencode/mtp-onpolicy/probe7win`): 24 train + 3 held-out windows from these prompts plus 4 windows
  from the 2500-token degenerate run - thin, so a round on it wants either another collection pass or
  oversampled degenerate windows.

## The engine rebuild re-verified against the 176B identity (2026-10-15 20:12)

The `--mtp-backoff` build changed the engine binary the 176B lane also runs, so the pinned identity cycle
was re-run (`moe-pin-cycle.sh engine-backoff`).  Its authoritative line - the trajectory hash - is
**md5 `c1aa2407cb`, len 558 on all 8 bodies**, identical to every pinned capture since Oct 14 19:33
(15 captures: pin1/pin4/pin5/pinref/pincheck-new/pincheck-ref/corr/rem/rows/single/176b-routerref/
176b-swdown/176b-swgate/176b-shdown and today's).  The build changed nothing for the 176B.

Caveat for future readers: `moe-pin-cycle.sh` calls `moe-exact.sh "$TAG"` with the default reftag `base`,
and `base` is an **Oct 14 19:15** capture (md5 `497b1ecec3`, len 564) - 18 minutes *before* the pinned
state was first recorded - so the cycle prints `identical: NO` against a pre-state-E reference even when
the state is exact.  The `traj ... md5` lines are the real check (or pass a state-E tag as the third
argument).  Same day: every 35B capture (plain, op1, op2, op3) sits at `6e1cb21f6e`, so the identity
prompt stayed constant across the drafter work as well.

## Round 5, measured: prose restored without losing structured (op5 <- op4, 2026-10-15)

Mixed replay: `probe6` = probe4's 77 prose windows + probe5's 45 structured windows (held-out = 12 + 20 = 32
windows), warm start op4, lr 1e-4, 400 steps; best checkpoint step 300 at held-out 0.5329, kept over the
final 0.5327.

Served A/B, one service, same day, 400-token protocol (plain 117.2-117.3; the 128-token row is the card
protocol):

| set | op3 | op4 | op5 |
| :--- | ---: | ---: | ---: |
| card prompt (128 tok) | 140.2-140.3, 74.0% | 140.9-142.7, 78.9% | **142.0, 76.4%** |
| structured old held-out | 48.9 / 48.3 / 46.2% | 57.7 / 61.3 / 50.0% | 57.7 / 61.9 / 49.4% |
| five fresh structured | 45.6 / 48.7 / 50.8 / 48.3 / 38.1% | 59.6 / 55.6 / 61.7 / 66.3 / 47.8% | 58.3 / 56.9 / 62.6 / 66.3 / 48.9% |
| held-out 11 | 53.2%, 121.5 | - | 54.9%, 123.6 |
| held-out factual 6 | 62.0%, 130.3 | 60.2%, 128.2 | 61.5%, 129.6 |
| technical 6 | 94.1% | 93.4% | 94.2% |
| round-4 prose 12 | - | 53.3% | 54.1% |
| 17 never-trained (11 + 6) | 124.4 (+6.2%) | - | 125.7 (+7.3%) |

Round 5 meets its goal: it keeps round 4's structured gains (within one window of op4 on every structured
prompt) and recovers the prose to within noise of op3 (factual -0.5 points with mixed per-prompt signs on
op5 against op4's -1.8 with five of six negative).  Its only trail against op4 is the card prompt's
acceptance (76.4% against 78.9%) at equal speed (142.0 against a 141.9 average).

Publication consequences, recorded because "numbers only rise":

- swapping the published head to op5 would raise the card prompt (139.1-139.3 -> 142.0), technical
  (153.5 -> 154.8) and the 17-prompt row (124.4 -> 125.7), but would lower the factual acceptance
  (63.1% -> 61.5%) and the backoff's degenerate line (+3% -> -4%).  So the swap needs a decision, not an
  automatic card edit.
- the backoff is costless on ordinary text: card prompt 51/76 windows, 133.3 vs 134.6 tok/s; technical
  148/251, 127.4 vs 127.5 (with vs without `--mtp-backoff 8`) - no streak of 8 rejections occurs, so it can
  be enabled wherever the drafter is.

The degenerate 800-token run with op5: plain 116.4, op5 88.9 (-23.6%, 89/710 windows), op5 + backoff 112.1
(-3.7%, 151 of 384 offered).  Across heads the backoff bounds that case to about plain (op3: +3%, op5:
-4%); the residual spread is where the pauses land relative to the head's own rejection pattern.

## Publication gate (updated 2026-10-14, both cards live)

Both models' results are on the live Hugging Face cards, as a **Performance** section near the top:

- `SkyIsNotGreen/Scion-35B-A3B`, revision `e5d120bb`: generation **118.0-118.2 tok/s** on one RX 7900 XT
  (warmed, greedy, 128 tokens per request), prompt read **480-482**, ~8.5 ms/token at ctx 4096, the
  determinism note (eight warmed requests byte-identical, cold restart the same).
  - its optional k=1 MTP drafter is published with it: README `8fc244be`, the head file `eb4c6cf6`
    (`Scion-35B-A3B-mtp-strata.gguf` = op5-300, lfs sha256 `c9d183d4`), checksums `eeb53a51`.  The
    Performance section now carries the enable block (flags and the serve-config form), the refreshed
    table (+20% / +32% / +6% structured / +7% held-out, worst case -24%), the `--mtp-backoff 8` row
    (-4% on the same degenerate run) and the near-tie caveat.
  - `SkyIsNotGreen/Scion-35B-A3B-mtp-drafter`, revision `5a5f4fa2`: its card gained the second-runtime
    section - Strata reads the model repo's head, the two files are not interchangeable, same enable
    block and numbers.  Its own file's sha256 was re-verified (`2dabf2cf`, matches the table).
  - both cards refreshed again with the control results: 35B README `73d73549` (enable block now carries
    `--mtp-period 4`; new table row "that same run with `--mtp-period 4`: 114.8 -> 132.3 (+15%)"; the
    loop bullet rewritten around the 0.0% separation), drafter README `b17a0e06` (its Strata section got
    the rule and the relax-the-backoff note).
  - and again with round 6 (op6-350): 35B README `b44a3e1b`, head `78064ec7` (lfs sha256 `a375a459`),
    sums `de285106`; drafter README `31c7b802`.  The table now reads 141.5-142.3 (+20%), 155.1 (+32%),
    126.7 (+8%), 125.9 (+7%), and the repeated-token row is **+33% at 96%** - the worst case became the
    best row, with the old -24% kept as the before/after in the bullet.
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

## Round 7, collected: a fresh long-form pass, and the merged set (op7 <- op6, 2026-10-16)

The remaining weak regime is long-form text (~51% acceptance at 1500-2000 tokens against 68-96% short),
so round 7 is a coverage pass over the lowest styles.  **60 fresh prompts** (`prompts9.txt`: dialogue,
Ruby/Lua/Zig/Haskell/OCaml/Elixir/Perl/Fortran code, essays, reviews, emails and the "begin now and do not
stop early" factuals), served through the 35B with the op6-350 head and both probes
(`STRATA_Q36_MTP_PROBE=probe10.bin`, `STRATA_Q36_JEV_PROBE=jev10.bin`), 2000 tokens each, greedy.  The
window ran only once the decision-heads overnight job had finished and the 176B was restored; it took
668 s and restored `scion-flashnext` (up=200) at the end.

Served acceptance: **46.8% over 51,777 windows** (24,246 accepted), **26-62% per prompt** - the weak
long-form regime, as expected.  The lowest rows are exactly the ones the round targets: the Ruby hash
class 26%, the Elixir supervisor 36%, the boundary email 36%, the conference email 35%, the board-game
review 40%, the Lua state machine 41%.

Assembly (the round-6 recipe, unchanged): cut into 60 requests by the `pos` deltas (`d != 1 and d != 2`
is a boundary), one 512-record window per `floor(records/512)` per request with `h` shifted by one
(`hh[:510] = h[i+1:i+511]`), train windows first and the last 9 prompts' windows last.  `probe10/` =
**69 windows (59 train, 10 held)**.  The assembler reproduces `probe7win`'s 23 windows byte-for-byte, so
the new windows use the probe9 convention exactly.  Merged with `probe9/` keeping the replay rule,
`probe11/` = **306 windows, n_train 259, n_eval 47** (probe9's 200 train + the new 59; probe9's 37 eval
+ the new 10 held).  Training runs from `head-op6.pt`, lr 1e-4, 400 steps, best-by-eval.

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
4. ~~**The drafter's economics**~~ (`ed3188d5`): wired, exact and opt-in at 35% served acceptance (118.0-118.2 ->
   109.3 tok/s).  (b) and (c) are priced empty above (the verify row is serial recurrence plus its own experts;
   there is no cheaper target).  (a) is **done**: fitted to the engine's own hiddens - and specifically to the *verify path's* pairs, which
   are the hiddens the drafter actually reads - the acceptance runs 53-94% by text and the drafter is
   **+30.8% on technical prose, +18.1% on the card's own prompt, +6.2% averaged over 17 never-trained
   prompts**, all served.  Two corrections since: the drafts are *not* bit-exact - the 2-row verify flips
   genuine near-ties (0.029 top-two margin measured; drafts forced to always reject still diverge from plain
   at token 61 of 800, and six of six free-running generations diverge by token 112) - so the card calls it a
   speed option, not a bit-exact one; and the degenerate tail is fixed engine-side (`--mtp-backoff`:
   89.0 -> 119.9 tok/s, 213 of 707 windows paused, -24% -> +3% against plain).  Round 4 attacked the weak
   styles: the structured prompts the old mix never trained (44.6-48.9%) reached 50.0-61.3% on the same
   prompts, 47.8-66.3% on five fresh ones, and the card's prompt 140.9-142.7 (78.9%), at the cost of ~2
   points of held-out factual prose; round 5 (mixed replay) is in flight and a head ships only if it beats
   op3 on every set.  Remaining levers: more data for the weak styles (the head saturates the training
   windows at acceptance ~1.0 while held-out plateaus at ~0.52), or the zero-padded width-1.5 head
   (`moe/mtp_pad_head.py`) if capacity turns out to bind.
4. The two-card arm for q36: scoped and ruled out for the 35B at T=1 (see the scope section).
5. The register-staged lens on `gr_up_multi_kernel` (gfx11 has no fast variant) and on the
   `native_quantize_q8_1` / `lora_round` small kernels that appear 3x per layer in the served graph.
6. A per-window timeline split (launch -> first ring vs last ring -> finish) to bound the ~6 ms hand-off
   exposure without adding kernel time.

