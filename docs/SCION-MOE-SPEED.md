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

## Next steps (session queue)

The continuation for these was `NEXT-SESSION-SCION-MOE-SPEED-2.md` (state + the copy-paste prompt); steps 1-2
are done above, steps 3+ resume below with a q36 decode budget first.

1. ~~**35B corrections**~~ done (Step 4, engine `dd673041`); ~~**35B serve lane**~~ done (Step 5).  The
   corrections' 2.5 ms/token is item 3 below's first measured number.
2. **35B decode budget**: a q36 timing instrument (events around the decode graph launch + a
   kernel-class split) and one published table *before* touching kernels; then the part-1 speed lens
   on this model (the corrections' reshape first - the 176B went 12.6 -> 2.6 ms/token on the same
   problem).
3. Grouped expert micro-bench at the served 176B shapes (T=3-4, ~28 experts/layer, PTQ1_0) against the
   ~190 GB/s implied by the in-graph stamps — decides scatter-vs-latency before touching the kernel.
4. The same register-staged lens on `gr_up_multi_kernel` (gfx11 has no fast variant) and on the
   `native_quantize_q8_1` / `lora_round` small kernels that appear 3x per layer in the served graph.
5. A per-window timeline split (launch -> first ring vs last ring -> finish) to bound the ~6 ms hand-off
   exposure without adding kernel time.

