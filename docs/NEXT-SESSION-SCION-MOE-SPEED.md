# Scion MoE speed lane — handoff (opened 2026-10-14)

**Goal.** Take the closed Bonsai-2 dense lane's speed methodology and apply it to the two Scion MoE models in
Strata: **Scion-FlashNext-176B-A6B** (live on the two cards, 68-71 tok/s with `--pipeline-windows 2`) and
**Scion-35B-A3B** (released, 11.34 GB, never served in Strata yet).  Engine work is speed work: measure the
budget, then attack the exposed latency, with the token stream as the gate.

**Read first:** `NEXT-SESSION-BONSAI2-SPEED.md` (the closed lane: instruments, measured facts, what transferred),
`P4-STRATA-FIT.md` (the fit, the corrections work, the VRAM policy, the quality instruments), then
`NEXT-SESSION-MULTIGPU-PERF.md` (the plan this lane continues).

## Where things are

- **Engine tree:** `work/strata`, branch `scion-0.1.44`, HEAD `42bef63a`, binary `work/strata/build-hip/strata`.
  The live service is `scion-flashnext` (port 8081, web chat + `/v1`), config `work/strata/strata-q6k-176b.json`
  with corrections on and `--pipeline-windows 2`.  The q4x160 sibling is `strata-q4x160-176b.json`.
- **This lane's docs:** `work/scion/docs/`.  The Bonsai lane's engine tree (`work/strata-bonsai2`, branch
  `bonsai2-dense`, HEAD `269add4e`) is closed and stays as the reference implementation of the techniques below.
- **Fixed protocol:** `serving/bench_strata.sh <label>` (4 requests, 128 tokens, temperature 0, run 1 warm-up;
  reads the server's `timings` JSON).  Warm-up is real here: graphs and the adaptive tier settle over 1-2 requests.
- **VRAM policy:** `serving/vram_reserve.py --apply <config>` / `--check`; keep `--vram-reserve-mib 3072` while
  measuring.  At the fringe (428 MiB free on the display card) decode fell to 25.8 tok/s.
- **Quality instruments (banked):** `serving/strata_ppl.py` + `STRATA_LOGPOS`; flash-next baselines 7.5728 (Q6K) /
  7.5741 (Q4X160).  Re-run only when data or arithmetic changes.

## What transfers from the Bonsai-2 lane (each measured there)

1. **Instrument the graph, not the wall clock.**  `STRATA_DENSE_GRAPH_TIME=1` printed the per-replay GPU span
   (events around the graph launch), the host wait/dequant/launch split, and the captured node count; with
   `STRATA_DENSE_ABLATE=1` the replay split into GEMVs (~74%) and everything else.  The MoE side already has
   `STRATA_VERIFY_PROFILE` (stage table per verify window) and `STRATA_DECODE_TIMING`; extend them with the
   Bonsai pair: one number for the verify window's replay (events around the launch), the node count, and an
   ablation arm that skips the expert GEMMs.
2. **In-graph latency is exposed; the micro-benchmark understates it 4-7x.**  Back-to-back micro launches
   overlap the latency the graph serializes.  The rms batch read 0.64 us faster in `dense_micro` and was worth
   0.5 ms/token served.  Confirm every candidate with a replay delta on a real run before believing it.
3. **The serialized-load lens.**  Slow small kernels showed `s_waitcnt` after every load in the ISA: a runtime
   trip count or an inline use stops the compiler from issuing loads together.  Batch them, and pin the
   arithmetic to the exact instruction the reference compiled to (`__fmaf_rn` where the plain loop is
   `v_fmac_f32`).  Check the ISA before rewriting a reduction.
4. **Never reorder a reduction without pinning it.**  A same-order 20-load batch changed the compiler's
   contraction and split the token stream at char 1196 of a 512-token run while random-input parity tests
   passed.  The served stream is the gate.
5. **Node counts are worth ~1.4 us/node in-graph** (not the 2.8 us standalone figure); pair/dual fusions pay
   where kernels share data or the fixed cost dominates (< ~20 MB matrices).  Small wins are still wins, but
   measure the delta: several "obvious" fusions measured flat and were reverted.
6. **Shape walls and warm clocks.**  PQ2_0 bandwidth varied 540-705 GB/s with matrix size, and the bench read
   390-580 GB/s cold vs 700 warm.  Never quote one big-matrix number as "the card's bandwidth".

## What the Scion lanes already know (do not re-derive)

- **Corrections:** reshaped (uint4 reads, hardware f16 conversions, A writes f16), served delta down to
  **2.6 ms/token** from ~9-11; `lora_corr_parity` bit-exact on the updated reference.  Folding B into the
  mmvq epilogue measured only 0.3-0.5 ms/window; cooperative A+B fusion is **RULED OUT** (it wedged both GPUs,
  see `COOP-CORR-SAFETY.md`).
- **Multi-GPU:** pipeline (layer) parallelism is what exists, one hand-off per window; `--pipeline-windows 2`
  measured **+62-68%** (68-71 tok/s) and is in the config.  The MTP drafter stays on the last card
  (`STRATA_MTP_STAGE0=1` measured worse, 52-59).  Tensor parallel via RCCL is not applicable to these custom
  containers.  **Expert-parallel MoE is the open structural design** (the cards already hold disjoint expert
  caches; HarMoEny reports +37-70% under imbalance) — scope it only after the per-window budget is instrumented.
- **The pipeline's steady state:** verify window ~32-35 ms warmed at ~2.2-2.4 tokens/window, with the
  **GPU-reach wait ~24-26 ms dominating**.  That wait is the first target, and it is the same class of problem
  the Bonsai lane fixed (in-graph serialized latency), not a weights problem.
- **f16 software round trips** were 75-80% of the correction kernel time at verify-window shapes; the hardware
  converters are bit-identical (`f16_hw_check`).  Grep `f16_bits.hpp` users in the hot path for the same
  pattern before assuming a kernel is memory-bound.

## First moves, in order

1. **Budget the live flash-next serve.**  Add the replay-span/node-count instrument and the expert-GEMM ablation
   to the verify window; publish one table: weights, corrections, small kernels, pipeline wait, host.  The
   number to beat is 32-35 ms/window warmed.
2. **Attack the 24-26 ms GPU-reach wait with the latency lens** (ISA, serialized loads, in-graph exposure).
   Keep it one step per commit, each accepted on the fixed protocol plus warmed token-stream identity.
3. **Serve Scion-35B-A3B in Strata.**  No config or service exists (the model card only has fork instructions):
   pack/load it (11.34 GB, fits one 20 GB card), add a config + service beside `scion-flashnext`, get the
   baseline with the fixed protocol, and only then apply moves 1-2 there.  One-card and two-card arms both,
   plus its MTP drafter.
4. **Only then the structural queue:** the correction-fold remainder (measured small), expert-parallel scope,
   and the 0.1.44 upstream sync if it moves RDNA3 numbers.

## Gates and rules

- **Token identity:** every change must leave warmed greedy requests byte-identical to the pre-change arm.  Keep
  a stash of the warmed full text for the compare (the fixed protocol plus one full-body diff).
- **One step per commit, numbers in the message.**  Never `git push`.  Model-load tests only in serve-down
  windows; stop one service before starting another on the same port.
- **Do not re-derive the rejected list:** correction cooperative fusion, RCCL tensor parallel, MTP stage-0
  placement, and (from Bonsai) the two-rows-per-warp GEMV ports and the doorbell-spin split wait.
- **Report when a measurement changes something.**  The Bonsai lane's rule worked: the useful output was the
  budget, the surprises, and the reverted dead ends, not the raw count of edits.

## Handoff acceptance

A fresh session is up to speed when it can (a) state the warmed verify-window budget for the live serve with
numbers, (b) name the current dominant cost and the next candidate, and (c) build and run the fixed protocol
without re-reading the whole history.  Steps 1-2 above produce exactly that.

## Continuation prompt (copy-paste)

```
New lane: Scion MoE speed in Strata.  Read work/scion/docs/NEXT-SESSION-SCION-MOE-SPEED.md first; it is the
handoff with the goal, the state, and the transferable findings from the closed Bonsai-2 dense lane.  Then read
NEXT-SESSION-BONSAI2-SPEED.md (closed lane: instruments, measured facts, methodology), P4-STRATA-FIT.md (the
fit, corrections, VRAM policy, quality instruments) and NEXT-SESSION-MULTIGPU-PERF.md (the plan this continues).

State to trust: the engine tree is work/strata, branch scion-0.1.44 (HEAD 42bef63a), binary build-hip/strata; the
live service scion-flashnext on 8081 runs the Q6K release with corrections and --pipeline-windows 2 at 68-71
tok/s steady; the verify window is 32-35 ms warmed with a 24-26 ms GPU-reach wait dominating; corrections cost
2.6 ms/token after the reshape; Scion-35B-A3B (11.34 GB) has no Strata serving lane yet.

Do this in order: (1) add a per-replay GPU-span + node-count + host-split instrument and an expert-GEMM ablation
to the verify window, and publish the warmed budget for the live serve; (2) attack the GPU-reach wait with the
ISA-guided serialized-load lens from the Bonsai lane; (3) bring up Scion-35B-A3B in Strata and apply the same.
One step per commit with numbers; every change must leave warmed greedy requests byte-identical (fixed protocol:
serving/bench_strata.sh; keep --vram-reserve-mib 3072); model-load tests only in serve-down windows; never
git push; do not re-derive the rejected items listed in the handoff.  Do not stop for approval; report when a
measurement changes something.
```
