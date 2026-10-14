# Bonsai-2 dense lane — record (closed 2026-10-14): 64.4 tok/s single card

**Closed.** The community note is posted: [Bonsai 2 in Strata: 64.4 tok/s on one RX 7900 XT with no MTP](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/94)
(Prism ML community tab, 2026-10-14).  The lane is finished; the "If the lane reopens" list below is not
scheduled work and nothing here needs to run again.

**Context.** The Bonsai-2 lane: PrismML's `Ternary-Bonsai-2-27B-PQ2_0.gguf` in Strata's dense engine on the two
RX 7900 XTs. All engine work is on branch `bonsai2-dense` in `work/strata-bonsai2`, HEAD `269add4e` (unmerged).
The previous state of this doc (HEAD 4e297beb, 55.8 tok/s) is the history; the kernels and the budget below
supersede it.

## Where the lane ended

- **Closing gate (same-day, same build): single card 64.4 tok/s (25-token 66.2), two cards 35.6 (38.9);
  `identical: YES` on both prompts.**  Best single run of the day was 64.6.  Started the lane at 44.9 and the
  day at 55.8; 918 -> 790 graph nodes, replay GPU 16.68 -> 15.30 ms, and by ablation the non-GEMV half went
  5.80 -> 3.89 ms.  Every commit below passed `bonsai-bisect.sh` (`identical: YES` on both prompts).
- **The card's real wall is shape-dependent.** Warm serve-down `pq2_bench 142 0 <n_in> <n_out> <reps>`:
  5120x10240 610 GB/s, 5120x6144 540, 6144x5120 538, 5120x17408 661, 17408x5120 655, 5120x12288 630,
  5120x248320 (head) 705. The 773 GB/s figure from the previous session is the 338 MB head matrix at full warm
  clock; the pass's matrices are all below it and the pass's GEMV total (~10.9 ms by ablation) is at those walls.
  Small matrices are launch-dominated (5120x1024: 226 GB/s warm) — reason enough for the pair kernel below.
- **Host is NOT in the way any more: dequant 0.07 ms, graph launch 0.03 ms, wait 0.19 ms** (measured,
  `STRATA_DENSE_GRAPH_TIME=1`). Device-side embedding would save <0.1 ms; it is off the list. The old budget's
  "host ~1.5 ms" does not describe the graph path.
- The two-card split: `STRATA_DENSE_SPLIT_EVWAIT=1` (event wait), byte-exact, 35.6 tok/s in the closing gate
  (38.9 on the short prompt).  The 29-32 tok/s in the earlier notes predates the kernel work.

## The nine commits (each with its own acceptance)

| commit | what | tok/s |
| --- | --- | ---: |
| 1e33cfa8 | FWHT butterflies: 5 shared stages instead of 20 `__syncthreads` (warp shuffles + ping-pong) | 56.7 |
| 95a21244 | GDN pre-block rms writes plain AND folded q8_1 (q8_1_ab); scaled write on the whole block | 57.2 |
| 890e7228 | attention partial: `per` folds at compile time, two cells per iteration (loads pipelined) | 59.3 |
| b32397b4 | residual add deferred into the next block's input rms; head rotation+quantize one launch | 59.0 (neutral) |
| 499b6a9c | `native_pq2_0_mmvq_pair`: qkv+z, ffn gate+up, k+v in one launch (ssm_alpha+beta are BF16, so they stay two f32 calls) | 59.9 |
| bb9bf384 | attention partial stages the cells through a per-warp shared slot (micro -10% more; pass sub-noise) | 59.8 |
| 277e6bae | rms sum: 20 loads batched, chain pinned to `__fmaf_rn` (the ISA's `v_fmac_f32`) | 62.0 |
| d7c779c0 | GDN alpha/beta f32 pair unrolled 4 -> 16 (type 30 is BF16: this is the *float* pair path) | 63.1 |
| 269add4e | attention partial: q load, cell dots and the stage copy stop waiting per load (63.75 -> 30.8 us) | 64.6 |

## Facts that cost time (do not re-derive)

1. **In-graph node cost is ~1.4 us, not 2.8.** Removing 128 nodes (the GEMV pairs) moved the replay by 0.18 ms.
   The 2.8 us figure is for standalone launches. Node-count fusions below ~100 nodes are not worth their risk.
2. **A GEMV's fixed cost dominates below ~20 MB** (launch + ramp + tail): the pair kernel exists exactly for the
   pass's small/medium matrices that share an activation, and it is bitwise the two single calls.
3. **Do not let the compiler batch/reorder a reduction in `dense_kernels.cu` without pinning the operation.**
   A 20-load register batch in `rms_fwht_quant_kernel`'s sum with the source still `acc += v * v` measured
   10.28 -> 9.71 us and **split the token stream at char 1196 of the 512 run** while `dense_fusion_parity`
   passed on its random inputs: the compiler contracted the batch differently.  The fix that shipped pins the
   chain to `__fmaf_rn` (what the plain loop compiles to: `v_fmac_f32`), and that version is byte-exact and
   worth **0.5 ms/token** in situ -- 7x its 0.64 us micro delta.  Check the ISA before rewriting a reduction.
4. **Attention partial: 63.75 -> 30.8 us at ctx=256.** Three stages got it there: templating `per` (8 for
   head_dim 256) so the `i < per` guards fold; two cells per iteration; and the serialized-load fixes below.
   NC=3 (48.2 us) and NC=4 (52.0) regress on register pressure.  The walk is still latency-bound: 24 blocks x 8
   warps for a 256-cell split, each warp walking its own cells in a fixed order (the merge's arithmetic forbids
   more warps).
5. **The serialized-load lens finds real wins.** The q load, the per-cell dot reads and the cell-stage copy in
   the attention partial each waited on one load at a time (visible in the ISA as an `s_waitcnt` after every
   load); batching them is what closed the gap above.  When a kernel measures slower than its memory footprint
   justifies, look for a runtime trip count or an inline use that stops the compiler from issuing the loads
   together.
6. **Per-shape GEMV alternatives were already tried and reverted** (two/four rows per warp, lane-per-block).
   `pq2_bench` variant 0 is the only mapping that survived testing.  Do not re-port them.
7. **`prism.hadamard.gdn_v_grouped` is TRUE in this file** (the first doc said off): the GDN V rotation runs the
   [hd][nk][rep] -> [hd][rep][nk] gather (`hd=128, nk=16, rep=3`) inside `fwht_quantize` and the parity test
   covers it.  `qwen35.block_count` 64 = 48 GDN (blk % 4 != 3) + 16 attention; `full_attention_interval` 4.
8. **Type 30 is BF16, not a custom quant.**  `ssm_alpha/beta` are float tensors, so `ab_quant` is FALSE here:
   the GDN layers run `dense_gemv_f32_rows_pair`, and the `q8_1_ab` dual-quant path from 95a21244 is dead for
   Bonsai (it exists for quantized-alpha/beta files).  The earlier "pair the type-30 GEMVs" item was void; the
   win was the unroll.
9. The GDN recurrence `rec` is at the state-bandwidth wall: 3.1 MB state read + written per layer = ~9 us at
   750 GB/s, measured 9.1. Leave it alone at T=1.

## Budget (single card, 64.4 tok/s = 15.5 ms/token; measured over the session)

| item | ms/token | notes |
| --- | ---: | --- |
| GEMVs (273 nodes after pairing) | ~11.4 | at the per-shape walls; the ablation says this is 74% of the 15.3 ms replay |
| rms_fwht_quant (128 calls) | ~0.6 | in situ after the batch; ~9 us still in dense_micro |
| attention partial+merge (16) | ~0.5 | 30.8 us at ctx=256 |
| GDN conv (48) + rec (48) | 0.19 + 0.44 | rec is state-bandwidth bound |
| swiglu_quant (64) | 0.29 | 4.5 us |
| fwht_quant V+attn (64) | 0.27 | 4.2 us |
| rope+split_q_rms+rms_k+kv (16 layers) | 0.25 | |
| node overhead (790) | ~1.1 | ~1.4 us/node |
| GDN f32 alpha/beta pair (48) | ~0.27 | 5.66 us each after the unroll |
| host (dequant+launch+wait) | 0.3 | |

vs the target's 14.1-14.7 ms (the 68-71 tok/s flash-next serve): the GEMV side is at the card's wall for these
shapes, so the remaining ~0.8-1.4 ms is kernel execution/tails.  The lever with the most left is the per-kernel
fixed cost, not node count.

## If the lane reopens (not scheduled)

1. **Attention partial** (30.8 us at ctx=256, ~0.5 ms/token): the load path is fixed; what is left is the
   per-cell serial chain (the 5-shuffle `warp_sum` then two `__expf` per cell) and the one dependent pair per
   iteration.  Try a load-regs / process / store-regs split so the next pair's staging loads are consumed after
   the current pair's updates, or check the ISA for the next serialized wait.
2. **rms_fwht around the butterfly**: the sum is done and pinned; the 5 butterfly+quant rounds and the kernel
   prologue are the rest (~9 us in the micro, ~5 in situ).  A two-kernel split (sum-only + a grid-parallel
   write/butterfly/quant) trades the serialized rounds for one more node; measure before keeping.
3. **q/k norm+rope fusion** (the first doc's item 2, still undone): split_q_rms / rms_k + the two rope launches
   are four nodes per attention layer.  The fused kernel must reproduce elementwise.cu's reduction exactly
   (see fact 1 of the first doc), so build it with a parity oracle first and only then run the acceptance.
4. **Device-side embedding gather**: the host dequant is 0.067 ms/token serialized before every graph launch
   plus the pinned memcpy and one node; a PQ2_0 row decode + inverse FWHT kernel would make the graph
   self-contained.  ~0.1 ms/token for a contained kernel.
5. ~~The community post~~ **done**: discussion #94, posted 2026-10-14.  If the lane reopens, the items above are
   each worth 0.05-0.5 ms; nothing else is queued for it.

**Calibration:** `dense_micro` numbers are a FILTER, not the result.  For the latency-bound kernels the in-situ
effect has run 4-7x the micro delta (the rms sum: 0.64 us micro, 0.5 ms/token pass), because the micro's
back-to-back launches overlap the latency the graph exposes; the attention staging was the exception (micro -10%,
pass flat).  Always confirm with a GRAPH_TIME replay delta on a real run before believing a kernel win.

## Instruments

- `bonsai-exact.sh` — acceptance: base vs split byte-identity (25 + 512 tokens) + timings.
- `bonsai-bisect.sh <tag> <config> '<env json>' [ref]` — one engine window vs the stored reference in
  `/tmp/opencode/exact/`.
- **`STRATA_DENSE_GRAPH_TIME=1`** — per-replay GPU span (events around the graph launch), the host wait/dequant/
  launch split, and the captured node count per bucket. Cheap, byte-exact, keep it off only when the numbers
  themselves are the measurement.
- **`STRATA_DENSE_ABLATE=1`** — every GEMV is skipped (bench-only, output garbage): replay time splits into
  weights and everything else.  With `GRAPH_TIME=1`: 5.80 ms at the start of the session, 3.89 ms on the final
  build (256-token average; the GEMV half is ~11.4 ms).
- `tests/dense_micro` — per-kernel serve-down numbers, now including gdn conv/rec, kv append and the attention
  part+merge at ctx 64/128/256/512.
- `tests/dense_fusion_parity` — the fused-vs-plain oracles: split_q_rms, rms_fwht (rot + plain q8_1),
  fwht_quant (incl. the GDN V gather), swiglu_quant, the four GEMV pair geometries. 45 cases.
- `tests/pq2_bench` — `142 0 <n_in> <n_out> <reps>` for in-situ shapes; needs a warm GPU (>100 ms of work) or
  the numbers read low (clock ramp).
- `STRATA_DENSE_PROF=1` (graphs off while profiling) — the old coarse sections.

## Gotchas

- Model-load tests only in serve-down windows; never build while a model loads.
- Build: `cmake --build build-hip --target strata-dense -j 10` (~1 min); test targets separate.
- `/tmp` is tmpfs and fills; never `git push`.
- The profiler's event records CANNOT be captured into a HIP graph: capturing them made the next
  `sample_tokens` launch fail with "invalid resource handle" (experiment reverted). Use GRAPH_TIME + ABLATE.
