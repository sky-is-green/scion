# P4 — the untouched release served in Strata (fit done, 2026-10-13)

**Status: the shipped Q6K release loads and serves in Strata on the local box
(2× RX 7900 XT). Measured 26–33 tok/s steady state, thinking works, all 192
corrections validated. The quality comparison against the fork numbers
(PPL/HS/WG) is the remaining half of P4.** Nothing pushed.

## What was needed

Everything built from what was already on the box; no pod time.

1. **Pack** — `python3 tools/iq_pack.py --gguf Scion-FlashNext-176B-A6B-Q6K.gguf
   --out /home/penis/models/scion-strata/pack-q6k --compat-bf16`. 1271
   tensors, 302 served natively from the GGUF (experts type 143 per layer,
   the Q4_0 table stays in the GGUF), 192 embedded corrections in the pack,
   arena 2.07 GiB. `native_experts.txt` records `143 143` for all 48 layers.
2. **Engine rebuild** — `cmake --build build-hip` in `work/strata` (the
   working tree carries the Scion support commit `4f7fb6b5` plus the P5
   table types 144–146).
3. **Embedding bridge** — the release has `token_embd.weight` as **Q6_K**;
   the engine's native embedding wants a type with a GPU row dequantizer
   (`is_iq()` + BF16), and Q6_K (14) is not one. The Q2_0 sibling uses Q8_0,
   which is why the earlier Q2_0 serving worked. Built the engine's
   one-tensor BF16 embedding from the local fp8 checkpoint
   (`tools/embd_bf16_pack.py --model work/models/qwen38-flashnext-fp8 --out
   /home/penis/models/scion-strata/scion-embd-bf16.gguf`, 1.27 GB, the bytes
   copied straight from `model.language_model.embed_tokens.weight`), passed
   as `--embd-gguf`. Note: the checkpoint's shards live in `shards/`, flat
   symlinks beside `model.safetensors.index.json` are needed for the tool.
4. **Engine bug fixed** (the blocker that would have hit any corrections
   file): `src/core/corrections.cpp` used its own layer map,
   `layer % qsa_interval == 0`, which makes layer 0 a QSA layer and rejects
   the `blk.0.ssm_out.*` corrections the file ships. `layout.hpp` already
   has the canonical rule (`layer % qsa_interval == qsa_interval - 1`;
   full attention on 3, 7, … 47 — what the file's `attn_output` set and the
   fork both use). The fix replaces the local predicate with
   `is_qsa_layer()`; the file's 192 pairs then all validate, and
   `lora_corr_parity` on the 4l file still reports 8 pairs bit-exact.

## Measured (RX 7900 XT ×2, 30 GB host RAM, the box)

- Startup: ~25–40 s, ending with "filling the GPU's expert cache (12022
  experts, 12.04 GiB of VRAM)"; layers split 0–25 (CUDA0) / 26–47 (CUDA1);
  **2670 MiB VRAM free** with everything loaded.
- First request (cold): 39 tokens at **11.3 tok/s** (MTP drafts 25/29
  accepted), prompt 66 tokens read at 8.4 tok/s.
- Steady state: **26–33 tok/s** generation, expert-cache hit **99.7–99.8%**,
  prompt reads 36–53 tok/s with prompt-cache hits (55/60 tokens cached).
- For scale: the fork's CPU-only local path is 0.15 tok/s; the fork's HIP
  partial offload peaked at ~1 tok/s and wedges at high offload (see the GPU
  bisect notes). The pod (A6000, CUDA fork) measured ~86 tok/s aggregate.

## How to run it

```sh
systemd-run --user --unit=p4-serve-q6k --collect \
  --property=WorkingDirectory=$HOME/Desktop/work/strata \
  .venv/bin/python serve/server.py --engine strata \
  --config strata-q6k-176b.json --port 8081
# OpenAI: http://127.0.0.1:8081/v1/chat/completions  (model scion-flashnext-176b-a6b-q6k)
```
Config: `work/strata/strata-q6k-176b.json` (pack-q6k, `--native` and
`--ple-gguf` = the Q6K file, `--embd-gguf` = scion-embd-bf16.gguf,
`--ple-io mmap`, `--mmap-experts`, MTP spec).
Unsloth Studio's custom provider row now points at this server
("Scion Flash-Next (Strata, local)", `http://127.0.0.1:8081/v1`).
The fork CPU bridge (unit `scion-flashnext-serve`, port 8903) is stopped; it
is a fallback only.

## Performance work (2026-10-13)

Fixed protocol: 4 requests, 128 tokens, temperature 0, one "history of
computing" prompt (run 1 warm-up), timings from the server JSON.

| arm | gen tps |
| --- | ---: |
| first serve, `--ple-io mmap`, prefill 512, reserve 3072, no env | ~20 |
| vendor preset: `--ple-io direct`, prefill 8192, reserve 1024, `STRATA_PREFILL_MMQ=1`, `STRATA_PREFILL_RING=96`, `STRATA_IO_THREADS=32`, `--pool-workers 15`, `--pcie-frac 0` | **25.8** |
| same, Q2_0 sibling (its Q5_0 table needs `--ple-io mmap`) | <=25 (noisy) |
| same, corrections removed from the pack (index lines only, hardlinked) | **47.6** |

So the stored `26-33 tok/s` was the correction cost, not the engine: 96
embedded correction branches (rank-512, two passes each) add ~22 ms/token —
25.8 tps with them, 47.6 without. The stock GSQ-RCO Q2_0 model (no
corrections) measured 51.6 tps on the same box in the earlier session; that is
the engine's real level here. The vendor's gfx1100 arm (RX 7900 XTX, 64 GB)
reports 55-59 tps on IQ3_XXS.

The branch cost is out of proportion to its math: ~0.9 GFLOP/token and
~0.8 GB/token of factor reads, which should be well under 1 ms on this GPU.
The two-pass design is one warp per output element with a long strided
reduction (lora_a: 80 serialized loads per warp; lora_b: 16), ~192 launches
per token on the critical path.

Kernel-only micro-benchmark (2026-10-13, `lora_corr_apply` off the engine,
`/tmp` scratch, gfx1100):

| branch shape | ms/call | split |
| --- | ---: | --- |
| ssm_out / attn_output (6144 x 512 -> 2560) | 0.167 | A ~0.093, B ~0.004, 2 launches ~0.016 fixed |
| ffn_moe_out (2560 x 512 -> 2560) | 0.068 | same pattern |
| ssm, 5 rows (window) | 0.148 | no more than 1 row -> latency- not work-bound |

So ~13 ms/token of the ~18 ms delta is the A pass itself: one warp per
output gives only 512 warps (about 6 per SM) with a 192-iteration dependent
chain, while the B pass and the arithmetic are cheap. A 4-way-split reduction
(4 warps per output, shared-memory combine in a fixed order) raises the
parallelism to ~2048 warps and should bring the A pass toward its
memory-bound ~10-20 us; that changes the documented reduction order, so the
reference in `lora_corr_parity` and the decode/prefill/verify agreement story
must be updated together. An unroll-4 ILP tweak alone was tried first and
measured no gain (grid size, not ILP, is the limiter).

Fix candidates, in order:
1. 4-way split reduction: 4 warps per output, each summing its quarter of the
   reduction axis in the same lane-strided form, combined in shared memory in
   a fixed order; update the `lora_corr_parity` reference to match. **DONE
   2026-10-13**: ssm branch 0.167 -> 0.066 ms and ffn 0.068 -> 0.028 ms
   isolated (2.4-2.5x); `lora_corr_parity` mirrors the new order and stays
   bit-exact (8 pairs, 0 of 7680). Served (fixed protocol, reserve 3072 on
   both sides): corrections arm ~30.6 tok/s median vs ~29.7 before, with
   no-corrections at ~46.5 - the served delta went ~12.6 -> ~11.0 ms/token.
   The served path does not exercise the rows=1 shape the split helps most
   (verify windows deliver rows>1, where the old kernel was already fine), and
   ~6 of those ms are per-call/engine overhead across 96 branches per token.
   The reserve fix captured most of the earlier 17.8 -> 12.6 ms step: at the
   old 1024 reserve the display card was running at the fringe.
2. fold the correction into the layer's existing GEMVs the way llama.cpp's
   LoRA path scales the matmul (no separate passes at all);
3. a fused A+B kernel: RULED OUT (2026-10-13). A counter in `corr_apply`
   shows the calls happen at GRAPH CAPTURE time (96 per captured window
   shape, ~480 total over 72 tokens), so there is no per-token host/launch
   overhead to remove; the residual ~11 ms/token is in-graph kernel time plus
   pipeline gaps between the small correction kernels and the big layer
   kernels. Grid-wide fusion would need cooperative launch and therefore
   `-fgpu-rdc`, which this build does not use - a global build change for a
   bounded win.
4. THE f16 ROUND TRIPS WERE THE KERNEL COST: **DONE 2026-10-13 (late)**. The
   served path runs the branches at verify-window shapes (rows up to `--spec`
   + 1 = 5), so a new bench (`src/kernels/lora_corr_bench.cu`, one branch per
   call, back to back, gfx1100) walked rows 1..8 for the two real shapes:
   cost grows ~linearly with rows, and diagnostic copies of the A pass (each
   conversion switched off) put ~75-80% of the time in `f16_bits.hpp`'s
   software converters - at rows=5 the ssm/attn A pass ran 117 us of which
   ~58 was the x round trip and ~35 the a widening, because every x element
   was round-tripped once PER RANK ROW. The round trips in `lora_corr.cu` are
   now the hardware conversions (`__float2half_rn` / `__half2float`), which
   `src/kernels/f16_hw_check.cu` shows bit-identical to the software ones:
   all 65,536 fp16 codes widening, 16,777,216 sweep values over 2^-44..2^44
   narrowing, every boundary value; the only differences are NaN payload
   bits, and no correction tensor holds a NaN. The accumulation order is
   untouched, so `lora_corr_parity` stays bit-exact (8 pairs, 0 of 7680,
   max|err| 0). Measured, same bench, per branch, before -> after: rows=5
   ssm/attn 162 -> 36 us, ffn 86 -> 27 us (rows=1: 80 -> 24 and 32 -> 12) -
   ~12 ms -> ~3 ms/token over the 96 branches at the served window mix.
   Engine rebuilt; the served A/B was pending (the box was running another
   lane's bench at the time) - it is in item 5, which also reshaped the
   passes and supersedes the in-place swap below.
5. THE PASS SHAPES, THE `t` ROUND TRIP AND B'S SCALAR LOOP: **DONE
   2026-10-13 (night)**. Item 4's in-place swap left the loop shapes and the
   f32 `t` alone; `lora_corr_bench` then showed where the rest was (rows=5,
   per branch, back to back: ssm A 117.6 us, B 40.9; ffn A 41.5, B 40.6 -
   the old-order copies in the same bench read ~162 / ~86 combined, which is
   item 4's before column). Three changes shipped:
   - A reads uint4 rows and converts f16 pairs with `__half22float2` (one
     hardware instruction per pair, exact widening), four accumulator chains
     per lane over the quarter split;
   - A writes its result as f16 (`t16`), so the B pass no longer re-rounds a
     f32 `t`; a small `round` pass is the only f32 -> f16 RNE step on x
     (`f16_from_f32`, the software converter, unchanged);
   - B reads `t16` and B's rows as uint4, pairs converted the same way.
   Same bench, rows=5: ssm/attn 162 -> 36 (item 4) -> **30.0 us**; ffn
   86 -> 27 -> **22.6 us** (rows=1: 80 -> 24.1 and 32 -> 17.2).  The
   reduction ORDER changed with the shape, so `lora_corr_parity`'s reference
   was rewritten to mirror it: 8 pairs, 0 of 7680, on both 4l fixtures; the
   difference against the pre-rewrite order (`corr_oldnew`, real factors) is
   max|delta| 5.96e-08, relative 1.19e-07 - f32 reordering noise, four orders
   under the PPL instrument's resolution.  `f16_hw_check` still passes.
   SERVED, fixed protocol, same box/config (reserve 3072, prefill 8192):
   before this session ~30.6-34.6 tok/s with corrections; after **41.4 /
   41.8 / 41.0 / 41.4** (75/103 drafts) and the clean no-corrections arm on
   the same engine build **46.5 / 46.3 / 46.5** - the served delta is
   **2.6 ms/token** (was ~9-11).  `STRATA_VERIFY_PROFILE` puts the
   correction-attributable stages at **1.94 ms/window** (out-proj 2.26,
   copy+combine 1.57, against 1.26 / 0.63 for the no-corrections arm), from
   7.66 ms/window before the session.  Scratch grew by `x16` + `t16` at the
   three call sites; prefill's `bytes_needed` also gained the `corr_t` and
   `grs` takes it had been missing.

Measured with `lora_corr_parity` still bit-exact (8 pairs) after the
`corrections.cpp` fix; the no-corrections pack above is a measurement arm only
and is not the shipped configuration.

## VRAM headroom (system stability)

A card at the fringe of its memory slows down, and the display card's free
VRAM shrinks when the desktop uses the GPU. Measured on 2026-10-13: with the
expert cache filled to the edge the display card (card1) sat at **428 MiB
free** and decode ran ~25.8 tok/s; with `--vram-reserve-mib 3072` both cards
keep >= 2.2 GiB free and the same workload measured **27-33 tok/s**.

Policy (applies to any card count):
- a card with a connected display keeps `max(2048 MiB, 10% of total)` free;
- other cards keep `max(1024 MiB, 5%)` free;
- the engine's flag is global, so use the largest per-card need **plus
  1024 MiB** for the buffers allocated after the expert-cache fit (measured:
  a 2560 MiB reserve still ended at 1781 MiB free after the desktop grew
  during load).

Tool: `work/scion/serving/vram_reserve.py` reads per-card VRAM and connected
connectors, prints the table, and `--apply` writes the recommendation into a
serve config (or `--check` for monitoring). The live server now runs with
`--vram-reserve-mib 3072`.

Engine feature candidate: do this natively and per card (reserve more on the
display card, less on the others); today the flag is one number for all cards.

## Validation: wikitext-2 PPL (in flight, 2026-10-13)

The engine's `STRATA_LOGPOS` dump (serve mode, `--short-read 600`) was built to
measure the end-of-turn mass a chat model puts on raw text: reading corpus
slices as *user* messages scores the model's anti-LM reflex (top token
`<|im_end|>` at mid-text; PPL 11-13 with corrections and 542 without - not
quality numbers). Sending each slice as an *assistant* message (user,
assistant = slice, user) has the model read its own words, which behaves like
text scoring: repeated text scores 85% top-1 agreement, and the sample chunks
land at (first token of each slice unscored, the llama.cpp convention):

| sample | per-chunk PPL | mean |
| --- | --- | ---: |
| chunks 0-3 | 3.14 / 5.35 / 5.58 / 4.71 | **4.71** |
| chunks 10-13 | 2.94 / 4.04 / 5.19 / 5.50 | **5.50** |

against the fork's wikitext-2 PPL of **5.4561** (llama-perplexity, 580 x 512,
chunk-reset) - in the expected band (the assistant conditioning is slightly
lighter than raw LM, so landing at or just below is expected). Scorer:
`work/scion/serving/strata_ppl.py`; the full 581-slice run is the
`p4-ppl-full` unit, results to `/tmp/opencode/ppl/ppl-full.jsonl`.

**Full run complete (2026-10-13): PPL 7.5728 over 296,636 tokens in 581
slices, 0 failures** (median chunk 7.61, p10 4.07, p90 14.03; the 4.7-5.5
samples above were clean early articles). The number is chat-conditioned and
runs ~+39% over the fork's raw-LM 5.4561, so the two are not directly
comparable: this instrument answers "how well does the model predict these
slices as its own reply", which is stable and usable for paired A/B (container
or correction arms), while an exactly comparable number needs the raw prefill
logits dump (the batched path computes no per-position vocabulary logits
today). Lighter conditioning shapes (assistant-only, system+assistant) are
rejected by the server with 400; `[user, assistant = slice, user]` is the
supported form.

**Q4X160 local A/B (2026-10-13, independent-engine cross-check).** Built the
shrunk file locally from the release (experts preserved, table streamed from
the prebuilt bytes): `Scion-FlashNext-176B-A6B-q4x160.gguf` 57.615 GB, export
check reproduces the pod exactly (`max|err| 6.537e-3`, rel rmse 0.0965); pack
`pack-q4x160`. Same instrument, same 581 slices, paired per chunk:

| arm | PPL | paired NLL/token delta |
| --- | ---: | --- |
| Q6K (release) | 7.5728 | - |
| Q4X160 | 7.5741 | **+0.00017 +/- 0.00079 (95%)** |

=> **+0.017% PPL, not significant** (95% bound within about +/-0.2%). The
smaller table is free on this instrument; the direction agrees with the fork
gate's +0.41% on the pod instrument. Files: `pack-q4x160` + the q4x160 GGUF in
`/home/penis/models/scion-strata/`; run config `/tmp/opencode/ppl/serve-q4x160.json`
(copy it into `work/strata/` to keep it).

## Remaining for P4

1. **Exact fork-comparable PPL (optional).** The local instrument (above) is
   chat-conditioned; the raw-LM number needs per-position logits in the
   batched prefill path (the head over all T rows exists in `verify.cpp`;
   `--dump-logits` today only fills the per-token loop). Then a 512-slice run
   compares directly to the fork's 5.4561.
2. Consider upstreaming the `corrections.cpp` layer-map fix (the fork never
   had the bug - its corrections follow the tensor's own layer).

## Two-card parallelism research (2026-10-13)

Why two 7900 XTs do not double decode: the engine uses **pipeline (layer)
parallelism** - layers 0-25 on card A, 26-47 on B, one hand-off per window -
so for a single conversation only one card computes at a time; the second
card's contribution is capacity (a second expert cache). Measured: 46.5 tok/s
without corrections (our card-pair ceiling) vs the vendor's 55-59 on one
7900 XTX with a larger cache, no split and no corrections.

What exists for real parallel compute:
- **llama.cpp upstream: experimental `-sm tensor`** (splits weights and KV via
  a meta-device abstraction); on ROCm it needs `-DGGML_HIP_RCCL=ON` (RCCL),
  which upstream keeps OFF by default - "not universally beneficial during
  testing". Not applicable to our custom containers (the fork's base has no
  tensor mode, and Studio's llama.cpp cannot load the release).
- **Vendor Strata is at 0.1.44** (released 2026-10-13); this tree is 0.1.37 +
  the Scion commit. Their multi-GPU work stays pipeline: draft layer on the
  first card, 8K prompt chunks across the split, measured PCIe share;
  `--pipeline-windows` is opt-in for two cards and loses ~3% on code there.
  RDNA3 (gfx11) keeps 0.1.42 behaviour ("we have no such card to measure on").
- **Expert-parallel MoE** (each card computes the experts it holds) fits this
  box: the caches already hold disjoint expert sets. HarMoEny
  (arXiv 2506.12417) reports +37-70% under load imbalance. Requires engine
  work - dispatch is per-layer-owner today.

Plan and next steps live in **`docs/NEXT-SESSION-MULTIGPU-PERF.md`**:
(1) fold the corrections into the layer matmuls (~11 ms/token, the certain
win), (2) merge upstream 0.1.44 and A/B, (3) scope expert-parallelism (and
measure card-2 idle time first).

## The 0.1.44 upgrade (2026-10-13 night)

**The merge is a port, not a merge.** Upstream rewrote the repository history
between 0.1.37 and 0.1.44: their v0.1.44 root is a fresh import, `git
merge-base db4f91a1 v0.1.44` is empty and a merge refuses with "unrelated
histories". The branch's net diff was applied with `git apply --3way` and 29
conflicted hunks were resolved by hand (PLE format registry, gate_attn's out32
through the new ld16 signature, the corr_apply sites around il_ready/mm and the
remote combine, PTQ1_0 through the new Fmt/Split templates, engine/CMake
sources). Branch `scion-0.1.44`: port `319cf7f5`, fixes `42bef63a`.

**Parity on 0.1.44** (4l files): `lora_corr_parity` 8 pairs, 0 of 7680;
`ple_q4_parity` PASS; `ptq1_0_parity` 0 failures (h 9.69e-08, out 3.70e-08).
Two fixes were needed first: upstream routes split formats through sub-16-lane
kernels whose activation walk is 80/20 q8_1 blocks (256-value IQ blocks), which
PTQ1_0's four-blocks-per-call split cannot use - it read far past the row
(NaN); `kSub16<143> = false` keeps it on the 32-lane multi kernels. The default
path now fuses SwiGLU + q8_1 (writes `hq` only), so the parity test reads the
f32 `h` off the v1 path and the quantized `h` off the default. `ple_x32_parity`
needs its transcode fixture, which was never committed (the P5 lane generated
it ad hoc); the tool is registered, the run is pending.

**Served A/B, fixed protocol** (release, Q6K, reserve 3072):

| engine | corrections | gen tok/s (runs 2-4) |
| --- | --- | --- |
| 0.1.37 tree (this session's port base) | on | 41.4 / 41.8 / 41.0 / 41.4 |
| 0.1.44 port | on | 41.1 / 42.2 / 43.0, then 42.4 / 43.2 / 43.5 |
| 0.1.44 + `--pipeline-windows 2` | on | 61.7 / **68.4 / 70.5** (warming through) |

Two runs of the pipelined engine at the bench prompt: steady **~70-71 tok/s**,
prompt reads 66-98 tok/s; the run-to-run spread is the first two requests after
a start (the same warm-up the non-pipelined engine shows). **The token stream
is exact:** the third warmed request's full reasoning text is byte-identical to
the non-pipelined engine's, and identical across repeated warmed runs; the
observed diffs were only the cold first/second requests, which differ the same
way without the flag. The vendor's "-3% on code, two cards" note does not
reproduce for this workload: this is a +62-68% decode win on the box, and
`--pipeline-windows 2` is now in `strata-q6k-176b.json` and
`strata-q4x160-176b.json` (284 MiB of CUDA0 moves out of the expert cache for
the second verifier and the GDN snapshots; the auto cache sizes around it).

gfx11-relevant items read out of 0.1.38-0.1.44: the gfx1100 100401 table now
covers the small-T expert GEMM (vendor: prompt +31.9%), prompt-path copies moved
through pinned bounce slots, the adaptive-tier and PLE reader work (upstream's
format registry already covers Q4_0, so the P5 port only added Q3X32/Q2X32/
Q4X160), and `--pipeline-windows`.

**Incident note.** The displayed card (card1) wedged twice on 2026-10-13: once
with the cooperative-kernel prototype (22:51, see `COOP-CORR-SAFETY.md`) and
again at 23:41 - a `gfx_0.0.0` ring timeout under kwin - while the freshly
ported `ptq1_0_parity` was exercising the out-of-range Split<143> path, minutes
after the first wedge and without a driver reset in between. No OOM; the box was
rebooted at 23:42. GPU tests after a wedge wait for a reset now.

## Rebalance under pipeline-windows (2026-10-14)

Under `--pipeline-windows 2` the busy counters read card0 63.6% / card1 82.8%
(memory 13-15%), so the auto layer split (26/22) leaves card1 the straggler and
the scalars said to rebalance.  Two paths, both with the fixed protocol:

- **`STRATA_MTP_STAGE0=1`** (the drafter on the first card, upstream's gfx12
  default): **worse** - 52-59 tok/s against 68-71 with the drafter on the last
  card.  The 26-layer card becomes the straggler when the draft chain moves
  onto it; on this box the vendor's gfx12 placement does not transfer.
- **an explicit split of 28** (card0 0-27): 69.5-72.8 tok/s warmed against
  68.4-71.2 auto; the acceptance-independent metric (`STRATA_DECODE_TIMING`)
  puts the warmed window at 32.6 ms against 34.6 - inside the run-to-run
  spread.  The auto split stays.

The warm-up both in speed and in the first requests' token streams is the new
adaptive tier at work: the timing line shows `2.9 swaps/window` on the first
requests settling to ~0.3, then the greedy stream is run-to-run deterministic
and identical to the non-pipelined engine's.

Current served shape: verify window ~32-35 ms warmed at ~2.2-2.4 tokens/window
(GPU-reach wait ~24-26 ms dominates), 68-71 tok/s steady, prompt reads 61-87
tok/s.
