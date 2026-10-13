# Shrink plan — below the 60.17 GB release

**Status (2026-10-11): expert pruning measured and closed. The next lane is
the PLE table.** Headline result: the 25% REAP-uniform prune is code-neutral
in paired tests but trades general quality for a small size win; 50% uniform
collapses the thinking mode. Measured trade: **-6.64 GB for wikitext PPL
+11.6%, HellaSwag -2.75, Winogrande -0.50, KLD +0.19 (same instrument)**.
Not worth it for the general release; it is only defensible as a coder
specialist (53.54 GB file, code-neutral, general side pays). Full evidence:
`docs/CODER-VARIANT-GATES-25.md`; survey: `docs/CODER-VARIANT-PRIOR-ART.md`.
**Next session: `docs/NEXT-SESSION-P5-PLE.md`** (unpruned table lever; no
expert deletion). Do not re-open pruning for the general release.

**Status (2026-10-12, P5 PLE lane):** **Q4X160 is gated and chosen as the
shipped container** (upload-path window, L40S, $0.62): KLD 0.7819 -> 0.7818
(-0.0001), top1 0.6566 -> 0.6588, PPL 5.4561 -> 5.4784 (**+0.41%**), HS
82.00 -> 82.00, WG 77.75 -> 78.00; file 57.615 GB table-only (~56.0 GB with
the body/corrections levers). The measured cost sits at/below the low end of
the +0.45-0.75% estimate and is ~1/3 of Q3X32's (+0.011 KLD / +1.30% PPL /
-0.25 HS / -0.25 WG at 53.76 GB). Q3X32 remains the small-size fallback
(52.2 GB with levers). Same-pod paired code arms for Q4X160 ran 2026-10-12
19:00Z (`paired-code-table.sh`, upload path): gen pass@1 73.2 -> 68.3 (all) /
71.4 -> 61.2 (test), hard 55.5 -> 53.0 / hard-4096 57.9 -> 56.1 (all); the
paired flips (22/14, 24/20, 23/20) sit inside the McNemar noise band - no
significant code/thinking regression. Balance **$9.78** before the arms
window; nothing pushed.

**Status (2026-10-13, local serving + validation):** the release runs locally
in Strata on the 2x7900XT box (~30.6 tok/s with corrections; 46.5 without, the
measured bottleneck), served as the user service `scion-flashnext` (web chat
`http://127.0.0.1:8081/`; Unsloth Studio's provider points at it). Q4X160
independently cross-checked locally: **+0.017% PPL paired (not significant)**
over 581 x 512 wikitext slices. VRAM headroom policy + tool in
`work/scion/serving/vram_reserve.py`. **Next lane: local speed - the
corrections fold, the Strata 0.1.44 upgrade, then expert-parallelism scoping:
see `docs/NEXT-SESSION-MULTIGPU-PERF.md`.** Balance **$6.45**, no pods,
nothing pushed.

**PLE lane (2026-10-11, local-first, zero pod spend):** the Q3X32/Q2X32
32-element table types are implemented and validated end to end locally: fork
readers (CPU + HIP, bit-exact on real 4l rows), Strata reader + fixture
parity (bit-exact against the fork's decoder), streamed exporter
(`repack_experts_q2_0.py --table-type q3x32|q2x32`), and `ple-table.patch`
verified on a pristine `qwen4exp-proto` + the 5-patch sequence. The codec is a
learned 8-level codebook with a per-block least-squares scale (rel rmse on
real rows 24.7% -> 19.1% against the first mid-rise rule; beats the 2-fp16
affine option at the same size). A release dry run reads **60.164 -> 53.764 GB
with all 512 experts intact** (table 28.800 -> 22.400 GB). Local 4l A/B on the
full 580-chunk wikitext corpus (deterministic, paired): the Q3X32 table is
**-0.278% PPL** vs the Q4_0 table (t +6.1) - no intelligence loss at the proxy
level. Design and evidence: `docs/PLE-TABLE-TYPES.md`.

**Pod gate (2026-10-12, L40S, one paired window, $3.61):** KLD 0.7819 ->
**0.7927 (+0.011)**, PPL 5.4561 -> **5.5269 (+1.30%)**, HS 82.00 -> 81.75,
WG 77.75 -> 77.50; the release side reproduced the historical baselines, so
the deltas are clean. For scale the 25% prune cost +0.186 KLD / +11.6% PPL.
**Open decision:** ship Q3X32 with the +1.3% PPL noted, or use a later window
for the (unmeasured) paired code arms and/or a mixed-precision table (keep a
fraction of rows at Q4_0). Row-hit map (4.9M tokens over wikitext/web/math/
coder): 90% of reads land in 6.0% of rows, so a hot-Q4_0/cold-Q3X32 table
would cost +0.4 GB and ~+0.1-0.3% PPL. Balance **$1.77**, no pods running.

**Uniform alternatives measured (2026-10-12):** the container ladder on real
rows now includes **Q4X160** (type 146, one fp16 scale per 160-element row,
16-level codebook): 82 B/row = 26.24 GB, 9.7% table error, **measured
+0.41% PPL** - implemented with full parity (fork CPU/HIP, Strata fixture) and
a `--table-type q4x160` exporter mode. The menu: **52.2 GB @ +1.3%** (Q3X32,
gated), **56.0 GB @ +0.41%** (Q4X160, gated 2026-10-12), **58.6 GB @ 0%**
(release table + the body/corrections levers). 16-level containers cannot go
below 4.1 bpw (the codes fill 80 B of the 82 B row).
**Q4X160 gate: DONE (2026-10-12), chosen.** Two early windows aborted cleanly
(a `HOST_TOO_SLOW` preflight, then the spend guard mid-parallel-transcode on a
pathologically slow host; spend ~$1.3), so the gate used the **upload path**:
the prebuilt stream `work/tmp-export/q4x160-table.bin` (26,240,125,952 B,
sha256 `915bae37c28915d0378772fffd17329c9046a606555cabfcb93f3e5cd0803015`) is
uploaded in 5 GB chunks, spliced pod-side (`--table-bytes`) and gated - no
pod-side CPU encode, host speed irrelevant. In-window result (release KLD
0.7819 / top1 0.6566 reproduced exactly): KLD **0.7818 (-0.0001)**, top1
0.6588, PPL 5.4784 (**+0.41%**), HS 82.00 (0.00), WG 78.00 (+0.25); file
57.615 GB table-only; window spend $0.62. The same-pod paired code arms for
Q4X160 ran 2026-10-12 19:00Z. A parallel pod-side encoder also exists
(`parallel_table_transcode.py`, byte-identical; needs progress logging before
reuse).
**Healing:** the QAT machinery already exists in `qwen4exp_proxy.py`
(`qat-screen`, `--ple-qat-bits` STE hook on the branches) and the screen ran
free locally: branch-only adaptation does not measurably reduce the 3-bit
damage (-1.6% at 2 bits, ~0% at 3 bits, 30 steps), so a correction re-fit pod
is not worth funding; plan in `docs/PLE-HEALING-PLAN.md`.

**Direction (decided):** the first pod gate window runs the **table only**
(Q3X32), measured against the release baseline with the same instrument; the
companions (PTQ1_0 repack -2.4 GB lossless, body K-quants -1.3 GB, corrections
mixed -0.3 GB) are bundled only after their own local parity/gates, so the
table signal stays attributable and the budget ($5.48) is spent on one clean
comparison first. Q2X32 stays behind the Q3 gate. No pod booking without
asking.

*Companion arithmetic check:* the PTQ1_0 repack lever (28 -> 25.4 B/128,
"-2.4 GB lossless") does not survive its own bound: 128 trits need
`128*log2(3)/8 = 25.36 B` at the entropy limit and the per-128 fp16 scale adds
2 B, so the best lossless container is 27.36 B/128 (about -0.6 GB, not
-2.4). Getting to 25.4 B/128 means dropping or sharing scales, which is lossy
and needs the gate. Re-derive before scheduling that lever.

Evidence trail from the executed P5:

- *Step 1 done, verified.* `router-stats.patch` (fork patch 4): env-gated
  per-expert routing counts, gate mass, full-softmax mass **and the REAP
  signal** (gate x expert-output L2 norm + norm moments) via
  `SCION_MOE_STATS=<path>`. Zero overhead when off re-verified on the 4l
  proof model after the REAP change (identical output, no stats file, wall
  2.00/1.88 s unpatched vs 1.94/1.85 s patched).
- *Step 2 ready.* Coder calibration corpus (`build_coder_corpus.py`):
  Evol-CodeAlpaca teacher-forced + on-policy generations, glaive
  tool-calling, fineweb/GSM8K ballast; vision impossible (release GGUF has no
  vision tower). Map window scripts stage the corpus as
  `router-corpus-coder.jsonl`.
- *Step 3 ready.* `repack_experts_q2_0.py` is prune-aware and its `--rank-by`
  now includes `reap` / `norm_sum`; tested on the 4l at keep 256. KLD gate
  (`llama-kld`, patch 5) verified against `moe/kld_eval.py`'s math; the
  48-layer park converted locally.
- *Gates for the coder variant* follow the survey, not the original order:
  functional code evaluation first (the 664-task suite's 164 code tasks on
  the pod, plus KLD/PPL diagnostics only), then the general/reasoning trade.
- *Paused on A6000 capacity* (2026-10-10): auth restored, but the map window
  booking hit "no instances available"; resume with
  `pod-moe-stats-book-retry.sh`. No spend incurred.
- *Map attempts 1-3 (L40S, 2026-10-10) all hit patch bugs, all fixed*:
  attempt 1 aborted because the stats patch reserved the scheduler mid-run
  (drops the GPU split); attempts 2-3 collected 1.4M+ tokens but maps carried
  sporadic `-inf` from post-compute readbacks racing buffer reuse. Final
  design: the collector now reads each captured tensor **at compute time**
  through the sched eval callback (`ggml_backend_sched_set_eval_callback`),
  with the ubatch serialized by a recursive mutex; no buffer pinning, no
  post-compute readback. Re-verified: zero-overhead (env off), CPU numbers
  identical to the old design, local HIP cli -ngl 99, and a 4-slot concurrent
  server stress test with per-64-token flushes (strict-valid map, zero
  inf/nan). Patch sequence 1-5 applies on a pristine fork.
- **Map complete (2026-10-11)**: `router-stats-coder.json` (sha256
  `3014d6495f0d6528f19399a3048efa9ba0e389b13a87d1cd233967a6d4b16236`,
  2,415,415 tokens, strict-valid; 4,184/4,186 rows). REAP-ranked 25% plan
  generated (`prune-plan-25.json`): the dropped quartile carries only
  **2.41% of the REAP signal, 3.36% of selected routing mass, 4.24% of
  counts**; worst layer 4.89%. Keep-set overlap vs mass/counts/norm_sum
  97-98% (the dense-softmax family differs at 80%). Paired re-run on one pod
  (2026-10-11) supersedes the cross-window code comparison: **at 25% the code
  gates are statistically indistinguishable** (gen test delta 0.0, hard -3
  to -5 points with flips inside the McNemar noise band, unchanged at a 4096
  cap), while the systematic costs are general-text (PPL +11.6%, HS -2.75,
  WG -0.5) and KLD (+0.19 in the fork instrument). Full table:
  `docs/CODER-VARIANT-GATES-25.md`. Three-way tier test (release / 25% / 50%,
  one pod): gen holds at both tiers, hard/thinking holds at 25% but collapses
  at 50% uniform (test 30.6% vs 49.0%, finished accuracy 68.2% vs 92.3%) -
  matching ISTA's uniform-50 warning. A 50% file needs nonuniform per-layer
  allocation or a correction re-fit; the 25% file (53.54 GB) is the safe
  coder candidate.
- **Pruning lane closed (2026-10-11).** 25% = code-neutral, general cost as
  above; 50% uniform breaks thinking (hard test 30.6% vs 49.0%, finished
  accuracy 68.2% vs 92.3%). Coder specialist remains available on demand;
  nonuniform allocation or a correction re-fit would be its next step, not
  the general release's.
- **Next: PLE table format** (`docs/NEXT-SESSION-P5-PLE.md`): Q4_0 28.8 GB
  (90 B/row, 160-element rows = 5 x 32-blocks) -> Q3x32 (70 B/row, 22.4 GB)
  or Q2x32 (50 B/row, 16.0 GB), plus optional lossless levers (ternary repack
  28 -> 25.4 B/128: -2.4 GB; body Q4_K/Q5_K: -1.3 GB; corrections mixed:
  -0.3 GB). Combined unpruned target ~46-52 GB with all 512 experts intact.

**Original plan (2026-10-09):** plan only, nothing measured yet. The released
Scion-FlashNext-176B-A6B is 60.17 GB / 2.72 bpw and is the smallest *unpruned*
Flash-Next quant published; the only smaller file on the Hub is ISTA's
**expert-pruned** coder IQ1_M at 58.4 GB. This document is the handoff for the
next quant pass: what to shrink, in what order, and the gates each step must
pass.

**Goal.** Cut the file below 60 GB with a measured quality cost, and publish
whatever passes the gates. Two honest targets: a conservative pass
(expert pruning only, ~53–54 GB) and an aggressive pass (pruning + table +
body, ~46–48 GB). Do not ship a file whose gates regress beyond the stated
bands.

## File budget (measured / derived)

| part | size | share | note |
| :--- | ---: | ---: | :--- |
| PLE n-gram table (Q4_0, 320,001,536 rows x 160) | 28.8 GB | 48% | 90 B per row, read per row |
| expert banks (PTQ1_0, 1.75 bpw) | ~26.4 GB | 44% | 48 layers x 512 experts, ~120.8B params |
| body (Q6_K + Q8_0) | ~4.1 GB | 7% | dense projections + head |
| corrections (F16, rank 512, 96 pairs) | ~0.68 GB | 1% | embedded adapter |

## Levers, by size impact

1. **PLE table: Q4_0 -> Q3_0 / Q2_0 (est. -6 to -13 GB).** The table is 48% of
   the file and is a lookup, not a dense weight. Needs a fork reader for the
   new table type (and Strata later), plus a quality gate — the PLE feeds
   every layer, so table error is not a small perturbation. No such table
   format is published for this model yet; this is the biggest single win.
2. **Expert pruning (25% -> ~-6.6 GB; 50% -> ~-13 GB).** Drop the least-used
   experts per layer, then re-index the bank and rewrite the router. Needs:
   - an **expert usage/importance map** — the missing input. Instrument the
     fork's `qwen4exp` router to count top-k selections (and weight mass) per
     expert per layer over a calibration set: fineweb-edu text plus the
     project's AYOT prompt set (`moe/build_ayot_prompts.py`, 128 math / 128
     coding / 256 web). The same 664-task community suite is the eval check;
   - an **exporter** that removes experts, rewrites `ffn_gate_inp` (and the
     shared-expert tensors), and renumbers the bank;
   - a **gate**, because the corrections were trained against the full bank —
     expect to either re-fit the corrections (the KD recipe with the deployed
     quantizer) or accept and measure the loss.
3. **Tighter ternary packing (~-2.5 GB).** PTQ1_0 spends 28 B per 128 trits
   against the 25.4 B entropy bound (~10% overhead). A denser container is a
   new ggml type id + fork + Strata kernel work; do it only after 1–2.
4. **Body: Q6_K+Q8_0 -> Q4_K/Q5_K (~-1.3 GB).** The fork already runs these
   types; the body is only 7% of the file, so this is the last easy lever.
5. **Corrections: per-layer mixed precision (~-0.3 GB).**
   `moe/branch_sensitivity.py` ranks which layers need F16 and which can go
   ternary; the saving is small but the method exists.

## Gates (in order, cheapest first)

1. **KLD gate** — `moe/kld_eval.py`, the 48-layer 8x512-token wikitext-2 gate
   the release already reports (mean 0.5800 for step-3000). A shrink step must
   not move it beyond its error band.
2. **HS/WG 400** — HellaSwag 82.00 / Winogrande 77.50 on the release; the
   comparison is paired over the same 400 tasks.
3. **Community arms** — the 664-task pod run (GSM8K-300 + MATH-200 +
   HumanEval-164). Compare like-for-like with the same caps (700 gen / 2048
   hard): the release numbers are floors (26% of think-harder records truncate
   at 2048), so a smaller model that answers sooner can *look* better — report
   both overall and finished-answer accuracy, and consider a larger cap for
   the comparison.

## First steps for the next session

1. Fork patch: per-expert routing counts + weight mass, env-gated, zero
   overhead when off.
2. Collect the map on a calibration corpus (pod, a few GPU hours); publish the
   map JSON alongside the results.
3. Prune at 25% and run the KLD + HS/WG gates; only if that holds, try 50%
   and/or the table format.
4. Decide on a correction re-fit after the first prune measurement, not
   before.
5. Re-benchmark on the pod with the fixed recipe
   (`hivebench/.../pod/remote-w1w2/recipe/pod-community-book.sh`) and update
   the model card with measured numbers.
