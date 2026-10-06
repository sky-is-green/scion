# Dense Clef-Flash ternary corrections — handoff (2026-10-03)

Everything a fresh session needs to continue without re-deriving. Read this
first, then `docs/DENSE-TERNARY-QAT.md` (the plan) and `dense/README.md`
(route + prior-art).

## CURRENT THREAD (2026-10-04): Goal B — shrink Clef into a postable community quant

The objective moved to **Goal B**: a genuinely good ternary Clef for HF (a
community quant, to fine-tune later).  The metric here is generation fidelity
(wikitext-2 PPL, c512, 100 chunks); the decision lane stays with **bf16 Clef**
(67/70, 26/30 — the ternary sidecar cannot fix decisions; see the history
below).

**The pipeline** (local, no rental):
- `dense/clef_v2_convert.py` re-converts `clef-flash-f16.gguf` into the rotated
  basis: `W' = W Rᵀ` on 176 attention/MLP linears, PQ2_0 pack,
  `prism.hadamard.*` metadata.  Key flags: `--sign-seed 1337` (PRF explicit
  signs; `-1` = identity), `--hessian-dir <dir>` + `--gptq-lloyd-scales` /
  `--gptq-act-order` / `--gptq-refine` (Hessian GPTQ; Lloyd or absmean group
  scales; RTN fallback), `--keep-f16 <regex>` (mixed precision), `--no-rotation`
  (control), `--rule lloyd|absmean`, `--self-test` (runtime convention + GPTQ
  API, seconds).
- `dense/clef_v2_hessians.py` captures `XᵀX/N` for the 200 ternary targets from
  the bf16 recurrent-GDN forward in layer-group passes (bounded memory).
  Sets on `/run/media/penis/30CE2C97CE2C577E/storage`: `clef-v2-hessians`
  (48x512-token wikitext windows, 29 GB, 24 min capture) and
  `clef-v2-hessians-128` (128x512, 29 GB, 54 min, group 4 — used for the 120.6
  run).  More windows is the dial (256 ≈ 2 h, group 8).
- PPL runs: `HIP_VISIBLE_DEVICES=1 /home/penis/llama.cpp/build/bin/llama-perplexity -m <model> -f <v2>/wiki.test.raw -c 512 --chunks 100 -ngl 99`
  (wiki.test.raw is built from the cached HF wikitext-2 dataset).
- Sweeps/controls: `dense/clef_v2_sweep_gpu.sh` (layer categories F16),
  `dense/clef_v2_ppl.sh` (f16/V2/deployed + probe), `dense/clef_v2_sweep_cpu.sh`
  (CPU fallback).

**The ladder so far** (f16 = 12.59; TAARDIS-27B to beat: 13.61 at 2.125 bpw):

| variant (signed basis unless noted) | PPL |
|---|---|
| deployed PQ2_0 (unrotated, Lloyd) | 8684 |
| V2 no-rotation control | 9516 |
| V2 identity signs, Lloyd RTN | 514.4 |
| V2 signed basis, Lloyd RTN | **476.1** |
| GPTQ + absmean scales | 2460.3 |
| RTN + absmean scales | 15936.8 |
| mixed precision nodown / noqkv / noedge | 265 / 280 / 273 |
| mixed precision nomlp (deleted, regenerable) | 846 (worse) |
| GPTQ + Lloyd scales | 277.2 |
| **GPTQ + Lloyd + act-order** | **141.6** |
| **+ 128-w Hessians (h128, act-order)** | **120.6** |
| **+ ffn_down F16 (h128 nodown)** | **75.2** |
| + flip-polish 4 passes (same recipe) | 3012.8 (falsified) |
| **QAT (rotation-in-the-loop KD, all-ternary)** | **23.3** |

**Findings:** rotation is the entire V2 gain (control = deployed); the **scale
rule dominates everything** (Lloyd vs absmean RTN = 33x); GPTQ's compensation
works (GPTQ+Lloyd 277 vs RTN+Lloyd 476; 6.5x over its own scale baseline) but
cannot fix a wrong scale rule; Hessian quality is the next live lever once
act-order is on (48 -> 128 windows: 141.6 -> 120.6, 1.17x), and mixed precision
stacks again on top of GPTQ+h128 (keeping `ffn_down` F16: 120.6 -> **75.2**,
1.60x, +2.6 GiB).  **Flip-polish is falsified** in the greedy per-trit form:
the Hessian proxy drops 35-95% per tensor but PPL explodes 75.2 -> 3012.8 —
per-layer proxy descent past GPTQ's constrained point does not survive
end-to-end compounding (the forensics' local-KD warning).  **QAT is the real
lever**: 120.6 -> **23.3** (5.2x) with 1200 steps of rotation-in-the-loop KD on
one L40S (~$3.9), in the same 5.52 GiB all-ternary container (1.85x f16).
Decisions are decoupled from body fidelity (nodown AUC 0.503).

**Next steps (ordered):**
1. PTQ is exhausted: rotation, Lloyd, GPTQ+act-order, h128 windows and
   `ffn_down` F16 are in (**75.2**, 8.13 GiB); flip-polish is falsified
   (3012.8), `--gptq-refine` is a no-op, and the lever ranking says damping,
   sign seeds and further window stacking cannot give measurable gains.
2. **Rotation-in-the-loop KD/QAT is DONE** (2026-10-05, L40S 48 GB rental,
   ~$3.9): the all-ternary h128 base trained 1200 steps against the f16 hidden
   cache -> **PPL 23.25** (5.2x better than the 120.6 PTQ ceiling; 1.85x f16)
   in the same 5.52 GiB container (`v2/clef-flash-v2-qat-a1.gguf`).  Export
   gotcha: the pod trains the reverse-loaded HF layout, so the export must
   re-apply the inverse of the loader's GDN V-head reorder (`_undo_gdn`) for
   attn_qkv/attn_gate/ssm_out before packing (`dense/clef_export_qat.py`).
   Remaining (gated on the generation blocker below): optional longer
   (2400-step) or mixed-base run, then release prep
   (`docs/HF-RELEASE-NOTES.md`).
   **Pre-post test caught a blocker (2026-10-05):** v1 (hidden KD only) passes
   PPL but its **free generation degenerates** (loops / wrong math / code
   garbage) while the f16 control is clean; teacher-forced NLL on the f16's
   own text: 2.02 vs 1.24.  Fix: QAT v2 with **logits KD** (chunked vocab
   projection; the forensics' objective) — needs a new pod (~$4-6).
   **Follow-up (2026-10-06):** logits-KD derisked at 0.8B (A hidden-only
   56.3 / B hidden+logits 45.5 / C mixed+logits 40.4 PPL) — all three still
   degenerate in free generation, so the objective alone is not the fix; the
   0.8B needs far more training.  Artifact-level check: the QAT GGUF's hidden
   cos through the local bridge is **0.838** (f16 control 0.9999), i.e. the
   export is faithful and the limit is training fidelity.  Free-generation
   samples are now part of the acceptance suite.
3. Frozen probe re-run for the record on the QAT body (**AUC 0.573**, p in
   ~[0.37, 0.50], best BA 0.500): decisions stay decoupled, so **bf16 Clef
   remains the validator** and the QAT artifact is the generation-fidelity
   community quant.  Release prep per `docs/HF-RELEASE-NOTES.md` (license
   clean) once release decisions are in scope.
4. **Community landscape (2026-10-06) + the gating experiment.**  Clef and
   Clef-Flash already have plenty of 4-8 bit community quants (bartowski GGUFs
   + imatrix, ggml-org official GGUFs, MLX 4/8-bit, FP8/NVFP4, EXL3, OpenVINO,
   and **W4A16 AutoRound/GPTQ for both sizes** by Vishva007) — but **no
   ternary Clef exists**.  The ternary ecosystem's bar: PrismML
   **Ternary-Bonsai-8B** (trained low-bit; Q2_0 g128 codec; own `prism` fork):
   2.03 GiB, benchmark avg 75.5 vs 79.3 base (≈1.44x PPL), 2nd of all 6-9B
   models; TAARDIS-27B 13.61 PPL; mainline TQ1_0/TQ2_0 (our fork also has
   **PTQ1_0 type 143**, the Prism 1.75 bpw codec); TurboQuant TQ3_1S/TQ4_1S
   (WHT-rotated, third-party fork).  Evaluation norm is **KL / top-1 vs bf16
   on ~250k mixed tokens incl. code** (localbench) or benchmark suites —
   wikitext PPL + a few samples is below it.  Our QAT is novel for Clef but
   not competitive, so **do not post yet**.
   **Gate done (2026-10-06, local, ~1.5 h, no rental).**  Built the
   baselines from `clef-flash-f16.gguf` with the fork's tools
   (`dense/clef_gate_chain.sh`: `llama-imatrix` on `v2/wiki.test.raw`, then
   quantize + 100x512 PPL + free generation via `dense/clef_gen_probe.py`):

   | plain baseline | size | PPL | free generation |
   |---|---|---|---|
   | TQ1_0 ± imatrix | 2.68 GiB | 1,860,827 | token soup (server 500) |
   | TQ2_0 + imatrix | 2.98 GiB | 1,860,827 | token soup |
   | PTQ1_0 + imatrix | 2.73 GiB | 2,097,117 | token soup |
   | Q2_K + imatrix (non-ternary ref) | 3.56 GiB | **13.06** | clean |
   | (QAT artifact, same harness) | 5.52 GiB | 23.25 | math loop 0.846 |

   The plain ternary codecs are **absmax** ternarization (`quantize_row_tq1_0_ref`
   sets `d = max|w|`, no Lloyd; PTQ1_0 is the same rule at group 128 and
   ignores `GGML_PQ2_0_LLOYD`) — imatrix is a no-op for them (TQ1_0 ±imatrix
   PPL identical to 4 dp), they are BitNet-oriented, and per-tensor cos 0.69
   vs f16 (deployed Lloyd ≈0.90) makes the collapse honest.  Two independent
   runtimes (fork + Prism build) produce the same soup.  **The QAT pipeline
   wins the ternary comparison by 4-5 orders of magnitude, but the real
   competition is elsewhere: a plain Q2_K beats the QAT artifact on PPL
   (13.06 vs 23.25), generation (clean vs math loop) and size (3.56 vs
   5.52 GiB), on stock runtimes.**  The community ternary bar (trained
   Bonsai ~1.44x) remains unmet.  **Decision: no release; proposed next
   (needs approval): one proper QAT run (mixed corpus incl. code/math,
   5-10M tokens, hidden + top-k logits KD, ~$8-15) accepted on clean
   generation on this probe suite, PPL ≤ ~18, and KL/top-1 vs bf16 on
   ~250k mixed tokens before any HF upload.**  Machine record:
   `hivebench/experiments/cascade/results/clef-flash-validator-20261003/community-gate-20261006.json`.
5. MoE aside: the Scion MoE route never used runtime rotation (its one rotation
   test was rotate-quantize-unrotate, a different scheme); if Goal B lands,
   porting is worth it — `build_lora_mm_id` supports expert rotation for
   qwen3moe/qwen35moe, but OLMoE is not in the fork's verified arch allow-list,
   and expert Hessians need routed-token capture.  Routers stay untouched.

**Do not:** delete `v2/v2-work/` while a conversion is running (per-tensor
packs); start GPU jobs without checking `power/runtime_status` (the card wedged
once — see the incident section); let `/home` drop below ~10 GB (models are
5-14 GB; everything scratch goes to the external drive or `/tmp/opencode`);
start hour-long runs inside the OpenCode app scope — systemd-oomd kills the
whole scope (three OOMs on 2026-10-04 cancelled the first h128 chain).  Use
`systemd-run --user --unit=clef-h128 /bin/sh dense/clef_v2_h128_chain.sh`
so the run survives app restarts.

## What this is

Train residual corrections for the **deployed** `PQ2_0` **Clef-Flash** body
(Qwen3.5-9B dense/hybrid + joint schema head, Apache-2.0) so the CPU validator
recovers bf16 decision parity. The method is Scion's: freeze the deployed
ternary body **in the loop**, add rank-512 branches on the attention output and
MLP output, distil the f16 hidden states (primary) and the head's decisions
(secondary). Dense route lives in `scion/dense/` — a sibling to `moe/`, not part
of it.

## Assets (local)

- `~/Desktop/work/models/clef-flash-ternary/`
  - `clef-flash-PQ2_0.gguf` — 3.26 GB deployed body (type 142 = PQ2_0/Q1_0_g128,
    `GGML_PQ2_0_LLOYD=1`).
  - `clef-flash-f16.gguf` — 17.9 GB reference (teacher states; regenerable ~70 s).
  - `hf-head/` — joint head + tokenizer + configs; `lm_head.safetensors`.
  - `corrections/cache-smoke/` — 158-sequence teacher cache (bench train+test +
    negatives; hidden + head option logits). One negative entry has non-finite
    teacher hidden; the trainer skips non-finite targets.
  - `corrections/pilot-rental/` — `branches-r512-g128-step78.pt` (1.07 GB) +
    `eval-base.json`, `eval-corr.json`.
  - `corrections/packaged/` — `clef-flash-corr-r512-g128-step78.lora.gguf`
    (71 MB, all-ternary `Q1_0_g128`), the merged single-file release
    `clef-flash-PQ2_0-corr-r512-g128-step78.gguf` (3.10 GiB,
    `adapter.embedded=true`), and `bridge-{uncorrected,corrected}.json` +
    `bridge-*/` scratch/logs from the honest benchmark.
  - `corrections/probe/` — old `clef_embed` binary, `tokens.txt`,
    `pq2_ref.bin` (stale 48-token reference from a pre-2026-10-03 build; the
    current bridge rebuild is bit-identical to the pre-patch `build-cpu`).
  - `v2/` — the Goal-B quant thread: `wiki.test.raw` (PPL corpus),
    `clef-flash-v2-pq2_0-rot.gguf` (identity signs, 514),
    `clef-flash-v2-signs.gguf` (signed, 476), `...-ctrl-norot.gguf`,
    `...-nodown.gguf` (265), `...-noqkv.gguf`, `...-noedge.gguf`,
    `...-signs-absmean.gguf`, `...-signs-gptq.gguf` (GPTQ+absmean, 2460),
    and (`running`) `...-signs-gptq-lloyd.gguf`; `ppl100-*.log` per variant.
  - Hessians: `/run/media/penis/30CE2C97CE2C577E/storage/clef-v2-hessians`
    (200 x fp32 `XᵀX/N`, 48 windows, manifest; `dense/clef_v2_hessians.py`).
  - Scripts: `dense/clef_v2_convert.py`, `dense/clef_v2_hessians.py`,
    `dense/clef_v2_ppl.sh`, `dense/clef_v2_sweep_gpu.sh`,
    `dense/clef_v2_sweep_cpu.sh`.
- Fork: `/home/penis/llama.cpp` branch `moe-corr-runtime` (PQ2_0 + TAARDIS
  virtual targets, dense `qwen35` hooks). `gguf-py` at `/home/penis/llama.cpp/gguf-py`.
- Code: `scion/dense/` (`clef_dense_load.py`, `clef_head.py`, `clef_cache.py`,
  `clef_corrections.py`, `clef_eval.py`, `clef_export.py`,
  `clef_bridge_eval.py`, `quant.py`, `preflight.py`, `RENTAL-RUNBOOK.md`).
  Bridge: `hivebench/tools/clef-bridge/clef_embed.cpp` (batch mode + embedded
  adapter attach). External prior art: `scion/docs/TAARDIS-PRIOR-ART.md`.

## Findings that constrain the next step

1. **Reverse loader is exact** [verified]: f16 GGUF → torch cos **0.99994** vs
   the CPU bridge; PQ2 dequant byte-identical to the fork's
   `dequantize_row_pq2_0`; joint-head sidecar reproduces bf16 `p_correct`
  exactly (0.9219).
2. **Local bf16 torch is unusable** [verified]: two identical evals gave
   different non-finite sets (23 then 50 of 70). f32 is stable; the f32 body is
   31.8 GB and needs a **48 GB** card.
3. **The deployed runtime is the recurrent GDN**; transformers defaults to the
   chunked one. Use `--gdn recurrent` for the deployment path (chunked only as a
   bf16-stability fallback).
4. **Ternary amplifies the torch-vs-ggml kernel gap** [verified]: PQ2 f32
   recurrent torch matches the CPU bridge at only **cos 0.948**. So the torch
   eval is a **proxy**. **Resolved 2026-10-03:** the packaged artifact through
   the bridge reproduces the proxy with zero verdict flips (max |Δp| 0.027);
   the cos gap does not reach the decision metric.
5. **Local GPU policy**: the non-display card is DRM `card0` / PCI `07` /
   torch-device `1`; run with `HIP_VISIBLE_DEVICES=1 --device cuda:0`. The
   display card (`card1`/PCI `03`) carries the desktop. One heavy ROCm process
   at a time; never run torch on the GPU beside a serving engine.

## Pilot result (proxy, 2026-10-03)

A40 48 GB rental, one epoch (78 steps), f32 recurrent, rank-512 both taps,
hidden KD + decision KD, `lr 5e-5 --upstream-clip 1.0`. Parity vs bf16 p and the
gold checker at 0.5:

| split | uncorrected | corrected | bf16 Clef | Tiny-Jev |
|---|---|---|---|---|
| train | 26/70 | **68/70** (2 FA, 0 FR) | 67/70 | 62/70 |
| test | 7/30 | **29/30** (1 FA, 0 FR) | 26/30 | 25/30 |
| hidden cos | 0.23 | 0.37 | 1.0 | — |

Cost ~$0.6; pod deleted.

## Packaging + honest CPU-bridge result (2026-10-03)

Both taps packaged and benchmarked through the deployed CPU path (local, no
spend): adapter export → `qwen35.cpp`/`llama-adapter.cpp` `blk.N.ffn_out`
virtual target → `build-cpu` rebuild → merge into the body
(`adapter.embedded=true`) → bridge (`hivebench/tools/clef-bridge/clef_embed`,
CPU) → joint head sidecar (CPU torch f32). The bridge reproduces the proxy
almost exactly, with zero verdict flips. **Read the held-out checkpoint below
first: the corrected verdict counts are the majority class.**

| split | uncorrected bridge | **corrected bridge** | corrected proxy | bf16 Clef | Tiny-Jev |
|---|---|---|---|---|---|
| train | 18/70 (0 FA, 52 FR) | **68/70 (2 FA, 0 FR)** | 68/70 (2 FA, 0 FR) | 67/70 | 62/70 |
| test | 8/30 (0 FA, 22 FR) | **29/30 (1 FA, 0 FR)** | 29/30 (1 FA, 0 FR) | 26/30 | 25/30 |
| mean p | 0.471 / 0.471 | 0.586 / 0.588 | 0.595 / 0.594 | 0.811 / 0.792 | — |
| hidden cos | 0.231 / 0.235 | 0.371 / 0.382 | 0.372 / 0.383 | 1.0 | — |

- Transfer proxy → bridge on the corrected body: **0 verdict flips on 100/100**,
  max |Δp| 0.027, max |Δcos| 0.004. The 0.948 body cos gap does not touch the
  decision metric.
- The uncorrected baseline on the bridge is more conservative than on the
  proxy (18/70 vs 26/70 train; 8/30 vs 7/30 test) — the module toggles cleanly.
- **CPU latency: ~49 ms/token** (recurrent GDN prefill; 13.8-28.6 s per
  278-589-token record), model load 7.4 s once, head sidecar 40-90 ms/check.
  bf16 Clef was ~3.2 s/check in transformers, so the CPU lane is ~5-6x slower
  per check; the HIP bridge on the non-display card is the latency fallback.
- **Harness backend done** (`clef-ternary`): `ClefTernaryValidator` + resident
  bridge serve + `judge_eval --family ternary`, bit-exact with the offline
  benchmark (max |Δp| 5e-5).  **But the held-out checkpoint below fails the
  artifact**: the corrected verdict counts are the majority class, so the
  training objective — not the plumbing — is what must change.
- Artifacts + JSONs: `models/clef-flash-ternary/corrections/packaged/`
  (`clef-flash-corr-r512-g128-step78.lora.gguf` 71 MB,
  `clef-flash-PQ2_0-corr-r512-g128-step78.gguf` 3.10 GiB,
  `bridge-{uncorrected,corrected}.json`); exporter `dense/clef_export.py`,
  benchmark `dense/clef_bridge_eval.py`.
- Gotcha for raw-API consumers: the embedded adapter is loaded at model load
  but a context only applies it after `llama_set_adapters_lora(ctx, nullptr, 0,
  nullptr)` (the bridge now does this; see the bridge README).
- Follow-up menu below; external prior art surveyed in
  [`../docs/TAARDIS-PRIOR-ART.md`](../docs/TAARDIS-PRIOR-ART.md).

## Held-out checkpoint: the parity number is majority-class (2026-10-03)

Scored the packaged artifact on 27 correct answers outside cascade-bench-v1
(reasoning/code/qa) plus 19 constructed negatives (same last-number
perturbation rule as training, verified wrong by the checker), corrected and
uncorrected bodies, through the harness backend:

| body | bench-train @0.5 | bench-test @0.5 | held-out 46 @0.5 | AUC fit / held-out |
|---|---|---|---|---|
| uncorrected PQ2_0 | 18/70 (0 FA, 52 FR) | 8/30 (0 FA, 22 FR) | 24/46 (6 FA, 16 FR) | 0.539 / 0.569 |
| uncorrected + calibration (t=0.48) | 37/70 (0 FA, 33 FR) | 12/30 (0 FA, 18 FR) | 25/46 (10 FA, 11 FR) | 0.539 / 0.569 |
| corrected (candidate) | 68/70 (2 FA, 0 FR) | 29/30 (1 FA, 0 FR) | 27/46 (19 FA, 0 FR) | 0.412* / 0.435 |
| f16 teacher (reference) | 0.939 (cached fit) | — | 34/46 (2 FA, 10 FR), BA 0.762 | 0.939 / **0.844** |
| always-accept | 68/70 | 29/30 | 27/46 | 0.500 |

\* 2-negative bench artifact; see below.  "fit" = bench-train (68 pos) + 55
usable training negatives; the corrected model accepts all 58 training
negatives (p 0.574-0.605) while the teacher rejects 50/58.

The corrected 68/70 & 29/30 equal always-accept on a 97%-positive bench: the
corrections flatten the head's ranking into p≈0.58 (AUC below chance, constant
accept) and never learned rejection on any data.  The uncorrected body is
miscalibrated and near chance against real negatives — the earlier "uncorrected
AUC 0.80" was a 2-negative artifact; with 55 real negatives it is 0.539.  The
teacher itself scores AUC 0.844 on the frozen probe, so the probe is valid and
the student gap is real.  Branch delta is O(hidden norm) (~110 vs ~120 per
token): a learned bias that swamps the already-weak signal.

Protocol from now on: AUC / TPR-at-fixed-FPR / balanced accuracy plus the
always-accept baseline; never verdict count alone on this bench.  Freeze the
46-record probe; fresh holdout per iteration.  **Do not post the current
corrections to HF** — the license chain is clean
([`../docs/HF-RELEASE-NOTES.md`](../docs/HF-RELEASE-NOTES.md)), the artifact is
not.

## V2 rotated-basis conversion — v1 done (2026-10-04)

`dense/clef_v2_convert.py` re-converted the **f16** body into the rotated
basis: `W' = W Rᵀ` folded into 176 attention/MLP linears (block-Hadamard 1024,
identity signs), Lloyd-g128 ternary, `ssm_out` unrotated, everything else
copied, `prism.hadamard.*` metadata emitted; `dense/clef_v2_ppl.sh` evaluates.
Artifact: `models/clef-flash-ternary/v2/clef-flash-v2-pq2_0-rot.gguf` (5.52
GiB).

| metric | f16 | deployed PQ2_0 | **V2 rotated** | V2 no-rotation control |
|---|---|---|---|---|
| wikitext-2 PPL (c512, 100 chunks) | 12.59 | 8684.23 | **514.35** | 9640.77 |
| hidden cos vs f16 (48 tok) | 1.0 | 0.336 | **0.498** | 0.311 |
| frozen-probe AUC through the head | 0.844 | 0.569 | **0.464** | — |
| probe best BA | 0.762 @0.5 | 0.533 (calibrated) | 0.519 @0.45 | — |

The no-rotation control rules out the exemption set: the fidelity gain is the
rotation, full stop (a PRF-signed basis, seed 1337, improves PPL to **476.08**
from 514.35 — the TAARDIS-style explicit-sign variant).  The **mixed-precision sweep** (`dense/clef_v2_sweep_gpu.sh`,
100 chunks) maps the residual damage and is non-monotone: keeping `ffn_down`
F16 gives PPL **265**, `attn_qkv` 280, edge layers 273 (vs V2 514) — but keeping
*all* FFN F16 gives **846** (worse), so ternary errors partially cancel across
the stack and the best local placement recovers only ~2x.  **The best variant
does not move the head**: `nodown` through the frozen decision probe is
**AUC 0.503** (pos med 0.433, neg med 0.435), exactly chance.  Body fidelity
and head decisions are decoupled in this regime, so local PTQ/mixed-precision
work is exhausted; the ternary validator needs a training run on the decision
channel (or bf16 Clef keeps that role).

For Goal B (the community quant), the quantizer investigation so far: the
**scale rule dominates GPTQ** — RTN+Lloyd 476.08, GPTQ+absmean 2460.25,
RTN+absmean 15936.78 (33x between the two Lloyd/absmean RTN runs).  GPTQ's
error compensation helps 6.5x within a scale rule but cannot fix a wrong one;
**GPTQ with the deployed Lloyd scales** (`--gptq-lloyd-scales`, Hessians from
`dense/clef_v2_hessians.py`) is the run in flight, with `act-order`/`refine`/
flip-polish queued and KD only if PTQ plateaus.

So: the rotation is real (16.9x PPL, hidden cos past even the trained
corrections' 0.372), but **the frozen head does not rank better on V2** — the
decision signal still needs training.  A controlled no-rotation variant and
GPTQ/Hessian placement are the cheap local follow-ups; a ranking-loss retrain
on V2 (rental, needs approval) is the route to a usable ternary validator.
Fork fix committed alongside (`e68c84a49`): rotation tensors now fall back to
the plain CPU buft when weights are repacked — rotated models no longer crash
the CPU path, and the bridge's `CLEF_EMBED_NO_REPACK` workaround is only for
older builds.

## Next-step menu (after calibration + diagnostics + V2 v1)

> Superseded by **CURRENT THREAD** at the top for the Goal-B quant work; the
> items below remain the decision-lane history and the optional retrain route.

1. ~~Package + bridge benchmark~~ — **done** (plumbing validated end to end).
2. ~~Harness integration~~ — **done** (`clef-ternary`; keep CPU lane or build
   the HIP bridge later).
3. ~~Calibration-only baseline~~ — **done** (`dense/clef_calibrate.py`,
   `calibration-uncorrected.json`, harness env `HIVE_TERNARY_CALIBRATION`):
   bench shift repaired, held-out BA 0.533 (chance).  Honest v0, not a usable
   validator.  Diagnostics (`dense/clef_probe_diag.py`) closed the "why":
   decision KD never learned rejection, and the body's hidden fidelity is the
   binding constraint.
4. ~~Clef V2 conversion + local follow-ups~~ — **done and exhausted**:
   rotation gives 16.9x PPL and cos 0.498; the no-rotation control rules out
   the exemption set; mixed precision recovers at most ~2x (`nodown` 265) and
   is non-monotone; and the best variant still scores **AUC 0.503** on the
   frozen probe.  Body fidelity and head decisions are decoupled here — PTQ
   cannot make the ternary validator.
5. **Retrain the decision channel** (the only route to a usable ternary
   validator; rental needs explicit approval, ~1-1.5 h / ~$1 for the pilot):
   ranking-aware decision loss (pairwise/AUC-style), class-balanced negatives,
   residual-magnitude regularization, hidden KD retained; base = V2 or nodown;
   evaluate on the frozen 46-record probe + a fresh holdout.  Acceptance:
   **held-out AUC/BA beat the calibrated baseline** (0.569 AUC / 0.533 BA),
   not bench verdict count.  If it does not move, keep **bf16 Clef** as the
   decision validator (67/70, 26/30) and treat the ternary line as the
   generation-fidelity artifact (V2 + mixed precision).

## Host incident (2026-10-04): non-display GPU wedged by runtime-PM resume

- `0000:07:00.0` amdgpu PSP resume timed out at 13:42:45 right after several
  back-to-back GPU PPL jobs (V2 / deployed / f16), leaving
  `power/runtime_status=error`; HIP then reports no ROCm device (KFD still
  enumerates both GPUs, PCIe AER clean).  Reboot to recover.
- Prevention once back: keep the card out of runtime suspend
  (`echo on > /sys/bus/pci/devices/0000:07:00.0/power/control`, root) and avoid
  rapid short-lived GPU process churn (keep one persistent context).
- The no-rotation control PPL (9640) was measured **on CPU** because the GPU
  was already wedged (CPU-accumulated; the attribution conclusion is
  unchanged).  The mixed-precision sweep was interrupted after `nodown` and
  `noqkv` finished converting; rerun `dense/clef_v2_sweep_cpu.sh` (or on GPU)
  after the reboot.

## Constraints (unchanged)

- **Never `git push`; commit locally.** Repos are ahead of origin by design.
- **Ask before heavy runs, downloads, and any rental.** The pilot rental was
  approved and closed; a longer run needs new approval.
- One heavy track at a time (30 GB host RAM); use the non-display card.
- hivebench/splinter pre-commit run a claims gate: stage only paths covered by
  `.claims/<id>.json` (create a local, gitignored claim if needed).
