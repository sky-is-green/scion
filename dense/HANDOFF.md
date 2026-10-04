# Dense Clef-Flash ternary corrections — handoff (2026-10-03)

Everything a fresh session needs to continue without re-deriving. Read this
first, then `docs/DENSE-TERNARY-QAT.md` (the plan) and `dense/README.md`
(route + prior-art).

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
rotation, full stop.  The **mixed-precision sweep** (`dense/clef_v2_sweep_gpu.sh`,
100 chunks) maps the residual damage and is non-monotone: keeping `ffn_down`
F16 gives PPL **265**, `attn_qkv` 280, edge layers 273 (vs V2 514) — but keeping
*all* FFN F16 gives **846** (worse), so ternary errors partially cancel across
the stack and the best local placement recovers only ~2x.  That is the PTQ
ceiling; the 514 -> 12.6 gap needs trained placement.  The best variant
(`nodown`) is run through the frozen decision probe to test whether more body
fidelity moves the head at all.

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

1. ~~Package + bridge benchmark~~ — **done** (plumbing validated end to end).
2. ~~Harness integration~~ — **done** (`clef-ternary`; keep CPU lane or build
   the HIP bridge later).
3. ~~Calibration-only baseline~~ — **done** (`dense/clef_calibrate.py`,
   `calibration-uncorrected.json`, harness env `HIVE_TERNARY_CALIBRATION`):
   bench shift repaired, held-out BA 0.533 (chance).  Honest v0, not a usable
   validator.  Diagnostics (`dense/clef_probe_diag.py`) closed the "why":
   decision KD never learned rejection, and the body's hidden fidelity is the
   binding constraint.
4. ~~Clef V2 conversion~~ — **v1 done** (see above): rotation+Lloyd gives
   16.9x PPL and hidden cos 0.498, but decisions stay at chance through the
   frozen head (probe AUC 0.464).  Local follow-ups, cheapest first:
   (a) **no-rotation control** (same exemptions, unrotated ternary) to
   attribute the fidelity gain; (b) **GPTQ/Hessian placement** on the rotated
   weights to push PPL toward f16; (c) mixed precision for the most sensitive
   linears (AUTOGRID-style noise floor).
5. **Retrain the decision channel on V2** (the only route to a usable ternary
   validator; rental needs explicit approval, ~1-1.5 h / ~$1 for the pilot):
   ranking-aware decision loss (pairwise/AUC-style), class-balanced negatives,
   residual-magnitude regularization, hidden KD retained; evaluate on the
   frozen 46-record probe + a fresh holdout.  The V2 base gives the correction
   more signal to work with (hidden cos 0.498 vs 0.336), and the runtime
   supports rotated-basis adapters if needed.  Acceptance: **held-out AUC and
   BA beat the calibrated baseline** (0.569 AUC / 0.533 BA), not bench verdict
   count.  Do not scale unless the pilot moves those numbers.

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
