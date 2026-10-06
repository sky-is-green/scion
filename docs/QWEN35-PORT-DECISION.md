# Qwen3.5 architecture port — decision record

**Date:** 2026-09-24
**Status:** decision recorded; confirmation run in progress
**Related:** [`MODEL-REGISTRY.md`](https://github.com/sky-is-green/bonsai2-ternary-forensics/blob/main/docs/MODEL-REGISTRY.md),
[`QUANTIZATION-LANDSCAPE.md`](https://github.com/sky-is-green/bonsai2-ternary-forensics/blob/main/docs/QUANTIZATION-LANDSCAPE.md),
[`REPRODUCIBILITY-AUDIT.md`](https://github.com/sky-is-green/bonsai2-ternary-forensics/blob/main/docs/REPRODUCIBILITY-AUDIT.md)

## Decision

The ternary QAT/KD recipe is extended to the **Qwen3.5 `qwen3_5` architecture**
through a tractable stand-in (`Qwen/Qwen3.5-2B`) rather than the 27B target,
and its retention is provisionally interpreted using the existing Qwen3
relationship **as a modelling assumption, not a measured result**.

This document records *why*, and exactly which parts are measured, assumed, or
still open. It exists so the decision cannot later be mistaken for a finding.

## Why Qwen3.5

- `Qwen/Qwen3.8-27B` (the Bonsai-2 base) is `Qwen3_5ForConditionalGeneration`
  (`model_type=qwen3_5`) — a hybrid Gated-DeltaNet/full-attention tower.  The
  legacy Qwen3 suffix selector cannot even address its `in_proj_qkv`/`in_proj_z`/
  `out_proj` matrices.
- The **Qwen3.5 small series (0.8B / 2B / 4B) shares that architecture**, so it
  exercises the real port on a node that cannot host the 27B pair.
- Preflight (`inspect_model_targets.py`, pinned revisions) confirms the profile
  is correct: Qwen3.5-0.8B/2B select **150/150** linears, Qwen3.5-4B **200/200**,
  `target_count_ok` true, vision tower and `in_proj_a`/`in_proj_b` controls
  correctly excluded.

### Working hypothesis about Prism's timeline (motivation only, not evidence)

Prism's older Ternary-Bonsai line (1.7B/4B/8B, unrotated `Q2_0_g64`, an
error-compensation signature) and Bonsai 2 (rotated `PQ2_0`, ≈naive scales,
trained residual) are ~2 months apart.  That is consistent with a team that had
the rotation + QAT/KD research ready, shipped the older PTQ line first, then
"sprinted" into Bonsai 2.  This is a **plausible story about Prism**, and it is
why porting to `qwen3_5` is the natural next step, but it is *not* evidence that the
implementation here matches theirs.

## Measured vs assumed

| item | status | value |
|---|---|---|
| Qwen3.5-2B is `qwen3_5` and addressable by the profile | **measured** | 150/150 tensors, widths 2048/6144 |
| Recipe runs end-to-end on `qwen3_5` | **measured** | smoke exit 0, `COMPLETE`, peak reserved 9.65 GiB |
| Smoke trajectory descends | **measured** | ratio 6261 → 6.78 over 500 steps |
| Effective bpw (text-only, embedding FP16) | **measured** | 6.79 / 5.82 / 4.14 (0.8B/2B/4B) |
| Converged Qwen3.5 retention | **unknown** | no run yet |
| Qwen3.5 ≥ Qwen3 resistance to ternarisation | **assumed, unmeasured** | — |
| "Our recipe is Prism's recipe" | **not established** | method proprietary; independent repro failed |

## Why the Qwen3 bit-density is not reused

The temptation is to read retention off the Qwen3 ladder at the Qwen3.5
bit-density.  This is unsafe for three reasons:

1. **On the Qwen3 ladder, bit-density and scale are the same variable.**  Bigger
   model → the embedding is a smaller fraction → lower bpw → better retention.
   Two points cannot identify which axis is causal.
2. **Qwen3.5 has *higher* bpw at matched size** (its vocab is 248320, so the
   FP16 embedding is 27% of the text model at 2B).  Using the Qwen3 bpw is
   therefore an **optimistic** bound, not a conservative one, and must be
   labelled as such.
3. **The Qwen3 ratios were measured with a different quantizer than the
   deliverable** (audit P0 `Q-MISMATCH`: `quantize_rtn_absmean` g128 in the
   pilot vs `quant.quantize` absmean+LS-refine in the packer).  They are not yet
   the deployed numbers.

## Matched-step control (the honest data)

Both legacy runs evaluate every 500 steps, so Qwen3.5-2B can be compared at
matched update budget.  Lower ratio is better.

| step | Qwen3-0.6B | Qwen3-1.7B | Qwen3.5-2B |
|---:|---:|---:|---:|
| 500 | 187.2 | 4.85 | **6.78** |
| 1000 | 68.5 | 4.60 | (running) |
| 2000 | 26.3 | 3.49 | (running) |
| 3000 | 16.2 | 2.67 | (running) |
| 20000 | 1.93 | 1.57 | — |

At 500 steps Qwen3.5-2B sits next to **Qwen3-1.7B**, not next to the 0.6B
model that shares its (roughly) effective bpw.  **Preliminary, non-converged
evidence favours scale-tracking over bit-density-tracking.**

Caveats that keep this preliminary: one noisy point; the legacy runs used
`residual` rotation while the Qwen3.5 run uses `input`; teacher PPL differs
(27.99 / 21.92 / 16.70); and none of these points is converged.

## Pre-registered prediction (to be scored by the 3k screening run)

| hypothesis | predicted converged retention for Qwen3.5-2B |
|---|---|
| **bit-density** `ret ≈ f(bpw)` | ≈ 50% (ratio ≈ 2.0, like Qwen3-0.6B) |
| **scale** `ret ≈ f(params)` | ≈ 65% (ratio ≈ 1.54, between 1.7B and 4B) |

The 3k run is a **screening point, not convergence** (the managed decay has a
10k warmup, so no decay events fire).  It can show which curve the trajectory is
tracking, but a converged number requires a full ~20k run.

## Result — 3k screening (2026-09-24)

`COMPLETE`, exit 0, teacher PPL
16.70, best ratio **3.0091 (33.2%)** at step 3000 (still descending), peak
reserved 9.65 GiB, 3669 s.  Matched-step mean ratio (steps 500–3000):

| comparison | mean ratio(Qwen3.5-2B) / ratio(other) |
|---|---:|
| vs Qwen3-1.7B | **≈ 1.02×** (indistinguishable) |
| vs Qwen3-0.6B | ≈ 0.03 – 0.19× (far better) |

**Verdict: the bit-density hypothesis is falsified.**  At matched update budget
Qwen3.5-2B tracks the same-size Qwen3-1.7B, *not* the Qwen3-0.6B model that
shares its effective bit-density (5.82 vs 5.66).  The trajectory is a function
of **size/architecture**, not of effective bpw.

Consequences:

- The "reuse the Qwen3 bit-density" shortcut is **not** valid — it predicted the
  wrong curve.  The size-matched Qwen3 relationship is the better working
  assumption, and even that is an assumption pending a converged run.
- **"Bit-density is a property of the recipe's value allocation, not its
  quality"** is confirmed: two models at the same bpw can retain very
  differently, because the embedding fraction (hence bpw) and the quality are
  driven by different things.

Caveats (why this is screening, not a converged claim): 3000 steps with no decay
event; the legacy runs used `residual` rotation while this run uses `input`;
teacher PPL differs (27.99 / 21.92 / 16.70); and architecture (`qwen3_5`) is
confounded with size, so "scale" vs "architecture" cannot be separated from one
architecture.  A converged (~20k) Qwen3.5-2B run is required to report a number.

## What this decision does *not* claim

- It does **not** claim Qwen3.5 parity with Prism, or that the recipe here is theirs.
- It does **not** claim a measured Qwen3.5 retention.
- It does **not** claim Qwen3.5 is more ternarisation-resistant than Qwen3 — that
  is an unmeasured assumption, and the one measured point does not support it.
- It does **not** replace the 27B port; the 27B still needs the sharded/blockwise
  path and the missing embedding/norm/export parity.
