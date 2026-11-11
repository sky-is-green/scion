# 003 - RoleQ: Role-Aligned Scales for Converting Language Models to Ternary

- arXiv: 2610.21808, 2026-10-15
- Authors: Johannes Knittel (Harvard), Alec Helbling (Georgia Tech),
  Hanspeter Pfister (Harvard)
- Link: https://arxiv.org/abs/2610.21808
- Status: analysed, actionable (scale axis + scale distillation + second plane)

## Claim

Below four bits, a perplexity that looks acceptable can hide large losses in
factual knowledge. Ternary conversions so far come from training from scratch,
long fine-tuning, or post-training methods that lose 40-50 MMLU points.
RoleQ places scales by the functional role of each weight vector rather than
by matrix layout: vectors that read from the residual stream get one scale per
output unit, vectors that write to it one per input unit. The rule is
structural, needs no search or calibration, and holds on integer and ternary
grids.

## Method

1. **Role-aligned scales**: readers (Q/K/V rows, gate/up rows) get per-output
   scales; writers (O columns, down columns) get per-input scales.
2. **RoleQ-T**: a data-free learned rotation (r1 with an l4 objective, r2 opt,
   r4), then ternary GPTQ with an exact per-vector ternary fit
   (support = prefix of |w| sorted descending, scale = mean of the support),
   then scale distillation.
3. **Scale distillation**: codes frozen, only scales (log a) and norm gains
   trainable (0.024% of an 8B), forward KL to the teacher on unlabelled data,
   400 steps, ~2 A100-hours at 8B.
4. **Optional second plane**: a sparse second ternary plane on units ranked
   by per-unit Fisher saliency (S_u = sum F_i delta_i^2), fitted as
   w = a1 t1 + a2 t2; up to nine levels, still two ternary matmuls.

## Numbers that matter

- Qwen3-8B-Base, ternary one plane, 1.590 bpw: WT2 11.33, zero-shot 60.33,
  MMLU 47.59 after scale distillation (PTQ before SD: 22.42 / 50.39 / 33.32).
  Layout-based placements at the same bits: per-OC 35.47 ppl, per-IC 33.39.
- Qwen3-8B-Base with sparse second plane, 2.157 bpw: 9.30 / 65.89 / 62.75.
- Qwen3-14B, 1.589 bpw + SD: 9.53 / 65.37 / 59.75. At 2.156 bpw with planes:
  8.05 / 69.21 / 68.31.
- At 14B, role-aligned scales lift MMLU from 25.4 (chance, per-OC) to 38.5
  before any distillation.
- Rotation matters: a random Hadamard alone takes 1.7B ternary PTQ
  perplexity from 1245.9 to 119.8.
- The second plane costs ~0.57 bpw at 8B and recovers about half the MMLU
  lost to conversion; Fisher selection beats random by 6.2 points and
  Hessian x weight by 9.0 at 1.99 bpw.

## What transfers to Scion v2

1. **Scale axis is free quality.** Our Q2_0 expert banks use one fp16 scale
   per 128 weights on one axis (layout-based). RoleQ's rule is data-free and
   format-compatible: gate/up are readers (per-output scales), down is a
   writer (per-input scales). If we ever re-quantize the body, this is the
   first thing to change.
2. **Scale distillation is the cheapest body-level improvement that is not
   QAT.** Codes stay frozen; only scales and norm gains move; the format and
   file size are unchanged. This fits between "frozen body" and "bf16 SFT
   escalation": it is a format-preserving calibration, not a body retrain.
   At 8B it is 2 A100-hours; for 35B budget roughly 6-12 hours on the pod.
3. **Second plane as a quality knob.** If corrections plateau, a sparse
   second ternary plane on Fisher-selected units is a way to spend ~0.5 bpw
   for a large knowledge recovery, without leaving the ternary format.
4. **Rotation + KD vs our D1.** D1 falsified rotation before RTN without a
   KD stage. RoleQ reports rotation is essential when followed by GPTQ and
   scale distillation. This does not contradict D1 but it does mean "rotation
   is dead" is too strong for a full conversion pipeline.

## What does not transfer directly

- The paper converts dense models (Qwen3-0.6B-14B, Llama-8B). Our target is
  a 35B-A3B MoE with expert banks, and the reader/writer roles apply per
  expert matrix but the routing dimension is new.
- RoleQ-T changes codes and scales (a conversion), which our frozen-body rule
  forbids for v2. Only the scale-distillation stage is compatible as-is.
- The learned rotations fold into weights; not applicable without
  re-quantizing.

## Experiments this suggests

1. Run scale-only distillation on the current body (frozen codes, train
   scales + norm gains, forward KL to the teacher). Measure hard-90, loop
   onset, and PPL. It is the lowest-risk body-side change we have.
2. If a re-quantization ever happens (v3): role-aligned scales + rotation +
   scale distillation, and compare with our PQ2_0 at matched bytes.
3. Prototype the Fisher second plane on the expert down projections (the
   writers) and measure hard-90 per added 0.5 bpw.

## Open questions

- How much of RoleQ's gain is the scale axis vs scale distillation? The paper
  separates them (Table 2 vs Table 1), but not on MoE.
- Does scale distillation survive a 35B-A3B with 3B active, or does the
  correction sidecar already absorb the available gain?
- Would a second plane on selected experts cost less than a correction round
  for the same accuracy?
