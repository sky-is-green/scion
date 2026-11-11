# 004 - Align, Then Correct: Training-Free Two-Stage Low-Rank Compensation for Extremely Quantized LLMs

- arXiv: 2610.08164, 2026-10-06
- Authors: Seobin Song, Geonho Lee, Janghwan Lee, Jungwook Choi
  (Hanyang University / Qualcomm AI Research)
- Link: https://arxiv.org/abs/2610.08164
- Status: analysed, actionable as a baseline and as a polishing step

## Claim

Every training-free low-rank quantization-error compensator (LoftQ, LQER,
QERA, EoRA, CLoQ, ProjQ, LQ-LoRA) shares two simplifications:

1. **Symmetric calibration**: they evaluate the full-precision and the
   compensated weights on the same activation, so the target is
   (W - W_hat) X_hat. That target is high-rank, so a fixed rank budget
   captures little of it. The asymmetric target W X - W_hat X_hat, built from
   the activations each model actually produces, is far more rank-compressible.
2. **Second-order only**: they minimize the quadratic loss term under the
   assumption that the model is stationary. Quantization displaces it, and the
   remaining first-order descent direction is larger than the compensation in
   every layer, by more than 10x in most.

## Method

- **Stage 1**: Fisher-weighted asymmetric alignment. For each layer, solve a
  reduced-rank regression whose closed-form solution is one truncated SVD of
  `H_z^0.5 C H_x^-0.5`, with C = W H~ - W_hat H_x. Blocks are calibrated front
  to back with refreshed input moments; the update is shrunk by a factor
  because calibration singular values are inflated.
- **Stage 2**: re-measure the statistics on the compensated model and take a
  single rank-constrained natural-gradient step, solved as one SVD of
  `Y0 - M` where Y0 is the whitened Stage-1 adapter and M is the whitened
  natural-gradient direction. A trust ratio scales the step.
- No iterative optimization: backward passes only collect statistics; every
  adapter is a closed-form SVD.

## Numbers that matter

- Qwen3-8B at 2 bits under QuIP#: WikiText-2 perplexity 12.43 -> 10.26.
- Qwen3-4B at 2 bits: 21.11 -> 13.22.
- Held-out C4: recovers 51% and 84% of the gap to FP16, versus 31% and 63%
  for the strongest baseline. Gains hold on the seven-task zero-shot average,
  at higher bit-widths, and under a different quantizer.
- Stage 2 contributes the largest gain and needs only 88K calibration tokens.
- Adapters are a rank-r matrix per layer, one SVD each, minutes on a single
  GPU.

## What transfers to Scion v2

1. **A strong, cheap baseline for our corrections.** Our rank-512 branches are
   trained with output-KD. Align-Then-Correct at the same rank and bytes is a
   closed-form alternative computed in minutes. If the trained corrections do
   not clearly beat it on the hard 90, the training budget is buying little.
   Run it as an ablation before the full FlashNext distill.
2. **The asymmetric-target insight applies to our KD.** Our corrections are
   trained against teacher logits on teacher traces; the paper's point is that
   the compensation target should be measured on the activations the
   quantized model actually produces, not on a shared input. That is the same
   drift Right On Q calls exposure bias, arrived at from a different direction.
3. **A closed-form polish.** After training, measure the remaining gradient on
   the corrected model and take one natural-gradient step. This is cheap and
   may remove residual error the KD loss leaves.
4. **Rank efficiency.** If the asymmetric target is more compressible, a much
   smaller rank (64-128 instead of 512) may match the current sidecar, cutting
   memory and decode overhead. Test this against the current artifact.

## What does not transfer directly

- The paper evaluates perplexity and zero-shot accuracy, not long-horizon
  reasoning; nothing here shows it fixes loop/non-termination behavior.
- It is a quantization-error compensator, not a distillation method: it has no
  teacher for reasoning style or long-horizon behavior, only the bf16 model.
- Their quantization is dense 2-bit (QuIP#/RTN), not our ternary MoE with
  expert banks and Q8_0 rest.

## Experiments this suggests

1. Implement Stage 1 + Stage 2 at our expert-bank granularity (rank 64 and
   512) as a training-free baseline, eval hard 90 and loop onset.
2. Compare: current trained corrections vs closed-form compensation vs both
   (closed-form init, then corrections).
3. Test rank 64/128 closed-form against rank 512 trained for the same bytes.
4. Add the Stage 2 natural-gradient polish to the trained correction
   checkpoint.

## Open questions

- Does the asymmetric target help when the "full-precision" model is the
  teacher (different family) rather than the same model at bf16?
- Can a closed-form compensator move long-horizon behavior at all, or is the
  loop collapse beyond what a per-layer SVD can reach?
- What is the decode cost of rank-512 adapters vs rank-64 in our engine?
