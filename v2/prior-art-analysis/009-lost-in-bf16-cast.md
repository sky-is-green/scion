# 009 - Lost in the bf16 Cast: Exporting Ternary Language Models Can Revert Most Low-Learning-Rate Code Changes

- arXiv: 2610.07853, 2026-10-06
- Link: https://arxiv.org/abs/2610.07853
- Status: analysed, actionable audit before any v2 result is trusted

## Claim

Ternary models are fine-tuned through higher-precision latents and deployed
as ternary codes by an export that, in the labs' documented pipelines, first
casts the latents to bf16. That cast sits exactly at the round-to-nearest
thresholds, so it can silently revert the fine-tuning. The paper audits
released checkpoints from three labs.

## Numbers

- In released checkpoints, fp32 quantization of the shipped latents disagrees
  with the deployed codes on 0.83-1.77% of codes (Falcon-E, BitCPM) and
  1.530% (2B-4T). Most Falcon-E/BitCPM disagreements are bf16 products
  exactly on the threshold, which ties-to-even maps to zero.
- At fine-tuned endpoints with ULP-matched rates, the documented export
  lowers greedy GSM8K strict accuracy from 58.79% to 0.78%
  (Falcon-E-1B-Base) and from 36.13% to 0.39% (BitCPM-CANN-0.5B); a bf16 save
  and reload lowers 2B-4T by 27.54 points while its last-number accuracy
  rises at that endpoint.
- The strict decrease recurs at three new seeds in all three models and on
  607 reserve test rows, but it is recipe-dependent: at Falcon-E's
  online-loss-selected rate the export was non-inferior, and after a longer
  recipe no difference was detected.
- Remedies: write the trained codes directly, or adjust bf16 inputs so the
  unchanged tools emit them. Both met a 4-point strict-accuracy
  non-inferiority criterion in all three models.

## What transfers to Scion v2

1. **Audit our export path.** Our pipeline converts training-time ternary
   codes into GGUF/Q2_0 through role_map and the repack tools. If any step
   rounds through bf16 or re-computes a threshold, codes can flip. Compare
   exported codes against the training-time codes on every tensor, not just
   by hash of the file.
2. **This is a cheap, high-value check.** The paper's export alone cost up to
   58 points of GSM8K. If our published artifact has a similar silent
   divergence, every eval and every correction result is built on the wrong
   codes.
3. **Write codes directly.** Our repack should copy integer codes verbatim
   and only store scales, never recompute a nearest grid from floats. The
   boro64 MLX->Q2_0 repack is a good reference for bit-exactness (it verified
   the dequantized blocks bit-identically).
4. **Recipe dependence** means the audit must use our actual recipe (ternary
   Lloyd g128, rank-512 corrections, the same exporter), not the paper's
   numbers.

## What does not transfer directly

- The audited models are BitNet-style latent QAT with onebitllms; our body is
  converted once from bf16 and has no latent fine-tuning, so the flip rate
  may be lower.
- The accuracy effect is measured on GSM8K strict accuracy; our hard-90
  coding and reasoning have different sensitivities.

## Experiments this suggests

1. Code-level diff: for every tensor, dequantize the exported GGUF and compare
   the ternary code to the training-time role_map code (count mismatches per
   million weights).
2. If mismatches exist, re-export from the codes (not the latents) and rerun a
   30-item subset to quantify the difference.
3. Add the code-level diff to the release checklist.

## Open questions

- Does the current 11.3 GB release have code-level mismatches? We have not
  measured this.
- Are the corrections applied in the same precision as they were trained
  (the sidecar export has its own cast risk)?
