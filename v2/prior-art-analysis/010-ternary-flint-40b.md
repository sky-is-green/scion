# 010 - Ternary-Flint-40B (artifact)

- Hub: kabcpc/Ternary-Flint-40B-GGUF, 2026-11-11 (WIP intermediate release)
- Base: Recursive Flint, 39.8B MoE / ~2.5B active, Apache-2.0, released
  2026-10-21. 49 layers (2 dense + 47 sparse), 256 routed experts, top-6 +
  a shared expert, 35 sliding-window (512) + 14 full-attention layers,
  vocab 150,144, 262K context (up to ~512K), knowledge cutoff June 2026.
  Card numbers: SWE-bench Verified 74.2, SWE-bench Pro v2 89.7,
  Terminal-Bench 2.1 54.9 at BF16.
- Status: analysed, competitive reference and technique source

## What the ternary build does

- Containers: PTQ1_0 (Type 143) and PQ2_0 (Type 142), i.e. our naming, in a
  custom Metal llama.cpp build with a PTQ1_0 LUT and a Hadamard H512
  transform.
- Variants (nlp_article PPL / size / bpw):
  - PTQ1_0 9.31 GiB, 1.750 bpw, PPL 10.95.
  - PQ2_0 10.97 GiB, 2.125 bpw, PPL 10.78.
  - Hybrid PTQ1_0 + Q4_1 11.04 GiB, 2.140 bpw, PPL 10.13.
  - Hybrid PQ2_0 + Q4_1 12.49 GiB, 2.450 bpw, PPL 10.13.
  - Lean 9.33 GiB, 1.880 bpw, PPL 8.37: token_embd Q8_0, attention Q5_K,
    shared experts Q8_0, lm_head Q6_K, top-2 sensitive down-projection
    layers (blk.8, blk.12) promoted to Q4_1, all other experts PTQ1_0.
- Recipe: Hadamard-512 folding; latent-STE KD with top-64 logits against the
  unquantized teacher; a scale-attenuation bias correction (BitNet's
  d = mean(|w|) underestimates non-zero trits by 1.3356x; unbiased LSQ
  scaling restores signal); selective per-layer promotion.
- A third-party OpenCL runtime (TheSanchezian, Intel Arc A770 16 GB) runs the
  PQ2_0 file at ~106 tok/s, PPL 13.38 on a 1,600-token WikiText-2 sample.

## Why it matters to us

1. **Someone else is using our containers, with calibration + KD, not trained
   corrections.** Their 9.3-12.5 GiB 40B MoE is the direct comparison point
   for our 11.3 GiB 35B. Their PPL is in the same range as our ternary work.
2. **Selective promotion works.** Promoting two sensitive down-projection
   layers to Q4_1 (and moving embeddings/head/attention to Q5-Q8) took a
   hybrid to PPL 8.37 at 1.88 bpw. That is the same per-layer allocation idea
   as our AUTOGRID, applied at conversion time.
3. **Scale attenuation is a concrete quantizer bug to check.** If our ternary
   scales use mean(|w|), they may be underestimating non-zero trits by a
   constant factor. Worth measuring on our body.
4. **Hadamard + KD is the competing conversion recipe.** Our D1 tested
   rotation before RTN without a KD stage; Flint's result is a full pipeline.
   If we ever re-quantize, this is the baseline to beat.

## What does not transfer

- Flint is a coding-focused MoE; its benchmark set (SWE-bench, Terminal-Bench)
  rewards different behavior than our hard-90 reasoning/coding/tools mix.
- Their custom Metal kernels and Type 143 LUT are not our Strata kernels; the
  numbers are on Apple Silicon, not ROCm.
- Their model has no trained correction sidecar.

## Baseline we could run

- Download the Lean or PQ2_0 GGUF and run the hard 90 (or the 30-item subset)
  through llama.cpp on this box. That gives a same-tier, same-container
  external comparison at ~9-11 GiB, which is exactly the consumer question.

## Open questions

- Does Flint's PPL advantage survive the hard-90 long-horizon items?
- Can we apply the 1.3356x scale correction to our current ternary scales as
  a scale-only recalibration?
- Is the Lean hybrid (Q4_1 for two layers) worth ~0.13 bpw over pure PQ2_0?
