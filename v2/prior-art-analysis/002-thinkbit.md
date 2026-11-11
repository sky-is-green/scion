# 002 - ThinkBit: Search Codewords, Recalibrate Scales, Select by Reasoning

- arXiv: 2610.34886, 2026-10-26
- Authors: Rhea Sanjay Sukthanker, Pashmina Cameron (Microsoft Applied Sciences)
- Link: https://arxiv.org/abs/2610.34886
- Status: analysed, actionable (scale recalibration + bit accounting + packing)

## Claim

Two-bit models are usually judged by perplexity or multiple-choice accuracy,
and those metrics "barely separate configurations whose sampled reasoning
accuracy differs by more than ten points". A 2-bit framework should be both
built for and evaluated on reasoning. ThinkBit is a strictly scalar 2-bit
format (integer codeword x one bf16 scale per 128 weights, no vector
codebooks, rotations or zero-points) with five stages.

## Method

1. **Allocate**: KronQ sensitivity gives each layer a ternary, 2-bit or 3-bit
   tier under a bit budget. Ternary costs 1.71 bpw (log2 3 + 16/128), 2-bit
   2.125, 3-bit 3.125.
2. **Search**: block-wise codeword search (GSQ/Gumbel for ternary and 3-bit,
   ReinMax for 2-bit) starting from GPTQ.
3. **Recalibrate**: freeze codewords, train only the stored scales and
   RMSNorm gains (<1% of parameters) with a forward-KL loss to bf16.
4. **QAT**: STE latents may reassign codewords.
5. **Select**: a paired reasoning test on training problems picks the
   recalibrated or the QAT checkpoint.

## Numbers that matter

- Qwen3-1.7B at 0.52 GB stores **2.42 bits per parameter** including the
  output head and embeddings, smaller than every published 2-bit GGUF of the
  model, and scores higher on MATH-500 than all of them.
- At matched size: recalibrated 46.7 MATH-500 vs EfficientQAT 30.0; after
  QAT on reasoning traces, 73.0.
- The 7-task QA average stays 53.7-54.8 when stages are removed, while
  sampled MATH-500 drops from 43.0 to 28.4-37.0. That is the PPL/MCQ blind
  spot in one number.
- Qwen3-14B at 3.74 GB scores 94.8 on MATH-500 vs 97.5 for 29.5 GB bf16.
- Allocation at 2.0 bpw for Qwen3-1.7B: 75.3% ternary, 6.0% 2-bit, 18.7%
  3-bit. The geometry of the format means 2 bits is dominated by the mix of
  ternary and 3-bit per layer; only q/k stay at 2+ bits.
- Community "2-bit" files are really 2.42-3.62 bits per parameter. PrismML's
  ternary Bonsai stores 2.15 bpw because every ternary weight sits in a 2-bit
  slot.

## What transfers to Scion v2

1. **Bit accounting.** We should report whole-file bits per parameter, not
   nominal quant names. Our 11.3 GB / 35B is ~2.58 bits per parameter
   including embeddings and head; the experts are 2.125 bpw in 2-bit slots.
2. **Packing is a free size win.** Ternary values need log2(3) = 1.585 bits;
   our Q2_0 puts one trit in a 2-bit slot, so the expert body is ~33% larger
   than the information content. A 5-trit-per-byte container (1.625 bpw)
   would take the 11.3 GB tier down to roughly 9 GB with identical weights.
   This is a format change only, no retraining, and it directly serves the
   consumer-hardware goal.
3. **Scale-only recalibration respects the frozen-body rule.** Stage 3
   trains only stored scales and RMSNorm gains with codewords frozen. That is
   not QAT and not a body retrain; it is a format-preserving calibration. It
   could be run on our body after corrections, or instead of a second
   correction round, before we consider any bf16 SFT escalation.
4. **Reasoning-based checkpoint selection.** We already have the hard subset;
   use it to pick between correction checkpoints instead of validation KLD.
5. **Mixed tiers are more efficient than uniform ternary** (75% ternary +
   19% 3-bit at 2.0 bpw). If we ever change the body format, per-layer tiers
   beat uniform PQ2_0 at the same average bits.

## What does not transfer directly

- Full-body QAT and codeword search change the body; that is out of scope
  while the body is frozen.
- Their models are dense 1.7-14B; the 35B-A3B expert layout and 3B active
  compute are different.
- The sensitivity allocation needs calibration passes over the body; our
  current pipeline has teacher caches that could serve.

## Experiments this suggests

1. Apply scale-only recalibration (RMSNorm + group scales, forward KL to the
   teacher) to the current ternary body with codewords frozen, then measure
   hard-90 and loop onset. Low risk, no body QAT.
2. Compute our real bits-per-parameter split (experts, attention, embeddings,
   head) and publish it next to the 11.3 GB figure.
3. Prototype a 5-trit packing of the existing Q2_0 experts and measure the
   file size and decode speed; the weights are unchanged.
4. Use the hard subset, not KLD, to select the final correction checkpoint.

## Open questions

- Does scale-only recalibration help a model whose failure is long-horizon
  repetition, or does it only recover local likelihood?
- Can the 5-trit packing fit the existing llama.cpp/Strata kernels, or does
  it need a new dot-product path (the 1.58-bit packing paper is the reference)?
- How much of our 2.58 bits per parameter is embedding/head, and would the
  frozen-vocabulary idea (entry 008) remove it?
