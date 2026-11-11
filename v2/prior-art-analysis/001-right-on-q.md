# 001 - Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories

- arXiv: 2610.29378, 2026-10-21
- Authors: Boyuan Zheng, Jing Xiong, Chenwei Wu, Hui Shen, Zesen Zhao,
  Hei Ting Una Chan, Xueshen Liu, Zhen Zhang, Liyue Shen, Zechun Liu,
  Z. Morley Mao (Michigan / UCSB / Meta AI)
- Link: https://arxiv.org/abs/2610.29378
- Status: analysed, actionable for the corrections training loop

## Claim

Low-bit reasoning models trained with fixed-response QAT still fail on
long-horizon tasks: they "often fail to terminate within a token budget that
suffices for the full-precision model", and they repeat themselves. The cause
is exposure bias: fixed-response supervision only covers teacher prefixes,
while at inference the student conditions on its own outputs, and the
divergence grows with response length. The fix is on-policy distillation on
the student's own rollouts plus a candidate-wise Dynamic KL.

## Method

1. Block-wise calibration (EfficientQAT Block-AP style) to get a working
   quantized student from RTN.
2. The quantized student samples responses with its own forward pass; the
   frozen full-precision teacher supplies next-token distributions on those
   same prefixes.
3. Dynamic KL: per candidate, mix forward and reverse KL with
   alpha = p/(p+q) (stop-gradient), so underpredicted candidates get FKL and
   overpredicted candidates get RKL.
4. Response-length schedule: cap rollouts at 4k tokens for steps 1-50, 8k for
   51-100, 16k afterwards. Early long rollouts are mostly repetition.

## Numbers that matter

- Qwen3-1.7B / 4B / 8B at W2 and W3; nine benchmarks including AIME,
  LiveCodeBench, HumanEval+, MBPP+, GPQA-Diamond, IFEval.
- OnQ has the best nine-benchmark average in all six settings; at 2 bits it
  wins every benchmark at every size. Qwen3-8B W2: 70.68 average vs 60.42 for
  the strongest baseline (+10.26).
- The gains concentrate on long-horizon benchmarks: AIME and LiveCodeBench
  are where every QAT method and the PTQ baselines fall apart, because those
  responses are ~3x longer than HumanEval ones.
- Repetition: fixed-response QAT baselines produce repetitive errors on
  11.4-54.0% of MATH-500 questions; OnQ 2.4%. On AIME, OnQ produced none.
- Truncation: lowest among QAT methods in all six settings; responses also
  get shorter.
- Repetition during early rollouts: 6.7% of tokens in the first 4k rise to
  21.1% beyond 8k; on-policy training removes it within ~100 steps.
- Ablations: on-policy beats fixed-response DKL by up to 8.40 points
  (Qwen3-1.7B W2: MATH-500 61.00 -> 69.40); Dynamic KL beats FKL/RKL.

## What transfers to Scion v2

1. **Our hard-90 failure is this paper's failure.** Loop onset around 5-10k
   tokens, truncation at the cap, and non-termination are exactly what they
   document. Their Figure 1 groups accuracy and truncation by full-precision
   response length; our accuracy-by-completion-length table (33% under 3k,
   19% 3-10k, 0% 10-20k, 8% over 20k) is the same shape.
2. **Fixed-response supervision is the likely flaw in our corrections.** Our
   corrections are trained by output-KD on teacher traces (fixed prefixes).
   That is precisely the exposure-bias setup they identify. The fix is to
   sample rollouts from the frozen ternary body and have the teacher score
   those prefixes.
3. **Response-length schedule** is directly usable: start corrections on 4k
   rollouts, extend to 8k, then 16k. It avoids spending supervision on early
   degenerate text.
4. **Dynamic KL** is a drop-in objective for the correction trainer instead
   of plain forward KL/top-k KD.
5. **Report repetition and termination**, not just accuracy. Their
   repetitive-error classification is a good model for our loop-onset metric.

## What does not transfer directly

- OnQ updates the quantized body (latent weights and scales). Our body is
  frozen; only ~35M correction parameters can move. The exposure-bias
  argument still applies to the corrections, but the ceiling is lower.
- Their models are dense 1.7-8B. A 35B-A3B with 3B active and a different
  teacher (FlashNext, Qwen4Exp) may behave differently.
- They need full-precision teacher logits on student prefixes. We have that
  (FlashNext top-k cache), but it means teacher passes over student rollouts,
  not just over teacher traces.

## Experiments this suggests

1. Corrections on teacher traces vs corrections on student rollouts, same
   token budget. The pilot already needs a teacher cache; add rollouts from
   the current ternary body.
2. Add Dynamic KL (top-K, alpha = p/(p+q)) to the correction loss.
3. Length schedule 4k/8k/16k for the pilot and the full run.
4. Add a repetitive-error classifier and a termination-rate column to the
   v2 eval, next to accuracy.

## Open questions

- Can rank-512 sidecars alone fix exposure bias, or does the body need to
  move? This is the main risk for the corrections-only plan.
- Does the teacher being a different family (FlashNext vs the empero 35B
  distill) change the Dynamic KL balance?
- How much of the 2.4% vs 11-54% repetition gap comes from on-policy data vs
  from the body update? A corrections-only ablation answers it.
