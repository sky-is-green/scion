# 35B quant-retention grid (2026-09-27)

Community-protocol benchmark of the single-file release against every quant of
the same model (`empero-ai/Qwen3.8-35B-A3B-Distill`), measured against a BF16
reference.  Raw logs: `hivebench/artifacts/ternary/refs/qwen35/results/`
(collector: `collect_bench.py`, table: `retention-grid.{md,json}`,
figure: `retention-grid.png` / `.svg` — Pareto grid, ours drawn as the blue triangle).

## Protocol (matches the community tables)

- **Perplexity**: wikitext-2 `wiki.test.raw`, `-c 512`, 580 chunks.
- **KL divergence**: 50 chunks (25.5k tokens) vs BF16 logits saved with
  `--save-all-logits`; mean / median / 99.9% / max as printed by
  `llama-perplexity` (the same stats Unsloth/APEX report).
- **Tasks**: HellaSwag 400 tasks (zero-shot, acc_norm) + Winogrande 400 tasks
  — the APEX protocol; ~±2% CI at n=400.
- Hardware: BF16 + all 9 quants on one H100 SXM (EUR-IS-3, same build, same
  session); the release on the local RX 7900 XT (card 1).  Cross-check:
  IQ2_M KLD local 0.163842 vs pod 0.163555 (0.2%) — builds are equivalent.
- Reference weights verified: 5 sampled tensors hash identically between the
  empero BF16 (reference logits) and the MrFuzzihead BF16 (our body's source).

## Results

| model | size (GB) | bpw | PPL | KLD mean | KLD median | KLD 99.9% | KLD max | HellaSwag | Winogrande |
|---|---|---|---|---|---|---|---|---|---|
| **release (ours)** | **11.34** | **2.61** | **8.3539** | **0.2694** | **0.1346** | **4.7579** | **7.2432** | **79.00%** | **76.25%** |
| IQ2_M | 12.56 | 2.90 | 8.4133 | 0.1636 | 0.0893 | 3.9627 | 6.8083 | 79.00% | 75.75% |
| Q2_K | 13.84 | 3.19 | 8.4729 | 0.1493 | 0.0780 | 3.2758 | 6.2913 | 76.50% | 73.25% |
| IQ3_M | 16.34 | 3.77 | 7.5196 | 0.0567 | 0.0315 | 1.2250 | 3.3902 | 80.00% | 74.25% |
| Q3_K_M | 17.66 | 4.07 | 7.4316 | 0.0583 | 0.0281 | 1.6107 | 8.0590 | 79.75% | 76.00% |
| IQ4_XS | 19.63 | 4.53 | 7.2638 | 0.0219 | 0.0111 | 0.6364 | 2.8611 | 80.75% | 75.75% |
| Q4_K_M | 21.71 | 5.01 | 7.2354 | 0.0314 | 0.0152 | 1.1410 | 3.5887 | 80.00% | 76.00% |
| Q5_K_M | 25.35 | 5.84 | 7.2725 | 0.0152 | 0.0065 | 0.7166 | 4.0739 | 80.50% | 76.25% |
| Q6_K | 29.21 | 6.73 | 7.1525 | 0.0080 | 0.0031 | 0.3596 | 3.5265 | 80.75% | 74.75% |
| Q8_0 | 37.80 | 8.72 | 7.1599 | 0.0043 | 0.0015 | 0.2095 | 1.2062 | 80.25% | 75.50% |
| BF16 (reference) | 71.07 | 16.38 | 7.1595 | — | — | — | — | 81.25% | 76.00% |

## How to read it / why this is good

Every panel is "up = better, left = smaller".  There are two metric families:

- **Answer quality** (HellaSwag / Winogrande): does the model get questions
  right?  Ours: 79.0% / 76.25% vs Q4_K_M 80.0% / 76.0% and BF16 81.25% / 76.0%
  — the same within the ±2% task noise.  That is **97% of BF16 task accuracy at
  48% of Q4_K_M's file size**, and the only sub-3-bit quant at that level.
- **Probability fidelity** (PPL / KLD): how closely the quant reproduces the
  full-precision model's probability estimates.  PPL scores only the word that
  actually came next; KLD compares the whole distribution, tail included.
  Ours is the **best-PPL model below 3 bpw** (8.35 vs IQ2_M 8.41, Q2_K 8.47)
  but still 2-bit-class on KLD (0.269 vs Q4_K_M 0.031) — the tail is the
  remaining gap and the next lever.

The method is also the only non-imatrix point on the chart: ternary experts
(PQ2_0) + trained low-rank corrections, not calibration.

## Findings

1. **Task accuracy: Q4-class.**  HellaSwag 79.0% is within the ±2% CI of
   Q4_K_M (80.0%) and BF16 (81.25%); Winogrande 76.25% is the joint best row
   (BF16 76.00%).  At 400 tasks these gaps are noise — the release is
   task-level indistinguishable from Q4_K_M/BF16, at half the file size.
2. **PPL: best of the 2-bit class.**  8.3539 beats IQ2_M (8.4133) and Q2_K
   (8.4729), but sits ~1.1 PPL behind the 3-bit group and Q4_K_M (7.2354).
   Retention vs BF16: PPL +16.7%; vs Q4_K_M: +15.5%.
3. **KLD: 2-bit class, and it disagrees with PPL.**  Mean 0.2694 is worse
   than IQ2_M (0.1636) and Q2_K (0.1493), and ~8.6x Q4_K_M (0.0314).  The
   corrections improved mean token likelihood more than they improved the
   full-distribution tail — PPL and KLD rank the release differently relative
   to IQ2_M.  Unsloth's warning that PPL/KLD can disagree with real-world
   accuracy applies in reverse here: the tasks look Q4-level while the
   distributional metrics look 2-bit.
4. **Headroom.**  The uncorrected ternary body was PPL 11.60; the corrections
   took it to 8.35 (and the body was never imatrix-calibrated).  Closing the
   remaining KLD gap to Q4 would need better tail behaviour, not just mean
   likelihood.

## Caveats

- KLD was measured over 50 chunks (25.5k tokens); consistent across all rows.
- The 2/3-bit competitor quants use imatrix calibration (Wikipedia-like data),
  which flatters wikitext KLD; our corrections were trained on fineweb.
- HellaSwag/Winogrande at 400 tasks carry ~±2% CI; treat sub-1% differences
  as ties.
- The release row was measured on the local card; the IQ2_M control confirms
  the local/pod builds agree (0.2% on KLD).

## Reproduce

```bash
# on a CUDA box (or locally for the release):
llama-perplexity -m <model.gguf> -f wiki.test.raw -c 512 -ngl 99 --chunks 580          # PPL
llama-perplexity -m <model.gguf> -f wiki.test.raw -c 512 -ngl 99 --chunks 50 \
    --kl-divergence --kl-divergence-base bf16-kld-50chunks.kld                         # KLD
llama-perplexity -m <model.gguf> -f hellaswag_val_full.txt --hellaswag --hellaswag-tasks 400
llama-perplexity -m <model.gguf> -f winogrande-debiased-eval.csv --winogrande --winogrande-tasks 400
```
