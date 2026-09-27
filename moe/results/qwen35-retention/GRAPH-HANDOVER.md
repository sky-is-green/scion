# Graph handover — 35B quant-retention data

Everything needed to redraw `retention-grid.*` without re-measuring.

## Data files (this directory)

- `retention-grid.csv` — the tidy table: `model,size_gb,bpw,ppl,kld_mean,kld_median,kld_999,kld_max,hellaswag,winogrande`
- `retention-grid.json` — same data keyed by model (the plot script reads this)
- `*.log` (46 files) — the raw `llama-perplexity` outputs per model/panel
  (`<model>-ppl.log`, `<model>-kld.log`, `<model>-hellaswag.log`,
  `<model>-winogrande.log`; `bf16-*` is the reference; `local-release-*` is
  the release measured on the local card)

## The data (for reference)

| model | size GB | bpw | PPL | KLD mean | KLD 99.9% | HellaSwag | Winogrande |
|---|---|---|---|---|---|---|---|
| **ours (release)** | **11.337** | **2.61** | **8.3539** | **0.2694** | **4.758** | **79.00** | **76.25** |
| IQ2_M | 12.558 | 2.90 | 8.4133 | 0.1636 | 3.963 | 79.00 | 75.75 |
| Q2_K | 13.839 | 3.19 | 8.4729 | 0.1493 | 3.276 | 76.50 | 73.25 |
| IQ3_M | 16.340 | 3.77 | 7.5196 | 0.0567 | 1.225 | 80.00 | 74.25 |
| Q3_K_M | 17.664 | 4.07 | 7.4316 | 0.0583 | 1.611 | 79.75 | 76.00 |
| IQ4_XS | 19.628 | 4.53 | 7.2638 | 0.0219 | 0.636 | 80.75 | 75.75 |
| Q4_K_M | 21.713 | 5.01 | 7.2354 | 0.0314 | 1.141 | 80.00 | 76.00 |
| Q5_K_M | 25.348 | 5.84 | 7.2725 | 0.0152 | 0.717 | 80.50 | 76.25 |
| Q6_K | 29.209 | 6.73 | 7.1525 | 0.0080 | 0.360 | 80.75 | 74.75 |
| Q8_0 | 37.802 | 8.72 | 7.1599 | 0.0043 | 0.209 | 80.25 | 75.50 |
| BF16 | 71.067 | 16.38 | 7.1595 | — (reference) | — | 81.25 | 76.00 |

## Method (so the plot can state it correctly)

- Model: `empero-ai/Qwen3.8-35B-A3B-Distill` (qwen35moe).
- **PPL**: wikitext-2 `wiki.test.raw`, `-c 512`, 580 chunks.
- **KLD**: 50 chunks (25.5k tokens) vs BF16 logits (`--save-all-logits` /
  `--kl-divergence`); BF16 reference weights hash-verified against the body's
  source (MrFuzzihead BF16).
- **Tasks**: HellaSwag 400 (zero-shot acc_norm) + Winogrande 400; ~±2% CI.
- BF16 + all quants measured in one H100 session (same build); the release on
  the local card, cross-checked (IQ2_M KLD local 0.1638 vs pod 0.1636).
- "ours" = ternary PQ2_0 experts + Q8_0 rest + trained corrections (single
  file, auto-applied); everyone else is an imatrix-calibrated K-quant / IQ
  quant. That distinction is the point of the chart.

## Files that produce the current version

- `../plot_bench.py` — matplotlib renderer (adjustText for label placement)
- `../collect_bench.py` — parses the logs into the CSV/JSON
- Rerun: `PYTHONPATH=... python collect_bench.py results && python plot_bench.py`
