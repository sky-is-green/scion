# Coder variant: 25% prune gate report (2026-10-11)

Release baseline vs the first prune: REAP-ranked uniform 25% (keep 384/512
experts per layer, `prune-plan-25.json`), PTQ1_0 expert container preserved,
corrections unchanged. Pruned file: **53.538 GB** (sha256
`26ef5a826984af9c8027f7bf15ca806c03b1293d6c380d740470720eb405dc29`), from
60.175 GB. Pod: L40S, one window, release and pruned measured with the same
fork build, same caps.

## KLD (48-layer, 8x512 wikitext, same parked bf16 teacher, `llama-kld`)

| | mean | p50 | p90 | p99 | p99.9 | max | top1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| release Q6K | 0.7819 | 0.4069 | 1.9871 | 5.2200 | 8.0044 | 15.1366 | 0.6566 |
| pruned 25% | 0.9675 | 0.4972 | 2.4937 | 6.6450 | 9.8439 | 12.7453 | 0.6333 |
| delta | +0.1856 | +0.090 | +0.507 | +1.425 | +1.840 | -2.391 | -0.023 |

Note: the PyTorch release gate reports 0.5800 on the same teacher; the fork
instrument reads the shipped PTQ1_0+corrections artifact at 0.7819. The
comparable number is the delta (this instrument, same model family).

## General gates (release protocol, same 400 tasks / 580 chunks)

| gate | release | pruned 25% | delta |
| :--- | ---: | ---: | ---: |
| wikitext-2 PPL (580 x 512) | 5.4571 | 6.0878 | +0.631 (+11.6%) |
| HellaSwag 400 | 82.00% | 79.25% | -2.75 |
| Winogrande 400 | 77.50% | 77.00% | -0.50 |

## Paired re-run on one pod (2026-10-11) - supersedes the cross-window code table

Same L40S pod, both models, release protocol server args; single-sample hard
arm at temperature 0.7. Flips are paired per task (release-only / pruned-only
correct):

| arm (paired, n=164 / test 49) | release | pruned 25% | delta | flips rel/pru |
| :--- | ---: | ---: | ---: | :--- |
| gen pass@1 all | 72.0% | 68.9% | -3.1 | 18 / 13 |
| gen pass@1 test | 71.4% | 71.4% | 0.0 | 4 / 4 |
| gen pass@any all | 86.6% | 83.5% | -3.1 | - |
| hard 2048 all | 56.1% | 51.2% | -4.9 | 29 / 21 |
| hard 2048 test | 61.2% | 57.1% | -4.1 | 8 / 6 |
| hard 4096 all | 60.4% | 57.3% | -3.1 | 24 / 19 |
| hard 4096 test | 65.3% | 59.2% | -6.1 | 6 / 3 |
| hard 2048 finished acc all | 94.8% | 93.3% | -1.5 | - |

The cross-window table above overstated the hard-arm regression (the release
baseline itself moves ~8 points on the test split between cards). Paired, the
generation arm is flat, the hard arm is -3 to -5 points overall with
discordant counts (29/21, 24/19) inside the McNemar noise band for these
sizes, and the 4096 diagnostic shows the gap does not grow with a bigger cap.
**At 25%, code is statistically indistinguishable between release and
pruned.** The systematic costs remain the general-text ones (PPL +11.6%,
HS -2.75, WG -0.5, KLD +0.19 in the fork instrument), which is the intended
code-calibration trade.

## Original cross-window code table (keep for the record)


| arm | release | pruned 25% | flips (rel-only / pruned-only) |
| :--- | ---: | ---: | :--- |
| gen pass@1 all | 68.9% | **71.3%** | 18 / 22 |
| gen pass@1 test (n=49) | 73.5% | **75.5%** | 5 / 6 |
| gen pass@any all (pass@3) | 84.1% | **86.0%** | - |
| hard pass@1 all | 58.5% | 50.6% | 32 / 19 |
| hard pass@1 test (n=49) | 69.4% | 44.9% | 15 / 3 |
| hard finished-answer acc all | 97.0% (99/164) | 90.2% (92/164) | - |
| hard finished-answer acc test | 100% (34/49) | 81.5% (27/49) | - |
| cap-hit (empty) hard test | 30.6% | 44.9% | - |

- Generation (single-shot code): **no detectable loss**; the flips are
  balanced and the point estimates favour the pruned model on both splits.
- Hard/thinking arm: a real regression, almost entirely cap truncation. Of
  the 15 test tasks the release solves and the pruned model loses, 11 end at
  the 2048-token cap with empty content (they used 508-1919 tokens on the
  release run); 4 answer but wrongly. This is the "sustained multi-step
  tasks accumulate error" effect ISTA's coder card also reports.
- The general-text cost is the intended trade of a code-calibrated prune,
  but it is present at 25% (PPL +11.6%).

## Three-way tier test (release / 25% / 50%, one pod, 2026-10-11)

| arm (n=164 / test 49) | release | 25% | 50% |
| :--- | ---: | ---: | ---: |
| gen pass@1 all | 68.3% | 68.3% | **69.5%** |
| gen pass@1 test | 61.2% | 63.3% | **69.4%** |
| gen pass@any all | 84.1% | 84.8% | 83.5% |
| hard 2048 all | 50.0% | **52.4%** | 42.7% |
| hard 2048 test | 49.0% | **51.0%** | 30.6% |
| hard finished acc all | 94.3% | 94.5% | 82.4% |
| hard finished acc test | 92.3% | 100% | 68.2% |
| hard cap rate test | 46.9% | 49.0% | 55.1% |
| gen flips vs release (test / all) | - | +1 / 0 | +4 / +2 |
| hard flips vs release (test / all) | - | +1 / +4 | **-9 / -12** |

- **Generation holds at both tiers** — flips vs release are positive/noise at
  25% and 50%; single-shot code is insensitive to removing half the experts.
- **Hard/thinking holds at 25%** (flips +1 test / +4 all; finished accuracy
  100% test) and **collapses at 50% uniform**: -18 points on the test split,
  flips -9, finished accuracy 92.3% -> 68.2% (and truncation up to 55%).
  This reproduces ISTA's Coder-Next warning: at 50%, uniform allocation
  breaks the sustained reasoning mode even when single-shot code is fine.
- Therefore a 50% file needs either **nonuniform per-layer allocation**
  (ISTA: 97% vs 55% HE recovery at 50%) or a **correction re-fit (healing)**
  before it can be shipped. The 25% file stays the safe coder candidate.

## Decision inputs (step 4: correction re-fit)



Paired, 25% is code-neutral; what remains is the general-text/KLD shift and
the fact that the corrections were trained against the full bank.  That makes
the re-fit optional at 25% and clearly relevant at 50%, where capacity loss
becomes real:

1. **Proceed to 50% uniform first** (recommended): the 25% measurement says
   the first quarter is free for code; the next quarter is where the
   literature says uniform allocation can collapse on this family (ISTA
   Coder-Next: 55% HE at 50% uniform vs 97% nonuniform).  A 50% REAP-uniform
   test is one ~2 h window and tells us directly whether the nonuniform
   engine work is needed.
2. **Correction re-fit**: continue the KD training (released recipe,
   `qwen4exp_proxy.py train` + `--resume` from step-3000) against the pruned
   ternary bank, export the 192 adapters, re-gate.  Targets the general/KLD
   shift; cost one 48 GB GPU window (~4-8 h, ~$5-7) plus local prune support
   in the training path.
3. **PLE table lever**: Q4_0 -> Q3_0/Q2_0 (-6..-13 GB, orthogonal to the
   expert prune); needs the fork/Strata reader plus a gate, i.e. local engine
   work before any pod time.
4. **Accept 25% as the coder candidate**: -6.64 GB, code-neutral, with the
   general-text trade stated on the card.

Artifacts: `pod/remote-gate-prune/bench/` (kld-*.json, pruned-PPL/HS/WG logs,
`community-code/scion-coder-code-gate.json`, samples/hard JSONL),
`scion/moe/artifacts/qwen4exp/{router-stats-coder.json,prune-plan-25.json}`.
