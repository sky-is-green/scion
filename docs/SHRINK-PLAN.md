# Shrink plan — below the 60.17 GB release

**Status (2026-10-09):** plan only, nothing measured yet. The released
Scion-FlashNext-176B-A6B is 60.17 GB / 2.72 bpw and is the smallest *unpruned*
Flash-Next quant published; the only smaller file on the Hub is ISTA's
**expert-pruned** coder IQ1_M at 58.4 GB. This document is the handoff for the
next quant pass: what to shrink, in what order, and the gates each step must
pass.

**Goal.** Cut the file below 60 GB with a measured quality cost, and publish
whatever passes the gates. Two honest targets: a conservative pass
(expert pruning only, ~53–54 GB) and an aggressive pass (pruning + table +
body, ~46–48 GB). Do not ship a file whose gates regress beyond the stated
bands.

## File budget (measured / derived)

| part | size | share | note |
| :--- | ---: | ---: | :--- |
| PLE n-gram table (Q4_0, 320,001,536 rows x 160) | 28.8 GB | 48% | 90 B per row, read per row |
| expert banks (PTQ1_0, 1.75 bpw) | ~26.4 GB | 44% | 48 layers x 512 experts, ~120.8B params |
| body (Q6_K + Q8_0) | ~4.1 GB | 7% | dense projections + head |
| corrections (F16, rank 512, 96 pairs) | ~0.68 GB | 1% | embedded adapter |

## Levers, by size impact

1. **PLE table: Q4_0 -> Q3_0 / Q2_0 (est. -6 to -13 GB).** The table is 48% of
   the file and is a lookup, not a dense weight. Needs a fork reader for the
   new table type (and Strata later), plus a quality gate — the PLE feeds
   every layer, so table error is not a small perturbation. No such table
   format is published for this model yet; this is the biggest single win.
2. **Expert pruning (25% -> ~-6.6 GB; 50% -> ~-13 GB).** Drop the least-used
   experts per layer, then re-index the bank and rewrite the router. Needs:
   - an **expert usage/importance map** — the missing input. Instrument the
     fork's `qwen4exp` router to count top-k selections (and weight mass) per
     expert per layer over a calibration set: fineweb-edu text plus the
     project's AYOT prompt set (`moe/build_ayot_prompts.py`, 128 math / 128
     coding / 256 web). The same 664-task community suite is the eval check;
   - an **exporter** that removes experts, rewrites `ffn_gate_inp` (and the
     shared-expert tensors), and renumbers the bank;
   - a **gate**, because the corrections were trained against the full bank —
     expect to either re-fit the corrections (the KD recipe with the deployed
     quantizer) or accept and measure the loss.
3. **Tighter ternary packing (~-2.5 GB).** PTQ1_0 spends 28 B per 128 trits
   against the 25.4 B entropy bound (~10% overhead). A denser container is a
   new ggml type id + fork + Strata kernel work; do it only after 1–2.
4. **Body: Q6_K+Q8_0 -> Q4_K/Q5_K (~-1.3 GB).** The fork already runs these
   types; the body is only 7% of the file, so this is the last easy lever.
5. **Corrections: per-layer mixed precision (~-0.3 GB).**
   `moe/branch_sensitivity.py` ranks which layers need F16 and which can go
   ternary; the saving is small but the method exists.

## Gates (in order, cheapest first)

1. **KLD gate** — `moe/kld_eval.py`, the 48-layer 8x512-token wikitext-2 gate
   the release already reports (mean 0.5800 for step-3000). A shrink step must
   not move it beyond its error band.
2. **HS/WG 400** — HellaSwag 82.00 / Winogrande 77.50 on the release; the
   comparison is paired over the same 400 tasks.
3. **Community arms** — the 664-task pod run (GSM8K-300 + MATH-200 +
   HumanEval-164). Compare like-for-like with the same caps (700 gen / 2048
   hard): the release numbers are floors (26% of think-harder records truncate
   at 2048), so a smaller model that answers sooner can *look* better — report
   both overall and finished-answer accuracy, and consider a larger cap for
   the comparison.

## First steps for the next session

1. Fork patch: per-expert routing counts + weight mass, env-gated, zero
   overhead when off.
2. Collect the map on a calibration corpus (pod, a few GPU hours); publish the
   map JSON alongside the results.
3. Prune at 25% and run the KLD + HS/WG gates; only if that holds, try 50%
   and/or the table format.
4. Decide on a correction re-fit after the first prune measurement, not
   before.
5. Re-benchmark on the pod with the fixed recipe
   (`hivebench/.../pod/remote-w1w2/recipe/pod-community-book.sh`) and update
   the model card with measured numbers.
