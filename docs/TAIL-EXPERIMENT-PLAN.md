# Tail experiment plan — KLD beyond the teacher's top-50

**Status:** queued (needs a free GPU).  Written 2026-09-27, CPU-only prep.
**Related:** `QUANT-RETENTION-35B.md`, `QUANTIZATION-LANDSCAPE.md` §3.6/§7,
`RELEASE-35B-MODEL-CARD.md`.

## Why

The correction training matches the teacher's **top-50 logits only**; the rest
of the student distribution is unconstrained.  Mean token likelihood (PPL)
improved 11.60 → 8.35, but full-vocabulary KLD vs BF16 stays 2-bit-class
(mean 0.2694 vs Q4_K_M 0.0314).  KLD is the community's retention metric and
the 1.7B canary showed the failure mode is *reliability*: 77% of losses had the
gold answer at rank 2, over half near-ties.  The tail is where that lives.

## Constraints

- Must stay **calibration-free** (teacher logits, not a corpus): that is the
  recipe's structural edge vs imatrix / GPTQ / AWQ (`QUANTIZATION-LANDSCAPE.md`).
- The full 35B student (66 GB) does not fit locally → prefix experiments first.

## Step 0 — baseline (local, ~10 min)

Measure the **uncorrected body's KLD** (never measured; only its PPL 11.60 is):

```bash
HIP_VISIBLE_DEVICES=1 llama-perplexity -m qwen35-body-pq2_0-q8rest.gguf \
  -f wiki.test.raw -c 512 -ngl 99 -t 8 --chunks 50 \
  --kl-divergence --kl-divergence-base bf16-kld-50chunks.kld
```

Plus body HellaSwag/Winogrande 400 (`/tmp/opencode/bench-body.sh` is ready;
it failed earlier only because card 1 was busy).

## Step 1 — prefix A/B (local, free)

Same recipe, same seed, two caches; everything else identical:

| arm | cache | top-k |
|---|---|---|
| A | existing prefix cache | 50 |
| B | new cache | **512** |

```bash
# B's cache (4-layer prefix, minutes, file ~10x the top-50 cache)
python moe/qwen35_moe_proxy.py cache --prefix-layers 4 --device cuda:0 \
  --windows 4096 --corpus-chars 50000000 --top-logits 512 \
  --cache-file $MOE/qwen35/prefix-top512.pt

# train each arm (same flags as the 35B recipe, prefix-scoped)
python moe/qwen35_moe_proxy.py train --prefix-layers 4 --device-map cuda:0 \
  --cache-file <arm cache> --ref-file $MOE/qwen35/eval-ref-w2.pt \
  --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
  --kd-weight 1.0 --temp 2.0 --windows 4096 --corpus-chars 50000000 \
  --epochs 1 --steps 4096 ...
```

Eval: harness PPL + router agreement **and** a full-vocabulary KLD comparison
on the prefix eval windows (small script; teacher vs student logits — the
prefix fits the local card, so this is free).

**Decision gate:** the KLD tail (99.9% and max) must improve materially.  PPL
alone is not the signal.

## Step 2 — tail term (only if Step 1 moves)

Pick one after seeing Step 1:

- **a. residual-mass term** — match the teacher's remaining probability mass
  (sample k low-probability vocab entries per token; stochastic but unbiased);
- **b. rank/margin term** — directly penalise near-tie flips (the canary mode);
- **c. top-k expansion only** — already Step 1.

Implementation sketch: `--kd-tail-weight` in `moe/qwen35_moe_proxy.py`'s train
loss; keep a CPU unit test.  Do not change the frozen v1 recipe.

## Step 3 — v2 full run (rental, only if justified)

Only if Steps 1–2 show a clear, reproducible tail win **and** a v2 release is
wanted: repeat the v1 rental recipe with the new objective (~$7–9).  Not
scheduled; v1 ships regardless.

## Explicit non-goal

Do **not** adopt imatrix-style corpus calibration.  It would close the KLD gap
the cheap way and erase the calibration-free edge that distinguishes this
method from every peer on the chart.
