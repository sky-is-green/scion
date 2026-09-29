# Tail experiment plan — KLD beyond the teacher's top-50

**Status:** Steps 0–1 done; Step 2a measured (works on the tail, handoff §3);
Step 2d (the tail-conditional D_KL2 term) built 2026-09-29 and queued.
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

  **Status: BUILT (2026-09-28, session 2), unmeasured.** Built as the
  *marginal* KL of the two-way support/complement split rather than a sampled
  estimator: `kd_loss.support_mass` + `kd_loss.residual_mass_kl`, wired as
  `--kd-tail-weight` (default 0.0) and `phase1-w1.sh train-tail <w>` (which
  pins `--kd-weight` at 1.0 so the arm is one variable against armC). The
  teacher-side mass is recorded per token by the cache stage as `w`, which is
  free there because the `topk` already consumed the full-width logits — so the
  aggregate is exact and per-token rather than sampled, with none of the
  variance and none of the 3.7x error a corpus-mean target would carry
  (measured support mass: 0.1704 mean, 0.0455 at p01, 0.0366 min). Read at
  temp 1, the deployed distribution, not the KD term's temp 2.

  **Blocker:** the existing caches (`prefix-top50.pt`, `prefix-top512.pt`) are
  both `['idx','router','val']` and have no `w`. The term refuses to substitute
  a constant and `SystemExit`s, so `phase1-w1.sh cache c` must be re-run once
  (~7 min) before `train-tail` can run at all.

  **Measured (2026-09-28/29).** The blocker was resolved with one
  `phase1-w1.sh cache c` re-run. tailw 1.0: mean KLD −10.5%, **p99 −26.0% /
  max −19.6%** for +6.6% PPL vs armC — the inverse trade to the kd-weight,
  which moves the mean and leaves the tail. tailw 2.0: p99 −40.6% / max −25.9%
  for +14.1% PPL. combo2t2 (kd 2.0 + tail 2.0): mean −38.6%, p99 −39.7%,
  max −30.8% at +57% PPL — the levers compose, and this is the current recipe
  candidate. The gate's chain-rule decomposition then measured 71–82% of the
  arms' KLD as the *tail-conditional* piece, which a top-k cache cannot hold;
  see (d).

- **b. rank/margin term** — directly penalise near-tie flips (the canary mode);
- **c. top-k expansion only** — already Step 1.
- **d. tail-conditional term (TAD's D_KL2)** — **BUILT (2026-09-29), queued.**
  The decomposition showed 71–82% of the gate's KLD is the tail-conditional
  piece, which no weight lever reached. The term is the (1−w_t)-weighted KL on
  the complement, estimated from 64 tokens per position sampled from the
  teacher's own tail conditional (Sparse Logit Sampling; unbiased, no
  importance weights): `kd_loss.sample_tail_tokens` / `tail_conditional_piece`,
  `--kd-tailcond-weight` (default 0.0), `phase1-w1.sh cache-tail` +
  `train-tailcond <w>`. Validated against `kld_eval --decompose-topk` (exact
  under enumeration; ratio ~1.00 when sampled). First arm: weight 1.0 on
  combo2t2, one variable.

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
