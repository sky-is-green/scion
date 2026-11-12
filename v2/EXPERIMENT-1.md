# Experiment 1 (v2): FlashNext-recipe corrections at 4k windows

Status: **FROZEN 2026-11-12** (local half of PLAN 13a step C; the teacher pass
is pod-gated behind E/F).  This document is the contract for the first v2
training run: one primary arm, one fallback arm differing in exactly one
variable, a cache spec the local path already produces, and a falsification
criterion that the 4k-window schedule can actually test.  Locked decisions
(PLAN section 5) are not relitigated here.

## Hypothesis

The shipped 35B loops after ~5-10k generated tokens because the v1
corrections never saw long-horizon data (4096-token windows, short corpus).
Training the same rank-512 corrections against Scion-FlashNext-176B traces
on 4k+ windows, with the deployed quantizer in the loop and the proven
FlashNext KD recipe, moves loop onset past 16k generated tokens and lifts
hard-subset accuracy.

## Arms (one variable, mirrored from the 176B run)

Both arms share everything; they differ only in `--kd-tailcond-weight`
(the D_KL2 weight FlashNext chose between for its primary/fallback):

| | primary `cur05-35B` | fallback `pred2.0-35B` |
|---|---|---|
| kd-weight | 2.0 | 2.0 |
| kd-tail-weight (marginal, D_KL1 marginal) | 2.0 | 2.0 |
| **kd-tailcond-weight (D_KL2)** | **3.0** | **2.0** |
| temp | 2.0 | 2.0 |
| balance | bias (delta 1e-3, z 1e-3) | bias |
| rank / placement | 512, both | 512, both |
| branch-quant | g128 | g128 |
| windows / seq | 4096-token cache records | same |
| memory switches | `--compact-banks --grad-checkpoint --checkpoint-mode group --expert-group-size 64` | same |
| body | deployed GGUF via `--bank-gguf` (0-ppm seed) | same |

Exact training command (the one E will cost and the prefix already ran):

```bash
python moe/qwen35_moe_proxy.py train --device-map auto --device cuda:0 \
  --bank-gguf "$MODELS/Scion-35B-A3B-PQ2_0-corr.gguf" \
  --cache-file "$Q/flashnext-traces-cache.pt" \
  --compact-banks --grad-checkpoint --checkpoint-mode group --expert-group-size 64 \
  --verify-banks --chunked-ce \
  --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
  --kd-weight 2.0 --kd-tail-weight 2.0 --kd-tailcond-weight 3.0 --temp 2.0 \
  --balance bias --log-entropy \
  --epochs 1 --steps 4096 --eval-every 500 --ckpt-every 500 --log-every 50 \
  --tag cur05-35B
# fallback arm: identical except --kd-tailcond-weight 2.0 --tag pred2.0-35B
```

`--log-entropy` stays ON on cards with headroom (the sharpening diagnostic
that caught the W1 collapse); OFF on a tight card (monitoring only, the
objective is unchanged).

## Data spec (what the teacher pass must produce)

1. Teacher: Scion-FlashNext-176B-A6B Q6_K via `llama-perplexity
   --save-all-logits` (the `qwen4exp-proto` fork; its half-chunk scoring is
   what `build_trace_cache.py` decodes - see `v2/REAL-CACHE.md`).
2. Corpus: the teacher's traces over the 360-prompt rollout corpus
   (`v2/rollout-corpus-manifest.json`; hard-90/easy-30 excluded by
   construction, 138 normalized-exclusion checks).  The 12 on-policy
   rollouts captured so far are on-policy traces but **not loop exemplars**
   (2026-11-12 correction: their recorded "coverage 1.0" was a broken
   metric; the texts are clean solutions) - the failure-driven data needs
   rollouts at the 32k budget where the hard-90 texts actually degenerate
   (all 30 items past 25k tokens show 0.65-0.97 duplication on the stored
   rows).  Steered revisions of failed tool/coding trajectories stay in
   scope (SteerDistill).  Eval sets never enter training.
3. `build_trace_cache.py --logits ... --out cache.pt --top-k 512 --seq 512
   --tail-logits 64 --seed 0` - produces `ids/idx/val/w/tidx/tlp` per
   record; the FULL recipe (including D_KL2) runs off this cache with no
   other data path.  Reproducible per seed (pinned by test).
4. Long windows: the pilot cache is 512-token records (the dump's native
   chunk); the 4k-window arm needs records of ~4096 tokens.  Two ways, in
   order of preference: (a) run the dump at `-c 4096` so each chunk is a
   4k window scored at its second half (2048 rows); (b) concatenate 512-
   token records into 4k groups (ids concatenated; idx/val/w rows joined -
   the KD targets stay position-aligned within each 512 block; the router
   balance sees whole windows).  (a) is the honest one and is what the pod
   command uses; the half-chunk scored rows then cover positions 2048-4094
   of each window.

## Falsification criterion (decided before the run)

The gate is capability, not KD loss (FlashNext lesson 3: the step-4000
extension was flat/worse on the gate and was not shipped):

- **Primary gate:** loop onset on the long reasoning items.  Generate (T=0.6,
  model default sampling, 32k budget) from the merged v2 model on the
  hard-90's long reasoning items (the 30 items past 25k tokens that ended at
  coverage 0.99 in v1): PASS if the 10-gram coverage onset moves past 16k
  generated tokens on >= 50% of the long items (v1: onset at ~5-10k, 26 of
  30 items at 0.99 coverage).
- **Secondary gate:** +10 points on subset30 over the current body
  (3/30 = 10.0% -> >= 13/30... i.e. >= 6/30 on the first pass given noise;
  the formal target stays PLAN section 4: >= 40% on the full hard 90).
- **Abort criteria:** NaN loss, load entropy collapse (H < 4.5 nats with
  peak > 0.5 - the W1 signature), a >50% deviation from the expected loss
  shape (lm 7-9 scale at full model, kd ~0.25-0.35 at the 176B's cache
  quality), or any eval-set contamination check failing.
- If neither gate moves after the full 4096-step run of BOTH arms: stop and
  report - do not ship the last step by default (FlashNext did not).

## Eval protocol (fixed before the run)

- subset30 (`v2/eval/subset30.json`) via `v2/scripts/hard_subset.py`:
  T=0.6, seed 1001, top_p 0.95 / top_k 20 / min_p 0.05, 32k budget,
  presence_penalty 1.5, one server at a time, streaming requests so the
  server's `reasoning_loop_recovery` actually runs (the v1 runs sent
  `stream: false` and the recovery was inert - fix the runner, not the
  server).
- Loop-onset coverage per item at every 512-token check, reported next to
  accuracy; truncation counts; seconds per correct.
- KLD/PPL are diagnostics only (FlashNext lesson 4: support-dominated KLD,
  weak code/logic categories).
- The merged artifact for eval is produced by the existing
  `export_branches_lora.py` + `merge_adapter_into_body.py` path and re-audited
  to **0 ppm** against the v2 training seed (`v2/scripts/export_audit.py`)
  before any eval number is recorded.

## Budget sketch (feeds step E; measured locally where possible)

| item | where | basis |
|---|---|---|
| teacher dump (30 traces ~4k tok avg -> ~120k tokens) | pod, 60.2 GB Q6_K | ~1 min/512-chunk on the local 35B; 176B ~4-6x slower -> order 1-2 h + load |
| cache build | local or pod | vectorized: minutes (was ~30 min scalar) |
| corrections 2 arms x 4096 steps | pod | local 4-layer prefix: ~13-15 GiB at 4k with chunked-ce; full model ~20.5 GiB -> 24-48 GB card; runtime extrapolation pending step E |
| export + merge + audit | local, free | the D loop exercises it |
| subset30 eval x 2 arms + loop onset | local or pod | ~2.5 h per arm on one 7900 XT (v1 subset timing), or faster on the pod |

## What would make us change this freeze

- If the 4k dump records are impractical on the pod (VRAM at 4096 ctx for the
  176B teacher), fall back to (b) cache concatenation - recorded, not silent.
- If the teacher mass `w` on real FlashNext traces behaves differently from
  the 35B self-dump (0.96 observed), re-check the decode before training.
- If FlashNext traces show the loop signature themselves (teacher rollouts
  looping), the hypothesis changes: that is a TEACHER problem, not a
  corrections problem - record and re-plan before spending the full run.
