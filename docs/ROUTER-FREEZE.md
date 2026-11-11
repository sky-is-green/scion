# Fixed-router corrections: what MiMo-V2.6 §5.4 does and does not imply for ternary PTQ

Status: research note, 2026-11-06.  The OLMoE proxy A/B has been run (two
seeds, 32 held-out windows); the result is in §4a, the protocol that produced
it is §6.  The 35B prefix/full-body transfer and the long-horizon task read are
still open.  Companion documents:
[`MOE-EXTENSION.md`](MOE-EXTENSION.md) (the MoE write-up),
[`SCION-RECIPE.md`](SCION-RECIPE.md) (the deployable recipe), and
[`../FAILURES.md`](../FAILURES.md) (D4: router-KD is a no-op).

## 1. The hypothesis

A reader of the Bonsai 2 / Scion comparison raised the following argument,
motivated by MiMo-V2.6's RL report:

> Once an MoE router has learned a useful expert specialization structure,
> preserving that structure and adapting the model to work within it may be
> more efficient than retraining the router and experts simultaneously.

In ternary PTQ terms: freeze the pretrained router, ternarise the experts, train
residual-stream corrections to restore the hidden states the router reads, and
add only small router adjustments if the remaining error cannot be recovered.
The claim is that this avoids a moving optimization target (the branch chases a
router that is also moving) and converges faster or more stably than joint
router/branch adaptation.

This is a real alternative, not a restatement of the current recipe: Scion's
deployable path trains router weights jointly with the correction branches.  The
question is whether that is earning its keep.

## 2. What MiMo-V2.6 actually measured

MiMo-V2.6 §5.4 ("Router Freezing for Stable RL") compares two RL runs that
differ only in whether the MoE router is frozen.  It tracks expert load at
decoder layer 9 (384 experts) with three statistics: coefficient of variation,
peak/mean load, and the fraction of cold experts below 0.1x the mean.

| run | CV | peak/mean | cold experts | benchmark |
|---|---:|---:|---:|---|
| router trainable, first 20 steps | 0.78 → 2.0 | 6x → 16x | 0.5% → 22% | — |
| step-20 router restored to pre-RL values | near initial | near initial | near initial | unchanged |
| router frozen | ~0.7, flat | ~5.5x, flat | ~1%, flat | improves normally |

The restore-router control is the important one: putting the step-20 router
weights back recovered load balance while benchmark performance stayed the same.
In that setting router drift was load-pathological and bought no quality.  This
is strong evidence about **load stability**, not evidence that router adaptation
is useless when the expert function itself has changed.

The analogy is also not exact.  In RL the experts move slowly, so the router's
pretrained specialization remains roughly valid.  Ternary PTQ changes the expert
function much more: on OLMoE the deployable Lloyd quantizer has expert weight
relative error 0.442 (0.517 for one-shot absmean), and the correction branches
recover ~40x of the resulting PPL damage.  The original top-k assignment may be
genuinely miscalibrated for the ternary experts.  Whether router adaptation
repairs that, or merely co-adapts to a different computation, is the experiment.

## 3. What Scion currently does

The deployable path is joint, not fixed-router:

- `moe/olmoe_corrections.py` unfreezes `...mlp.gate.weight` with the branches;
  `--eval-router base` exists as an inference-time ablation.
- `moe/qwen35_moe_proxy.py` and `moe/qwen4exp_proxy.py` do the same in
  `attach_branches`; checkpoints keep the router weights, and
  `export_branches_lora.py --routers` writes exact rank-64 `ffn_gate_inp` LoRA
  deltas.
- The 35B and 176B model cards' "router deltas" line is therefore literal, not
  decorative.  Router weights are not negligible in parameter count either:
  the 35B has ~21M (40 layers x 256 experts x 2048 hidden), roughly 12% of the
  rank-512 branch budget.  "Small adjustment" in the hypothesis refers to the
  weight change, not the optimizer footprint.
- The 35B v2 recipe additionally trains a per-expert router bias
  (`--balance bias`); `SCION-RECIPE.md` §6 records a material difference when
  the bias is omitted (35B in-run 8.28 vs deployed eval 10.5), so the bias is
  load-bearing.  A "fixed-router" arm that keeps the bias is not fully fixed;
  one that drops it may lose quality for a different reason.  The 35B A/B
  therefore needs a 2x2 (router weights x bias).

Router-KD is a verified no-op (D4): distilling the teacher's top-k decisions did
not change the trajectory.  That says the router *loss* is not the lever; it
does not say router *weights* are irrelevant.  The branch repairs the state the
router reads, but the router can still be moved.

## 4. Evidence that already exists

All rows are OLMoE-1B-7B, 8-window wikitext unless marked; they are not one
matched A/B, so read them as context, not as an answer.

| arm | PPL | top-8 agreement | note |
|---|---:|---:|---|
| FP teacher | 11.02 | 1.000 | reference |
| RTN, absmean, no corrections | 40,121 | 0.463 | collapses |
| RTN, Lloyd, no corrections | 804.39 | 0.691 | deployable quantizer |
| + rank-512 branches + trained routers, Lloyd/g128 | 19.80 | 0.846 | current recipe, 4096 windows |
| old absmean rank 512 + routers | 71.14 | 0.710 | not matched to Lloyd |
| old absmean rank 64 + routers | 253.85 | 0.506 | capacity control |

The runtime protocol gives a cleaner read on the router delta's **inference**
value.  Same OLMoE body, same `attn_out` adapter, 563 wikitext chunks, only the
`ffn_gate_inp` deltas added:

| build | PPL |
|---|---:|
| uncorrected mixed GGUF | 566.59 |
| + branches, no router deltas | 16.17 |
| + branches + router deltas | 15.61 |

That is a 3.5% relative PPL improvement from the router deltas on the documented
rank-1024 `attn_out` pair, against a ~35x recovery from the branches.  The
deltas are small in weight space: on the 16-layer OLMoE adapter, the per-layer
delta RMS is 0.00066-0.00127; layer 0 is 1.20e-3 against a base router RMS of
3.21e-2, about 3.7%.  Small weights can still flip top-k decisions, so this is
not evidence that the deltas are unimportant - only that they are small
adjustments, which is exactly the regime the hypothesis is about.

What is missing: no frozen-router **training** arm, and no saved
`--eval-router base` result that would separate the branch's contribution from
the router delta on the harness protocol.  The A/B below fills both gaps.

### 4a. The OLMoE A/B result (2026-11-06)

Two seeds, one cache per seed, 4096 steps, rank 512, Lloyd/g128, both
placements, 32 held-out wikitext windows.  `joint` trains the routers; `base`
is the same joint checkpoint evaluated with the pretrained routers (the
branch-only inference ablation); `frozen` trains only the branches.

Per seed:

| seed | arm | PPL | top-1 | top-8 | router KL | load CV | cold |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | joint | 16.0104 | .6916 | .8461 | 1.8001 | .8616 | 10.42% |
| 0 | base | 16.1161 | .6924 | .8471 | 1.7890 | .8579 | 10.31% |
| 0 | frozen | 16.0683 | .6946 | .8476 | 1.7854 | .8578 | 10.31% |
| 1 | joint | 15.9166 | .6925 | .8467 | 1.7931 | .8578 | 10.33% |
| 1 | base | 15.9886 | .6927 | .8475 | 1.7860 | .8542 | 10.24% |
| 1 | frozen | 16.0273 | .6925 | .8479 | 1.7836 | .8554 | 10.32% |

Mean:

| arm | PPL | top-1 | top-8 | router KL | load CV | cold |
|---|---:|---:|---:|---:|---:|---:|
| joint | **15.9635** | .6921 | .8464 | 1.7966 | .8597 | 10.38% |
| base | 16.0524 | .6926 | .8473 | 1.7875 | **.8561** | 10.28% |
| frozen | 16.0478 | **.6936** | **.8478** | **1.7845** | .8566 | 10.32% |
| teacher | 8.5739 | — | 1.000 | — | .847 | 8.35% |

What the result says:

- **Joint keeps a small, consistent PPL edge**: 0.36% in seed 0 and 0.70% in
  seed 1, ~0.53% on the mean.  It is not noise-free, but the sign is stable.
- **Frozen is slightly better on routing fidelity in both seeds**: higher
  top-1/top-8 and lower teacher-router KL.  The differences are small
  (~0.001-0.002 agreement, ~0.01 KL) but consistent.
- **The joint router delta is co-adaptation, not routing recovery.**  Within
  the joint arm, adding the delta improves PPL by 0.66% (seed 0) and 0.45%
  (seed 1), while top-8 agreement falls and KL rises in both.  It buys
  language-model loss by moving routing slightly away from the teacher.
- **No load collapse, unlike MiMo's RL run.**  Both corrected arms repair the
  RTN damage (CV 1.08 -> ~0.86, cold 16% -> ~10%, dead 5.2 -> ~2); frozen's
  load advantage over joint is marginal, and `base` has the best CV.
- **The strong MiMo effect does not transfer to this PTQ setting.**  The
  router is small and the branches carry almost all of the recovery; training
  it is a last-percent trade, not a stability prerequisite.

The practical read: freezing is a legitimate, simpler recipe that preserves
routing fidelity slightly better at a ~0.5% PPL cost.  Whether that trade is
worth it depends on long-horizon behavior, which this proxy cannot answer.

## 5. Why the analogy needs care

1. **Freezing weights is not freezing routing.**  The commenter notes this, and
   it is the central measurement problem: quantization changes the hidden states
   entering the router, so top-k can flip even with the router weights bit-exact.
   The fixed-router arm must therefore be judged on routing agreement and load,
   not on the fact that a parameter tensor did not move.

2. **The router may be compensating for the experts, not drifting.**  A frozen
   router forces the branch to make the old assignment work.  If the ternary
   experts are systematically better at different tokens, the branch may need
   more capacity or may never close the gap.  The oracle-routing ceiling
   (evaluate the quantized body with teacher top-k forced) measures whether the
   residual gap is routing or expert approximation.

3. **The current recipe already has a load-balancing router modification.**  The
   bias can hide a collapse that pure router-weight training would show, or it
   can be the thing that makes joint training viable.  The 35B 2x2 separates the
   two; on OLMoE the bias is not in the correction path, so the proxy A/B is
   clean.

4. **MiMo's dependent variable is load, not long-horizon quality.**  The report
   shows frozen-router RL improves normally, but it does not compare long-horizon
   task performance under a fixed router.  A PPL tie can still hide a difference
   in compounding behavior; position-bucketed agreement and a long-context task
   set are part of the protocol for that reason.

## 6. The A/B

### 6.1 Primary: OLMoE, matched compute

One cache per seed, one body build, one step budget per arm.  Two arms:

- **J (joint, current):** branches + trainable routers, `--router-weight 0`
  (D4 already showed the router-KD term is a no-op; dropping it makes the arm
  one variable cleaner).
- **F (fixed):** branches only, `--freeze-router`.

Both use `--quant lloyd --branch-quant g128 --branch-target both --rank 512`,
the same LR schedule, and a matching seed-specific cache.  Two seeds were run
(§4a); add a third if a result is near the noise floor.  A secondary
`--router-lr` sweep (0.1x and 1x the branch LR) is the fair joint arm if J
loses badly; without it, a single shared LR can make joint training look worse
than it needs to.

The script is [`../moe/router-freeze-ab.sh`](../moe/router-freeze-ab.sh):

```sh
# one cache per seed: the cache is built from seed-specific text windows
for s in 0 1; do
  python moe/olmoe_proxy.py cache --windows 4096 --corpus-chars 50000000 \
      --seed "$s" --out "moe/artifacts/olmoe/teacher-cache-s$s.pt"
done
MOE_ARTIFACTS=/path/to/artifacts CACHE=/path/to/teacher-cache-s0.pt \
    STEPS=4096 SEEDS=0 EVAL_WINDOWS=16 bash moe/router-freeze-ab.sh
MOE_ARTIFACTS=/path/to/artifacts CACHE=/path/to/teacher-cache-s1.pt \
    STEPS=4096 SEEDS=1 EVAL_WINDOWS=16 bash moe/router-freeze-ab.sh
```

It trains both arms and evaluates J twice: with the trained routers and with
`--eval-router base` (branches + pretrained routers).  The second eval is the
inference decomposition: if J-with-base is close to F, the router delta is not
carrying the gain and freezing during training is promising; if J-with-base is
close to J, the branch is doing the work and the router delta is mostly
cosmetic.

### 6.2 Secondary: 35B / 176B transfer

The proxy answers the mechanism question; it is 1B active and not a quality
forecast (D8 already showed prefix tail gains do not transfer).  For the real
target:

1. **4-layer `qwen3_5_moe` prefix, 2x2:** router weights frozen/trainable x
   `--balance bias` on/off, same cache and steps.  Read load, routing agreement,
   PPL and convergence, not absolute quality.
2. **Full 35B, two arms:** the winning cell of the 2x2 against J, on one cache,
   under the community gate protocol (wikitext-2 PPL, KLD, HellaSwag,
   Winogrande).  Only run the full arm if the prefix and proxy agree on a
   direction, or if the frozen arm is within noise and the artifact
   simplification is the goal.
3. **FlashNext 176B** only after the 35B gate; the routing structure is much
   wider (512 experts, top-10) and the load question is more severe.

`--freeze-router` is wired into `qwen35_moe_proxy.py` and
`qwen4exp_proxy.py`.  The full 35B recipe keeps `--balance bias` in both arms
for the first comparison; the 2x2 prefix is where the bias is isolated.

### 6.3 Metrics

| axis | metric | why |
|---|---|---|
| quality | PPL, full-vocab KLD mean/median/p99/max, task accuracy | the deployed objective |
| routing | top-1 and top-8 agreement per layer; teacher-router KL over the union of supports | top-8 alone hides a changed top-1 |
| routing stability | agreement per position bucket (0-63, 64-127, ...) | compounding errors are a position profile |
| load | CV, peak/mean, cold fraction <0.1x mean, dead experts, load entropy | MiMo's actual dependent variable; PPL can hide collapse |
| router movement | relative Frobenius delta per layer from the base router | zero for F, small for J; separates "small adjustment" from "new router" |
| convergence | loss/KLD vs step, best checkpoint vs last, seed variance | the hypothesis is about optimization, not only the endpoint |
| cost | tokens/s, wall-clock, trainable parameters | matched-compute claim |

The eval stage now writes these into `branches-eval-<tag>.json`; the `layers`
array keeps the per-layer profile, and the top-level fields are the aggregate
used in the table above.  `router_agree` remains the top-8 number for continuity
with the earlier results.

### 6.4 Decision rule

- **Joint is justified** only if it improves held-out quality at matched compute
  without a material regression in load or teacher routing agreement.  A lower
  training KD with a lower top-k agreement or a higher load CV is co-adaptation,
  not recovery.
- **Frozen is preferred** if it is non-inferior within the noise band (working
  thresholds: PPL within 2%, KLD mean within 5%, top-8 agreement within 0.01,
  task CI includes 0) and has equal or better load stability.  It removes the
  router deltas, the bias interaction, and one moving part from the recipe.
- **If frozen loses badly** (>5% PPL or >10% KLD mean), router adaptation is
  load-bearing.  Then the next question is the smallest sufficient adjustment:
  rank-64 vs rank-8 router deltas, and whether the bias alone recovers most of
  it.
- **If both are within noise**, choose frozen for the simpler artifact and the
  lower load risk, and keep joint as a documented fallback.
- **If frozen converges faster but joint wins at the endpoint**, the binary
  choice is the wrong experiment: test a freeze-then-unfreeze schedule, or a
  rank-8 delta applied only after the branch has restored the hidden states.
  That is the commenter's step 4 ("small router adjustments only if the
  remaining error cannot be recovered") in its natural form.

## 7. What would change the conclusion

- Frozen wins on quality and load: drop router deltas from the deployable
  recipe; the export gets simpler and the 2 bpw claim loses a 4.3 MB sidecar.
- Joint wins on quality but not load: keep the router delta, add a load guard
  (bias or MiMo-style monitoring), and report both.
- Joint wins only with a tuned router LR: the honest conclusion is "joint
  training needs its own optimizer budget", not "freezing is worse".
- The proxy and the 35B disagree: trust the full body, as D8 did for the tail
  objective.  The proxy is a mechanism screen, not a release gate.

## 8. Reproduce

```sh
# OLMoE cache and eval refs (see moe/README.md)
export MOE_ARTIFACTS=$PWD/moe/artifacts
python moe/olmoe_proxy.py cache --windows 4096 --corpus-chars 50000000
python moe/olmoe_corrections.py ref --ref-file $MOE_ARTIFACTS/olmoe/eval-ref-w2.pt

# matched A/B, one cache per seed
for s in 0 1; do
  python moe/olmoe_proxy.py cache --windows 4096 --corpus-chars 50000000 \
      --seed "$s" --out "$MOE_ARTIFACTS/olmoe/teacher-cache-s$s.pt"
  CACHE="$MOE_ARTIFACTS/olmoe/teacher-cache-s$s.pt" \
      STEPS=4096 SEEDS=$s EVAL_WINDOWS=32 bash moe/router-freeze-ab.sh
done

# the inference decomposition is in each run:
#   branches-eval-rf-s0-joint-trained.json  (trained routers)
#   branches-eval-rf-s0-joint-base.json     (base routers)
#   branches-eval-rf-s0-frozen-trained.json (frozen arm)
```

The metric helpers are in [`../moe/router_metrics.py`](../moe/router_metrics.py)
with CPU tests in [`../moe/tests/test_router_metrics.py`](../moe/tests/test_router_metrics.py).
