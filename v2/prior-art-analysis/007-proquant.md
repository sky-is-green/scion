# 007 - ProQuant: Preserving Expert Quality over Quantity in Ultra-Low-Bit MoE

- arXiv: 2610.30779, 2026-10-22
- Link: https://arxiv.org/abs/2610.30779, code: github.com/ThrowBug/ProQuant
- Status: analysed, relevant to expert allocation

## Claim

Mixed-precision PTQ for MoE is hard under tight bit budgets because
(1) nonlinear quantization errors at very low precision are underestimated
when cross-layer propagation is ignored, and (2) directly fine-tuning the
router can cause routing drift, and calibration gains may not generalize.
ProQuant is a weight-only PTQ framework with two parts.

## Method

- **PoP (Prune-or-Prioritize Mixed-Precision Allocation)**: jointly optimizes
  expert pruning and bit-width assignment, so higher-precision experts can
  substitute for severely degraded ones, improving budget utilization.
- **RoCo (Routing-Consistent RMSNorm Finetune)**: adapts expert input
  representations through mathematically equivalent transformations,
  compensating quantization error while reducing routing drift.

## Numbers

- Outperforms existing MoE PTQ methods across generative tasks (the abstract
  gives no single headline number; the code is public).
- The two contributions are exactly the two failure modes our own D2 and D4
  identified: router updates drift, and router-targeted training is a no-op
  in our setting.

## What transfers to Scion v2

1. **Allocation, not uniform bits.** Our Q2_0 uses the same precision for
   every expert bank. PoP's prune-or-prioritize view matches our AUTOGRID
   per-expert sensitivity work: spend bits where they matter, and let a
   higher-precision expert absorb the damage of a pruned neighbor.
2. **Routing-consistent normalization** is a cheap, format-preserving trick
   we can test on the current body: adjust expert input scales rather than
   train the router. It aligns with the RoleQ/ThinkBit scale-only idea.
3. **Router drift is real** (their point 2), which supports our decision not
   to train router deltas in v1 and to keep the body frozen.

## What does not transfer directly

- It is a PTQ conversion framework, not a correction/distillation method; no
  long-horizon reasoning evaluation in the abstract.
- MoE layouts differ (routed experts per layer, top-k); our 35B-A3B has its
  own expert shape and the Strata native packing.

## Experiments this suggests

1. Compare uniform Q2_0 against a per-expert mixed allocation (promote the
   most sensitive expert banks, prune the least useful) at the same total
   bytes.
2. Try RoCo-style input-scale adaptation on our experts as a scale-only
   calibration step.
3. Use the hard-90 to choose the allocation, not calibration loss.

## Open questions

- How much of our long-horizon collapse is driven by a small number of
  damaged experts? A per-expert ablation would show it.
- Does mixed allocation interact with the correction sidecar (does it reduce
  the correction burden)?
