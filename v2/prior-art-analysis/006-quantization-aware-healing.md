# 006 - Quantization-Aware Healing: A Practical Recipe for Recovering Compressed, 4-Bit LLMs

- arXiv: 2608.20953, 2026-08-21
- Link: https://arxiv.org/abs/2608.20953
- Status: analysed, confirms the no-QAT decision

## Claim

Shipping a cheap model now means structural compression plus 4-bit
quantization. Both degrade reasoning, math, coding and long-context behavior
enough to need a recovery stage. The default recipe, QAT against hard labels,
converged slowly and collapsed past its peak in their pipeline. Their fix:
distill the compressed, quantized student directly from the **original
uncompressed model**, not from a bf16 checkpoint that was itself recovered
from the compression.

## Method

- Pipeline example: GPT-OSS 120B -> structurally compressed 60B -> MXFP4.
- QAH: the 4-bit student is trained by distillation from the original 120B
  model (the true teacher), not from the 60B bf16 intermediate.
- Practical notes: a large, reproducible quality gap between distributed
  training backends; avoid hand-tuned early stopping, QAH is stable under
  continued training.

## Numbers

- The QAH student matches or beats its own bf16 source on 7 of 9 benchmarks
  at ~4x less weight memory and half the teacher's parameter count.
- Released as Hypernova-60B.
- Reaches the 120B model on LiveCodeBench.
- About 7x faster to a comparable peak than matched QAT, and does not
  collapse under continued training.

## What transfers to Scion v2

1. **The teacher rule.** Our current body was distilled from empero's 35B
   distill. For v2 the teacher should be the strongest original model we can
   score on, FlashNext, not another recovered intermediate. This matches the
   v2 plan.
2. **No QAT for the body, again.** Their QAT baseline "collapsed past its
   peak"; QAH's stable alternative is distillation. That is consistent with
   our frozen-body, corrections-only decision, and with D2/D5.
3. **Monitor for collapse in correction training.** If we ever extend
   training, keep the early checkpoint selected on reasoning, not loss.
4. **Backend differences are real.** Their note about distributed-training
   backends changing quality is a warning for the runpod-side distill:
   pin the backend and compare a small eval before scaling.

## What does not transfer directly

- 4-bit MXFP4, a 60B/120B dense-ish pipeline, not ternary 35B-A3B.
- QAH still trains the student body; our v2 trains corrections only. The
  teacher-selection and collapse lessons transfer, the mechanics do not.

## Experiments this suggests

1. In the v2 pilot, compare teacher candidates on a fixed 30-item subset
   before generating the full trace corpus (FlashNext vs the empero bf16):
   pick the teacher by hard-subset quality, not by size.
2. If corrections training is extended, checkpoint-select on the hard subset
   at every save, not on KD loss.

## Open questions

- Is the "original uncompressed" teacher still available for our body
  (empero bf16 is a distill, not the original Qwen3.6-35B-A3B)?
- Would distilling from FlashNext into corrections inherit the same
  advantage, or does the corrections bottleneck cap it?
