# Lessons for v2

Status: planning reference, 2026-11-11. Sources: [FAILURES.md](../FAILURES.md),
[docs/MOE-EXTENSION.md](../docs/MOE-EXTENSION.md),
[docs/ROUTER-FREEZE.md](../docs/ROUTER-FREEZE.md),
[docs/WEIGHT-PATCHING.md](../docs/WEIGHT-PATCHING.md), and the heldoutlab
hard-90 evaluation of the shipped 35B.

## 1. What the negative register settles

| id | lesson | status |
|---|---|---|
| D1 | Rotating expert banks into a basis before RTN does not help MoE. Falsified: PPL 40,121 raw vs 56,611 rotated. | closed |
| D2 | In-place ternary QAT at local budgets recovers ~1.9x and leaves routing unchanged (0.463 to 0.464). Even fp32 masters did not change it. Whole-body QAT is not the route; the corrections do the work. | closed |
| D3 | Correction branches inside experts are ~25x worse than on the residual stream. | closed |
| D4 | Router-KD matched a zero-weight control within ~1.5%. Do not spend on it. | closed |
| D5 | Post-hoc ternarised branches are 14-25x worse than STE-trained branches. Corrections must be trained with the quantizer in the loop. | closed |
| D6 | MOTE-style shared-expert up-cycling does not transfer to Qwen3.6-35B-A3B. | not applicable |
| D7 | A static hot/cold expert split is performance-neutral by construction. | rejected |
| D8 | Tail-conditioned KD did not transfer to the full body at full scale. | closed |

## 2. What the hard-90 eval added

- Shipped 35B: 13/90 = 14.4% at model-default sampling. Families: reasoning
  4/30, coding 1/30, tools 8/30.
- The failure is length-dependent: accuracy 33% under 3k completion tokens,
  19% at 3-10k, 0% at 10-20k, 8% above 20k.
- Loop onset is at roughly 5-10k generated tokens and is temperature-invariant
  (T=0.1 and T=0.6 both). 26 of 30 items past 25k tokens end at 0.99 coverage.
- 12/30 coding items truncated at 32k; 9 were lost to an unclosed code fence
  because thinking consumed 24.4k of the 32k budget. That is answer-room
  arithmetic, not a capability miss.
- Runtime ruled out: needle recall is perfect at 2k-26.5k prompt tokens.
- Harness ruled out: no tool-call repairs, lenient equals strict, answers are
  real.
- Control: Bonsai 27B at 5.95 GB scores 67.8% on the same items. Two bits per
  se is not the limit; the recipe is.
- The current corrections never saw long-horizon data: 4096-token windows and
  a short corpus, while the failures happen at 8k+ generated tokens.

## 3. Rules for v2 (do not relitigate)

1. Body frozen. Corrections are the only trainable part.
2. No full-body QAT, not even as a fallback. Escalation is bf16 SFT/LoRA of
   the body, then a single quantize, then corrections (D2).
3. Corrections sit on the residual stream (D3), are trained with the deployed
   quantizer in the loop (D5), and v1 uses output-KD only (D4).
4. Ternary only, ~11 GB class. A 20 GB release contradicts the product.
5. Hard 90 and easy 30 never enter the training corpus.
6. Evaluation: loop recovery must actually run for non-streaming requests
   (today it only runs in the streaming path), presence_penalty 1.5, budget
   sweep 4/8/16/32k, and the loop-onset coverage metric next to accuracy.
7. Report seconds per correct answer and the loop onset, not just accuracy.
8. **Corrections train on failures.** Use rollouts from the frozen body and
   teacher-steered revisions of failed tool/coding trajectories, not only
   teacher traces (Right On Q, SteerDistill). Length schedule 4k/8k/16k.
9. **Export fidelity is a release gate.** Compare exported codes to
   training-time codes per tensor; never recompute codes from float latents
   (Lost in the bf16 Cast).
10. **Format-preserving body moves are allowed.** Scale-only distillation
    (scales + norm gains, codes frozen) is not QAT and can run before any
    bf16 SFT escalation (ThinkBit, RoleQ).
11. **Run the tier baselines.** Ternary-Flint (9.3-12.5 GB) and the ternary
    Nemotron (8.82 GB) on the hard subset, before claiming the tier.
12. **Packing is a kernel project.** The on-disk PTQ1_0 type exists and the
    CPU path supports it, but the q36 HIP expert kernels are Q2_0-specific.
    Prototype before promising the ~9.6 GB tier.

## 3a. v2 execution lessons (2026-11-12)

1. **The shipped body is valid C++ Lloyd output, not Python-rule output.**
   3,660.9 ppm of expert codes differ from the Python training rule; the
   artifact is internally faithful and re-quantising its dequantised values is
   exact.  Seed the frozen training banks from the deployed GGUF and re-audit
   every v2 export to 0 ppm.  This is the code-level version of D9/009.
2. **Size the hardware from measured bytes, not assumed working sets.**  The
   35B body is ~69 GB in bf16; the current qwen35 proxy has no compact bank or
   gradient checkpointing.  The old "18 GB local / one 4090" line is invalid.
   Port `CompactBank` and measure 2k/4k/8k peaks before naming a card.
3. **A 512-token pilot cannot test a >16k loop-onset gate.**  Choose an
   experiment whose length schedule can falsify the hypothesis, or change the
   gate before spending.
4. **Plumbing smoke is not a data-path test.**  The 4-layer prefix ran on a
   random cache.  A real `_logits_` dump must be parsed and trained on before
   any full run.
5. **No pod before PLAN section 13a passes.**  The runbook is a sketch until
   its commands, memory, runtime, resume and cost cap are measured.

## 4. Open questions

- Can corrections alone patch long-horizon behavior, or is the body the hard
  ceiling?
- Teacher choice: Scion-FlashNext at 2.60 bpw (cheap, local) vs the official
  Qwen3.8-Flash-Next-FP8 (cleaner, 3x H100 for the cache).
- Router deltas: skipped in v1; worth it only if output-KD plateaus and the
  expensive teacher is justified.
- How much of the gap is the empero distill body vs the ternary quantization?

## 5. Related records

- [../FAILURES.md](../FAILURES.md) - full entries with method, evidence and
  cost to revisit.
- [../docs/MOE-EXTENSION.md](../docs/MOE-EXTENSION.md) - routing drift,
  placement rule, AUTOGRID, sidecar formats.
- [../docs/ROUTER-FREEZE.md](../docs/ROUTER-FREEZE.md) - fixed vs joint router
  A/B.
- [../docs/WEIGHT-PATCHING.md](../docs/WEIGHT-PATCHING.md) - layer/expert
  sensitivity map.
- [../docs/QUANT-RETENTION-35B.md](../docs/QUANT-RETENTION-35B.md) - retention
  grid against community quants.
