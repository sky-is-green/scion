# Prior art analysis - Scion v2

One file per paper/artifact, focused on what transfers to our work: ternary
body, frozen-body corrections, long-horizon reasoning, consumer size, tools.

Status values: `to-read`, `skimmed`, `analysed`, `actionable`, `not-applicable`.

| # | item | date | status | one-line takeaway |
|---|---|---|---|---|
| 001 | [Right On Q (2610.29378)](001-right-on-q.md) | 2026-10-21 | analysed | Quantized reasoning fails to terminate on long tasks; on-policy corrections on the student's own rollouts fix repetition and truncation. |
| 002 | [ThinkBit (2610.34886)](002-thinkbit.md) | 2026-10-26 | analysed | 2-bit for reasoning; PPL/MCQ are the wrong metrics; packing and scale-only recalibration are levers. |
| 003 | [RoleQ (2610.21808)](003-roleq.md) | 2026-10-15 | analysed | Ternary scales by functional role, not matrix layout; scale distillation with frozen codes. |
| 004 | [Align, Then Correct (2610.08164)](004-align-then-correct.md) | 2026-10-06 | analysed | Training-free two-stage low-rank compensation; a cheap baseline for our corrections. |
| 005 | [SteerDistill (2611.13410)](005-steerdistill.md) | 2026-11-10 | analysed | Closed-loop teacher steering on failed agent rollouts; data recipe for tools/coding. |
| 006 | [Quantization-Aware Healing (2608.20953)](006-quantization-aware-healing.md) | 2026-08-21 | analysed | QAT collapses past its peak; distill from the original teacher instead. |
| 007 | [ProQuant (2610.30779)](007-proquant.md) | 2026-10-22 | analysed | Mixed-precision MoE allocation + routing-consistent norm adaptation. |
| 008 | [Embeddings Are Overpriced (2610.22428)](008-embeddings-overpriced.md) | 2026-10-15 | analysed | The vocab table dominates ternary file size; a frozen codebook works. |
| 009 | [Lost in the bf16 Cast (2610.07853)](009-lost-in-bf16-cast.md) | 2026-10-06 | analysed | Export can revert ternary codes; audit code-level fidelity. |
| 010 | [Ternary-Flint-40B (artifact)](010-ternary-flint-40b.md) | 2026-11-11 | analysed | A 40B ternary MoE in our containers, hybrid precision + KD, no corrections. |
| 011 | [Ternary Nemotron 30B-A3B (artifact)](011-ternary-nemotron.md) | 2026-11-10 | analysed | 1.58-bit 30B-A3B at 8.82 GB, 89% teacher recovery, 368K context. |

Add entries as we go. Each file follows the same shape: citation, claim,
method, numbers, what transfers to v2, what does not, experiments we could
run, open questions.

## Cross-cutting notes so far

- **Fixed-response supervision is a recurring flaw.** Right On Q (exposure
  bias on long responses) and Align, Then Correct (symmetric calibration)
  independently conclude that targets built on a shared/reference input miss
  what the quantized model actually does. Our corrections are trained on
  teacher traces, so this is the first thing to change in the pilot.
- **The cheapeast body-side moves are format-preserving.** ThinkBit and RoleQ
  both recover significant quality by training only scales and norm gains
  with codes frozen, in hours on one GPU. This sits between "frozen body" and
  "bf16 SFT escalation".
- **Bit accounting and packing are free size wins.** Our ternary expert banks
  sit in 2-bit slots (2.125 bpw) instead of 1.585-1.625. A 5-trit packing
  could take the 11.3 GB tier to roughly 9 GB with identical weights.
- **Reasoning is the right metric.** ThinkBit shows PPL/MCQ can stay flat
  while sampled MATH-500 drops 6-15 points. Our hard-90 and loop-onset
  protocol is the correct axis.
- **Export fidelity is not a detail.** Lost in the bf16 Cast (009) and the
  Nemotron repack (011) both show the integer codes are the artifact: write
  them verbatim, verify bit-exactness, never re-round from floats.
- **The consumer bar moved.** Flint (010) and the Nota Nemotron (011) are
  8.8-11 GB ternary models with published numbers. We should run both on the
  hard 90 as external baselines before claiming a tier.
- **Correction data should come from failures.** Right On Q (rollouts) and
  SteerDistill (teacher revision of failed trajectories) both say the training
  signal lives where the quantized model actually fails, not in teacher
  traces.
