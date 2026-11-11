# 005 - SteerDistill: Quantizing LLM Agents via Closed-Loop Teacher Steering

- arXiv: 2611.13410, 2026-11-10
- Link: https://arxiv.org/abs/2611.13410
- Status: analysed, actionable for corrections data

## Claim

Quantized agents fail by overlooking task requirements or environmental
conditions, producing environment transitions that diverge from successful
full-precision references. Offline data misses the student's failure
contexts, and the student's own rollouts contain few correct answers in those
contexts. A full-precision teacher can instead guide the student to revise
failed responses, with the guidance iteratively refined from what happens when
the student executes it.

## Method

1. Run the quantized student; collect failed rollouts.
2. The teacher steers a revision of the failed trajectory (environment states,
   local verification, steering prompts, feedback).
3. Execute the revised trajectory; feed the outcome back to refine the
   guidance (closed loop).
4. Train the student without guidance to match the steered student at the
   corrected turns and the teacher at other turns, so inference is unassisted.

## Numbers

- Five agentic benchmarks, three models: +1.67 to +2.18 average score over the
  strongest quantized baseline at W4A16, within 1 point of BF16.
- tau2-Bench: +1.84 percentage points success over QeRL with 67% fewer
  training GPU-hours.
- Produces shorter successful trajectories than the quantized baselines.

## What transfers to Scion v2

1. **This is the missing data recipe for our tools failures.** Our tools
   family is 8/30, and the failure mode is exactly "overlooks task
   requirements or environment conditions". The hard-90 tools environment
   gives us execution feedback, so we can run the loop: body rollout, teacher
   revision, re-execute, keep the corrected turns.
2. **It also targets the coding truncations.** A teacher-steered revision of a
   truncated or looping attempt is a positive example at the exact prefix
   where the body fails, which is what Right On Q's on-policy point asks for.
3. **Shorter successful trajectories** is the behavior we want: the current
   model burns 24k tokens and then fails. Steering rewards early, correct
   termination.
4. **Corrections-only adaptation**: instead of training the body, train the
   sidecar on the corrected turns. The paper trains the student body; our
   version puts the corrected-turn loss on the corrections.

## What does not transfer directly

- W4A16 and dense-ish agent models; our ternary 35B-A3B is much lower bit and
  the corrections are small.
- Needs a reliable executor/verifier per task (we have it for tools and code,
  not for open reasoning).
- The teacher must be able to solve the failed context; FlashNext can for
  many, but not all.

## Experiments this suggests

1. Take the current body's failed tools rollouts, have FlashNext steer a
   revision, re-execute, and train corrections on the corrected turns only.
2. Same for coding: failed/truncated attempts become steered revisions; keep
   only those that pass the checker.
3. Measure success rate and trajectory length, not just accuracy.

## Open questions

- Does steering data still help when only rank-512 sidecars train?
- How many steered trajectories are needed before the tools score moves?
- Can the same loop fix reasoning items, where there is no executor?
