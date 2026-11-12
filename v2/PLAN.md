# Scion 35B v2: FlashNext distill plan

Status: **PLANNING** (nothing started, current model unchanged)
Date: 2026-11-11

This document is the working plan. It is not a run log. The desktop copy at
`~/Desktop/scion-35b-v2-plan.md` is the same scope.

---

## 1. What we have today

The current Scion-35B-A3B artifact:

- Ternary Q2_0_g128 expert banks (2.125 bpw), Q8_0 for the rest, rank-512
  trained corrections. 11.3 GB, one GGUF for llama.cpp / Strata.
- Base: empero-ai/Qwen3.8-35B-A3B-Distill, built on Qwen/Qwen3.6-35B-A3B.
- Full-vocabulary KLD vs the BF16 teacher: 0.269 mean (Q4_K_M of the same
  model is 0.031).
- Short tasks: 23/30 on the Hive 30-task suite (76.7%), ~110 tok/s decode,
  ~1k tok/s prefill on one 7900 XT.

## 2. What we measured on the hard 90 (heldoutlab round 2)

Setup: heldoutlab harness, 90 hard items, 2x 7900 XT, T=0.6 (model default
sampling), 32k budget.

| metric | value |
|---|---|
| overall | 13/90 = 14.4% |
| reasoning | 4/30 |
| coding | 1/30 |
| tools | 8/30 |
| truncations at 32k | 17 (12 of them coding) |
| coding answers lost to an unclosed code fence | 9 |
| items past 25k tokens ending at 0.99 loop coverage | 26 of 30 |

Accuracy by completion length: <3k 33%, 3-10k 19%, 10-20k 0%, >20k 8%.

- The long-generated text derails into repetition after roughly 5-10k tokens,
  at T=0.1 and T=0.6 alike (temperature-invariant). Onset example: coverage
  0.06 at 4k words, 1.00 at 8k words.
- 24.4k of the 32k budget goes to thinking, so answers either arrive from a
  degenerate state or are cut mid-code. The 9 unclosed fences are pure
  answer-room losses.
- Code that does complete carries coarse errors (IndentationError,
  UnboundLocalError, ZeroDivisionError, missing attributes).

Controls: needle-in-haystack through the same engine is perfect at 2k-26.5k
prompt tokens (runtime fine); no repairs, lenient equals strict, answers are
real (harness fine); Bonsai 27B PTQ1_0 at 5.95 GB scores 61/90 = 67.8% on the
same items (2-bit per se is not the limit).

## 3. Why the current model fails at long horizon

The corrections were trained on 4096-token windows and a short corpus. Nothing
in the recipe ever saw a 10-30k-token reasoning chain, so the corrections cannot
patch the tail where the body derails. The ternary body is frozen and only
~35M correction parameters can steer it.

The 176B Scion-FlashNext proves the method works at 2.60 bpw when the body is
good and the corrections are trained against a strong teacher: 28/30 on the
easy suite, 80.5 tok/s on two cards.

## 4. Goal

A better, more open Scion 35B, still ternary and still consumer hardware
(~11 GB), with long-horizon behavior trained in.

Targets:

- hard 90: >= 40% (from 14.4%), within ~10 points of the teacher at
  comparable precision on a 30-item subset.
- No loop collapse before 24k generated tokens (coverage < 0.25 at every
  512-token check).
- Coding truncation < 10%.
- Same 11.3 GB tier, ternary only.

Openness: publish the trace corpus (teacher is Apache-2.0), the correction
recipe, the eval suite and raw results.

## 5. Decisions (locked)

1. **Ternary-specific.** No Q4/Q6K tier.
2. **No full-body QAT.** Quantizing the body is what damages the tail; the
   body is the most important part. Scion-FlashNext proves corrections alone
   are enough.
3. **Body frozen.** The existing ternary body stays untouched. Corrections are
   the only trainable part, trained with the deployed quantizer in the loop.
4. **Escalation is not QAT.** If corrections plateau, the next step is a bf16
   SFT/LoRA of the body on FlashNext traces, then quantize once, then retrain
   corrections.
5. **Eval sets are training-excluded.** heldoutlab hard 90 and the easy 30 are
   never in the trace corpus.
6. **v1 is output-KD only.** Router deltas need teacher router logits, which a
   GGUF teacher cannot expose. The pipeline has a --freeze-router arm.

## 5a. Prior-art inputs (adopted 2026-11-11)

Full analyses: [prior-art-analysis/](prior-art-analysis/). What changes:

1. **Failure-driven correction data.** Train on rollouts from the frozen body,
   not only on teacher traces. Two variants: on-policy KD with Dynamic KL and
   a 4k/8k/16k length schedule (Right On Q), and teacher-steered revisions of
   failed tool/coding trajectories (SteerDistill).
2. **Closed-form baseline.** Align, Then Correct (asymmetric low-rank
   compensation, one SVD per layer) at rank 64 and 512, evaluated before the
   full distill. If it matches the trained corrections, cut the training.
3. **A format-preserving body-side stage (optional).** Scale-only
   distillation: scales and norm gains trainable, ternary codes frozen,
   forward KL to the teacher (ThinkBit, RoleQ). Not QAT. Add routing-consistent
   norm adaptation only if router drift appears (ProQuant).
4. **Export audit is a gate.** Verify the exported codes equal the
   training-time codes on every tensor; never recompute codes from float
   latents (Lost in the bf16 Cast). Add the code-level diff to the release
   checklist.
5. **Eval additions.** Repetitive-error rate, termination rate, trajectory
   length, real bits per parameter and whole-file size (ThinkBit), plus two
   external baselines.
6. **Size track, in parallel.** A Q2_0 -> PTQ1_0 repack (same trits, tighter
   packing) and an embedding/head reduction. Measured: embedding + head are
   0.540 GB each, 1.08 GB total, 9.5% of the 11.3 GB. Caveat: the q36 HIP
   expert kernels are Q2_0-specific; PTQ1_0 exists on disk and in the CPU
   expert path, but the GPU path is a kernel prototype, not a free repack.

## 6. Feasibility checks (done)

- No FlashNext distills exist on the Hub at any size (searched 2026-11-11).
- Teacher: SkyIsNotGreen/Scion-FlashNext-176B-A6B, base
  Qwen/Qwen3.8-Flash-Next-FP8, Apache-2.0. 57.6 GB on HF; local copies at
  models-p310/scion-strata and c10e404b-.../ab-models.
- Student: Qwen3.8-35B-A3B does not exist. Options:
  empero-ai/Qwen3.8-35B-A3B-Distill (current Scion body) or
  Qwen/Qwen3.6-35B-A3B base.
- Tokenizer: identical. Both 248,320 tokens, identical merges, same special
  IDs (im_start 248045). Top-k logit KD is possible.
- Architectures differ: teacher Qwen4Exp, student Qwen3_5Moe. Traces plus
  logit KD, no weight surgery.
- Pipeline exists: moe/olmoe_corrections.py, moe/olmoe_proxy.py,
  moe/role_map.py, export via the Strata bridge.

## 7. Plan

### Phase 0: pilot and gates (local, $0-10)

1. **Export audit (gate).** Re-run the exporter on the current model and diff
   the exported ternary codes against the training-time codes, per tensor.
   Count mismatches per million weights. Also confirm the branch exporter
   (`pack_q1_0_g128`) is fed deployed ternary values, never the float master.
2. **External baselines.** Run Ternary-Flint (Lean or PQ2_0) and the ternary
   Nemotron 30B-A3B (8.82 GB) on the 30-item hard subset with the fixed
   protocol, on this box. Same-tier references for the 11.3 GB artifact.
3. **Failure rollouts.** Capture the current body's failed tools and coding
   rollouts (plus its looping long items) for the steering and on-policy data.
4. Corpus: 300-500 hard reasoning/coding prompts from the downloaded benches
   and work/corpora, eval sets excluded.
5. Teacher pass: Scion-FlashNext 176B at 4-8k context. **Correction
   (2026-11-12): the local 2-card 20 GB box cannot hold the 60 GB Q6_K
   teacher; the teacher pass needs a rented 80 GB+ card (see 13a).**
6. Corrections: retrain rank-512 branches, Dynamic KL, long windows (8k),
   quantizer in the loop, on (a) teacher traces and (b) rollouts plus steered
   failures. **Correction (2026-11-12): the current qwen35 proxy loads the full
   69 GB bf16 body and has no compact-bank or gradient-checkpoint path. The
   "18 GB local / one 4090" assumption is invalid until the compact-bank port
   in 13a is done and measured; do not size a rental from this line.**
7. Closed-form baseline: Align, Then Correct at rank 64 and 512 on the same
   calibration data.
8. Eval: 30-item hard subset, fixed protocol (non-streaming loop recovery,
   presence_penalty 1.5, budgets 8k/16k), plus repetition and termination.
9. Gate: loop onset past 16k tokens on at least half the long items, or +10
   points on the subset over the current body. If neither, stop and report.

### Phase 1: full corrections (runpod, $40-80)

1. 5-8k verified traces and steered rollouts (16k cap), one teacher pass with
   the logit cache.
2. Corrections on a 48 GB card with long windows and the length schedule;
   compare against the closed-form baseline and against a rank sweep
   (64/128/512) at matched bytes.
3. Checkpoint selection on the hard subset, not KD loss.
4. Export GGUF + Q2_0 pack, full eval: hard 90, easy 30, loop onset, budget
   sweep 4/8/16/32k, repetition/termination, seconds per correct.

### Phase 2: only if Phase 1 plateaus

Try, in order, before touching the body:

1. Scale-only distillation (codes frozen, scales + norm gains, forward KL) on
   the current body; eval the hard subset. Format-preserving.
2. A sparse second ternary plane on Fisher-selected expert units (+~0.5 bpw),
   eval hard 90 per added byte.
3. bf16 SFT/LoRA of the body on FlashNext traces (no QAT), merge, quantize
   once, retrain corrections.

### Size track (parallel, no training)

1. Prototype a Q2_0 -> PTQ1_0 transcode on one expert tensor and measure the
   file saving (~1.7 GB expected) and decode speed. Only proceed to the full
   model if a q36 HIP kernel runs at parity.
2. Embedding/head: 1.08 GB is 9.5% of the file. Options: Q6K head/emb
   (~0.3 GB saving) or a frozen codebook (entry 008, up to ~1.0 GB). Any
   change needs a short recovery and a hard-subset check.

### Phase 3: release

Ternary model, trace corpus dataset, eval suite, recipe, raw results.

## 8. Budget

| item | where | cost |
|---|---|---|
| pilot traces | local, 80 tok/s | $0 |
| pilot corrections | local or 1x 4090 | $0-5 |
| full traces (8k prompts, 16k cap) | A100 80 GB spot | $25-50 |
| full corrections, long windows | A6000 48 GB or 4090 | $10-30 |
| reruns | | $10-20 |
| **total** | | **~$45-105** |

No full-body training line item.

## 9. Risks

- Corrections may not move long-horizon behavior enough; the body could be the
  hard ceiling. Phase 2 is the answer, not QAT.
- The teacher is the 2.60 bpw Scion build, itself distilled from the FP8. A
  cleaner FP8 teacher needs 3x H100 and multiplies the trace cost.
- Long-window corrections training is slower and needs more cache memory.
- Eval contamination must be enforced.

## 10. Next step

The ordered task list is in section 13. Start with the export audit (it gates
every number), then the two external baselines, then the failure-driven pilot.

## 11. Repo cleanup note (2026-11-11)

The 54 GB untracked artifact tree under moe/artifacts was removed during this
planning pass. Preserved: every small evaluation JSON (now under
moe/results/olmoe and moe/results/qwen4exp), the two final router-freeze
correction checkpoints, and eval-ref-w2.pt. The OLMoE model snapshot and the
partial empero teacher download are re-fetchable. Teacher caches, intermediate
step checkpoints and the qwen4exp kld-park windows were deleted as
reproducible.

## 12. Open questions register

Closed or decided:

| question | answer |
|---|---|
| How much of the file is embedding + head? | **Closed:** 0.540 GB each, 1.08 GB total, 9.5% of 11.3 GB (hidden 2048, vocab 248,320, Q8_0). |
| Can the 5-trit packing fit the kernels? | **Partly:** on-disk type 143 and the CPU expert path exist; the q36 HIP expert kernels are Q2_0-specific. Treat as a kernel prototype, not a repack. |
| Is scale-only distillation allowed under the frozen-body rule? | **Decided yes:** codes frozen, scales + norm gains only (<0.1% params), format-preserving. Not QAT. |
| Do we train the router? | **No:** D2/D4 and ProQuant's routing-drift finding. |
| Do we QAT the body? | **No:** D2, QAH's collapse, and the Scion-FlashNext evidence. |
| Is export fidelity a real risk? | **Yes:** the branch exporter recomputes codes from float weights; exact only when fed deployed ternary values. Audit is a gate. |

Still open, each with a decisive test:

| question | test and pass condition |
|---|---|
| Can rank-512 corrections fix exposure bias? | Pilot A/B (traces vs rollouts vs steered failures); pass if loop onset >16k on >=50% of long items or +10 points on the 30-item subset. |
| Does the teacher family matter for Dynamic KL? | FlashNext vs empero bf16 teacher on the 30-item subset, same data and rank. |
| Does closed-form compensation match trained corrections? | Align, Then Correct at rank 64/512 vs trained at the same bytes, hard subset. |
| Does scale-only distillation move long-horizon behavior? | Apply to the current body; hard subset and loop onset before/after. |
| Do Flint and ternary Nemotron beat Scion on the hard 90? | External baselines, 30 items, same box and protocol. |
| Does PTQ1_0 packing cost decode speed? | Single-tensor transcode plus a one-layer q36 HIP kernel run; require parity before the full repack. |
| Is the current shipped GGUF code-faithful? | Dequantize every tensor, compare codes to training-time role_map codes, count mismatches per million weights. |

## 13. Next actions, in order

1. Export audit on the current 11.3 GB artifact (gate for everything else).
2. External baselines: Ternary-Flint and ternary Nemotron on the 30-item hard
   subset.
3. Capture failure rollouts from the current body (tools, coding, long items).
4. Build the pilot trace + steered-revision corpus; run the corrections A/B
   (traces vs rollouts vs steered), closed-form at rank 64/512, rank sweep.
5. Scale-only distillation on the current body (parallel, cheap).
6. PTQ1_0 single-tensor transcode and kernel parity check (size track).

## 13a. Readiness gate and execution status (2026-11-12)

**No pod is authorised until the readiness criteria below pass.** This section
records the audit result and the memory/ops reassessment; it does not change
the locked decisions in section 5.

### Status against section 13

| # | action | status | evidence |
|---|---|---|---|
| 1 | Export audit | **Done; v1 strict gate failed, v2 fix proven** | `v2/export-audit/README.md`: 3,660.9 ppm code mismatch vs the Python training rule; artifact is faithful C++ Lloyd output; re-quantising deployed values is 0 ppm. `moe/seed_banks_from_gguf.py`, `--bank-gguf`. |
| 2 | External baselines | **Partial** | Nemotron **0/24**, stopped at 24/30 (22 truncations, 4 server errors); Flint blocked (Metal-only `flint` runtime, no Linux/ROCm arch, only the NonaCore file). `v2/BASELINES.md`. |
| 3 | Failure rollouts | **Pilot sample** | 360-prompt corpus with hard-90/easy-30 excluded; 12 on-policy rollouts; loop coverage 1.0 on the long reasoning items. `v2/PROGRESS.md`. |
| 4 | Pilot corrections / ATC / rank sweep | **Not started** | Only a 4-layer prefix smoke with a synthetic cache; not a memory or data validation. `v2/PREFIX-SMOKE.md`. |
| 5 | Scale-only distillation | **Not started** | No code, no run. |
| 6 | PTQ1_0 + kernel parity | **Done as prototype** | 11.337 -> 9.827 GB; lossless trits/scales; HIP parity PASS; cross-type KLD 0.0030; PPL tie. `v2/PTQ1_0.md`. |

### Why the old Phase 0 hardware line is wrong

`qwen35_moe_proxy.py` loads the full 35B bf16 body (~69 GB) and has no
`CompactBank` or gradient-checkpoint path. The qwen4exp proxy has both, and
its bank layout is the same fused `gate_up_proj`/`down_proj`. Therefore:

- the 20 GB local card and the 48 GB pod cannot hold the frozen body;
- an 80-96 GB card is at best borderline for 4k sequences once branch
  gradients and activations are included;
- the 4-layer smoke (7 GB) proves plumbing only, not full-model memory;
- teacher-pass and training estimates in section 7 / Phase 0 were not
  grounded in measured numbers.

### Readiness criteria (all must pass, all local/free)

1. **Compact student.** Port `CompactBank` to qwen35; decode bit-exact against
   the current bank; measure peak VRAM/RAM at 512/2k/4k/8k on the 4-layer
   prefix; extrapolate to 40 layers and state the smallest viable card.
2. **Real cache path.** Produce a real `llama-perplexity --save-all-logits`
   dump from a small local model, parse it with `moe/build_trace_cache.py`,
   load the cache, and complete one prefix training step. Check teacher/student
   tokenizer alignment.
3. **Experiment choice.** Pick the first experiment so it can actually test
   the section 7 gate. A 512-token-only trace-KD run cannot test "loop onset
   past 16k"; either use 4k+ windows or change the gate before training.
4. **Arms.** Implement or explicitly descope: traces, rollouts, steered
   revisions, Dynamic KL / length schedule, ATC rank 64/512, rank sweep,
   scale-only.
5. **Full loop on the prefix.** Seed -> real cache -> train -> export ->
   merge -> subset30 -> code audit, end to end.
6. **Ops plan.** Exact commands, measured runtime, peak memory, disk budget,
   resume points, artifact sync/checksum, cost cap and abort conditions.
7. **Protocol freeze.** n, seeds, metric, gate and decision tree fixed before
   the run.

### Order

A. CompactBank port + memory study.
B. Real-logits cache validation on a small local model.
C. Choose the first experiment and write its falsification criterion.
D. Implement/descope the arms; exercise the full loop on the prefix.
E. Write the run plan and cost model from measured numbers.
F. Ask for approval; only then create a pod.

`moe/runpod-pilot.sh` is a sketch for E/F, not an approved runbook. The
previous "minimal pilot, $2-8" estimate is withdrawn.
