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

### Phase 0: pilot (local, $0-5)

1. Corpus: 300-500 hard reasoning/coding prompts from the downloaded benches
   and work/corpora, eval sets excluded.
2. Teacher pass: Scion-FlashNext 176B on the local 2-card box at 4-8k
   context, ~80 tok/s. Traces plus top-k output logits in one pass.
3. Corrections: retrain rank-512 branches, output-KD only, long windows (8k),
   quantizer in the loop. Local 7900 XT if it fits (~18 GB working set), else
   one rented 4090 ($2-5).
4. Eval: 30-item hard subset, fixed protocol (non-streaming loop recovery,
   presence_penalty 1.5, budgets 8k/16k).
5. Gate: does loop onset move past 16k tokens, and does the subset score rise
   above the current body?

### Phase 1: full corrections (runpod, $40-80)

1. 5-8k verified traces (16k cap), one teacher pass with the logit cache.
2. Corrections on a 48 GB card with long windows.
3. Export GGUF + Q2_0 pack, full eval: hard 90, easy 30, loop onset, budget
   sweep 4/8/16/32k, seconds per correct.

### Phase 2: only if Phase 1 plateaus

bf16 SFT/LoRA of the body on FlashNext traces (no QAT), merge, quantize once,
retrain corrections.

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

Build the 300-prompt corpus manifest (eval sets excluded), start the local
teacher trace pass, then wire the long-window corrections pilot.

## 11. Repo cleanup note (2026-11-11)

The 54 GB untracked artifact tree under moe/artifacts was removed during this
planning pass. Preserved: every small evaluation JSON (now under
moe/results/olmoe and moe/results/qwen4exp), the two final router-freeze
correction checkpoints, and eval-ref-w2.pt. The OLMoE model snapshot and the
partial empero teacher download are re-fetchable. Teacher caches, intermediate
step checkpoints and the qwen4exp kld-park windows were deleted as
reproducible.
