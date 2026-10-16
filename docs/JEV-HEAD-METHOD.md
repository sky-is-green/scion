# The JEV-head method — fit, serve, policy (2026-10-15)

A per-model correctness readout that rides the served model's own hidden state.  Same lifecycle as the
MTP sidecar (fit per body, shipped with the model), but the method is one thing: three recipes below,
finalised on the q36 engine with the 35B as the reference.  Porting to another model is a refit plus
the engine's two hooks; nothing about the readout is model-specific except the weights.

## 1. Fit (per body, in the serving format)

The study's rule holds: a head does not transfer across bodies or prompt formats, so every body gets
its own fit, on the prompt it will actually be served.

1. **Judge prompts.**  The decision prompt (system "You are a decision function...", then
   `Question / Candidate answer / Is the candidate answer correct? / A. Yes / B. No`) rendered with the
   body's own chat template, thinking off, tokenized.  378 judge-bench items (262 train / 116 test,
   sha `3289dcb8...`) is the frozen set; community tasks work the same way.
2. **States.**  The engine's own extraction: q36 `EMB path=<file> <ids>`, main engine `GEN 1 emb=<file>`.
   Both write the last prompt token's normed hidden, which is the unembedding input: the same vector
   the head reads at serving time, so there is no train/serve skew to discover later.
3. **Fit.**  Ridge on standardised features (lambda 1) plus Platt calibration on the train split, then
   export in the study's JSON schema and convert to the engine's binary (`head_bin.py --from-json`,
   magic `SJH1`, 12 KB per 1000 dims).  The two forms agree in Python to 5.6e-8 on all 378 vectors.
4. **Gate.**  Frozen-split AUC of the fitted head.  Our numbers: 1.0000 (35B, q36 states),
   0.9941 (176B, main-engine states), against 0.903 for the prompted self-judge.  A weak body or a
   wrong extraction shows here; do not ship below ~0.95 without a look.

## 2. Serve (the sidecar)

The engine loads the binary (`STRATA_Q36_JEV_HEAD=<file>` on q36; the main engine's equivalent is not
wired yet) and an EMB request answers **`EMB <dim> <file> <p_correct>`**: the vector written raw, then
the readout computed from that raw vector on the host (a dot product; no generation-time cost, no extra
resident model, no torch).  A request without a head loaded answers as before.

Measured on q36 + Scion-35B-A3B-PQ2_0-corr, one 7900 XT, 2026-10-15:

| what | number |
| --- | --- |
| judge pass, 378 items | 123 s (325 ms mean; min 101 ms; max 22.9 s, one cold outlier) |
| engine `p` vs the same head in Python, same vector | worst 5.0e-7 |
| engine verdicts, frozen test split (116 items) | AUC 1.0000, BA 1.0000 at 0.90 |

The test is `dh_verdict.py`.  On the main engine the readout reads `head_mixed_` (the last stage's
head input, what `emb=` already writes) and the same ten lines of arithmetic apply.

**The API path (any body, no engine build):** a run config may name a `decision_head` (the JSON this
directory exports), and `POST /v1/verdict {"task", "candidate"}` then answers `p_correct`, the head's
threshold and `accept` over one judge pass, computed in the server (pure Python, no numpy).  That is
the delivery for the 176B and the 35B alike: a client that already knows how to call the engine gets
the verdict from the same server, and the engine-side `STRATA_Q36_JEV_HEAD` above stays the
zero-copy fast path where it exists.

## 3. Policy (what pays, and what does not)

The head is an **end-of-trace verifier**.  The study's negative result stands: early cancellation saves
10-19% of tokens while cancelling 3-15% of correct samples, and wrong answers are committed upstream, so
steering generation with it is a losing trade.  The two paying uses:

- **veto**: majority answer plus head override (switch to the head's pick only when it beats the
  majority by a margin tuned on train): the study's +5-7 tasks over 199 held-out per body;
- **gating**: sample sequentially and accept the first candidate with `p >= thr` (thr tuned on train):
  best-of-3 accuracy at about half the pool cost.

Both live above the engine (the harness or the server); the engine's job is only to answer `p_correct`.
For the cascade suite the natural client is the validator adapter: one call to the resident engine, no
separate judge model.

**Measured on the 176B** (2026-10-16 night, the full 664-task community protocol, held-out 199):
within-pool AUC **0.965**; veto over majority **+11 tasks (14 fixes / 3 breaks, p=0.013)** with the
in-domain head; best-of-3 151/199 recovers 73% of the single-to-oracle headroom; gating the weakest arm
(+3.6 points at 1.43x tokens).  On this body the in-domain refit beats the J-Bench-frozen head for both
selection and veto, so the study's "validate per model" caveat is the rule.  Full table:
`docs/DECISION-HEADS-PORT.md`, "The 176B community arms".

## What is not done

- The 176B's engine-side sidecar readout (the main engine's `head_mixed_` behind the same protocol);
  the vector path (`emb=`) is in and the head exists, the API route covers it, and the in-engine
  verdict on the main path would only save a file read.
- In-stream scoring (the engine running the judge readout on the trace it just produced) is a product
  option; the reference path scores a judge prompt the client sends.
- The think-harder arm (B) on the 176B: left out of the night on purpose (dominated by gating in the
  study; about five hours of 2,048-token thinking for an arm already answered).

## Published (2026-10-16)

- `SkyIsNotGreen/decision-heads`, commit `d991c74`: `PORT-REPORT.md` plus the three heads
  (176B J-Bench and community, 35B q36), README updated.
- `SkyIsNotGreen/Scion-35B-A3B`, commit `ac1572d2`: a `decision-head/` folder (JSON + engine binary +
  usage) and a section in the model card with the measured numbers.
- Strata side: `POST /v1/verdict` + the `decision_head` config key (commit `8234da4c`), the q36
  engine sidecar (`STRATA_Q36_JEV_HEAD`), all in the local trees, not pushed upstream.
