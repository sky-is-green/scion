# v2 progress log - 2026-11-11

Execution of PLAN section 13, in order.  Commits are on `scion-test`.

## 1. Saved-state audit (done)

- HEAD `494cfb8`, clean, synced with origin.
- `sha256sum -c v2/SHA256SUMS`: 24/24 OK.
- EVAL.md hashes match both JSONL files; summaries recompute exactly
  (90: 13/90 = 14.44%, R 4 C 1 T 8, 17 trunc; 38: 10/38 = 26.32%).
- Model sha256 matches `4e9e3292...cffa040`.
- Report: `v2/AUDIT-2026-11-11.md`.  No inconsistencies.

## 2. Export-code audit (gate, done)

`v2/export-audit/README.md`, commit `7ed4a1b` (manifest fix `3c12d95`).

- Branch factors: 160 tensors, 209.7M weights, 0 fourth-state codes, 0
  non-max scales.  `pack_q1_0_g128` is fed `deploy_weights()` output, not the
  float master (code path + packed invariants).
- Expert banks: 120 tensors, 32.21B weights.  **117,926,147 code mismatches =
  3,660.9 ppm** vs the Python training rule.  19.24% of groups have a
  different fp16 scale; the mismatches are 9.27 ppm where the scale agrees and
  18,984 ppm where it differs.
- Provenance: a literal C++ transcription reproduces the deployed layer-0
  scales bit-for-bit and the codes to 3.66 ppm; the shipped file is the
  C++ `GGML_PQ2_0_LLOYD` output.
- Remediation: re-quantising the deployed dequantised banks with the Python
  rule gives **0 code mismatches** at f32 and bf16 across all 120 tensors.
  `moe/seed_banks_from_gguf.py` makes the deployed GGUF the training-time
  source of truth; `--bank-gguf` is wired into the trainer.  No body re-pack.
- **v1 strict gate: fail; v2 path: pass with the seed fix.**  Re-audit after
  the first v2 export and require 0 ppm.

## 3. External baselines (partial)

30-item subset definition: `v2/eval/subset30.json` (first 10 per family by
the harness `stable_order`, interleaved).  Scion's own result on that subset,
extracted from the saved T=0.6 run: **3/30 = 10.0%** (R 0/10, C 0/10, T 3/10),
7 truncations, median 291 s, 77.1 tok/s.

- **Ternary Nemotron 30B-A3B** (8.82 GB): **0/24, stopped at 24/30** with the
  same protocol (T=0.6, seed 1001, 32k, 65k server context, q8 KV); 22
  truncations, 4 server errors, 5 repairs.  All four long reasoning items end
  at 10-gram coverage 1.0.  Results: `v2/eval/hard-subset30-nemotron/`.  This
  is a same-tier reference only; it is a community Q2_0 repack.
- **Ternary-Flint-40B**: blocked on this box.  The Hub repo currently has one
  file, `Ternary-Flint-40B-NonaCore-Q4_0-9.8GB.gguf` (10.58 GB, arch `flint`),
  and its card ships a macOS-arm64 Metal runtime only (custom `flint` arch +
  `prism.hadamard.*` + Metal PTQ1_0 kernels).  The Linux/ROCm tree has no
  `flint` architecture.  The Lean/PQ2_0 variants named in the prior-art note
  are not in the repo.  Recorded, not run.

## 4. Failure rollouts (done for the pilot sample)

- Corpus builder: `v2/scripts/build_rollout_corpus.py`; 360 prompts
  (160 reasoning from AIME/MATH-500, 100 coding from MBPP/HumanEval/
  LiveCodeBench, 100 synthetic tools).  Heldoutlab hard-90 and the 30-task
  test split are excluded by normalized exact match (138 exclusions).
  Manifest: `v2/rollout-corpus-manifest.json` (sha256 of the prompts).
- 12 rollouts captured from the shipped body (4 per family, 16k cap):
  `v2/rollouts/body/` (gitignored), summary + analysis.  **Correction
  2026-11-12: the analysis.json "cov" values for these rollouts are wrong.**
  Recomputed with the 10-gram duplicate-fraction and covered-position
  metrics on the stored texts, the three "long reasoning rollouts" are
  *clean solutions*, not loops: PR-001 (2,935 tok) dup 0.003 / covered 0.013
  (reasoning-only), PR-002 (14.1k tok) dup 0.089 / covered 0.175
  (reasoning+content); both end with a wrap-out and a real answer, and a
  byte-identical rerun of PR-001 through the same server reproduced it.  The
  "coverage 1.0" record came from a broken analysis computation and is
  retracted; the rollouts are on-policy traces, **not** loop exemplars.  The
  hard-90 loop evidence is unaffected (verified independently, see below).
- The hard-90 loop evidence re-verified 2026-11-12 on the stored rows: all
  30 items past 25k tokens show 10-gram duplicate fractions 0.65-0.97
  (e.g. R07 0.948, C07 0.955) - genuinely degenerate completions.  The
  primary gate's basis stands.
- The hard-90/easy-30 text is not in the corpus; no eval data enters training.

## 5. Pilot corrections (blocked on GPU stock)

- Runbook: `moe/runpod-pilot.sh` (setup / teacher / score / train / export).
  Teacher = Scion-FlashNext-176B-A6B Q6_K (60.2 GB), cache via
  `llama-perplexity --save-all-logits` -> `moe/build_trace_cache.py` ->
  `qwen35_moe_proxy.py train --bank-gguf ...`.
- Runpod has had no 2xA6000, A100-80 or H100 capacity since 18:25; the account
  balance is $6.45 and the spend limit $80, so it is stock, not budget.  Retry
  the one-liners in `moe/runpod-pilot.sh` when a pod appears.
- Local 4-layer prefix smoke passed end-to-end: seeded banks, `ids`-carrying
  cache, both placements, 2 steps, checkpoint, F16 adapter export.  Bugs found
  and fixed: `layer_type` detection on transformers 5.5, int32 `ids` to CE,
  missing checkpoint directory.  Details: `v2/PREFIX-SMOKE.md`.
- Rank 64 cannot be packed as Q1_0_g128 (the group axis is the rank and must
  be a multiple of 128); rank-64 sweeps need F16 factors or a g64 container.

## 6. PTQ1_0 size track (done)

`v2/PTQ1_0.md`.

- Full 120-tensor transcode: **lossless** (0 trit/scale mismatches),
  11.337 GB -> 9.827 GB (saving 1.510 GB), 1.75 bpw experts.
- Real-weight HIP parity: `ptq1_0_parity` PASS (gu 2.2e-11, down 9.3e-10;
  decode-once vs per-entry bitwise; 0 failures).  The tool needed a 3-line
  geometry patch for the 35B.
- Engine probe: 8/8 generated tokens identical, top-1 8/8, but Q2_0 vs PTQ1_0
  logits have KLD mean 0.0030 / max 0.0158 (rel-L2 3.6%).  Prototype only;
  hard-subset A/B required before a release repack.

## Path reassessment (2026-11-12)

The 35B v1 path is the base, not something to rebuild.  Scion-FlashNext-176B
is the same corrections-only method at 176B and supplies the concrete deltas:

- the KD recipe (top-512 + tail-64, tail weights, support weighting, temp 2.0,
  balance bias, one variable between arms);
- the memory switches already proven at 176B: `--compact-banks`,
  `--grad-checkpoint`, group checkpointing (the bank layout is shared);
- checkpoint selection on the capability gate, not KD loss (the step-4000
  extension was flat/worse and not shipped);
- the evaluation discipline: accuracy, truncation, repetition and loop onset
  first; KLD/PPL are diagnostics; the teacher is itself a 2.60 bpw distill.

The old "18 GB local / one 4090 / 48 GB pod" line is invalid; the known 35B
path is H100/H200 80 GB+, and the FlashNext switches are what make 4k/8k
windows fit.  The 4-layer smoke used a random cache, so it is plumbing only.

Local-first order (PLAN 13a): (A) port/tune the three memory switches on
qwen35 and measure 2k/4k/8k on the prefix; (B) real `_logits_` cache test;
(C) build the first v2 trace cache and freeze the experiment/gate; (D) full
prefix loop (seed -> train -> export -> merge -> subset -> audit); (E) exact
command, measured cost and stop rule; (F) approval.  `runpod-pilot.sh` is a
sketch for E/F; the previous "minimal pilot, $2-8" estimate is withdrawn.

## Next

1. Port the FlashNext memory switches to qwen35; measure the prefix.
   **Done 2026-11-12 - `v2/MEMORY-PORT.md`.**  The three switches are ported
   (all default off), the compact container is read verbatim from the
   deployed GGUF, and the bit-exactness gate passed on real weights: 3.22B
   prefix weights, 0 mismatches, decode == deployed dequantised values.
   Measured on the 4-layer prefix (20 GB card, recipe geometry): dense v1
   OOMs at 2k; compact+ckpt fits 2k (15.4 GiB) and, with the chunked-CE
   lever, 4k (13.9 GiB); 8k is 1-2.5 GiB short with the levers identified
   (causal-conv1d, GDN ssm dtype, split-window accumulation).  Full-model
   extrapolation: the compact model is ~13.6 GiB resident (vs 69.4 dense),
   4k/8k windows need a 24-48 GB card.  28 new unit tests pass; the 41
   failing qwen4exp tests are pre-existing (transformers 5.17 vs the 5.5-era
   rig) and unchanged by the port.
2. Real `llama-perplexity --save-all-logits` cache test (local, free).
   **Done 2026-11-12 - `v2/REAL-CACHE.md`.**  Real dump from the deployed
   35B through the prism fork, decoded, aligned and trained on (2 prefix
   steps through the compact-banks port).  Key finding: the prism forks
   score only the SECOND half of each chunk (255 rows, `first = n_ctx/2`),
   same in the production teacher fork - `build_trace_cache.py` now detects
   the layout and aligns ids so trainer row i predicts `ids[i+1]` (4 unit
   tests).  Alignment: dump tokens == HF ids 2048/2048; teacher NLL from
   the cache 2.232 vs the engine's 2.263; teacher support mass 0.96; the
   FlashNext GGUF tokenizer matches the student (248,320 tokens, im_start
   248045).  Upgraded the same day (`c0c84ab`): vectorized decode,
   sampled teacher-tail tokens (tidx/tlp) so the FULL FlashNext recipe
   (D_KL2 included) runs off real dumps, truncated-dump tolerance.
3. Freeze experiment 1.  **Done 2026-11-12 - `v2/EXPERIMENT-1.md`**
   (cur05-35B vs pred2.0-35B, one variable, 4k windows, falsification
   criterion, eval protocol, budget sketch).
4. Full prefix loop.  **Done 2026-11-12 - `v2/PREFIX-LOOP.md`.**  Seed ->
   train -> export -> merge (--replace) -> audit PASS -> serve -> harness
   smoke, all local; plus the recovery findings (detection non-streaming OK;
   recover mode inert for the 35B pack - protocol options recorded) and the
   rollout-coverage correction (`fb26b6e`).
5. Remaining local work before E: eval-protocol decision on the recovery
   options (a)/(b) and the same-protocol baseline rerun; E draft (exact
   full-model command, measured runtime/memory, cost cap, stop rule); then
   F (approval).
