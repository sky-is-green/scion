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
  `v2/rollouts/body/` (gitignored), summary + analysis.  All three long
  reasoning rollouts (12.6k, 14.1k and 2.9k tokens) end at **10-gram coverage
  1.0**; coding/tools are short (75-741 tokens), one tool call emitted.
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

## Readiness reassessment (2026-11-12)

**Not pod-ready.**  The qwen35 training path loads the full ~69 GB bf16 body
and has no compact-bank or gradient-checkpoint mode; the qwen4exp proxy has
both and the same expert layout.  The old "18 GB local / one 4090 / 48 GB pod"
assumptions are invalid, and the 512-token runbook cannot test the >16k
loop-onset gate.  The 4-layer smoke used a random cache, so it is plumbing
only.

Readiness criteria (PLAN 13a): compact-bank port + measured memory at
2k/4k/8k; real `_logits_` cache validation; an experiment that can test the
gate; arms implemented or descoped; full prefix loop (seed -> train -> export
-> merge -> subset -> audit); measured ops plan with cost cap; frozen
protocol.

Local-first order: (A) CompactBank port + memory study, (B) real-cache
validation on a small model, (C) experiment choice, (D) arms + full prefix
loop, (E) run plan from measured numbers, (F) approval.  `runpod-pilot.sh` is
a sketch for E/F; the previous "minimal pilot, $2-8" estimate is withdrawn.

## Next

1. CompactBank port + memory measurement (local, free).
2. Real `llama-perplexity --save-all-logits` cache test (local, free).
3. Choose and freeze the first experiment; then implement/descope its arms.
4. Only after PLAN 13a passes, write the run plan and ask for approval.
