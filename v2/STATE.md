# Project state - 2026-11-11

Status: **audit passed; section 13 started.**  Export gate found a 3,661 ppm
v1 training/deployment code mismatch and the seed-from-GGUF fix is in;
external baselines and rollouts are partly done; pilot training is blocked on
runpod GPU stock.  See [PROGRESS.md](PROGRESS.md).

## Repository

- `sky-is-green/scion`, branch `scion-test`, synced with origin.
- 440 tracked files; working tree 771 MB, of which 522 MB is the preserved
  OLMoE artifacts (two final router-freeze checkpoints, `eval-ref-w2.pt`,
  `strata-ppl/`). The rest is the 54 GB -> 771 MB cleanup from this session.
- `v2/` is the project space:
  - `README.md` - scope and layout
  - `PLAN.md` - plan, prior-art adoption, open-questions register, next actions
  - `LESSONS.md` - negative register digest and v2 rules
  - `prior-art-analysis/` - 11 analyses + the survey longlist
  - `scripts/hard90_scion.py` - the eval runner (patched to record seed/T/arm)
  - `configs/scion35b-strata.json` - the server config
  - `eval/` - the two hard-90 result files and their summaries
  - `EVAL.md` - evaluation state, hashes, reproduction steps

## Models on this box

- Scion-35B-A3B-PQ2_0-corr.gguf, 11.3 GB,
  sha256 `4e9e3292c2af320b436b9c0cc10d10aee86689b0f0ce9d390be9d8365cffa040`
  at `models-p310/scion-strata/`.
- Scion-FlashNext-176B-A6B Q6_K, 60.2 GB, at
  `c10e404b-.../ab-models/Scion-FlashNext-176B-A6B-Q6_K.gguf` (v2 teacher).
- BF16 body source for the export audit, 69.4 GB, sha256 `ecbe9e21...666660`
  at `models-p310/bf16/Qwen3.8-35B-A3B-Distill-BF16.gguf`.
- Ternary Nemotron 30B-A3B, 9.47 GB, at
  `models-p310/nemotron/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf`.
- `empero-prefix/empero-hf` holds the first three HF shards for the 4-layer
  local pipeline smoke (not a full model copy).
- The partial empero bf16 teacher (13 GB) was deleted in the cleanup; it is
  public and re-fetchable.

## Evaluation

- `v2/eval/hard90-scion-32k/` - 90 items, T=0.6, seed 1001, 13/90 (14.4%),
  17 truncations.
- `v2/eval/hard90-scion-32k-t01/` - 38 items, T=0.1, seed 1001, 10/38 (26.3%).
- Full detail, hashes and reproduction: `v2/EVAL.md`.

## Other repos and assets

- `hivebench` (`hive_test`): the hard-90 results were gitignored there; they
  are now copied into `v2/eval/`. The runner was untracked there; it is now in
  `v2/scripts/`. Other modified/untracked files in that repo are from separate
  work and were left untouched.
- Public dataset (Bonsai reproduction, unrelated to v2):
  https://huggingface.co/datasets/SkyIsNotGreen/bonsai2-27b-round2-repro
- Desktop: `scion-35b-v2-plan.md` is synced with `v2/PLAN.md`;
  `scion-v2-research-survey.md` is mirrored at
  `v2/prior-art-analysis/SURVEY-2026-11-11.md`.

## Runtime state

- No scion processes or user units are running.
- `/tmp/opencode/bonsai2-27b-test` (heldout harness clone) was wiped by the
  12:08 reboot; re-clone before the next eval.
- `/tmp/opencode/scion35b-strata.json` is preserved in `v2/configs/`.

## Section 13 checklist

1. **Done, gate failed for v1 / fixed for v2.** 3,661 ppm code mismatch vs the
   Python rule; the artifact is faithful C++ Lloyd output; seed the training
   body from the deployed GGUF (0 ppm after re-quantisation).  See
   [export-audit/README.md](export-audit/README.md).
2. **Partial.** Nemotron running on the 30-item subset; Flint blocked (Metal-
   only `flint` runtime, no Linux/ROCm arch, only the NonaCore file on the
   Hub).  See [BASELINES.md](BASELINES.md) / [PROGRESS.md](PROGRESS.md).
3. **Pilot sample done.** 360-prompt corpus (hard-90/easy-30 excluded), 12
   on-policy rollouts, 3 long reasoning loops at coverage 1.0.  See
   [PROGRESS.md](PROGRESS.md).
4. **Blocked on runpod stock.**  Runbook ready: `moe/runpod-pilot.sh`;
   4-layer local pipeline smoke passed ([PREFIX-SMOKE.md](PREFIX-SMOKE.md)).
5. **Not run** (same pod path).  Scale-only needs the full HF student.
6. **Done, prototype.** 1.51 GB saving, lossless trits/scales, HIP parity
   PASS, KLD 0.0030 cross-type.  See [PTQ1_0.md](PTQ1_0.md).

## Known gaps / risks

- The two result files do not carry seed/T fields (the runner patch applies to
  future runs); the values are recorded in `v2/EVAL.md`.
- v1 correction training did not see exactly the deployed codes (3,661 ppm);
  the v2 pipeline must use `--bank-gguf`.  Re-audit the first v2 export.
- PTQ1_0 is not a free repack: cross-type KLD is 0.0030, so it needs a
  hard-subset A/B before release.
- `reasoning_loop_recovery` is inert for non-streaming requests; the harness
  must stream, or the server needs the check in its non-streaming path.
- Runpod had no 80/96 GB capacity during this session; the full teacher-trace,
  corrections and scale-only runs are still pending.
