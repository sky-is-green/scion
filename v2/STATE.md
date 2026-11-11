# Project state - 2026-11-11

Status: **planning complete, ready for a fresh start after audit.**

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

## Ready-to-start checklist (from PLAN section 13)

1. Export audit on the current 11.3 GB artifact (gate).
2. External baselines: Ternary-Flint and ternary Nemotron on the 30-item
   subset.
3. Failure rollouts from the current body (tools, coding, long items).
4. Pilot: traces vs rollouts vs steered data; closed-form ATC at rank 64/512;
   rank sweep.
5. Scale-only distillation on the current body.
6. PTQ1_0 single-tensor transcode and q36 kernel parity check.

## Known gaps / risks

- The two result files do not carry seed/T fields (the runner patch applies to
  future runs); the values are recorded in `v2/EVAL.md`.
- The q36 HIP expert kernels are Q2_0-specific; PTQ1_0 remains a kernel
  prototype, not a promised size win.
- Export code fidelity has not been audited yet.
- `reasoning_loop_recovery` is inert for non-streaming requests; the harness
  must stream, or the server needs the check in its non-streaming path.
