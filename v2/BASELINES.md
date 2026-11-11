# External baselines on the 30-item hard subset

Protocol: heldoutlab `items_r2`, 10 items per family (`v2/eval/subset30.json`),
T=0.6, seed 1001, top_p 0.95 / top_k 20 / min_p 0.05, 32k completion budget,
one server at a time.  The Scion number is extracted from the saved hard-90
T=0.6 run; external models use the same harness and graders.

| model | size | score | R | C | T | notes |
|---|---:|---:|---:|---:|---:|---|
| Scion-35B-A3B PQ2_0-corr (reference) | 11.34 GB | **3/30 (10.0%)** | 0/10 | 0/10 | 3/10 | 7 truncations; median 291 s; 77.1 tok/s |
| Ternary-Nemotron-3.5-Lightning-30B-A3B | 8.82 GB | running | | | | Q2_0, stock llama.cpp ROCm, 65k context |
| Ternary-Flint-40B-NonaCore | 10.58 GB | **blocked** | | | | Metal-only `flint` runtime; no Linux/ROCm arch |

## Ternary Nemotron

`v2/eval/hard-subset30-nemotron/` (runner `v2/scripts/hard_subset.py`).
The model is a reasoning model and spends most of the 32k budget thinking;
the early reasoning/coding items truncate.  Final numbers are appended here
when the 30 items finish.

## Ternary Flint - why it is blocked

- The Hub repo `kabcpc/Ternary-Flint-40B-GGUF` currently contains exactly one
  GGUF: `Ternary-Flint-40B-NonaCore-Q4_0-9.8GB.gguf` (10,580,056,000 bytes).
  The Lean/PQ2_0 variants named in the prior-art analysis are not in the repo.
- The file's header says `general.architecture = flint`; the card requires a
  custom Metal runtime (`flint` arch, `prism.hadamard.*`, Metal PTQ1_0 LUT and
  multi-column GEMV) and ships only macOS-arm64 binaries.
- The local Linux/ROCm forks (`llama-stock`, `prism-moe-corr`,
  `prism-ptq1-rdna3`, `prism-qwen4exp-shrink`, `strata`) have no `flint`
  architecture.  Porting the arch + Hadamard initialization is a separate
  kernel project, not a baseline run.
- Result: no same-protocol Flint number on this box; do not substitute the
  author's Apple-Silicon PPL/speed claims.

## Teacher reference

Scion-FlashNext-176B-A6B Q6_K (60.2 GB) does not fit the local 2x20 GB box;
the teacher-on-subset number needs the same runpod path as the corrections
pilot (blocked on stock this session).  The plan's "within ~10 points of the
teacher" target is therefore still unmeasured.
