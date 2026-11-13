# External baselines on the 30-item hard subset

Protocol: heldoutlab `items_r2`, 10 items per family (`v2/eval/subset30.json`),
T=0.6, seed 1001, top_p 0.95 / top_k 20 / min_p 0.05, 32k completion budget,
one server at a time.  The Scion number is extracted from the saved hard-90
T=0.6 run; external models use the same harness and graders.

| model | size | score | R | C | T | notes |
|---|---:|---:|---:|---:|---:|---|
| Scion-35B-A3B PQ2_0-corr (reference) | 11.34 GB | **3/30 (10.0%)** | 0/10 | 0/10 | 3/10 | 7 truncations; median 291 s; 77.1 tok/s |
| Ternary-Nemotron-3.5-Lightning-30B-A3B | 8.82 GB | **0/24 partial** | 0/6 | 0/10 | 0/8 | stopped at 24/30; 22 truncations, 4 server errors |
| Ternary-Flint-40B-NonaCore | 10.58 GB | **blocked** | | | | Metal-only `flint` runtime; no Linux/ROCm arch |

## Ternary Nemotron (stopped at 24/30)

`v2/eval/hard-subset30-nemotron/` (runner `v2/scripts/hard_subset.py`).

- **0/24 correct**, by family 0/6 reasoning, 0/10 coding, 0/8 tools.
- **22/24 truncated**, 4 server errors, 5 tool-call repairs.
- Median 224 s; mean 35,199 completion tokens; max 86,624; mean decode
  144.8 tok/s.
- All four long reasoning items that produced enough text end at 10-gram
  coverage **1.0** (R09, R21, R29, R05): the same loop shape as Scion.
- Stopped by decision at 24/30.  The remaining six cannot change the
  qualitative result (the model is at zero with almost everything truncated).
- This is a community Q2_0 repack, not a controlled quantization study; treat
  it as a same-tier reference, not a verdict on the model family.

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

## Baseline rerun under the recovery protocol (2026-11-13)

`v2/eval/hard-subset30-scion-recover/` - the shipped v1 model, the same
protocol, plus the recovery sentence in the system prompt (which makes the
server's "recover" loop mode live; see `v2/PREFIX-LOOP.md`):

| model / protocol | size | score | R | C | T | truncated | median wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scion v1, original protocol (extracted from the hard-90 run) | 11.34 GB | 3/30 (10.0%) | 0/10 | 0/10 | 3/10 | 7 | 291 s |
| **Scion v1 + recovery sentence (new reference)** | 11.34 GB | **5/30 (16.7%)** | **2/10** | 0/10 | 3/10 | **3** | **171 s** |
| Ternary-Nemotron (original protocol, stopped at 24/30) | 8.82 GB | 0/24 | 0/6 | 0/10 | 0/8 | 22 | 224 s |

The recovery alone moves the shipped model +2 items and cuts truncations
7 -> 3 - direct evidence that the loop is a primary failure mode and that
the v2 gate must be measured against **5/30**, not 3/30 (the freeze's
"+10 points" secondary gate is now >= 8/30).  Coding stays 0/10 either way:
those failures are capability/truncation in the answer, not just loops.

## Teacher reference

Scion-FlashNext-176B-A6B Q6_K (60.2 GB) does not fit the local 2x20 GB box;
the teacher-on-subset number needs the same runpod path as the corrections
pilot (blocked on stock this session).  The plan's "within ~10 points of the
teacher" target is therefore still unmeasured.
