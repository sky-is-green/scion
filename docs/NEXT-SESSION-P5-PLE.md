# Next session: P5 PLE-table lever (continuation prompt)

**Superseded 2026-10-12 by `docs/NEXT-SESSION-Q4X160-GATE.md`** (kept for the
record; the PLE lane advanced past this prompt: Q3X32 is built and gated,
Q4X160 is built and gate-ready).

**Context.** Read `~/Desktop/SCION-LOCAL-ARCHITECTURE.md`, then
`docs/SHRINK-PLAN.md` (status at top), then `docs/CODER-VARIANT-GATES-25.md`
(why expert pruning is closed). Expert pruning is measured and **closed for
the general release**: 25% was code-neutral but cost general quality
(-6.64 GB for PPL +11.6%, HS -2.75, WG -0.50, KLD +0.19); 50% uniform broke
the thinking mode. Do not re-open pruning for the general release; it stays a
coder-specialist option only.

## Goal

Cut the **unpruned** Scion-FlashNext-176B-A6B (60.17 GB, all 512 experts)
below 60 GB with quality intact. The only large lever left is the PLE n-gram
table (28.80 GB, 48% of the file); the companions are lossless or standard.

## Size accounting (verified from the release header, decimal GB)

| part | release | note |
| :--- | ---: | :--- |
| experts PTQ1_0 | 26.42 | 512/layer; leave alone |
| **PLE table Q4_0** | **28.80** | 320,001,536 rows x 160 elems, 90 B/row |
| body Q6_K/Q8_0 | 3.21 | |
| corrections F16 | 0.68 | |
| other | 1.05 | |
| total | 60.16 | |

Table geometry: row = 160 elements = **5 blocks of 32** (80 B nibbles + 5 x
fp16 scales = 90 B). K-quants (256-element blocks) do not fit this row.
Proposed 32-block containers:

- **Q3x32**: 3-bit codes + fp16 scale per 32 -> 14 B/block, 70 B/row ->
  22.40 GB (**-6.40**)
- **Q2x32**: 2-bit codes + fp16 scale per 32 -> 10 B/block, 50 B/row ->
  16.00 GB (**-12.80**)

Companion levers (bundle after the table gates):

- PTQ1_0 28 -> 25.4 B/128 (entropy bound): -2.4 GB, **lossless** (new type +
  readers only)
- body Q6_K/Q8_0 -> Q4_K/Q5_K: -1.3 GB (fork already supports; gate it)
- corrections F16 -> per-layer mixed (`branch_sensitivity.py`): -0.3 GB

Combined unpruned target: **~49.7 GB (Q3 table) / ~43.3 GB (Q2 table)**.

## Work items (local first, zero pod spend)

1. **Define the table container(s)** in the fork: new ggml type id(s) (check
   `ggml/include/ggml.h` + the type maps), 32-element blocks, 2/3-bit
   LSB-first codes, one fp16 scale per block; document the quantizer rule
   (affine min/max per 32 is the simple choice; anything fancier needs the
   gate).
2. **Fork reader**: the qwen4exp PLE path (`src/models/qwen4exp.cpp`) gathers
   `per_layer_token_embd` rows; the new type must be readable by the
   CPU/CUDA paths there (or route the lookup through a dedicated kernel).
   Synthetic-GGUF round trip first.
3. **Parity test** next to `tools/kld` or the kernels: bit-exact decode and
   row dogfood vs a reference decoder on real 4l table rows (pattern to
   follow: Strata's `ple_q4_parity.cpp`).
4. **Strata side** (serving): copy the P2 precedent in
   `src/kernels/ngram.cpp` (q4_0 flag + decode branch + type acceptance) and
   add a `ple_<T>_parity` test; the table feeds every layer, parity must be
   bit-exact against the fork's decoder.
5. **Exporter**: extend
   `hivebench/tools/strata-bridge/repack_experts_q2_0.py` (or a sibling) to
   transcode the table Q4_0 -> new type, streamed, never materialising the
   28.8 GB table.
6. **Pod gates**, in order: `llama-kld` vs the parked teacher
   (`scion/moe/artifacts/qwen4exp/kld-park`; release baseline fork-instrument
   mean **0.7819**, top1 **0.6566**), paired code arms (release baselines:
   gen 68.3 all / 61.2 test, hard 50.0 / 49.0), then PPL 580 (**5.4571**),
   HS 400 (**82.00**), WG 400 (**77.50**). Q2 likely needs the full gate to
   decide; Q3 first.

## Staged infrastructure

- Fork patches in `hivebench/artifacts/ternary/moe/qwen4exp/pod/remote-w1w2/recipe/`:
  `router-stats.patch`, `kld-gate.patch`; the full 1-5 sequence and the three
  release patches apply on a pristine `qwen4exp-proto` (verified).
- Map/plans: `scion/moe/artifacts/qwen4exp/` (`router-stats-coder.json`,
  `prune-plan-25.json`, `prune-plan-50.json`, `kld-park/`).
- Pod bookers (guards, dead-man, systemd units built in):
  `pod-moe-stats-book-retry.sh` (map), `pod-prune-gate-book.sh`,
  `pod-paired-code-book.sh`, `pod-tier-code-book.sh`; A6000 $0.59/h
  preferred, L40S $1.09 fallback.
- Corpus builders: `scion/moe/{build_coder_corpus.py, build_router_corpus.py,
  build_ayot_prompts.py}`.
- The 53.54 GB pruned coder file and the 60 GB release are **not local**;
  a pod regenerates/ downloads them (exporter + plan, ~5 min on-pod).

## Constraints

- **Never push git unless asked.** Refs unchanged: strata fork
  `scion-support @ 4f7fb6b`, scion `main @ 16aa7e7`, hivebench
  `hive_cascade @ d5f76aa`, HF README `a50c8591`.
- Other agent lanes share the box: long jobs via
  `systemd-run --user --unit=...`; never kill their processes/units.
- Pod balance **$5.48** (2026-10-11); **ask before booking**; the local
  parity work above costs nothing.
- P4 (serve the untouched release in Strata and compare against the fork
  numbers) is still pending - do not start it unless asked.

## First commands

```sh
cd ~/Desktop/work
ls scion/moe/artifacts/qwen4exp/            # map, plans, teacher park
git -C prism-qwen4exp-shrink status         # fork worktree, patches applied
sed -n '1,80p' scion/docs/SHRINK-PLAN.md    # current status
```
