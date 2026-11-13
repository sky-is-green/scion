# Scion 35B v2: FlashNext distill

Status: **local phase done (PLAN 13a A-D), run plan drafted, pilot not
authorised.**  The shipped Scion-35B-A3B is unchanged.

The v2 goal is a better, more open Scion 35B that stays ternary and stays in
the ~11 GB consumer tier: the body is quantized once and frozen, and
corrections are retrained against the Scion-FlashNext 176B teacher with
long-horizon data. No full-body QAT.

- [PLAN.md](PLAN.md) - scope, phases, budget, risks, open-questions register,
  and 13a: the execution status, FlashNext lessons and the local-first order
- [LESSONS.md](LESSONS.md) - what the negative register and the hard-90 eval
  settle before we start
- [EVAL.md](EVAL.md) - evaluation state, hashes, reproduction steps
- [STATE.md](STATE.md) - audit-ready inventory of the whole project state
- [export-audit/README.md](export-audit/README.md) - the v1 export-code audit
  (gate) and the seed-from-GGUF fix
- [MEMORY-PORT.md](MEMORY-PORT.md) - the FlashNext memory switches ported to
  qwen35; the bit-exactness gate and the 4-layer prefix measurements
- [REAL-CACHE.md](REAL-CACHE.md) - the real `_logits_` cache path validated
  end to end (half-chunk layout, tokenizer alignment, tail tokens)
- [EXPERIMENT-1.md](EXPERIMENT-1.md) - the frozen first experiment: arms,
  cache spec, falsification criterion, eval protocol
- [PREFIX-LOOP.md](PREFIX-LOOP.md) - the full local loop (train -> export ->
  merge -> audit -> serve -> harness) and the recovery findings
- [RUN-PLAN.md](RUN-PLAN.md) - the step-E draft: pod sizing from local
  measurements, commands, resume points, cost cap, stop rules
- [BASELINES.md](BASELINES.md), [PTQ1_0.md](PTQ1_0.md), [PREFIX-SMOKE.md](PREFIX-SMOKE.md)
- [../FAILURES.md](../FAILURES.md) - the full negative register (D1-D8)
- [../docs/MOE-EXTENSION.md](../docs/MOE-EXTENSION.md) - the method write-up
- [../docs/ROUTER-FREEZE.md](../docs/ROUTER-FREEZE.md),
  [../docs/WEIGHT-PATCHING.md](../docs/WEIGHT-PATCHING.md) - the latest research

Planned working directories (gitignored, created when used):

| path | contents |
|---|---|
| `corpus/` | prompt manifests, eval sets excluded |
| `traces/` | teacher traces + top-k logits cache |
| `corrections/` | trained correction checkpoints |
| `runs/` | training runs |
| `export/` | GGUF / Q2_0 builds |

Tracked here:

| path | contents |
|---|---|
| `scripts/hard90_scion.py` | the full-90 eval runner |
| `scripts/hard_subset.py` | the 30-item subset runner (recovery sentence on by default) |
| `scripts/prefix_gate.py` | the real-weights bit-exactness gate for the compact banks |
| `scripts/prefix_mem_probe.py` | the 4-layer prefix memory probe (one condition per process) |
| `scripts/audit_merged_adapter.py` | the merged-release audit (byte identity + 0-ppm factors) |
| `configs/scion35b-strata.json` | the v1-model server config |
| `eval/` | the hard-90 result files, the subset30 baseline and its runs |
