# Scion 35B v2: FlashNext distill

Status: **PLANNING**. Nothing in this folder has run. The shipped
Scion-35B-A3B is unchanged.

The v2 goal is a better, more open Scion 35B that stays ternary and stays in
the ~11 GB consumer tier: the body is quantized once and frozen, and
corrections are retrained against the Scion-FlashNext 176B teacher with
long-horizon data. No full-body QAT.

- [PLAN.md](PLAN.md) - scope, phases, budget, risks, open-questions register
- [LESSONS.md](LESSONS.md) - what the negative register and the hard-90 eval
  settle before we start
- [EVAL.md](EVAL.md) - evaluation state, hashes, reproduction steps
- [STATE.md](STATE.md) - audit-ready inventory of the whole project state
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
| `scripts/hard90_scion.py` | the eval runner (records seed/temperature/arm) |
| `configs/scion35b-strata.json` | the server config used for the evals |
| `eval/` | the two hard-90 result files and summaries |

Next step: build the 300-prompt pilot manifest (hard 90 and easy 30 excluded),
run the local teacher trace pass, then the long-window corrections pilot.
