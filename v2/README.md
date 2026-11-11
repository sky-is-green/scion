# Scion 35B v2: FlashNext distill

Status: **PLANNING**. Nothing in this folder has run. The shipped
Scion-35B-A3B is unchanged.

The v2 goal is a better, more open Scion 35B that stays ternary and stays in
the ~11 GB consumer tier: the body is quantized once and frozen, and
corrections are retrained against the Scion-FlashNext 176B teacher with
long-horizon data. No full-body QAT.

- [PLAN.md](PLAN.md) - scope, phases, budget, risks
- [LESSONS.md](LESSONS.md) - what the negative register and the hard-90 eval
  settle before we start
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
| `eval/` | result JSONs and reports |
| `export/` | GGUF / Q2_0 builds |

Next step: build the 300-prompt pilot manifest (hard 90 and easy 30 excluded),
run the local teacher trace pass, then the long-window corrections pilot.
