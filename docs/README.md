# Documentation index — Scion (MoE track)

| document | what it is |
|---|---|
| [../README.md](../README.md) | start here: the idea, the reference release, quickstart |
| [MOE-EXTENSION.md](MOE-EXTENSION.md) | the full MoE write-up: routing drift, the placement rule, AUTOGRID, sidecar formats, deployment routes |
| [QUANT-RETENTION-35B.md](QUANT-RETENTION-35B.md) | the 35B release measured against every community quant (protocol + table) |
| [RELEASE-35B-MODEL-CARD.md](RELEASE-35B-MODEL-CARD.md) | the published card for Scion-35B-A3B |
| [QWEN35-PORT-DECISION.md](QWEN35-PORT-DECISION.md) | the `qwen3_5_moe` (Qwen3.5/3.8) port decision record |
| [TAIL-EXPERIMENT-PLAN.md](TAIL-EXPERIMENT-PLAN.md) | **current path:** KLD beyond the teacher's top-50 (queued; needs a GPU) |
| [FAILURES.md](FAILURES.md) | negative register for this track (D1–D8) |
| [launch/](launch/) | public launch copy: Reddit body, community notes, discussion replies (copy-paste blocks) |
| [archive/](archive/) | superseded drafts kept for provenance |

Harness and experiments:

| path | what it is |
|---|---|
| [../moe/README.md](../moe/README.md) | the harness: proxies, trainers, port, rental runbook, results |
| [../serving/](../serving/) | serving experiments: placement sweep + expert-cache handoff (negative result) |
| [../retention-grid.png](../retention-grid.png) | the release figure (regenerate with `../moe/plot_bench.py`) |

Companion docs that live in the forensics repo (they are sources of the TMLR
paper and stay there): `WHITEPAPER.md`, `FORENSIC-ARCHIVE.md`, `SCALING-PROTOCOL.md`,
`QUANTIZATION-LANDSCAPE.md`, `RETENTION-VS-SCALE.md`, and the TBR failure register —
see <https://github.com/sky-is-green/bonsai2-ternary-forensics/tree/main/docs>.
Links inside the documents here point at their canonical URLs there.
