# Documentation index — Scion (MoE track)

| document | what it is |
|---|---|
| [../README.md](../README.md) | start here: the idea, the reference release, quickstart |
| [MOE-EXTENSION.md](MOE-EXTENSION.md) | the full MoE write-up: routing drift, the placement rule, AUTOGRID, sidecar formats, deployment routes |
| [SCION-RECIPE.md](SCION-RECIPE.md) | **the training recipe, consolidated and model-agnostic** — objective/flags, stages/gates, failure modes, drafter lane, Qwen-4.0 port checklist |
| [QUANT-RETENTION-35B.md](QUANT-RETENTION-35B.md) | the 35B release measured against every community quant (protocol + table) |
| [RELEASE-35B-MODEL-CARD.md](RELEASE-35B-MODEL-CARD.md) | the published card for Scion-35B-A3B |
| [QWEN35-PORT-DECISION.md](QWEN35-PORT-DECISION.md) | the `qwen3_5_moe` (Qwen3.5/3.8) port decision record |
| [TAIL-EXPERIMENT-PLAN.md](TAIL-EXPERIMENT-PLAN.md) | tail-conditioned KD: **resolved (negative)** — the prefix tail terms did not transfer to the full body |
| [PLATFORM-SCOPE.md](PLATFORM-SCOPE.md) | verified runtime paths and what a backend-specific report must carry |
| [QUANT-METHODS-PLAN.md](QUANT-METHODS-PLAN.md) | CAT-Q / AYOT / SignRoundV2 / TernaryQuench adoption: CPU side landed, GPU queue |
| [../FAILURES.md](../FAILURES.md) | negative register for this track (D1–D8) |
| [launch/](launch/) | public launch copy: Reddit body, community notes, discussion replies (copy-paste blocks) |
| [archive/](archive/) | superseded drafts kept for provenance |

Harness and experiments:

| path | what it is |
|---|---|
| [../moe/README.md](../moe/README.md) | the harness: proxies, trainers, port, rental runbook, results |
| [../serving/](../serving/) | serving experiments: placement sweep + expert-cache negative result |
| [../retention-grid.png](../retention-grid.png) | the release figure (regenerate with `../moe/plot_bench.py`) |

Companion docs that live in the forensics repo (they are sources of the TMLR
paper and stay there): `WHITEPAPER.md`, `FORENSIC-ARCHIVE.md`, `SCALING-PROTOCOL.md`,
`QUANTIZATION-LANDSCAPE.md`, `RETENTION-VS-SCALE.md`, and the TBR failure register —
see <https://github.com/sky-is-green/bonsai2-ternary-forensics/tree/main/docs>.
Links inside the documents here point at their canonical URLs there.
