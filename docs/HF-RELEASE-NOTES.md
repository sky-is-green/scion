# HF release notes — clef-flash ternary corrections (pre-post checklist)

Checked 2026-10-03, before the local generalization track.  Everything in the
chain is permissive; posting is a documentation exercise, not a legal one.

## License matrix

| component | license | evidence |
|---|---|---|
| `Cloudflare/clef-flash` (backbone + joint head + tokenizer) | **Apache-2.0** | HF API `cardData.license=apache-2.0`; repo ships `LICENSE` (11.5 KB); no `NOTICE` file; not gated |
| `Qwen/Qwen3.5-9B` (Clef's base) | **Apache-2.0** | HF API + `license_link` to the repo `LICENSE` |
| our PQ2_0 body + merged release | Apache-2.0 | GGUF metadata `general.license=apache-2.0` (inherited at conversion); derivative of Clef |
| trained corrections (adapter / merged tensors) | Apache-2.0 (ours) | derivative weights; scion repo is Apache-2.0 |
| `scion` code | Apache-2.0 | `scion/LICENSE` |
| `hivebench` code | MIT | `hivebench/LICENSE` |
| `llama.cpp` fork | MIT (+ TAARDIS additions MIT) | upstream license; we redistribute no fork binaries |
| TAARDIS artifacts (27B etc.) | Apache-2.0 model / MIT autogrid | not redistributed; cited as prior art |

## Obligations if we post the model

Apache-2.0 (both Clef and Qwen lineage) requires:

1. **Ship the license** — include the Apache-2.0 text (`LICENSE`) in the repo.
2. **Attribution** — credit Cloudflare (Clef-Flash) and the Qwen team
   (Qwen3.5-9B), link both cards.
3. **State changes** (§4b) — this is a modified derivative.  Write down:
   - quantized the text backbone to `PQ2_0` (`GGML_PQ2_0_LLOYD=1`, 2.90 BPW,
     3.26 GB); vision tower and MTP head not converted (text-only body);
   - trained rank-512 residual corrections (attn output, MLP output; all
     merged into the body, 3.10 GiB, `adapter.embedded=true`);
   - the joint schema head + tokenizer distributed **unmodified** (or linked
     to the original);
   - evaluation conditions (cascade-bench-v1, threshold 0.5, CPU bridge) with
     the honest numbers, including the probability-compression caveat.
4. **No trademark implication** — a clear disclaimer like TAARDIS's: this
   release is not endorsed by or affiliated with Cloudflare / the Qwen team.
   Don't brand the repo as an official "Clef" release.
5. No NOTICE file exists in the Clef repo, so none needs to be propagated
   (re-check at post time).

## Current release candidate (2026-10-06)

**Not ready to post.**  The ternary quant thread reached **PPL 23.25** (5.52
GiB all-ternary QAT of the Clef-Flash text backbone; f16 12.59) but its free
generation degrades (code/math; the f16 control is clean) — a release would
fail on first use.  The sidecar corrections remain decision-degenerate
(constant accept) and must not be posted.  Community context: Clef/Clef-Flash
already have many 4-8 bit quants (bartowski/ggml-org GGUFs, MLX, FP8/NVFP4,
EXL3, OpenVINO, W4A16 AutoRound/GPTQ for both sizes) but **no ternary Clef**;
the ternary bar is PrismML's trained Ternary-Bonsai-8B (~1.44x PPL at
2.03 GiB, near-base benchmarks).  **Community ternary gate measured
(2026-10-06):** plain TQ1_0/TQ2_0/PTQ1_0 are naive absmax codecs and collapse
to 1.9-2.1M PPL (token soup; imatrix no-op), while a plain **Q2_K beats our
artifact outright — 13.06 PPL, clean generation, 3.56 GiB, mainline** vs
23.25 / math loop / 5.52 GiB / fork-only.  Gate outcome: **no release**; the
QAT pipeline still wins the ternary comparison, but the release bar (trained
Bonsai ~1.44x, or the Q2_K-class baseline) is not met.  Next (needs approval)
is one proper QAT run (mixed corpus + logits KD, 5-10M tokens, ~$8-15)
accepted on clean generation, PPL <= ~18 and KL/top-1 vs bf16 before any
upload.  Record: `hivebench/experiments/cascade/results/
clef-flash-validator-20261003/community-gate-20261006.json`.

## Suggested upload set (once a quant passes the gate)

- the ternary Clef-Flash GGUF (currently `v2/clef-flash-v2-qat-a1.gguf`, when
  its generation passes) + a runtime note (rotation metadata needs the fork);
- `README.md` (model card: attribution, changes, eval table incl. KL/top-1,
  generation samples, caveats, non-affiliation disclaimer);
- `LICENSE` (Apache-2.0).
- The joint head/tokenizer are Cloudflare's — link rather than re-host unless
  a runnable end-to-end repo is intended.  Do **not** include the dead sidecar
  corrections.

## Code credits state (for the card)

The conversion/eval tooling lives in `scion/dense/` (Apache-2.0) and
`hivebench/tools/clef-bridge/` (MIT); the runtime is the local llama.cpp fork
(MIT), itself carrying TAARDIS virtual-target support (MIT, Cody Dixon).
