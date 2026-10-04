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

## Current release candidate (2026-10-04)

**Not ready to post.**  The Goal-B quant thread (`../dense/HANDOFF.md`
CURRENT THREAD) is at PPL 476 (signed RTN) with GPTQ+Lloyd in flight; the
earlier sidecar corrections are decision-degenerate (constant-accept) and must
not be posted.  Upload the set below once the quant passes acceptance: PPL
approaching f16 (12.59) and the frozen decision probe re-run for the record.

## Suggested upload set (single directory)

- `clef-flash-PQ2_0-corr-r512-g128-step78.gguf` (3.10 GiB, single-file
  release; corrections embedded)
- `clef-flash-corr-r512-g128-step78.lora.gguf` (71 MB, standalone adapter)
- `hf-head/` + `lm_head.safetensors` only if we intend to make the repo
  runnable end-to-end; otherwise link to the Cloudflare release (keeps us from
  re-hosting files we didn't change)
- `README.md` (model card: attribution, changes, eval table, caveats)
- `LICENSE` (Apache-2.0)

## Code credits state (for the card)

The conversion/eval tooling lives in `scion/dense/` (Apache-2.0) and
`hivebench/tools/clef-bridge/` (MIT); the runtime is the local llama.cpp fork
(MIT), itself carrying TAARDIS virtual-target support (MIT, Cody Dixon).
