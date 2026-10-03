# Dense-model ternary QAT (new route, 2026-10-03)

Scion's recipe was built for MoE: ternary expert banks plus residual-stream
corrections that repair routing damage. This note opens the dense variant,
using Cloudflare's **Clef-Flash** (Qwen3.5-9B + joint schema head, Apache-2.0)
as the test bed. No training code here yet; the run happens in a dedicated
session.

## Why dense is a different problem

- There is no router, so there is no routing drift to repair. The damage is
  uniform function noise across every quantized linear, plus the hybrid
  linear-attention layers (Qwen3.5 `in_proj_qkv` / `in_proj_z` / `out_proj`).
- The correction placement rule (§2.4 of the MoE work) must be re-derived:
  candidates are per-block `attn-out` and `mlp-out` branches on the residual
  stream, trained against the deployed quantizer.
- Clef's joint schema head gives an end-to-end metric the MoE work did not
  have: typed decision parity (accept/reject, p_correct) against bf16, not only
  PPL/KLD. The head itself stays bf16/int8; only the backbone is quantized.

## Measured on this box (2026-10-03)

| step | result |
|---|---|
| `llama-quantize PQ2_0` default (absmax) | broken: hidden states cos 0.007 vs f16/HF |
| `llama-quantize PQ2_0` with `GGML_PQ2_0_LLOYD=1` | cos 0.417 vs f16/HF; **3.26 GB** (2.90 BPW) |
| uncorrected ternary decisions | every p shifted down ~0.35-0.49 (e.g. 0.43 vs 0.92 bf16) |
| hidden-state bridge (no quantizer change) | exact on f16: cos 1.0000 vs HF `last_hidden_state` |
| bf16 validator baseline (CPU) | 67/70 correct verdicts train, 26/30 test at 0.5; ~3.2 s/check |

Full notes: `hivebench/experiments/cascade/results/clef-flash-validator-20261003/TERNARY-NOTES.md`.
The comparison matches the MoE record: uncorrected body is not usable, the
corrections are the recipe.

## Plan for the training session

1. **Teacher cache.** Run the bf16 backbone over the calibration corpus and
   cache per-token final hidden states (and, where affordable, per-layer
   states) plus the head's target decisions. The 9B teacher fits the box on
   CPU or split across the two 7900 XT; cache size is the dial.
2. **Student forward with the deployed quantizer.** The ternary backbone is
   the frozen student; the rank branches and the frozen joint head sit on top.
   Train with the real `GGML_PQ2_0_LLOYD=1` weights, exactly as deployed
   (§2.4c: post-hoc ternarisation is 14-25x worse).
3. **Placements.** Start with the two residual-stream taps that carried the
   MoE result: branch on the attention output and on the MLP output of each
   block. If the hybrid linear-attention blocks need their own tap, add
   `linear_attn.out_proj` as a third candidate. Rank 512 is the working
   default; sweep rank only if parity is short.
4. **Losses.** Primary: hidden-state KD to the teacher cache (cosine + MSE per
   token). Secondary: decision KD on the head's `noul` probability for the
   bench records (the metric we ship). Both with the deployed forward in the
   loop.
5. **Evaluation.** (a) per-layer hidden-state cosine vs bf16; (b) decision
   parity on the recorded bench labels: p_correct deltas, accept/reject
   confusion at 0.5 on train and test; (c) MODEL-CARD numbers. Success
   bar: ternary Clef at least as good as Tiny-Jev on the label benchmark and
   close to bf16 Clef (67/70 train, 26/30 test at 0.5).
6. **Packaging.** Merge the branches into the body the way
   `moe/merge_adapter_into_body.py` embeds MoE corrections; for dense the taps
   are plain residual branches, so an offline fold or a small embedded adapter
   both work. Ship: PQ2_0 body + joint head + tokenizer, one directory.
7. **Harness integration.** Add a `clef-ternary` backend to
   `harness/cascade/validator.py` (bridge: `tools/clef-bridge`, head sidecar
   in torch, CPU-only, ~3.3 GB + 0.2 GB). Re-run the validator benchmark and
   compare against Tiny-Jev and bf16 Clef.

## Open questions

- Are the hybrid linear-attention tensors safe under PQ2_0 Lloyd, or should
  they be held at Q8_0 like Scion's "Q8_0 rest"? Measure per-tensor
  sensitivity before the full run.
- Does the joint head tolerate the residual distortion better or worse than
  plain generation? The head reads final states and option spans; the decision
  metric will tell.
- Corpus: bench tasks alone may overfit the validator; mix in generic text
  (wikitext/fineweb slice) for backbone fidelity and keep a held-out decision
  set.
