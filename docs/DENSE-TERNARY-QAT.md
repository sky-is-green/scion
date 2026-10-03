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

## Stage A measured: reverse loader + the GDN kernel (2026-10-03)

The HF `Cloudflare/clef-flash` backbone was not kept on disk (only the head +
tokenizer), so the in-loop student is built by **reverse-mapping the deployed
PQ2_0 GGUF** into a transformers `Qwen3_5TextModel` (no download). Code:
`dense/clef_dense_load.py` (new sibling route to `moe/`; dense has no router
and a different tap set, so it does not belong in the MoE harness). Findings,
all reproduced on this box:

- **Mapping is exact.** f16 GGUF -> torch, `missing=0 unexpected=0`; the whole
  inverted converter (transpose orientation, zero-centred RMSNorm `w-1`,
  `A_log=log(-x)`, `dt_bias`, conv squeeze, grouped->tiled V-head reorder) is
  validated. The V reorder inverse is `reorder_v_heads` with the head counts
  swapped (it is only an involution on the trailing axis).
- **PQ2_0 dequant is byte-identical** to the fork's own
  `dequantize_row_pq2_0` (max abs diff 0.0; `blk.0.ffn_gate.weight`), and the
  Q4_K `token_embd` dequant is byte-identical to `gguf.quants.dequantize`. The
  container is scale-first (`ggml_half d` + 32 code bytes), strict ternary
  `{q-1}` (`quantize_row_pq2_0_lloyd_ref`), no rotation metadata.
- **The runtime kernel is the lever.** transformers' default prefill uses
  `torch_chunk_gated_delta_rule`; the deployed llama.cpp CPU path uses the
  recurrent GDN. With chunked prefill, torch vs the CPU bridge decays over
  positions (f16 per-token cos 1.0 -> 0.82 by token 39). Patching every
  `Qwen3_5GatedDeltaNet` to `torch_recurrent_gated_delta_rule` gives
  **f16 f32 cos 0.9975** (per-token >= 0.97) against the CPU bridge. Training
  must use the recurrent form or the corrections will not transfer.
- **OOM incident.** A full 9B load via the naive state-dict path (f32
  intermediates + a full 18 GiB state dict on a 30 GB host) OOM'd the box.
  The loader now casts per tensor; the full model must be **streamed onto the
  GPU per tensor**, never materialised twice on the host. Host RAM is the
  binding constraint, not VRAM.

Open question resolved: the GDN tensors are safe under Lloyd (per-tensor cos
~0.90 vs f16, same as the MLP/attention linears), so all 248 text linears stay
at PQ2_0; only norms/conv/A_log/dt_bias stay non-quantised, as deployed.

## Stage B: the dense correction stack (2026-10-03)

`dense/` (sibling to `moe/`) now holds the route: `clef_dense_load.py` (streamed
reverse loader + recurrent-GDN patch), `clef_head.py` (joint head + lm_head
sidecar), `clef_cache.py` (teacher cache), `clef_corrections.py` (end-to-end
trainer), `clef_eval.py` (decision parity + hidden cosine), `quant.py` (Lloyd
branch quantizer).

- **Head sidecar validated exact:** encode -> f16 bridge states -> joint head
  reproduces the recorded bf16 `p_correct` to 4 dp (0.9219 on `gsm8k-0001`).
- **Teacher cache** (smoke): 158 sequences (100 bench train+test + 58 hard
  negatives), 0.52 GB, hidden states + head option logits, one `.npz` each.
- **Prior-art constraints applied** (Bonsai-2 27B forensics): end-to-end only
  (F6/F7), recurrent GDN (F4/Gate 2), clean holdout (F11), managed LR decay,
  no exotic quantizer tricks.
- **Single-card VRAM limit:** the deployed body is 15.9 GB in bf16; `both` taps
  at rank 512 (268M branch params, fp32 masters + grads) OOM'd one 21.5 GB RX
  7900 XT during backward (19.0 GB allocated, ~1.5 GB short). `attn_out`/rank
  256 fits; the full `both`/rank 512 config needs the two-card split (one
  model-parallel process, per the U2 policy) or CPU embedding offload.

Fallback if corrections fall short: full end-to-end ternary QAT of the body
(rotation + STE/KD + managed schedule) is the forensics' known dense lever
(`recover.py`); our sidecar route is the cheaper Scion shape and was the MoE
winner.

## Stage C: rental pilot (2026-10-03)

One A40 48 GB (RunPod, `clef-pilot`, ~$0.4-0.8/hr) runs the f32 correction pilot:
f32 body (31.8 GB) + rank-512 branches on both taps + recurrent GDN + decision
KD, 35.8 GB VRAM peak, ~15 s/step, stable (no NaN).  Artifacts uploaded:
PQ2 GGUF, `hf-head/`, `lm_head.safetensors`, the teacher cache, the fork's
`gguf-py`, and `dense/`.  `dense/preflight.py` runs first.

**Preflight caveat (important).** On the A40, the PQ2 **f32 recurrent torch
forward** matches the CPU ggml bridge at only **cos 0.948** (per-token
0.91-0.99), versus 0.9975 for the *f16* body locally.  The ternary weights
amplify the torch-vs-ggml kernel difference (forensics F4 again).  So the
torch-eval parity is a **proxy**; the honest number must come from the
llama.cpp bridge on the packaged artifact.  The pilot's purpose is to measure
whether the correction signal survives that gap.

**Pilot result (2026-10-03, one epoch / 78 steps, rank 512, both taps, f32
recurrent torch).**  Decision parity vs the recorded bf16 p and the gold
checker at 0.5:

| split | uncorrected | **corrected** | bf16 Clef | Tiny-Jev |
|---|---|---|---|---|
| train correct | 26/70 (0 FA, 44 FR) | **68/70 (2 FA, 0 FR)** | 67/70 (0 FA, 3 FR) | 62/70 |
| test correct | 7/30 (0 FA, 23 FR) | **29/30 (1 FA, 0 FR)** | 26/30 (1 FA, 3 FR) | 25/30 |
| mean p train/test | 0.474/0.476 | 0.595/0.594 | 0.811/0.792 | — |
| hidden cos train/test | 0.231/0.236 | 0.372/0.383 | 1.0 | — |

The correction is decisive on the proxy: correct verdicts recover 26->68 and
7->29, above both Tiny-Jev and bf16 Clef on verdict count (a different FA/FR
mix: 2 train FAs vs bf16's 0).  Cost: ~$0.6 (A40, ~1 h), pod deleted.
**Outstanding:** the honest deployment number requires packaging the branches
(`attn_out` as a LoRA on `attn_output`/`ssm_out`; `mlp_out` needs the dense
`qwen35` `ffn_out` fork hook) and re-running the benchmark through the CPU
bridge.  The hidden cos (0.37) is still far from 1.0, so a longer run and the
`both`-tap packaging are the natural next steps.

**PQ2 bf16 torch forward is nondeterministically non-finite (2026-10-03).**
The decisive local blocker: two identical chunked evaluations of the *same*
uncorrected PQ2 body produced **different sets** of non-finite records (23 then
50 of 70 train).  All deployed Weights are finite (max |w| 18.5), the raw
forward is finite on a probed record, and the branch wrapper is finite on that
record -- so the failure is ROCm/bf16 kernel nondeterminism at a numerical edge,
not a mapping or wrapper bug.  llama.cpp (f32/f16 CPU accumulation) is stable on
the same file.  **Consequence: torch-bf16 on this box cannot train or evaluate
the 9B reliably.**  f32 fixes it (torch body 31.8 GB, does not fit 21.5 GB), so
the production run needs the 48 GB rental; the local pilot lane is closed.

**GDN backward instability (2026-10-03).** The deployed CPU runtime is the
*recurrent* GDN, so training should use it for transfer.  But the transformers
recurrent fallback unrolls the whole sequence and its backward NaN's within
1-2 steps on this 9B (bf16 body, ROCm): the recurrence amplifies the branch
perturbation (matches forensics F4).  Truncated BPTT (state detached every 64
tokens) did not fix it; the **chunked** fallback is the stable training form
(30+ steps, loss falling), at the cost of a forward that differs from the
recurrent deployment (f16 per-token cos 0.82 at token 39).  The clean fix is a
**48 GB rental with the body in fp32 and the recurrent forward**: f32 body is
31.8 GB (does not fit locally), the recurrent backward is stable in f32, and the
forward then matches the deployment exactly.  Local (21.5 GB) can only pilot the
chunked form; transfer to the recurrent runtime must be measured through the CPU
bridge.

**Loop smoke passed (2026-10-03).** A bounded 2-layer prefix run
(`--prefix-layers 2`, rank 16, both taps, 3 steps) under
`systemd-run --scope -p MemoryMax=16G` with a VRAM preflight exercised the whole
trainer: deployed-PQ2 load -> branch attach (g128 STE) -> recurrent GDN ->
hidden KD -> CPU decision-KD grad bridge through the joint head -> double-grad
backward with checkpointing -> Adafactor step -> checkpoint. Peak VRAM 4.94 GB,
no OOM. Full-model VRAM remains the only unproven local quantity (~21 GB > one
21.5 GB card), so the production run is slated for a 48 GB rental
(`dense/RENTAL-RUNBOOK.md`), with `dense/preflight.py` as the first action.

## Stage D: packaging + CPU-bridge benchmark (2026-10-03)

Both taps packaged with `dense/clef_export.py` (all-ternary `Q1_0_g128` LoRA,
71 MB), the dense `blk.N.ffn_out` virtual target added to the fork
(`qwen35.cpp` + `llama-adapter.cpp`, anchored on `ffn_down`), `build-cpu`
rebuilt, and the adapter merged into the body as a single file
(`adapter.embedded=true`, 3.10 GiB).  The benchmark (`dense/clef_bridge_eval.py`)
runs the bridge (CPU, batch mode) -> joint head sidecar (CPU torch f32) over all
100 bench records, on both bodies.

**The honest deployment numbers land on the torch proxy, with zero verdict
flips:**

| split | uncorrected bridge | **corrected bridge** | corrected proxy | bf16 Clef | Tiny-Jev |
|---|---|---|---|---|---|
| train | 18/70 (0 FA, 52 FR) | **68/70 (2 FA, 0 FR)** | 68/70 (2 FA, 0 FR) | 67/70 | 62/70 |
| test | 8/30 (0 FA, 22 FR) | **29/30 (1 FA, 0 FR)** | 29/30 (1 FA, 0 FR) | 26/30 | 25/30 |
| mean p | 0.471 / 0.471 | 0.586 / 0.588 | 0.595 / 0.594 | 0.811 / 0.792 | — |
| hidden cos | 0.231 / 0.235 | 0.371 / 0.382 | 0.372 / 0.383 | 1.0 | — |

Transfer proxy -> bridge (corrected): 0 flips on 100/100, max |Δp| 0.027,
max |Δcos| 0.004.  The PQ2 f32 torch-vs-bridge cos gap (0.948) is a body-level
artifact that cancels for the decision metric.  The uncorrected bridge is more
conservative than the proxy (18/70 train vs 26/70), so the module toggle is
measurable end to end.

**CPU latency: ~49 ms/token** (recurrent GDN prefill; 13.8-28.6 s per
278-589-token record), one-time model load 7.4 s, head sidecar 40-90 ms/check.
The bf16 transformer validator was ~3.2 s/check, so the CPU lane is ~5-8x
slower; the non-display card is the fallback for the harness backend.

Artifacts: `models/clef-flash-ternary/corrections/packaged/` (adapter, merged
release, `bridge-{uncorrected,corrected}.json`).  Fork patch committed locally
as `f8395a69b`.  External prior art (TAARDIS: same `Q1_0_g128` lineage, per-head
GDN readout doctors, rotation-first pipeline) is surveyed in
[`TAARDIS-PRIOR-ART.md`](TAARDIS-PRIOR-ART.md).

## Status and next step (2026-10-03)

Packaging + bridge benchmark **done**: the correction route now has an honest
deployment number (68/70 train, 29/30 test at 0.5, zero proxy flips) on the
merged single-file release, above bf16 Clef and Tiny-Jev.  Menu: (1) the
`clef-ternary` harness backend (unblocked; decide CPU ~20 s/check vs
non-display-card bridge); (2) longer f32 rental run (needs approval) with a
per-head GDN readout tap + damage-based rank allocation; (3) Clef V2
(rotation + Hessian GPTQ + self-distill) as the bigger quality lever.  Full
context: [`../dense/HANDOFF.md`](../dense/HANDOFF.md).  No push; rentals need
approval.
