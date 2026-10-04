# `dense/` — dense-model ternary corrections (Clef-Flash test bed)

This is the **dense route**, a sibling to `moe/` rather than part of it.  The
MoE recipe repairs *routing* damage; a dense (or dense/hybrid) model has no
router, so the failure is uniform function noise across every quantised
linear, the placement rule is re-derived, and the shipped metric is
**typed-decision parity** (accept/reject, `p_correct`) instead of PPL/KLD.

The target is Cloudflare's **Clef-Flash** (Qwen3.5-9B + a joint schema head,
Apache-2.0): a ternary `PQ2_0` body (`GGML_PQ2_0_LLOYD=1`) plus rank-512
residual branches, evaluated through the joint head.  Plan:
[`../docs/DENSE-TERNARY-QAT.md`](../docs/DENSE-TERNARY-QAT.md).

## Contents

| file | what |
|---|---|
| `clef_dense_load.py` | reverse-map a Clef GGUF (f16 or deployed `PQ2_0`) into a transformers `Qwen3_5TextModel`; streamed, memory-gated |
| `clef_cache.py` | teacher hidden-state / decision cache builder |
| `clef_corrections.py` | dense correction trainer: residual taps on `attn_out`/`mlp_out`, deployed quantizer in the loop |
| `clef_eval.py` | torch-proxy decision parity + hidden cosine |
| `clef_export.py` | export trained branches as an all-ternary `Q1_0_g128` llama.cpp LoRA (`ssm_out`, `attn_output`, `ffn_out` targets) |
| `clef_bridge_eval.py` | honest CPU-bridge benchmark: bridge -> CPU-f32 head sidecar -> parity + latency |
| `clef_v2_convert.py` | Goal-B quantizer: rotated-basis PQ2_0 conversion from the f16 GGUF (signs, mixed precision, Hessian GPTQ, `prism.hadamard.*` metadata; `--self-test`) |
| `clef_v2_hessians.py` | per-linear `XᵀX/N` capture for GPTQ (bf16 recurrent-GDN forward, layer-group passes) |
| `clef_v2_ppl.sh` / `clef_v2_sweep_gpu.sh` / `clef_v2_sweep_cpu.sh` | wikitext PPL harness, mixed-precision sweep, CPU fallback |

## Established on this box (2026-10-03)

- **No HF backbone is kept on disk** — only the head/tokenizer — so the
  in-loop student is built by reverse-mapping the deployed GGUF.  The inverted
  converter is validated (`missing=0 unexpected=0`).
- **`PQ2_0` dequant is byte-identical** to the fork's `dequantize_row_pq2_0`
  (verified through a C harness), including the Q4_K embedding.
- **The runtime kernel matters:** the deployed llama.cpp CPU path uses the
  *recurrent* GDN prefill, while transformers defaults to the *chunked* one.
  Training must switch `chunk_gated_delta_rule` →
  `torch_recurrent_gated_delta_rule`, or the corrections will not transfer to
  the CPU validator (chunked decays to cos 0.82 by token 39; recurrent holds
  0.9975).
- **Memory discipline:** never build a full fp32 state dict (36 GB) and never
  hold two copies of the 9B model on the 30 GB host.  `load_text_model_streamed`
  builds bf16 and adopts one tensor at a time.

Runtime fork: `/home/penis/llama.cpp`, branch `moe-corr-runtime` (PQ2_0 +
TAARDIS virtual targets, including the dense `qwen35` readout/`ffn_out` hooks).

## Prior art: Bonsai 2 27B dense forensics (must-read)

The dense route is not new here. `~/Desktop/work/bonsai2-ternary-forensics`
(Scion's parent) already ran the experiment on a dense/hybrid Qwen3.5 27B and
recorded what does and does not work. Load-bearing results for us:

- **End-to-end training only.** Per-layer / block-wise KD is a *dead-end*
  (F6 student-stream 1.25 M PPL; F7 teacher-forced 1.06 M PPL): "local per-layer
  KD cannot control global compounding." Do **not** train the branches
  layer-locally; the frozen body must be in the loop.
- **The GDN recurrence amplifies small weight perturbations.** Patching only
  layers 0+3 of 402 tensors cost 2.6x PPL (Gate 2/F4). This matches our finding
  that the torch chunked GDN diverges from the CPU runtime; allocates correction
  capacity to the recurrent (`linear_attn`) layers and match the runtime kernel.
- **Clean holdout discipline.** The 1.7B STE+KD "1.10x" was retracted twice
  (F11) to ~48% retention once evaluation stopped reading training windows.
  Tune on the 70 train records, report the 30 test records, and never sample
  the eval region.
- **Full-master QAT needs rotation in the loop + a managed LR decay; higher LR
  is worse; mirror-descent / one-well / gating / reprojection tricks are
  falsified** (`docs/RECIPE-LEDGER.md`). Our route is different — residual
  *sidecar* branches on an already-trained frozen ternary body — so rotation is
  not available (the deployed Lloyd container has none), but the LR/decay and
  no-exotic-tricks lessons carry over.
- **Data selection (entropy/excess-loss) lost to random** (F8). Use a simple
  mixed corpus; do not over-engineer selection.
- **Structure is not quality** — matching sparsity/trit layout does not predict
  retention (`RECIPE-LEDGER`); measure the deployed metric.
- **`in_proj_a`/`in_proj_b` are exempt (BF16) in the released Prism format** and
  "permutation plus drift" in Gate 1. Clef's deployed PQ2_0 *does* quantise them
  (type 142), a Clef-specific difference worth watching in per-tensor sensitivity.
- **One heavy ROCm process at a time** (U2: two contexts hang GPU1).

Reusable code: `bonsai_forensics/recover.py` (`ternary_ste`, Adafactor recipe,
holdout discipline), `bonsai_forensics/targets.py` `QWEN3_5` profile (the exact
hybrid projection inventory), `bonsai_forensics/pq2_0.py` (codec).

## Local GPU policy

Use the **non-display** card: it carries no desktop VRAM and is more reliable.
On this box the mapping is:

| DRM | PCI | torch (unpinned) | role |
|---|---|---|---|
| card0 | `07:00.0` | device 1 | **free / use this** (`HIP_VISIBLE_DEVICES=1`) |
| card1 | `03:00.0` | device 0 | display (desktop VRAM) |

So every run pins `HIP_VISIBLE_DEVICES=1` and uses `--device cuda:0`.  Check
free VRAM before any run: `cat /sys/class/drm/card0/device/mem_info_vram_used`.

## Status: packaged and benchmarked honestly (2026-10-03)

One-epoch (78-step) f32 recurrent pilot on an A40 48 GB (rank-512 branches on
both taps), then packaging + the CPU-bridge benchmark.  The honest deployment
numbers on the merged single-file release **match the torch proxy with zero
verdict flips** (max |Δp| 0.027):

| split | uncorrected bridge | corrected bridge | bf16 Clef | Tiny-Jev |
|---|---|---|---|---|
| train | 18/70 (0 FA, 52 FR) | **68/70 (2 FA, 0 FR)** | 67/70 | 62/70 |
| test | 8/30 (0 FA, 22 FR) | **29/30 (1 FA, 0 FR)** | 26/30 | 25/30 |
| hidden cos | 0.231 / 0.235 | 0.371 / 0.382 | 1.0 | — |

CPU latency ~49 ms/token (recurrent GDN prefill): 13.8-28.6 s per record plus
40-90 ms head.  Artifacts: `models/clef-flash-ternary/corrections/`
(`pilot-rental/` for the checkpoint, `packaged/` for the adapter, merged body,
and bridge JSONs).  Cost ~$0.6; pod deleted.  Hidden cos 0.37 still leaves
headroom; see [`HANDOFF.md`](HANDOFF.md) (CURRENT THREAD) and
[`../docs/TAARDIS-PRIOR-ART.md`](../docs/TAARDIS-PRIOR-ART.md) for the next
levers (per-head GDN readout tap, damage-based rank allocation, the Goal-B
quantizer work).

## Goal B: ternary community-quant thread (2026-10-04)

Goal B is a genuinely good ternary Clef for HF (community quant, fine-tune
later), metric = wikitext PPL.  The local pipeline is rotation +
signed basis + PQ2_0 + optional Hessian GPTQ, with the deployed Lloyd scale
rule.  Ladder (f16 = 12.59): deployed 8684 -> V2 identity-sign RTN 514 ->
signed RTN **476** -> mixed-precision `nodown` **265**; GPTQ with absmean
scales 2460 vs absmean RTN 15937 (**the scale rule dominates**, 33x), so the
current run is GPTQ + Lloyd scales.  Details, exact commands, assets and the
next-step ladder: [`HANDOFF.md`](HANDOFF.md) **CURRENT THREAD** section.
Decision side (for the record): body fidelity is decoupled from the frozen
head (best variant probe AUC 0.503); bf16 Clef remains the validator.

## Validated on this box (2026-10-03)

Full f16 reverse-load, streamed to one RX 7900 XT: **11.7 s, 426 params, VRAM
15.9 GB (peak 17.9), host peak 17.4 GB (mostly reclaimable mmap page cache)**.
With the recurrent GDN prefill, the torch post-norm hidden states match the
**CPU ggml bridge at cos 0.99994** (GPU bridge 0.9997). That is the training
forward we will use.

## Handoff

Continuing this work?  Start at [`HANDOFF.md`](HANDOFF.md) — assets, findings,
the pilot number, and the next-step menu.
