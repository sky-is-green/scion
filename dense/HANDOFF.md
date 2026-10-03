# Dense Clef-Flash ternary corrections — handoff (2026-10-03)

Everything a fresh session needs to continue without re-deriving. Read this
first, then `docs/DENSE-TERNARY-QAT.md` (the plan) and `dense/README.md`
(route + prior-art).

## What this is

Train residual corrections for the **deployed** `PQ2_0` **Clef-Flash** body
(Qwen3.5-9B dense/hybrid + joint schema head, Apache-2.0) so the CPU validator
recovers bf16 decision parity. The method is Scion's: freeze the deployed
ternary body **in the loop**, add rank-512 branches on the attention output and
MLP output, distil the f16 hidden states (primary) and the head's decisions
(secondary). Dense route lives in `scion/dense/` — a sibling to `moe/`, not part
of it.

## Assets (local)

- `~/Desktop/work/models/clef-flash-ternary/`
  - `clef-flash-PQ2_0.gguf` — 3.26 GB deployed body (type 142 = PQ2_0/Q1_0_g128,
    `GGML_PQ2_0_LLOYD=1`).
  - `clef-flash-f16.gguf` — 17.9 GB reference (teacher states; regenerable ~70 s).
  - `hf-head/` — joint head + tokenizer + configs; `lm_head.safetensors`.
  - `corrections/cache-smoke/` — 158-sequence teacher cache (bench train+test +
    negatives; hidden + head option logits). One negative entry has non-finite
    teacher hidden; the trainer skips non-finite targets.
  - `corrections/pilot-rental/` — `branches-r512-g128-step78.pt` (1.07 GB) +
    `eval-base.json`, `eval-corr.json`.
  - `corrections/packaged/` — `clef-flash-corr-r512-g128-step78.lora.gguf`
    (71 MB, all-ternary `Q1_0_g128`), the merged single-file release
    `clef-flash-PQ2_0-corr-r512-g128-step78.gguf` (3.10 GiB,
    `adapter.embedded=true`), and `bridge-{uncorrected,corrected}.json` +
    `bridge-*/` scratch/logs from the honest benchmark.
  - `corrections/probe/` — old `clef_embed` binary, `tokens.txt`,
    `pq2_ref.bin` (stale 48-token reference from a pre-2026-10-03 build; the
    current bridge rebuild is bit-identical to the pre-patch `build-cpu`).
- Fork: `/home/penis/llama.cpp` branch `moe-corr-runtime` (PQ2_0 + TAARDIS
  virtual targets, dense `qwen35` hooks). `gguf-py` at `/home/penis/llama.cpp/gguf-py`.
- Code: `scion/dense/` (`clef_dense_load.py`, `clef_head.py`, `clef_cache.py`,
  `clef_corrections.py`, `clef_eval.py`, `clef_export.py`,
  `clef_bridge_eval.py`, `quant.py`, `preflight.py`, `RENTAL-RUNBOOK.md`).
  Bridge: `hivebench/tools/clef-bridge/clef_embed.cpp` (batch mode + embedded
  adapter attach). External prior art: `scion/docs/TAARDIS-PRIOR-ART.md`.

## Findings that constrain the next step

1. **Reverse loader is exact** [verified]: f16 GGUF → torch cos **0.99994** vs
   the CPU bridge; PQ2 dequant byte-identical to the fork's
   `dequantize_row_pq2_0`; joint-head sidecar reproduces bf16 `p_correct`
  exactly (0.9219).
2. **Local bf16 torch is unusable** [verified]: two identical evals gave
   different non-finite sets (23 then 50 of 70). f32 is stable; the f32 body is
   31.8 GB and needs a **48 GB** card.
3. **The deployed runtime is the recurrent GDN**; transformers defaults to the
   chunked one. Use `--gdn recurrent` for the deployment path (chunked only as a
   bf16-stability fallback).
4. **Ternary amplifies the torch-vs-ggml kernel gap** [verified]: PQ2 f32
   recurrent torch matches the CPU bridge at only **cos 0.948**. So the torch
   eval is a **proxy**. **Resolved 2026-10-03:** the packaged artifact through
   the bridge reproduces the proxy with zero verdict flips (max |Δp| 0.027);
   the cos gap does not reach the decision metric.
5. **Local GPU policy**: the non-display card is DRM `card0` / PCI `07` /
   torch-device `1`; run with `HIP_VISIBLE_DEVICES=1 --device cuda:0`. The
   display card (`card1`/PCI `03`) carries the desktop. One heavy ROCm process
   at a time; never run torch on the GPU beside a serving engine.

## Pilot result (proxy, 2026-10-03)

A40 48 GB rental, one epoch (78 steps), f32 recurrent, rank-512 both taps,
hidden KD + decision KD, `lr 5e-5 --upstream-clip 1.0`. Parity vs bf16 p and the
gold checker at 0.5:

| split | uncorrected | corrected | bf16 Clef | Tiny-Jev |
|---|---|---|---|---|
| train | 26/70 | **68/70** (2 FA, 0 FR) | 67/70 | 62/70 |
| test | 7/30 | **29/30** (1 FA, 0 FR) | 26/30 | 25/30 |
| hidden cos | 0.23 | 0.37 | 1.0 | — |

Cost ~$0.6; pod deleted.

## Packaging + honest CPU-bridge result (2026-10-03)

Both taps packaged and benchmarked through the deployed CPU path (local, no
spend): adapter export → `qwen35.cpp`/`llama-adapter.cpp` `blk.N.ffn_out`
virtual target → `build-cpu` rebuild → merge into the body
(`adapter.embedded=true`) → bridge (`hivebench/tools/clef-bridge/clef_embed`,
CPU) → joint head sidecar (CPU torch f32). **The honest numbers land on the
proxy almost exactly, with zero verdict flips:**

| split | uncorrected bridge | **corrected bridge** | corrected proxy | bf16 Clef | Tiny-Jev |
|---|---|---|---|---|---|
| train | 18/70 (0 FA, 52 FR) | **68/70 (2 FA, 0 FR)** | 68/70 (2 FA, 0 FR) | 67/70 | 62/70 |
| test | 8/30 (0 FA, 22 FR) | **29/30 (1 FA, 0 FR)** | 29/30 (1 FA, 0 FR) | 26/30 | 25/30 |
| mean p | 0.471 / 0.471 | 0.586 / 0.588 | 0.595 / 0.594 | 0.811 / 0.792 | — |
| hidden cos | 0.231 / 0.235 | 0.371 / 0.382 | 0.372 / 0.383 | 1.0 | — |

- Transfer proxy → bridge on the corrected body: **0 verdict flips on 100/100**,
  max |Δp| 0.027, max |Δcos| 0.004. The 0.948 body cos gap does not touch the
  decision metric.
- The uncorrected baseline on the bridge is more conservative than on the
  proxy (18/70 vs 26/70 train; 8/30 vs 7/30 test) — the module toggles cleanly.
- **CPU latency: ~49 ms/token** (recurrent GDN prefill; 13.8-28.6 s per
  278-589-token record), model load 7.4 s once, head sidecar 40-90 ms/check.
  bf16 Clef was ~3.2 s/check in transformers, so route D needs to accept a
  ~5-8x slower CPU check or move the bridge to the non-display card.
- Artifacts + JSONs: `models/clef-flash-ternary/corrections/packaged/`
  (`clef-flash-corr-r512-g128-step78.lora.gguf` 71 MB,
  `clef-flash-PQ2_0-corr-r512-g128-step78.gguf` 3.10 GiB,
  `bridge-{uncorrected,corrected}.json`); exporter `dense/clef_export.py`,
  benchmark `dense/clef_bridge_eval.py`.
- Gotcha for raw-API consumers: the embedded adapter is loaded at model load
  but a context only applies it after `llama_set_adapters_lora(ctx, nullptr, 0,
  nullptr)` (the bridge now does this; see the bridge README).
- Follow-up menu below; external prior art surveyed in
  [`../docs/TAARDIS-PRIOR-ART.md`](../docs/TAARDIS-PRIOR-ART.md).

## Next-step menu (updated)

1. ~~Package + bridge benchmark~~ — **done** (above); attn_out-only shortcut
   (old item 2) is moot.
2. **Harness integration** (`clef-ternary` backend in
   `hivebench/harness/cascade/validator.py`): bridge subprocess + head sidecar
   on CPU. Decided: keep CPU (~20 s/check) or use the non-display card (needs
   the HIP build and the one-heavy-process policy). Unblocked.
3. **Longer f32 rental run** (needs explicit approval): more epochs, LR decay,
   generic corpus. Per TAARDIS prior art, spend the extra capacity on
   (a) a **per-head GDN readout tap** (`blk.N.ssm_readout`; runtime hook already
   in the fork, training/export missing) and (b) **rank allocation by measured
   damage** instead of uniform rank 512. Corpus: reasoning-heavy rails +
   sparse generic lanes with chunk-level holdout, not the validator tasks.
4. **Clef V2 conversion** (the larger lever, no rental needed to prototype):
   re-quantize the f16 body with per-linear block-Hadamard rotation + Hessian
   GPTQ + self-distill + Doctors, the TAARDIS Qwen3.5-0.8B recipe. The deployed
   PQ2_0 has no rotation; TAARDIS's 27B/0.8B evidence says rotation is the
   quality lever. Fork runtime support (`forge.rotation.*`, `taardis-lora`)
   already exists locally.

## Constraints (unchanged)

- **Never `git push`; commit locally.** Repos are ahead of origin by design.
- **Ask before heavy runs, downloads, and any rental.** The pilot rental was
  approved and closed; a longer run needs new approval.
- One heavy track at a time (30 GB host RAM); use the non-display card.
- hivebench/splinter pre-commit run a claims gate: stage only paths covered by
  `.claims/<id>.json` (create a local, gitignored claim if needed).
