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
  - `corrections/probe/` — `clef_embed` (fork bridge binary), `tokens.txt`,
    `pq2_ref.bin` (48-token CPU-bridge reference).
- Fork: `/home/penis/llama.cpp` branch `moe-corr-runtime` (PQ2_0 + TAARDIS
  virtual targets, dense `qwen35` hooks). `gguf-py` at `/home/penis/llama.cpp/gguf-py`.
- Code: `scion/dense/` (`clef_dense_load.py`, `clef_head.py`, `clef_cache.py`,
  `clef_corrections.py`, `clef_eval.py`, `quant.py`, `preflight.py`,
  `RENTAL-RUNBOOK.md`).

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
   eval is a **proxy**; the honest number must come from the llama.cpp bridge on
   the packaged artifact.
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

Cost ~$0.6; pod deleted. **This is a proxy** — the honest number requires
packaging + the bridge benchmark.

## Next-step menu (decide in the new session)

1. **Package + bridge benchmark (recommended; local, no spend).**
   - Export `dense/clef_export.py` (adapt `moe/export_branches_lora.py`):
     `.linear_attn.out_proj.branch` → `blk.N.ssm_out.weight` LoRA;
     `.self_attn.o_proj.branch` → `blk.N.attn_output.weight` LoRA;
     `.mlp.branch` → dense `blk.N.ffn_out` branch.
   - Patch the fork `src/models/qwen35.cpp` to apply `blk.N.ffn_out` (mirror
     `qwen35moe.cpp:558` `build_lora_branch`, and add the name to the TAARDIS
     virtual target/anchor in `src/llama-adapter.cpp`); rebuild.
   - Merge the adapter into the body (`moe/merge_adapter_into_body.py`) so the
     bridge loads it via `adapter.embedded=true`.
   - Run the benchmark through the bridge (`dense/clef_embed`, CPU) → head
     sidecar (CPU torch f32) → p_correct; report parity + CPU latency. Optionally
     use the fork's per-layer capture for per-layer cos vs the f16 bridge.
2. **attn_out-only package** (no fork patch) as a faster first read.
3. **Longer f32 rental run** (needs approval) — more epochs, LR decay, generic
   corpus — to push hidden cos and the held-out parity before packaging.
4. **Harness integration** (after packaging): add the `clef-ternary` backend to
   `hivebench/harness/cascade/validator.py` (bridge subprocess + head sidecar,
   CPU) and re-run the label benchmark.

## Constraints (unchanged)

- **Never `git push`; commit locally.** Repos are ahead of origin by design.
- **Ask before heavy runs, downloads, and any rental.** The pilot rental was
  approved and closed; a longer run needs new approval.
- One heavy track at a time (30 GB host RAM); use the non-display card.
- hivebench/splinter pre-commit run a claims gate: stage only paths covered by
  `.claims/<id>.json` (create a local, gitignored claim if needed).
