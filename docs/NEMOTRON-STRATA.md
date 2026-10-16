# Nemotron 3.5 in Strata — the support build (work notes, 2026-10-16)

The plan for serving NVIDIA's Nemotron 3.5 Bolt and Super on Strata, ahead of the scion treatment
(corrections, MTP/spec decoder, JEV head).  Everything below was read from the model files and this
tree; no numbers are guessed.

## What the two models are

- **Bolt**: `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` (NVIDIA's own GGUF repo).  GGUF architecture
  **`nemotron_h_moe`**, 88 blocks, embedding 4096, 262,144 context.  The blocks are a recipe, not a
  stack: `block_configs` in the HF config and per-block arrays in the GGUF
  (`expert_feed_forward_length`, `expert_used_count`, `attention.head_count_kv` are 88-element
  lists with zeros off-type), so every block is read on its own.  MoE blocks: 496 experts, 9 used,
  1 shared (FF 5376), `expert_weights_norm`, `expert_weights_scale = 5.0`, gating func 2.
  Mamba blocks: `ssm.conv_kernel 4`, `state_size 128`, `inner_size 8192`, `group_count 8`,
  `time_step_rank 128`.  Attention blocks: 32 heads, 2 KV, key/value 128, rope 128.  The quant mix
  of the 24 GB QAT file: **IQ2_XXS** for the routed experts, **NVFP4 (ggml type 40)** for the SSM
  weights, Q8_0 for embeddings/output, F32 for norms/scales (843 tensors).
- **Super**: `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B` (120B total, 12B active, vision; quants from
  UD-IQ1_M up in the unsloth repo, plus NVFP4).  Same architecture family; a heavier body for later.

## What this tree already has

- GPU MMVQ for **both trick quant types**: `vec_dot_iq2_xxs_q8_1` and `vec_dot_nvfp4_q8_1` with
  `Fmt<40>`/`Split<40>` in `src/kernels/cuda/iq_kernels.cu`, plus CPU NVFP4 blocks
  (`src/kernels/cpu/nvfp4_avx2.cpp`, `nvfp4_avx512.cpp`).  Q8_0 embeddings/output exist.
- Grouped MoE kernels (`native_moe.cu`, `s2_expert_grouped.cu`, `shared_expert.cu`) and the
  attention/rope kernels the other families use.
- A hybrid block already: the q36 path's GDN layers (conv1d + gated recurrent state +
  projections) are the *plumbing* a Mamba block needs; the math differs.
- One family switch (`src/program/generate.cpp` ~1868) routes `qwen35moe` to the q36 engine and
  `qwen4exp` to the main path; `nemotron_h_moe` is a third entry there.

## The gaps (the actual work)

1. **Loader** for `nemotron_h_moe`: the per-block arrays above, the block-type pattern, and the
   per-block variable FF sizes (2048/2304/2560, not one number per model).
2. **Mamba-2 block**: conv1d (with the GDN one as a base), the selective scan over A/B/C/D, dt
   (`time_step_rank 128`), the group layout (`group_count 8`), and the gating.  No Mamba code
   exists in this tree today.
3. **MoE variants**: experts are squared-ReLU FF (the fork's `LLM_FFN_RELU_SQR`), gating is
   sigmoid with an `exp_probs_b` bias and the 5.0 scale, and the per-block FF size breaks the
   current fixed-geometry assumptions.
4. **Geometry checks**: q36's guard (`head_dim != 256`, etc.) is qwen35moe's; Nemotron's attention
   blocks are 128-wide, so the checks move per-block.
5. **Vision (omni wrapper)**: deferred; text-only serve first, mmproj later (the engine's vision
   path is a separable piece).

## Bring-up ladder, parity-gated at every step

The reference is the fork's own implementation (`prism-moe-corr/src/models/nemotron-h-moe.cpp` plus
the llama.cpp graph/hparams behind it), so every step compares against a working implementation on
the same GGUF:

1. **metadata**: load the 88-block recipe, print it against the fork's hparams (a CPU test).
2. **embed + one Mamba block**: teacher-forced hidden/logit comparison against the fork's graph on
   a fixed prompt (the `q36ref.cu` pattern).
3. **one attention block, one MoE block**, then the **full forward**: logits parity (max-abs and
   KL against the fork) on the same prompt.
4. **generate + serve**: tok/s, VRAM/RAM split, context growth; then the scion stages (corrections,
   MTP sidecar against NVIDIA's own LiLiCorr spec decoder, JEV head fitted on the served body).

## Status (2026-10-16, ~04:15)

- The 24 GB Bolt QAT GGUF + mmproj are downloading to `/home/penis/models/nemotron-bolt` (on
  `/home`, a different disk from the busy decision-head run).
- The cards are held by the overnight arms run until about 05:45 (`scion-flashnext` returns then);
  the first load test happens after it.
- Super's quant choice is still open: UD-IQ2_XXS (~30 GB) vs UD-IQ2_M vs NVFP4.
