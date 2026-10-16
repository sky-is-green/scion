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

## Reference status (2026-10-16)

- The user's forks predate Nemotron 3.5: `/home/penis/llama.cpp` (the decision-head fork) refuses the
  file with `key nemotron_h_moe.expert_used_count has wrong type arr but expected type u32` - the 3.5
  "puzzle" recipe carries per-layer arrays where the fork's reader wants scalars.
- Upstream `ggml-org/llama.cpp` master supports it fully (`src/models/nemotron-h.cpp`,
  `nemotron-h-moe.cpp`, `mamba-base.cpp`).  A stock HIP build of master is in
  `work/llama-stock/build` and is this project's reference implementation.
- The upstream contract, read from those files:
  * hparams: `get_key_or_arr` for `expert_feed_forward_length` and `expert_used_count` (per-layer,
    88 entries); `moe_latent_size`; SSM constants; a layer is **recurrent iff `n_head_kv(i) == 0 and
    n_ff(i) == 0`**; 88 layers with a heterogeneous top-k array is `75B_A9B` (Bolt), uniform is
    `120B_A12B` (Super).
  * the **per-layer expert count comes from the tensor shapes** (the router's `ne[1]`, cross-checked
    against `n_expert_used(i)`), never from the header's global `expert_count`.
  * Mamba2 layer (`build_mamba2_layer`): `ssm_in` -> `z [d_inner] | xBC [d_inner + 2*n_group*d_state]
    | dt [dt_rank]`; conv over xBC with a `d_conv-1`-row state, bias, SiLU; split x/B/C; `dt += dt_b`;
    selective scan with A and the per-head D skip (`y += x*D`); `swiglu(z, y)`; grouped RMS norm
    `[d_inner/n_group, n_group]`; `ssm_out`.  d_in_proj = `2*d_inner + 2*n_group*d_state + dt_rank`
    (4096 -> 18560 on Bolt).
  * MoE layer: `latent_down` -> experts on the 1024-latent (`up_exps`/`down_exps`, squared-ReLU, no
    gate matrix) -> `latent_up`; shared expert `up_shexp`/`down_shexp` from the *pre-latent* input;
    sigmoid gating with `exp_probs_b` bias, `expert_weights_norm`, `expert_weights_scale = 5.0`;
    add both branches.  Attention layer: plain GQA (32/2 heads x 128), kq_scale `1/sqrt(128)`, rope 128.
- Bolt's file has no embedded NextN/MTP block (blocks 0..87 only); NVIDIA ships the **LiLiCorr spec
  decoder as a separate GGUF** - the same slot as the MTP sidecar.

## Where it goes in Strata

The q36 engine is the right host: it is the family engine for hybrid-MoE bodies, and it already has
the pieces this model needs - per-layer dispatch (GDN vs attention), a **hybrid expert mode** (VRAM
slots + the CPU pool; Bolt's 22 GB does not fit one 20 GB card resident), the serve/MTP/sidecar
machinery, and access to the shared kernels that already decode IQ2_XXS and NVFP4 on gfx1100.  The
build is therefore: the `nemotron_h_moe` loader branch (per-layer arrays, tensor shapes as the
authority), a third block kind (mamba2) beside GDN and attention, and the latent-MoE variant in the
expert path.  Worktree: `work/strata-nem`, branch `nemotron-hip`.

## Bring-up ladder (revised)

1. **Reference up**: stock build serves Bolt (speed baseline, and the top-k logprobs to compare against).
2. **Loader + Mamba2**: the q36 loader branch plus the mamba2 block, parity-tested per block against a
   small numpy reference computed from the same weights (the `q36ref.cu` pattern).
3. **Full forward parity**: teacher-forced top-k logprobs against the stock build on fixed prompts.
4. **Serve with hybrid experts**, measure tok/s and the VRAM/RAM split.
5. Then the scion stages: corrections, the LiLiCorr draft (or our MTP sidecar) and the JEV head.

## Status (2026-10-16)

- **Done: the metadata layer.**  The `nemotron_h_moe` family, the per-layer recipe (block kinds from the
  `head_count_kv` / FF arrays, per-MoE top-k and FF width, `moe_latent_size`, the mamba-2 SSM keys) and
  the architecture guard are in `work/strata-nem` (commit `5b7f16a7`).  Validated against the real Bolt
  file: 88 = 40 mamba + 8 attn + 40 moe, latent 1024, top-k max 9, FF max 2560, ssm 8192/128/8/128/4.
  The suite's `model_family_test` still passes; the GGUF reader now keeps small numeric arrays whole
  (it sampled 64 entries before - the recipe indexes all 88).
- **Reference: running.**  Stock upstream (`work/llama-stock`, HIP build) loads Bolt on **CPU** in 35 s
  and answers (`"The capital of France is"` -> `" Paris."`, top-5 logprobs captured in
  `/tmp/opencode/bolt-ref-lp1.json`: ` Paris` -0.025, ` **` -5.53, ...).  On the two cards the stock
  **HIP upload path is unusably slow**: ~21 GB of tensors took 14.5 minutes (about 24 MB/s, one core
  87% busy in a per-tensor host loop, RSS 17 GB); a second attempt did not finish in 35 minutes.  The
  reader/pipeline is fine - the host->device copy path is the problem, which is exactly what the
  engine's batched DMA (docs/BATCHED_DMA.md) exists for.  The CPU reference stays up on port 8896 as
  the parity target; speed work waits for the engine path.
- Next: the `nemotron_h_moe` loader branch in q36 (tensor names per block kind, per-layer expert counts
  from the router shapes) and the mamba-2 block (conv + selective scan + grouped RMS norm), parity-
  tested per block against the captured reference.
