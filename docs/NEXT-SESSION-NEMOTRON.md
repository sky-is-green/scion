# Next session: Nemotron 3.5 on Strata (continuation prompt, 2026-10-16)

**Context.** Read `~/Desktop/SCION-LOCAL-ARCHITECTURE.md`, then `work/scion/docs/NEMOTRON-STRATA.md`
(what the models are, the upstream contract, the plan), then this doc. The operator's direction:
build support for NVIDIA Nemotron 3.5 via **Strata first** (not the fork), then the scion treatment.

## Where the lane stands (2026-10-16)

- **The metadata layer is in.** `work/strata-nem`, branch **`nemotron-hip`**, based on the clean
  `scion-0.1.44` tip `2b5d28d0` (same base as `ep-hip` and `decision-head`), commit **`5b7f16a7`**:
  - `nemotron_h_moe` is a `ModelFamily` (arch string, title) with no engine path yet;
  - `ModelGeometry` carries the per-layer recipe (`nem_kind` / `nem_used` / `nem_ff`, `moe_latent`,
    `ssm_dt_rank`), parsed from the file's three 88-entry arrays: a block is mamba-2 when
    `head_count_kv` and FF are both 0, attention when only FF is 0, MoE otherwise; each MoE block
    keeps its own top-k and FF width (the "puzzle");
  - the architecture guard requires the arrays and refuses the pre-3.5 scalar form by name;
  - the GGUF reader keeps small numeric arrays whole (it kept a 64-entry sample before - the recipe
    indexes all 88; do not regress this).
  - Validated against the real Bolt file: 88 = 40 mamba + 8 attn + 40 moe, latent 1024, per-layer
    top-k max 9, FF max 2560, ssm inner 8192 / dt 128 / groups 8 / state 128 / conv 4, shexp 5376.
    `tests/core/model_family_test.cpp` still passes (build it with the AMD_HIP.md recipe, or compile
    it directly: `g++ -std=c++20 -I include -I tests -I tests/core tests/core/model_family_test.cpp`).
- **The layout table + test are in.** `c72a4b59`: `family_layout.hpp` gains `LayerKind::Mamba/Moe`,
  `nemotron_layer_tensors()` (attn_norm + the mamba-2 / attention / latent-MoE wheels per `nem_kind`;
  the per-layer expert count and FF width are the block's own, so the expert/router tensors carry a
  wildcard dim `-1` that `check_gguf_layout` matches against any value), and `layout_table_complete()`
  covers the family; `geometry_from_gguf` sets `top_k` from the per-layer recipe before its guard.
  `model_family_test.cpp` adds a tiny 6-block recipe (mamba/moe/mamba/attn/moe/mamba) covering the
  geometry, a complete layout, the wildcard dim, and the wrong-shape / missing / unnamed / scalar
  refusals.  **Verified against the real Bolt file: 843 tensors, layout ok.**
- **The q36 loader branch + family switch are in.** `db78ad34`: `src/qwen35/q36_engine.cu` accepts
  `nemotron_h_moe`; `Layer` carries `nem` (0 mamba-2 / 1 attention / 2 latent MoE) with the mamba-2
  wheels (`ssm_in`/`ssm_out` as raw NVFP4 device copies + their per-tensor scales) and the latent MoE
  (router, `exp_probs_b`, latent down/up, shared expert, and the routed up/down experts kept as
  mapped-file pointers with their own types; expert count from the router's `ne[1]`, FF from the down
  tensor's `ne[0]`).  `mat_any` loads type 40 (NVFP4) via the iq path.  `generate.cpp` routes the
  family to q36 beside `qwen35moe`; `run()` **refuses a Nemotron run after the dense-only load**
  ("no compute path yet") rather than mis-run.  On Bolt: 843 tensors, 5.15 GiB on the GPU in 2.7 s.
- **Pitfall found:** Bolt's experts are **IQ2_XXS**, which `STRATA_D_FMTS` does **not** list for the
  grouped native `down` kernel (no `X(16)`) - so `native_expert_supported(16,16,..)` is false.  The
  latent experts must go through `iq_mmvq` / an iq-backed grouped path (the loader gates on
  `iq_supported`).  The mamba-2 `ssm_in`/`ssm_out` are **NVFP4 (type 40)**: also the iq path, not
  `native_mmvq`.
- **The mamba-2 block is in and parity-tested.** `7ee0aabd` (geometry reads `layer_norm_rms_epsilon`
  - Bolt's is 1e-5) + `80feeeab`: `strata::kernels::nem_mamba2_block` (conv + selective scan +
  SwiGLU/grouped-RMS-norm) in `include/strata/kernels/nemotron_mamba.hpp` +
  `src/kernels/cuda/nemotron_mamba.cu`; `tests/hip/nemotron_mamba.cpp` (a real Bolt mamba layer
  against a double host reference, GPU vs host rel 4e-7) and `tools/nemotron/mamba_parity.py` (an
  independent numpy reference GPU vs numpy rel 3.3e-7).  The ssm_in / ssm_out projections stay the
  caller's (the iq kernels'): step 3 wires the mm around `nem_mamba2_block`.
- **The model is on disk.** `/home/penis/models/nemotron-bolt/` (NVIDIA's own GGUF repo):
  `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` 22.0 GB (74.0B params: IQ2_XXS experts 17.5 GB,
  **NVFP4 type-40** SSM weights 2.47 GB, Q8_0 attention/latent/embed 1.8 GB, F32 0.24 GB) plus the
  1.73 GB mmproj.  Super (120B A12B VL) is **not** downloaded; the quant is open (UD-IQ2_XXS ~30 GB /
  UD-IQ2_M / NVFP4).
- **The reference must be regenerated.** Stock upstream `work/llama-stock` (shallow clone of master,
  HIP build for gfx1100) served Bolt **on CPU** on port **8896** (`-ngl 0 -c 1024`; loads in 35 s,
  ~0.5 tok/s, correctness only).  **The box rebooted since and `/tmp` was cleared**: the server is
  down and the captured target `/tmp/opencode/bolt-ref-lp1.json` is gone.  Regenerate before step 4
  (`"The capital of France is"` -> `" Paris."`, top-5 logprobs ` Paris` -0.025, ` **` -5.53,
  ` located` -5.80; the `curl` below).  **On the two cards, stock's upload path is unusably slow**:
  ~21 GB took 14.5 minutes (~24 MB/s, one core 87% busy in a per-tensor host loop, RSS 17 GB); a
  second attempt did not finish in 35 minutes.  That is a stock-path pathology, not a model problem -
  the engine's batched DMA (`docs/BATCHED_DMA.md`) is the fix.  If a fast GPU reference is wanted,
  merge upstream's 3.5 support into a **scratch branch of the fork** (`~/Desktop/work/prism-moe-corr`);
  never touch the working branches.
- **Box state:** `scion-flashnext` is **down** (cards free for this lane; restore with
  `systemctl --user start scion-flashnext` when done).  The CPU reference holds ~14 GB RAM of 30 GB.
  Another lane's CPU training may run; check `free -g` and `ps` before heavy steps.

## Plan, in order

1. **The loader branch in q36** (`src/qwen35/q36_engine.cu`): accept `nemotron_h_moe` beside
   `qwen35moe`; build the `Layer` list from `nem_kind`; load the tensor names/shapes upstream's
   `load_arch_tensors` uses (mamba: `ssm_in`, `ssm_conv1d(.b)`, `ssm_dt.bias`, `ssm_a`, `ssm_d`,
   `ssm_norm`, `ssm_out`; attention: `attn_norm`, `attn_q/k/v/output`; moe: `ffn_gate_inp`,
   `ffn_exp_probs_b.bias`, `ffn_latent_down/up` (Q8_0), `ffn_up/down_exps` (IQ2_XXS, per-layer
   expert count **from the router tensor's own shape**, 240..496), `ffn_up/down_shexp`).  Add the
   nemotron entries to the `family_layout.hpp` layout tables so `check_gguf_layout` validates Bolt,
   and a synthetic test (the `gguf_fixture.hpp` pattern; `u32s` writes the arrays).
   **DONE** (`c72a4b59` layout table + test, `db78ad34` loader + family switch; `run()` refuses a
   Nemotron run until step 3 lands, so a mis-run cannot slip through).
2. **The mamba-2 block** (the only new math): `ssm_in` -> `z [d_inner] | xBC | dt`; conv over xBC
   with a `d_conv-1`-row state + bias + SiLU; split x/B/C; `dt += dt_b`; selective scan with A and
   the per-head D skip (`y += x*D`); `swiglu(z, y)`; grouped RMS norm `[d_inner/n_group, n_group]`;
   `ssm_out`.  State per layer: conv `[3, 10240]` + ssm `[128, 64, 128]` per sequence (~4 MB/layer).
   Parity-test the block alone against a numpy reference computed from the same weights (the
   `q36ref.cu` pattern), not against the whole model.
   **DONE** (`7ee0aabd` + `80feeeab`): `include/strata/kernels/nemotron_mamba.hpp` +
   `src/kernels/cuda/nemotron_mamba.cu` (`nem_mamba2_block`: conv + scan + gate_norm; the ssm_in/ssm_out
   projections stay the caller's), `tests/hip/nemotron_mamba.cpp` (real Bolt layer, T 1/2/4/16, a
   double host reference) and `tools/nemotron/mamba_parity.py` (numpy reference straight off the GGUF).
   GPU vs numpy rel 3.3e-7 on Bolt block 0.  Two bugs the numpy check caught: `ssm_conv1d.weight` is
   GGUF `[d_conv, channels]` **tap-fastest** (element (k,c) at `c*d_conv + k`), and `geometry_from_gguf`
   never read `attention.layer_norm_rms_epsilon` (Bolt's is 1e-5, not the 1e-6 default).
3. **Attention + MoE variants in the q36 dispatch**: the third block kind and the latent MoE
   (experts on 1024, squared-ReLU, sigmoid gating with `exp_probs_b` + `expert_weights_norm` +
   `scale = 5.0`, shared expert from the pre-latent input, add both).  The engine's kernels already
   decode IQ2_XXS and NVFP4 (type 40) on gfx1100 (`src/kernels/cuda/iq_kernels.cu`); the grouped
   MoE kernels are `native_moe.cu` / `s2_expert_grouped.cu` / `shared_expert.cu`.
4. **Full-forward parity**, teacher-forced top-k logprobs against the captured reference (and/or a
   fast GPU reference if one is built).  Then **serve with the hybrid expert mode** (Bolt's 22 GB
   does not fit one 20 GB card resident; q36's hybrid slots + CPU pool is the path), measure tok/s
   and the VRAM/RAM split - the batched DMA should make loading minutes, not the stock's 15.
5. **Scion stages**: keep NVIDIA's QAT quant as the body (post-hoc re-quant is the known-bad path);
   corrections/adapters only where measured; the **LiLiCorr draft** (NVIDIA ships it as a separate
   GGUF; upstream supports a draft model for this arch) versus our MTP sidecar; then the JEV head
   (the `emb=` / `STRATA_Q36_JEV_HEAD` path from the decision-head lane is the template).

## Quick commands

```sh
# the parser, against the real file (header only)
g++ -std=c++20 -I include /tmp/opencode/nem_check.cpp -o /tmp/opencode/nem_check   # from work/strata-nem
/tmp/opencode/nem_check /home/penis/models/nemotron-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf

# the CPU reference (rebuild/restart: the box rebooted, the server is down)
cd work/llama-stock && ./build/bin/llama-server -m /home/penis/models/nemotron-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
  -ngl 0 -c 1024 --host 127.0.0.1 --port 8896
curl -s http://127.0.0.1:8896/completion -d '{"prompt":"The capital of France is","n_predict":2,"temperature":0,"logprobs":5}'

# restore the user's service when the cards are no longer needed
systemctl --user start scion-flashnext
```
