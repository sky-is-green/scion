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
- **The model is on disk.** `/home/penis/models/nemotron-bolt/` (NVIDIA's own GGUF repo):
  `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` 22.0 GB (74.0B params: IQ2_XXS experts 17.5 GB,
  **NVFP4 type-40** SSM weights 2.47 GB, Q8_0 attention/latent/embed 1.8 GB, F32 0.24 GB) plus the
  1.73 GB mmproj.  Super (120B A12B VL) is **not** downloaded; the quant is open (UD-IQ2_XXS ~30 GB /
  UD-IQ2_M / NVFP4).
- **The reference is running.** Stock upstream `work/llama-stock` (shallow clone of master, HIP build
  for gfx1100) serves Bolt **on CPU** on port **8896** (`-ngl 0 -c 1024`; loads in 35 s, ~0.5 tok/s,
  correctness only).  Captured parity target: `/tmp/opencode/bolt-ref-lp1.json` - `"The capital of
  France is"` -> `" Paris."` with top-5 logprobs (` Paris` -0.025, ` **` -5.53, ` located` -5.80).
  **On the two cards, stock's upload path is unusably slow**: ~21 GB took 14.5 minutes (~24 MB/s,
  one core 87% busy in a per-tensor host loop, RSS 17 GB); a second attempt did not finish in 35
  minutes.  That is a stock-path pathology, not a model problem - the engine's batched DMA
  (`docs/BATCHED_DMA.md`) is the fix.  If a fast GPU reference is wanted, merge upstream's 3.5
  support into a **scratch branch of the fork** (`~/Desktop/work/prism-moe-corr`); never touch the
  working branches.
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
2. **The mamba-2 block** (the only new math): `ssm_in` -> `z [d_inner] | xBC | dt`; conv over xBC
   with a `d_conv-1`-row state + bias + SiLU; split x/B/C; `dt += dt_b`; selective scan with A and
   the per-head D skip (`y += x*D`); `swiglu(z, y)`; grouped RMS norm `[d_inner/n_group, n_group]`;
   `ssm_out`.  State per layer: conv `[3, 10240]` + ssm `[128, 64, 128]` per sequence (~4 MB/layer).
   Parity-test the block alone against a numpy reference computed from the same weights (the
   `q36ref.cu` pattern), not against the whole model.
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

# the CPU reference (already up; restart if it died)
cd work/llama-stock && ./build/bin/llama-server -m /home/penis/models/nemotron-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
  -ngl 0 -c 1024 --host 127.0.0.1 --port 8896
curl -s http://127.0.0.1:8896/completion -d '{"prompt":"The capital of France is","n_predict":2,"temperature":0,"logprobs":5}'

# restore the user's service when the cards are no longer needed
systemctl --user start scion-flashnext
```
