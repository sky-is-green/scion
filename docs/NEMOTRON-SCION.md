# Scioning Nemotron 3.5 — what is possible, how far losslessly, and the cheap gains

Work notes, 2026-10-16.  The ask: can we give NVIDIA's Nemotron 3.5 variants the scion treatment (a
smaller `.gguf`), and how far can we go.  Everything below is measured from the real Bolt file
(`/home/penis/models/nemotron-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`, 22.01 GB, 843 tensors)
unless said otherwise.

## Where the bytes are (Bolt)

| role | size | share | type |
|---|---|---|---|
| routed experts (`ffn_up_exps`+`ffn_down_exps`, 80 tensors) | **17.05 GB** | 77.5% | IQ2_XXS |
| mamba `ssm_in`/`ssm_out` (80) | 2.47 GB | 11.2% | NVFP4 (type 40) |
| `token_embd` + `output` | 1.14 GB | 5.2% | Q8_0 |
| latent + shared expert (160) | 0.81 GB | 3.7% | Q8_0 / IQ2_XXS |
| attention (32) | 0.30 GB | 1.4% | Q8_0 |
| norms / biases / scales (409) | 0.24 GB | 1.1% | F32 |

**The experts are the whole game.**  Per-layer counts (from the routers): 240 x12, 304 x6, 368 x9, 432 x10,
496 x3 — 13,824 experts over 40 MoE blocks.

## How far losslessly

**Nothing.**  A Scion "pack" (`tools/strata_pack.py`, `tools/iq_pack.py`) is a *bit-exact* re-encode
(`decode(pack) == dequant_ggml(source)`), so it cannot be smaller than the source blocks.  Checked, and
there is no redundancy to exploit:

- `output.weight` is **not** tied to `token_embd.weight` (different sha256) — no embed/head sharing.
- 99.97% of the F32 values are exactly BF16, so narrowing those tensors is nearly lossless but only saves
  ~0.12 GB (and 19k values of 0.24 GB are not exact, so it is not bit-exact).
- The one clearly lossless trim is outside the model: the **separate `mmproj` (1.73 GB)** can simply be
  dropped if we do not serve vision.

So the model file is essentially incompressible at its current precision: any real shrink re-quantizes or
prunes, which is lossy and needs the corrections stage (Scion's whole mechanism).

## The levers (Bolt, non-expert body fixed at 4.96 GB)

Expert re-quant (same expert count):

| target | bpw | total | delta |
|---|---|---|---|
| IQ2_XXS (now) | 2.06 | 22.01 GB | — |
| PTQ1_0 ternary (Scion's) / IQ1_M | 1.75 | 19.43 GB | **-2.58 GB** |
| IQ1_S | 1.56 | 17.88 GB | **-4.13 GB** |

Pruning (drop low-value experts, IQ2_XXS kept):

| prune | total | delta |
|---|---|---|
| 25% | 17.75 GB | -4.26 GB |
| 40% | 15.19 GB | -6.82 GB |

Combined (ternary + prune 25%) lands at **~15.8 GB** — under a 19 GiB card with room for a full resident
expert cache (today only 55% fits at 17.05 GB of experts).

## What transfers from the existing Scion tooling, and what is new

- **Per-type canonical repack** (`tools/canonical_xcheck.py` MAPPINGS, `pack_layer.py`): the *per-ggml-type*
  block encoders are architecture-agnostic and reusable.
- **The pack format + expert arena**: qwen4exp-specific.  `iq_pack.py`'s expert arena is "512 blobs of
  `[gate | up | down]`, blob 1,382,400 bytes", and `pack_layer.py` asserts that.  Nemotron has **no gate**,
  **latent 1024 experts**, and a **per-layer FF width** (1536/2048/2304/2560), so a Nemotron arena is new.
  Same for the index/`native_experts.txt` layout.
- **Corrections**: the sites are `FfnMoeOut` / `SsmOut` / `AttnOutput` (`src/core/corrections.cpp`), and
  they map cleanly onto Nemotron's MoE output (after `latent_up`), mamba `ssm_out`, and `attn_output`.  The
  engine work is a correction hook in `forward_nem` (the q36 corr path is qwen-only today).
- **PLE table**: none — Nemotron has no PLE.  So a Nemotron Scion is experts + corrections only.
- **Expert types**: already supported — the engine's `Nemotron` MoE runs any `iq_supported` expert type
  through `iq_mmvq`, so a re-quanted (IQ1_M / PTQ1_0 / IQ1_S) expert file needs no kernel work, only the
  exporter.

## Findings that shape the plan

- **A sub-IQ2_XXS expert re-quant needs an imatrix (measured).**  `llama-quantize --allow-requantize` with a
  `--tensor-type-file` that pins every non-expert tensor to its own type and sends the experts to `iq1_m`
  *is* expressible (manual overrides bind per tensor, and `nvfp4` / `q8_0` / `f32` are accepted targets), and
  the dry-run sizes it: all-IQ1_M 16.3 GB (1.85 bpw), experts-only ~19.4 GB.  But the run is refused:
  `this quantization requires an imatrix!` - the i-quants below IQ2_XXS want an importance matrix, so this is
  a calibration-pass job, not a free re-pack.
- `exp_probs_b` is **near-constant per layer** (spread 0.014-0.3 within a layer, but 4.4-51.3 across
  layers; 0 experts fall more than 3 sigma below the mean).  It is a soft tie-breaker, **not** a per-expert
  importance signal - so pruning cannot be read off the file; it needs a live routing map over a
  calibration set (the REAP pattern the qwen4exp lane already has).
- Bolt is **already a 2-bit QAT** model.  Post-hoc re-quant of a QAT body is the known-bad path
  (`SCION-LOCAL-ARCHITECTURE.md` P5), so an expert re-quant must be paired with corrections to be shippable.
- **Super (BF16 120B) is the bigger, cleaner target**: no QAT to fight, and UD-IQ2_XXS is 51 GB untuned -
  a ternary/pruned Scion Super would be the release, not a re-quant of a re-quant.

## Measured (2026-10-16)

**An expert re-quant to IQ1_M, on the real Bolt.**
- `llama-imatrix` works: a GGUF imatrix over 64 wiki chunks, and the **expert entries are present** (~99.7%
  coverage; the `no preceding-layer tensor` warnings are for the non-expert tensors only).
- A plain `--tensor-type-file` pin + base `IQ1_M` picks **Q2_K for the down experts** (llama.cpp's own
  IQ1_M recipe), which Strata cannot decode; forcing `--tensor-type ffn_up_exps=iq1_m
  --tensor-type ffn_down_exps=iq1_m` gives a uniform IQ1_M file.  Bolt **22.01 -> 19.43 GB** (-2.58 GB,
  2.38 -> 2.10 BPW) in ~11 min.
- **Engine drift (teacher-forced, first prompt token): top-1 unchanged (`6993` " Paris"), top-5 4/5,
  top-20 18/20, top-100 76/100, KL(bolt||exp) over the top-100 = 0.0161 nats.**  The greedy text diverges
  at token 2 (a low-confidence near-tie), so a Scion release would want corrections, but the distribution
  barely moves.

**The routing-map prune probe, and a keep-list simulation.**
- The engine gains `STRATA_Q36_NEM_ROUTE=<file>` (per MoE block per token it appends the chosen ids and
  weights) and `STRATA_Q36_NEM_KEEP=<file>` (a per-block keep-list; a pruned expert is never selected - the
  weights stay, so the file is not rewritten and the *routing* quality is what is measured).
- Routing over 6 wiki prompts (1328 tokens): the mass is spread - **90% of the routed mass is 49% of the
  experts (6810 of 13824), 99% is 78% (10775); 50% is 12%.**  Bolt's per-layer counts (240..496) are
  already Star-Elastic-compressed from Super's uniform 512.
- **Engine drift of the simulated prune:**

  | change | size | top-1 | top-5 | top-20 | KL |
  |---|---|---|---|---|---|
  | keep 78% (`keep99`) | ~18.3 GB (-3.7) | same | 4/5 | 19/20 | **0.0029** |
  | keep 49% (`keep90`) | ~13.4 GB (-8.6) | same | 3/5 | 15/20 | **0.0132** |

  keep99 matches Bolt's greedy for 5 of 6 tokens and only diverges at the 6th.  Both are far inside a
  correctable budget, so a pruned (and/or re-quanted) Bolt is a real path - and the same probe applies to
  Super, where the 512-expert uniform recipe has more to give.

**How far losslessly: still nil** (the pack is bit-exact; no tied tensors); these are the cheap *lossy*
gains, measured.

## Recommended next steps

1. **Output-drift of an expert re-quant, measured on the engine we already have.**  Needs an imatrix first
   (see above); then re-quant the experts (IQ2_XXS -> IQ1_M / PTQ1_0) into a copy of the file and run the
   engine's teacher-forced logits against the original, reporting KL / top-5 overlap.  The engine path
   already accepts the types, so this is the cheapest *real* quality signal once the imatrix exists.  An
   imatrix is a GPU-calibration pass (rented GPU) and is reusable for both Bolt and Super.
2. **Routing map for pruning** (needs a calibration pass): capture the per-layer expert histogram over a
   few hundred prompts, then size pruning by the *routed* mass, not `exp_probs_b`.  The engine can dump the
   ids (a small probe) or we add one.
3. Then, if (1) is acceptable or correctable: corrections training on rented GPUs (the qwen4exp
   `export_branches_lora.py` + `lora_corr` parity harness are the template) and a Nemotron pack arena.

Nothing here is committed to a release; this is the map and the cheap measurements to take first.
