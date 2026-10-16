# Scioning Nemotron 3.5 **Super** — plan and first measurements

2026-10-16.  Super (`NVIDIA-Nemotron-3.5-Super-VL-120B-A12B`, the parent Bolt was compressed from) is the
better Scion target: unlike Bolt it is not already a 2-bit QAT model - it ships as BF16, and its big UD
quants are ~3 bpw.  Worktree `work/strata-nem`; the UD-IQ2_XXS file is on disk (3 shards, 51.1 GB).

## The recipe (read from the file)

89 blocks = **88 trunk (40 mamba + 8 attn + 40 moe, the same pattern as Bolt) + 1 NextN/MTP block**;
`expert_count` 512, **uniform** `expert_used_count` 22 / `expert_feed_forward_length` 2688 (scalars),
latent 1024, SSM inner 8192 / dt 128 / groups 8 / state 128 / conv 4.  The engine runs it (commit
`3006c03c`): `"The capital of France is"` -> " Paris." the same greedy as Bolt.

## Where the bytes are, and the projections

| | bytes | share |
|---|---|---|
| routed experts (80 tensors) | **44.34 GB** | 86.8% |
| everything else (SSM, attention, embed, norms) | 6.77 GB | 13.2% |

Expert types: IQ3_XXS 16.7 GB, Q2_0 11.1, IQ4_NL 9.5, Q8_0 3.0, IQ2_S 1.8, IQ2_XXS 1.5, IQ4_XS 0.75 -
**115.6 B expert params**.  So:

| expert container | bpw | experts | total |
|---|---|---|---|
| as shipped (UD-IQ2_XXS mix) | ~3.07 | 44.34 GB | 51.11 GB |
| **Q2_0** (Strata-native, no imatrix) | 2.25 | 32.5 GB | **39.3 GB (-11.8)** |
| **ternary PTQ1_0 / IQ1_M** | 1.75 | 25.3 GB | **~32.0 GB (-19.1, -37%)** |

## Measured: a blind Q2_0 re-quant (local, no imatrix)

`llama-quantize --allow-requantize` with every non-expert tensor pinned to its own type (699 tensors) and
the experts to `q2_0` rewrote Super 51.1 -> **39.3 GB in 2.8 min** (`nemotron-super-exp/super-exp-q2_0-experts.gguf`).

Drift (teacher-forced first token): top-1 unchanged (" Paris"), top-5 3/5, top-20 14/20, top-100 66/100,
**KL(super||q2_0) = 0.3521 nats**.  That is 20x the cost of Bolt's imatrix-guided IQ1_M (0.016) - a blind
re-quant of Super's higher-precision experts (IQ3_XXS, IQ4_NL, Q8_0) is too lossy to ship as-is.

**So Super needs the imatrix (or corrections):** the size headroom is large, but a blind re-quant spends it
on quality.  That is the opposite of Bolt, where the QAT body was already close and the re-quant was cheap.

## The pipeline, and what is local vs a pod

1. **Imatrix** (`llama-imatrix`, GPU): needs the model resident - Super is 51 GB, so this wants a >= 48 GB
   card (or a partially-offloaded run).  On the two 19 GiB cards it cannot fit, so this is the pod step.
   (The Bolt imatrix ran in ~2 min once resident; the tools are built in `work/llama-stock/build/bin`.)
2. **Ternary re-quant** with the imatrix, `--tensor-type ffn_up_exps=iq1_m --tensor-type ffn_down_exps=iq1_m`
   (or the Scion PTQ1_0 path, `scion/moe/ptq1_0.py` + `repack_experts_q2_0.py`), CPU + disk, ~10-30 min.
3. **Corrections** to recover the quality: the sites map onto Nemotron's MoE output (after `latent_up`),
   `ssm_out` and `attn_output`; the qwen4exp lane's `export_branches_lora.py` + `lora_corr` harness is the
   template, but the `forward_nem` correction hook is new engine work.  Training = the pod (rented GPUs).
4. **Prune** (optional, stacks): a routing map over calibration prompts.  The probe is built
   (`STRATA_Q36_NEM_ROUTE`) and works on Super, but Super decodes at ~0.1 tok/s on this box (~880 expert
   gathers/token), so a useful map wants the pod or a warm run.  For Bolt, 22% of the experts held 1% of
   the mass (keep-78% cost 0.003 nats); Super's 512 uniform experts likely prune comparably.
5. **Pack/arena**: a Nemotron expert arena is new (no gate, latent 1024, per-layer FF, the NextN block);
   the per-type block encoders transfer.

## Local artifacts

- `models/nemotron-super-exp/super-exp-q2_0-experts.gguf` (39.3 GB) - the blind re-quant, kept for reference.
- `/tmp/opencode/{super_types.txt, lg-super-orig.f32, lg-super-q2.f32}` - the pin list and the two logit dumps.
- Bolt was deleted (22 GB + the 19 GB exp); 94 GB free.

## Measured: imatrix-guided re-quant (local, no pod)

Two engine hooks make this local: `STRATA_Q36_NEM_IMATRIX=<file>` accumulates, per MoE block, the latent
energy (the `ffn_up_exps` input) and the squared-ReLU FF energy (the `ffn_down_exps` input) and writes the
legacy `.dat` imatrix `llama-quantize` reads; `STRATA_Q36_NEM_RESIDENT=N` is a host-side page-cache
residency policy (below).  A 128-token wiki calibration costs ~22 min on this box.

`llama-quantize --imatrix` with the non-experts pinned and **`ffn_up_exps=iq1_m`,
`ffn_down_exps=q2_0`** (IQ1_M needs `n_in % 256 == 0`, and Super's FF 2688 is not a multiple of 256, so the
down experts take the 64-block Q2_0) rewrote Super **51.1 -> 35.71 GB** (2.31 BPW) in ~11 min.

| container | size | top-1 | top-5 | top-20 | KL(super‖exp) |
|---|---|---|---|---|---|
| blind Q2_0 (no imatrix) | 39.3 GB | same | 3/5 | 14/20 | 0.3521 |
| **IQ1_M / Q2_0, imatrix** | **35.7 GB (-15.4, -30%)** | same | 4/5 | 16/20 | **0.1018** |

The imatrix buys 3.5x on the divergence **and** 3.6 GB (the mixed IQ1_M ups).  Top-1 is unchanged
(" Paris.") in every case.  A fuller calibration (more tokens) and corrections would push 0.10 lower.

## Host-side expert residency (CachyLLama's `--moe-expert-residency`, adapted)

Super overflows RAM as well as VRAM, so every expert gather was a **disk fault** - the calibration ran at
~10 s/token.  CachyLLama's MoE residency is the fix: keep the hot experts' **mmap pages** in RAM with
`madvise(MADV_WILLNEED)` and release the cold ones with `MADV_COLD` (non-destructive; `MADV_DONTNEED` is
destructive on a file mapping and `MADV_FREE` is `EINVAL` on `MAP_SHARED|PROT_READ`), steered by a
per-layer recency+frequency cache.  `STRATA_Q36_NEM_RESIDENT=N` implements exactly that in `forward_nem`
(slot cache + the two `madvise` hints on the experts' `src_u`/`src_d` ranges); it sits **under** the VRAM
expert cache, so it only pays for the gathers the VRAM cache misses.

Measured at 64 experts/block: policy hit rate **58.6%** (112,640 touches, 44,056 evictions).  CachyLLama
reports 83-95.5% on 256-expert/top-8 models; Super is 512 experts / **top-k 22**, so its working set is
much larger - a higher `--moe-resident-per-layer` (RAM budget ~10-15 GB) is the knob.

## Bottom line

Super has the headroom Scion wants (**-37% to ternary**), but reaching it needs the imatrix (a >= 48 GB
card / pod) and corrections; the local blind re-quant overshoots the quality budget (0.35 nats).  Next:
either the pod imatrix + ternary + a first corrections pass, or a local warm routing-map prune (cheap, but
slow) to stack with it.

## UPDATE 2026-10-16 (the pod was not needed)

The imatrix, the re-quant and the drift all ran **locally**: 51.1 -> 35.71 GB (-30%) at KL **0.1018**,
imatrix-guided, with a 128-token calibration (~22 min) and the CachyLLama-derived host residency.  See
"Measured: imatrix-guided re-quant" above.  The pod is only for corrections **training** throughput.

## UPDATE 2026-10-16 (b): chunked calibration, and the metric was lying

A single long calibration costs ~10 s/token (a 51 GB file inside 30 GB of RAM: measured **0.69-0.91 GB of
expert reads per token**, so it is RAM-bound, not policy-bound - see the residency note above; at
256 experts/block the policy wants ~29 GB of hot expert pages and the kernel evicts them anyway).
Calibrating as **many short 32-token sequences in separate processes, merging the per-entry
`sums`/`ncall`** runs 1024 tokens in **~40 min against ~6 h** for one long sequence
(`tools/nemotron/calib_chunks.sh` + `merge_imatrix.py`).

The first reading said the merged 1024-token imatrix was *worse* (KL 0.1808 vs 0.1018) - but that was a
**single-token** KL (one prompt, one distribution).  Teacher-forcing 32 tokens (`--force`, which dumps
K+1 full rows; `tools/nemotron/drift.sh`) reverses it:

| imatrix | KL p50 | KL p90 | steps > 1 nat | top-1 |
|---|---|---|---|---|
| 128 tokens, one sequence | 0.0935 | 0.99 | 4/33 | 84.8% |
| **1024 tokens, 32-token chunks** | **0.0569** | 0.80 | 3/33 | 78.8% |

Both are good; the mean is dominated by 3-4 **confident token flips** (the reference entropy there is
0.1-0.2, so it is a different-but-plausible argmax, not a blurred distribution).  Lesson: **the
single-token drift is not a usable metric - teacher-force >= 32 steps.**  The 32-token resets do distort
the deep (`ffn_down_exps`) statistics (cosine 0.60-0.84 against the 128-token run; the latent-driven
`ffn_up_exps` correlate 0.96-0.99), so the gain is from the extra tokens, not from the chunking as such.

## Next: ternary experts (PTQ1_0, ggml type 143)

`scion/moe/ptq1_0.py` is the container (**28 bytes / 128 = 1.75 bpw**, `qs[24] | qh[2] | fp16 d[2]`), and
`scion/moe/qwen4exp_export.py` already writes it - `--experts ptq1_0` via a patched `gguf` that defines
`GGMLQuantizationType.PTQ1_0` and `LlamaFileType.MOSTLY_PTQ1_0` - so a Super exporter is an adaptation,
not new ground.  The engine reads type 143 (`vec_dot_ptq1_0_q8_1_multi` + the CPU kernels), so the drift
harness above applies unchanged.  Both expert roles qualify at 128 wide (up: 1024 % 128 == 0, down:
2688 % 128 == 0 - the reason IQ1_M could only take the ups).

