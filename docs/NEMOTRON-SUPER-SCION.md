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

## Bottom line

Super has the headroom Scion wants (**-37% to ternary**), but reaching it needs the imatrix (a >= 48 GB
card / pod) and corrections; the local blind re-quant overshoots the quality budget (0.35 nats).  Next:
either the pod imatrix + ternary + a first corrections pass, or a local warm routing-map prune (cheap, but
slow) to stack with it.
