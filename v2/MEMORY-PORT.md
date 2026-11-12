# Memory port (FlashNext P2 switches) - PLAN 13a step A

Date: 2026-11-12.  Status: **ported and gated on real weights; 2k/4k fit the
20 GB card, 8k is ~1-2.5 GiB short with the identified levers listed.**  No pod
was created; everything ran on the local 2x RX 7900 XT.

## What was ported (moe/qwen35_moe_proxy.py)

The three proven FlashNext switches (`box-run-flashnext.sh train`), ported
from `moe/qwen4exp_proxy.py`; all default off, v1 paths untouched:

1. `--compact-banks` - the frozen expert banks are held in their *deployed*
   form (2-bit codes + fp16 scale per 128-group, the Q2_0_g128 container) and
   the training forward decodes only the expert slices a token hits
   (`CompactBank`, `compact_experts_forward`, `compact_banks`).
2. `--grad-checkpoint` - the compact decode and the decoder layers are
   recomputed in backward (`enable_grad_checkpointing`,
   `enable_layer_checkpointing`); exact, costs one extra forward per step.
3. `--checkpoint-mode group --expert-group-size 64` - the decode checkpoint is
   one node per expert group, bounding the decoded-weight saves at
   ~group x 6.3 MB (the FlashNext runbook value).

Supporting levers in the same pass, each measured or pinned:

- `--verify-banks` (gate, see below).
- `--chunked-ce` (qwen4exp P2 lever, ported verbatim, off by default): the LM
  CE over the 248,320 vocab is computed and recomputed in chunks instead of
  one full-vocab fp32 softmax (same loss to fp32 reduction order; pinned by
  test).
- `kd_loss.support_mass_lse` now processes rows in checkpointed chunks
  (~0.25 GiB fp32 transient instead of seq/268 GiB).  Each row's
  logsumexp/gather/exp/sum is independent, so the values are **bitwise
  identical** to the unchunked form (tested on CPU).
- Bug fixes: `del logits` after `loss.backward()` in the train loop (the next
  record's forward allocated its full logits before the old one was rebound -
  invisible at 512-token v1 windows, ~4-8 GiB at 4k/8k);
  `memory_report` residency no longer includes `expert_bf16_equivalent`
  (faithfully copied from qwen4exp, wrong there too);
  `memory_report` reads the model's device, not the current device.

## The v2 seed fix applied to the compact container

With `--compact-banks --bank-gguf` the container is **read verbatim from the
deployed GGUF** (`seed_banks_from_gguf.read_bank_blocks`, new): the GGUF's
Q2_0_g128 blocks (fp16 scale + 32 packed-code bytes per 128-group) map 1:1
onto the `CompactBank` buffers, so training sees exactly the deployed codes
*scales*, not a re-derivation.  This is strictly closer to deployment than the
v1 dense path, which re-derives fp16 scales from the bf16 seeded values (the
export audit measured 86.3% scale-bit changes there, 0 code changes).  The
measured forward consequence: v1 dense-in-place vs compact-from-GGUF on the
prefix is rel-L2 logit drift 0.0052 (max |diff| 0.188), router top-8 agreement
0.9972 - small, but the compact path is the one that IS the shipped artifact.

Streaming design: per layer - seed dense from the GGUF, build the compact
bank from the same file's blocks, verify, free the dense bank.  Peak is one
dense bank (~1.6 GiB bf16) instead of the whole model (~64 GiB), which is
what will let the full 40-layer model train from the deployed GGUF on cards
that cannot hold the dense banks at all.

## Bit-exactness gate (real weights, 4-layer prefix)

`v2/scripts/prefix_gate.py` (GPU 1), empero HF prefix shards +
`Scion-35B-A3B-PQ2_0-corr.gguf` (sha `4e9e3292...cffa040`):

- 3,221,225,472 weights checked (4 layers x gu `[256,1024,2048]` +
  dn `[256,2048,512]`): **0 element mismatches** - the compact decode is
  bit-for-bit the seeded dense bank, i.e. the deployed dequantised values.
- The builder's internal verify (per expert `torch.equal` vs the seeded
  values + code re-derivation with `ternary_lloyd`) passed on all 8 banks.
- 28 new CPU/GPU unit tests in `moe/tests/test_qwen35_compact.py` pass
  (pack layout == Q2_0_g128 byte layout, encode/decode == in-place banker,
  block->container fidelity, forward equality, group checkpointing exactness
  at group sizes 1/2/3/8/64-class, chunked CE value+gradient equality, CLI
  defaults).  The 41 pre-existing `test_qwen4exp_*` failures in the local
  suite are unchanged before/after (transformers 5.17 vs the 5.5-era tiny
  rig; not touched by this port).

## Memory measurements (4-layer prefix, one 7900 XT 20 GB, steps=2)

Real recipe geometry: rank 512, `both` placements, `--balance bias`,
`--kd-tail-weight 2.0 --kd-tailcond-weight 3.0`, top-512 + tail-64 synthetic
cache (values random; LM/KD numbers meaningless by design).  Peak =
`torch.cuda.max_memory_allocated` over the whole process.

| condition | 2k | 4k | 8k |
|---|---|---|---|
| dense v1 (seed + quantise-in-place) | OOM (peak 18.3, +1.9 needed) | OOM | OOM |
| dense + layer ckpt | 15.91 OK | OOM (19.14, +1.9 needed) | OOM |
| compact + ckpt g64 | 15.45 OK | OOM (16.48, +3.8 needed) | OOM |
| compact + ckpt g64 + chunked CE | not run | **13.90 OK** | OOM (18.64 peak, +1.74 needed, 2.72 reserved-unallocated fragmentation) |

Resident model after build (compact): banks 0.80 GiB (codes 0.75 + scales
0.05) vs 6.0 GiB dense bf16; other params 2.16 GiB; branches 0.078 GiB;
routers 0.004 GiB.  Peak during build: 9.16 GiB (the 4-layer prefix load).

Decomposition at 8k (memory-history snapshot + OOM messages): resident model
~3.0 GiB, full-vocab logits + grad ~7.6 GiB (2 x 3.79 GiB, the binding
contiguous blocks), checkpoint boundaries ~0.13 GiB, GDN recompute transient
~2.5 GiB (fp32 `mamba_ssm_dtype` states, one layer at a time), LSE chunks
0.25 GiB.  The KD tail terms are NOT the binding term at 8k (same failure
with `--no-tail`).

## Extrapolation to the full 40-layer model (what is measured vs assumed)

Measured/derived: compact banks 8.56 GiB (codes+scales; = the audit's expert
payload), other bf16 ~5.0 GiB -> resident ~13.6 GiB vs 69.4 GiB dense (the
dense model cannot fit any single card locally at all).  Vocab logits+grad
and GDN transients are layer-count independent; boundary saves scale x10
(assumed linear): 0.13 -> 1.3 GiB at 8k.

| seq | full-model estimate | 20 GB card |
|---|---|---|
| 2k | ~18.3 GiB | marginal |
| 4k | ~20.5 GiB | no (24 GB class) |
| 8k | ~25-28 GiB | no (32-48 GB class) |
| 16k | ~34+ GiB (logits+grad alone 15.2) | 48 GB class |

This is the number step E sizes the pod with: **4k/8k windows on the full
model need a 24-48 GB card; the three switches make the model itself fit
anywhere >= 16 GB.**  Explicit assumptions: activation-side x10 linearity,
no flash-attention regression on the pod, teacher in a cache (never
resident).

## Identified levers, not applied (with reasons)

- `causal-conv1d` + `flash-linear-attention`: in `box-run-v2.sh setup`; fla
  0.6.0 is installed locally but causal-conv1d does not build here (native
  ROCm extension).  The GDN fallback conv path is part of the 8k gap and
  dominates step time (the warning fires every run).
- `mamba_ssm_dtype bf16` (the host config pins fp32): would cut the ~2.5 GiB
  GDN recompute transient, but changes attention-path numerics - a decision
  for the experiment freeze, not a default.
- Sequence-split accumulation (two half-window passes per step): the lever
  for 16k windows and for the logits-grad pair; changes loss-accumulation
  order and interacts with the bias-balance update - defer to the
  experiment design.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` (set in the runbooks): on
  this ROCm 7.2 build it reports lower "allocated" but hits driver mapping
  failures; the plain allocator measured more honestly.  Re-evaluate on the
  pod's CUDA runtime.

## Reproduce

```bash
# bit-exactness gate (GPU 1, ~3 min)
cd work/scion
MOE_ARTIFACTS=../models-p310/empero-prefix \
SCION_GGUF=../models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf \
../.envs/scion-rocm/bin/python v2/scripts/prefix_gate.py --device cuda:1

# one memory condition (~2 min each)
MOE_ARTIFACTS=../models-p310/empero-prefix \
SCION_GGUF=../models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf \
../.envs/scion-rocm/bin/python v2/scripts/prefix_mem_probe.py \
  --seq 4096 --mode compact --ckpt --group-size 64 --chunked-ce --device cuda:1

# unit tests (28)
../.envs/scion-rocm/bin/python -m pytest moe/tests/test_qwen35_compact.py -q
```

Files: `moe/qwen35_moe_proxy.py`, `moe/seed_banks_from_gguf.py`,
`moe/kd_loss.py`, `moe/tests/test_qwen35_compact.py`,
`v2/scripts/prefix_gate.py`, `v2/scripts/prefix_mem_probe.py`.
