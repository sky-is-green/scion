# The MoE extension: does the ternary recipe survive routing?

Status: research note, 2026-09-25.  This document re-frames the dense-27B
forensics (Parts 1 and 2 of the public write-up) onto the question that
follows it: **can ternary-class compression reach Bonsai-level retention on a
Mixture-of-Experts model?**

The dense analysis is settled and is not repeated here (see `WHITEPAPER.md`,
`FORENSIC-ARCHIVE.md`, `FAILURES.md`).  The one-line summary that matters for
this extension: the container is public, the residual is ordinary
rotation-in-the-loop ternary QAT, and the public PTQ route collapses
off-calibration (1,178x to ~35,000x).  Quality is trained placement.

## 1. Why MoE is a different problem

Dense models have one failure mode under ternarisation: the per-weight
approximation error, absorbed by the redundancy of the whole network.  MoE
models have two, and the second one is not a capacity problem:

1. the expert weights are ternarised (same as dense), and
2. **routing is a discrete decision** made from the hidden state, so any drift
   in that state flips expert selection and compounds downstream.

97% of MoE weights are dormant per token, so per-token gradient coverage is
sparse, and top-k selection is non-differentiable.  Neither property exists in
the dense case.

## 2. Measurements (all local; the harness is in [`moe/`](../moe/) and the result JSONs under [`moe/results/`](../moe/results/))

### 2.1 Routing drift is real and compounds

Prefix probe on a 35B-A3B-class target (`qwen3_5_moe`, 256 experts, top-8 +
shared; embedding + layers 0-3; experts rotate+absmean RTN at g128; router and
attention exact):

| layer | router top-8 agreement vs FP | hidden-state rel L2 drift |
|---|---|---|
| 0 | 1.000 | 0.202 |
| 1 | 0.807 | 0.311 |
| 2 | 0.760 | 0.346 |
| 3 | 0.652 | 0.373 |

Expert weight rel err 0.511, zero share 0.309.  Layer 0 cannot flip (its input
is the exact embedding), so 1.000 is a correctness check.  From layer 1 the
router weights are still exact; the flips come entirely from hidden drift.

### 2.2 Rotation is not the missing lever for MoE

Rotating the expert banks before RTN does not improve the quantizer:
weight rel err 0.5156 (raw) vs 0.5109 (rotated), zero share 0.309 in both.
In-place RTN on OLMoE-1B-7B: PPL 11.02 -> 40,121 (3,640x) raw, 56,611 (5,135x)
rotated.  The archive's own Gate-2 result (Prism's exact basis + RTN on the
dense 27B -> 1,270x) predicted this: the basis buys the container, the trained
placement buys the model.

### 2.3 In-place QAT at local budgets does not recover it

Full-model bf16 QAT (6.4B trainable, 0.5M tokens): 1.9x recovery, routing
unchanged (0.463 -> 0.464).  fp32 masters on a 4-layer subset removed the
precision question but training still did nothing (23.94 -> 23.79).  bf16
masters partially swallow the updates (bulk update RMS ~5e-6 vs bf16 ULP
~6e-5), but precision is not the binding constraint.

### 2.4 Trained corrections recover it, and placement decides how much

TAARDIS-style low-rank correction branches on the same frozen RTN body, same
loss (LM + output KD + router KD), same 1024 steps unless noted:

| arm | trainable | PPL (8 windows) | router agreement |
|---|---|---|---|
| RTN, no corrections | - | 40,121 | 0.463 |
| per-layer branches (residual stream), rank 64 | 6.29M | 253.85 | 0.506 |
| per-layer branches, rank 256, 2048 steps | 18.9M | 85.50 | 0.674 |
| per-layer branches, rank 512, 4096 steps (512-window recipe) | 35.7M | 71.14 (best, step 2000; 75.64 at 4096) | 0.710 (0.726 at 4096) |
| per-expert branches (inside experts), rank 8 | 52.4M | 6,551.74 | 0.425 |

Trainable counts include the trainable routers (2.1M of the totals).  The
per-expert arm was evaluated in a separate pass whose RTN baseline was 39,727;
re-evaluating the same frozen body at this collapsed scale moves the number by
about 1%.

8x more parameters inside the experts, 25x worse PPL, and routing ends below
the untrained baseline.  Mechanism: routing is decided from the residual stream
entering the MoE block, so a correction there can steer the next layer's
decisions; a correction inside an expert acts after the decision and cannot
influence it.

**Capacity in the right place keeps helping, with sharply diminishing
returns.**  Rank 64 -> 256 took PPL from 253.85 to 85.50 and routing agreement
from 0.506 to 0.674; rank 512 improves the best 8-window PPL again, to 71.14
(564x recovery from the in-place collapse), on 35.7M trainable parameters
(33.6M branches plus routers) against a 6.4B frozen body.  But the rank-512
run turns over: PPL is 71.14 at step 2000 and 75.64 at step 4096, while router
agreement keeps climbing (0.710 -> 0.726).  More capacity buys a lower floor;
the last stretch is not a capacity problem.  At the rank-512 best the teacher
gap is 6.5x (11.02 vs 71.14), the best checkpoint is not the last one, and the
remaining levers are schedule and data, below.

**Design rule: for the routing bottleneck, corrections must live on the
residual stream, not inside the experts.**  TAARDIS's per-matmul placement was
derived on a dense model with no routing, so it does not transfer.

### 2.4a The last levers: schedule, data, and the router-KD control

Two levers — LR decay and more unique distillation data — took the same
rank-512 recipe from 71.14 to 37.11 (8-window PPL, 1081x recovery, teacher gap
3.4x):

| change | PPL (8 windows) | router agreement |
|---|---|---|
| rank 512, 512 windows, constant LR (reference) | 71.14 (best, step 2000; 75.64 at 4096) | 0.710 |
| + LR decay after half-time | 68.32 (step 4096) | 0.721 |
| + 2,048 unique windows (2 epochs) | 51.59 (step 4096) | 0.726 |
| + 4,096 unique windows (1 epoch) | 49.27 (step 4096) | 0.724 |
| **+ 8,192 unique windows (1 epoch, decay from half-time)** | **37.11 (step 8192)** | **0.752** |

The 512-window run was overfitting to the fixed cache: PPL turned over after
step 2000 (71.14 -> 75.64) while routing agreement kept rising.  LR decay
removed the turnover; more unique distillation text removed most of the
residual gap.  The data slope is still positive: -24% for the first 4x of
data, -4% for the next 2x at fixed steps, and **-25% for 8,192 windows when
the step budget is doubled with it** (37.11, still descending at 8,192).  The
port's data budget should scale with its compute budget, not stop at the proxy
budget.

The router-KD term turned out to be a no-op on this stack: a `--router-weight
0` control matched the KD-on trajectory within ~1.5% at every checkpoint,
including the same late turnover.  Routing-agreement recovery comes from
repairing the state the router reads, not from distilling its decisions.  The
recipe drops the term.

### 2.4b The AUTOGRID map (noise-floor classification)

CodeMasterCody3D/autogrid (MIT) classifies every tensor as FREE@k (conversion
quieter than the container's own rounding), TERNARY (already on the grid), or
STEER (needs a steered pipeline).  Run locally on OLMoE:

| model | max-k | result |
|---|---|---|
| OLMoE FP | 8 | 100% free, recommended k=6 (err 2.3e-3 vs bf16 floor 3.9e-3) |
| OLMoE FP | 2 | **100% STEER** (0 free, 0 ternary) |
| OLMoE ternary build | 2 | **93.1% already ternary** (the 3,072 expert tensors, 6.44B params); 6.9% STEER (attention, embeddings, head, norms, router, left FP by design) |

The k=2 row states the same boundary this document reaches from the other
side: at ternary budgets nothing converts for free, and the STEER set is
exactly what the correction pipeline has to carry.

### 2.4c The sidecar at 2 bits

The design rule puts the corrections on the residual stream; the remaining
artifact question is whether they can be stored at ternary-class bit widths.
TAARDIS V3 ships its branches ternarised per rank component at ~2 bits per
factor.  Both that per-rank format and our g128 expert format were trained with
straight-through estimation on the rank-512 recipe.  At the 512-window
operating point, at matched steps:

| branch format | PPL | router agreement | deployed size (35.7M branches) |
|---|---|---|---|
| fp32 (reference) | 71.14 | 0.710 | 134 MB |
| g128 ternary, post-hoc | 1,014.34 | 0.560 | 8.9 MB (2.125 bpw) |
| per-rank ternary, post-hoc | 1,759.11 | 0.528 | 8.4 MB (2.004 bpw) |
| **g128 ternary, STE-trained** | **93.47** | **0.651** | **8.9 MB (2.125 bpw)** |
| per-rank ternary, STE-trained | 107.18 | 0.626 | 8.4 MB (2.004 bpw) |

Post-hoc ternarisation alone is not viable (14-25x PPL).  Training in the
deployed format recovers most of it: the g128 sidecar costs 1.31x over the fp32
reference (93.47 vs 71.14 at the same 2000-step point) and the per-rank format
1.51x.  The cost is not a constant: on the stronger schedule and data of §2.4a
the g128 sidecar reaches 76.75 at 4,096 steps (1.56x) and then plateaus at
**73.67 by step 8,000 — 1.50x converged** against the 4,096-window fp32 run
(49.27), 545x recovery from the in-place collapse.  The plateau says the
ternary constraint is a representational tax at that operating point, not an
optimisation shortfall.  (The fp32 recipe has since improved to 37.11 on
8,192 windows, §2.4a; the sidecar at that newest operating point is not yet
measured.  Repeats of the 4,096-step point differ by ~3%, so the ratio is
~1.5x rather than an exact figure.)  Either way the sidecar is ~9 MB, under
0.5% of a ternary artifact, so the ~2 bpw total claim holds; g128 is the
better size/fidelity trade at equal training.

**The tax is edge-dominated, and reducible.**  A post-hoc scan of the
8,192-window model ternarises one layer's branch at a time: layers 0 and 15
cost ~3.3 PPL each, layer 13 only +0.09, the rest +0.1 to +2.8.  Single-layer
costs are not additive (all-ternary post-hoc is catastrophic), so the test is a
retrain.  On the same recipe, at the same 8,192-step budget:

| branch format | PPL (8 windows) | deployed (35.7M branches) |
|---|---|---|
| all g128 ternary | 73.67 (1.50x) | 8.9 MB (2.125 bpw) |
| mixed: layers 0, 15 fp16 | 67.43 (1.37x) | 24.6 MB (5.86 bpw) |
| **mixed: layers 0, 1, 3, 15 fp16** | **62.84 (1.28x)** | **40.2 MB (9.59 bpw)** |
| fp32 reference (4,096-step run) | 49.27 | 134 MB |

Each fp16 layer buys back ~2.5-3 PPL for ~8.4 MB, so the tax is tunable almost
linearly, and even the 4/16 mix stays a small fraction of a ternary 35B build
(~0.6%).  The choice for the port is a size/quality dial, not a binary:
2 bits everywhere, a handful of sensitive layers at fp16, or all fp32 if the
~2 bpw total claim is relaxed.

### 2.4d The deployable quantizer

The TAARDIS fork's ternary container (`Q1_0_g128`) does not use a one-shot
absmean scale: its quantizer refines each group scale to the Lloyd-Max / TWN
fixed point (default-on).  On OLMoE experts that rule gives weight rel err
0.442 against 0.517 for one-shot absmean (zero share 0.465 vs 0.312).

Everything above this point used the one-shot absmean rule, so the collapse
was measured against a harder base than deployment.  Re-running the recipe
with the deployable quantizer:

| stage | PPL (8w) | router agreement |
|---|---|---|
| FP teacher | 11.02 | 1.000 |
| RTN (deployable quantizer) | 804.39 | 0.691 |
| **+ rank-512 branches, g128 ternary sidecar** | **19.80** | **0.846** |

That is a **1.8x teacher gap** with a 2.125 bpw body and an 8.9 MB sidecar,
against 3.4x for the absmean-trained recipe.  The container matches too: a
GGUF with ternary experts and Q8 attention measures 566 uncorrected in the
fork, and its expert codes reproduce the torch Lloyd rule to 0.001 rel err.

### 2.4e Branch delivery: standard LoRA vs runtime op

Two placements were trained against the deployable quantizer (8-window PPL,
teacher 11.02):

| placement | PPL (8w) | router agreement | delivery |
|---|---|---|---|
| RTN (deployable quantizer) | 804.39 | 0.691 | — |
| `moe_out`, rank 512 / 1024 | 19.80 / 19.57 | 0.846 / 0.850 | fork extension (validated) |
| `attn_out`, rank 512 / 1024 | 21.42 / 20.93 | 0.828 / 0.829 | standard llama.cpp LoRA |
| both placements, rank 512 each | **19.11** | 0.847 | LoRA + fork extension |

`attn_out` costs 1.7 PPL (~8%) against the best placement and needs no
runtime change: the branch is a rank-512 update on `attn_output.weight`, which
the runtime already applies from a LoRA adapter.  Exporting ternarises the
factors with the deployed rule; `--dtype q1_0_g128` packs them in the fork's
native 2.125 bpw container (8.9 MB for 16 layers, bit-exact with the f16
export).  The trained routers ride along as exact rank-64 `ffn_gate_inp` LoRA
pairs (dense f16, +4.3 MB); they are part of the evaluated model and are worth
including.

The `moe_out` branch has no base tensor to attach to, so the fork grows a
small extension: the adapter may carry `blk.N.ffn_moe_out.weight.lora_a/b`,
and the OLMoE graph adds `lora_b(lora_a(normed_block_input))` to the MoE output
(`build_lora_branch` in `llama-graph.cpp`, hooked in `models/olmoe.cpp`; fork
branch `moe-corr-runtime`, ~90 lines).  The placement gain transfers to the
runtime: the rank-512 `moe_out` adapter reaches 14.63 with routers against
15.77 for the `attn_out` adapter on the same 563-chunk protocol.

Training the two placements jointly (rank 512 each; the same 22.2 MB adapter
as `moe_out` rank 1024) beats either alone — 19.11 / 0.847 on the 8-window
protocol — because the `attn_out` branch repairs the attention path before the
residual add while the `moe_out` branch corrects the block output.  A small
hyperparameter screen kept the inherited temperature (2.0) and LR schedule and
rejected faster decay, rank 2048, and AdamW (which diverges at the shared LR);
`--kd-weight 1.0` was the one win over the inherited 0.5 (21.78 vs 21.99 on
the screening protocol).  Rank 512 per placement is sufficient for the
combined model: rank 1024 each gives the same 19.11 on the 8-window protocol
at 2x the adapter size.

The deployed-format tax on this base is small: training the same `attn_out`
branches in fp32 (no STE) gives 21.11 vs 21.42 on the 8-window protocol (1.4%),
so the 8.9 MB compact adapter is the right trade against a 67 MB dense one.
Rank is another dial: rank 1024 reaches 20.93 / 0.829 at 2x the sidecar
(17.8 MB compact), closing a third of the `moe_out` gap.  The data slope,
however, is flat under Lloyd: 8,192 unique windows at matched steps did not
beat 4,096 windows x 2 epochs (2-window step 5000: 23.99 vs 23.85), so the
4,096-window budget is sufficient for this recipe.

End-to-end in the TAARDIS fork (mixed Q1-expert/Q8-rest GGUF,
`wiki.test.raw`, c512, 563 chunks, GPU):

| build | PPL | size |
|---|---|---|
| mixed GGUF, uncorrected | 566.7 | 2.12 GB |
| + `attn_out` branches (rank 512) | 16.17 | +8.9 MB |
| + `attn_out` branches + routers (rank 512) | 15.77 | +13.2 MB |
| + `attn_out` branches + routers (rank 1024) | 15.61 | +22.2 MB |
| + `moe_out` branches (rank 512) | 14.87 | +8.9 MB |
| + `moe_out` branches + routers (rank 512) | 14.63 | +13.2 MB |
| + `moe_out` branches + routers (rank 1024) | 14.61 | +22.2 MB |
| + both placements + routers (rank 512 each) | **14.48** | +22.2 MB |

CPU and HIP agree (uncorrected 566.59 vs 566.74; corrected within 0.002 PPL),
and the ternary expert kernels plus the fork extension run on the local GPUs
at ~10x the CPU speed.
The correction transfers to the GGUF base at the same ratio as the harness
(~35x on the sequential-chunk runtime protocol vs ~38x on the harder
random-window harness protocol), so the runtime absolute PPL is lower than the
harness number; the two protocols are not directly comparable.

### 2.5 Where the cheap route already works

MoTE-style up-cycling (pretrained FFN kept as a frozen BF16 shared expert,
ternary routed experts on top): 1.048x with **zero training**, and training the
experts further on a small corpus made it worse (1.122x).  The frozen FP
component carries the function; the ternary experts only learn a correction.
For Qwen3.6-35B-A3B this is not directly applicable (its shared expert is
~0.4% of weights), so it implies re-architecture rather than in-place work.

## 3. Prior art, and the cliff

- **APEX** (localai-org/apex-quant): MoE-aware mixed-precision PTQ; 35B-A3B at
  10.9-12.2 GB with ~98% benchmark retention.  Bonsai-class retention on a MoE
  already exists, at ~2.8 bpw rather than ~2.2.
- **MoTE** (arXiv 2506.14435): ternary routed experts on a frozen BF16 shared
  expert; matches/beats a full-precision MoE baseline at 1.5B and 3B.  Shared
  expert precision is worth 7.6 points.
- **DeepSeek-V4-Flash** (284B total, 13B active, natively FP8/FP4, heavily
  trained): per the MLX conversion measurements, **3-bit experts collapse the
  model (PPL 1.4e7, degenerate loops)** and 4-bit sits just above a cliff.  The
  same source finds non-expert weights ~50x more quantization-sensitive per
  parameter.  Over-training and parameter count do not buy tolerance at these
  bit widths.
- **EAQuant / ExpertQuant / RSPO**: independent confirmation that router
  sensitivity dominates MoE quantization loss, and that rank flips around the
  top-k boundary are the mechanism.

## 4. The research path

The target: `qwen3_5_moe` (35B total, 3B active), ternary experts, correction
sidecar, evaluated teacher-relative on held-out windows with the archive's
fairness rules.

| route | mechanism | cost | status |
|---|---|---|---|
| A | in-place ternary + trained residual-stream corrections | single 80 GB card or 2x40 GB for the correction run; no 263 GB fp32-master QAT | placement rule established; with the deployable quantizer: 19.80 (`moe_out`, fork extension validated) / 20.93 (`attn_out` rank 1024, standard LoRA) on the 4,096-window recipe, 13.2-22.2 MB sidecar; delivery verified end-to-end in the fork (566.7 uncorrected -> 14.63 with the extension, 15.61 as standard LoRA, on 563 wikitext chunks) |
| B | MoTE-style re-architecture (frozen FP component carries the function) | cheap training, larger artifact | demonstrated at 1.5B/3B |
| C | MoE-aware mixed-precision PTQ (APEX-style) | cheapest, ~2.8 bpw | published, Bonsai-class retention |

Route A is the one this work opens: it reaches ternary bit budgets without the
4xH100 QAT bill, because only the corrections train.  Open items:

1. correction capacity, schedule, and data on the residual stream (rank 512
   plus LR decay and more unique windows took the absmean recipe to 37.11; the
   deployable quantizer then took it to 19.80, and the data/step budget is
   still paying and should scale with compute),
2. router-aware losses: the router-KD term is a verified no-op (the
   `--router-weight 0` control matched within ~1.5%), so the correction repairs
   the state the router reads rather than distilling its decisions,
3. ternarising the correction sidecar: STE-trained ternary branches cost
   1.2-1.5x PPL over fp32 on the absmean base; the tax is tunable with a mixed
   format (2/16 and 4/16 fp16 layers reach 1.37x / 1.28x at 24.6 MB / 40.2 MB).
   On the deployable-quantizer base the all-g128 ternary sidecar is included in
   the 19.80 result, so the ~2 bpw total claim is met as shipped,
4. serving: ternary experts run today on the TAARDIS fork (verified on CPU and
   GPU: 2.12 GB mixed OLMoE GGUF, Q1_0_g128 expert banks, ~10x CPU speed on the
   local cards); branch delivery is resolved for both placements: `attn_out`
   ships as a standard LoRA (8.9-22.2 MB, 566.7 -> 15.61), and the better
   `moe_out` placement has a validated fork extension (branch
   `moe-corr-runtime`, ~90 lines; 566.7 -> 14.63).  Publishing the fork
   extension is the remaining step,
5. the iso-compute ladder for capacity-vs-compute separation (0.6B 51.9% and
   1.7B 63.6% on WikiText; the primary ladder closed without the 4B rung, see
   [`SCALING-PROTOCOL.md`](https://github.com/sky-is-green/bonsai2-ternary-forensics/blob/main/docs/SCALING-PROTOCOL.md)).

The port plan for the real target lives with the harness:
[`../moe/PORT-QWEN35.md`](../moe/PORT-QWEN35.md).

## 5. Reproduce

The harness is in [`../moe/`](../moe/) (see [`moe/README.md`](../moe/README.md)),
with the result JSONs under [`../moe/results/`](../moe/results/):

```
moe/
  gguf_header.py        remote GGUF header census over HTTP range requests
  role_map.py           MoE role map + ternary projection (8.82 GiB / 2.186 bpw)
  probe_router.py       E1 routing-drift prefix probe
  olmoe_proxy.py        in-place ternary QAT on OLMoE (negative control)
  moe_proxy.py          MoTE-style up-cycle proxy (Qwen3-1.7B)
  olmoe_rotate_rtn.py   rotation-vs-RTN quantizer comparison
  olmoe_corrections.py      per-layer residual-stream corrections (route A)
  olmoe_experts.py      per-expert corrections (placement control)
  eval_ckpts.py         checkpoint trajectory + router diagnostics
  save_ternary_olmoe.py materialise a ternary build to an HF dir
  qwen35_moe_proxy.py   qwen3_5_moe (35B-A3B) port: fused-bank STE patch, smoke
  branch_sensitivity.py per-layer sidecar sensitivity scan
  export_branches_lora.py pack branch checkpoints as llama.cpp LoRA adapters
  results/              the JSON evidence quoted in this document
```

The deployable recipe trains against the deployment quantizer:
`--quant lloyd` for the frozen body and the branches (the Q1_0_g128 rule), and
`--branch-quant g128` for the shipped sidecar.  See `moe/README.md` for the
full run sequence, including the `export_branches_lora.py` adapter step.
