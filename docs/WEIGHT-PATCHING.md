# Weight patching / weight-space programming — what it is, and what it means for Scion

Status: investigation note, 2026-11-06.  The first two Scion probes have now run
(layer-level and expert-bank split, below); the per-expert screen and the
targeted-vs-sidecar comparison are still open.

## 1. Two different things are being conflated

**Weight Patching** (arXiv 2604.13694, Sun et al., USTC) is a *mechanistic
localization* method.  Given a base model and a behavior-specialized counterpart
with the same architecture, it replaces selected parameter slices (an attention
head's Q/O, an MLP neuron's gate/up/down columns) from the specialized model into
the base model under a fixed input, and measures whether the transplant recovers
a behavioral anchor state.  It also gives a first-order attribution score,
`sum_p (theta_sft - theta_base)_p * dF/dtheta_p`, to screen components without
running every exact patch.  The recovered component scores can guide
component-wise model merging.  It is not an LLM writing weights, and it is not a
training-free capability editor.

**Weight-Space Programming** ("Sparks of Weight-Space Programming", Takehi et
al., MIT CSAIL, 2026-11-02) is the viral claim: a frontier LLM reads raw weights
and writes edits directly, with no gradients.  The headline result is
Qwen3.5-0.8B MATH-500 7.9% -> 48.1% after editing 0.016% of the weights.  This
is a different technique from the arXiv paper, although both are parameter-space
interventions.

## 2. What the weight-space-programming evidence actually shows

The Qwen result is real but narrower than the headline:

- The base 0.8B model loops; 93% of its MATH-500 answers hit the 32k token
  limit.  Astra traced the loop to one direction written by the MLPs of layers
  15 and 19 and subtracted it from 64 output columns in each (131,072 weights).
- The edit removes the loop, so the model can finish answers it could already
  work out.  It is not adding mathematical capability.
- The same blog reports that on Gemma 3 270M, which rarely loops, the edits did
  not improve MATH-500 beyond noise: "it could remove an obstacle that hid an
  ability, but it could not add an ability the model lacked."
- A controlled replication by Kevin Bondzio (`kevinpl07/weight-patching`) finds
  the same shape: the edit works as a narrow patch on a frozen model when the
  model already represents the bad case (probe AUC 0.99-1.0); it does not
  survive 20 steps of unrelated fine-tuning; with enough data LoRA fixes more of
  the bug, at the cost of changing other answers.  Much of the recipe is classic
  probing + steering, implemented as a one-neuron write.
- Critiques describe the Qwen result as a refusal/loop-direction ablation with
  standard ML engineering.  That is not a dismissal: a targeted direction
  removal that leaves 69-100% of normal outputs unchanged is useful.  It is just
  not a general weight optimizer or self-improvement loop.

The practical repos:

| repo | what it is |
|---|---|
| `kevinpl07/weight-patching` | the controlled recipe: probe -> first word -> one-neuron edit; four real bugs on 1B-4B models; honest LoRA comparison |
| `vukrosic/llm-weight-editing` | Opus writes weights for toy rule tasks and small MLPs; mixed results; explicitly warns it has not been tried on a language model |
| `RikiyaT/lm_understand-logs` | the original MIT logs and edited Qwen3.5-0.8B checkpoint |

The arXiv Weight Patching paper has no public code at the time of writing
("Request Code" on the paper page), but the method is simple enough to
reimplement.

## 3. Why direct editing is a poor fit for Scion

Scion's body is ternary: each expert weight is `{-s, 0, +s}` with one fp16 scale
per 128 weights.  The viral edits assume full-precision weights where a small
real-valued change is representable.  On a ternary body:

- a real-valued weight edit either breaks the container or has to be encoded as
  a code/scale flip, which is a coarse intervention;
- a dense patch would have to live in a sidecar (which is exactly what Scion's
  correction branches already are);
- the damage we care about is broad quantization/routing error, not one loop or
  refusal direction;
- reading 35B/176B raw weights is impractical for the agent; the published
  experiments are <=4B or toy networks.

So "Claude edits the frozen ternary weights" is not the useful framing.

## 4. Why parameter-space localization *is* a good fit

The useful adaptation is the arXiv method's *localization* half, pointed at
quantization instead of instruction following:

1. **Quantization sensitivity map.**  Pair the FP teacher with the ternary
   student.  For each candidate component (attention head, MLP neuron, expert,
   or whole layer), compute the first-order score
   `(theta_teacher - theta_student) * dL/dtheta_student` for a held-out loss
   (LM or output KD), then validate the top candidates with exact patching:
   replace the student slice with the teacher slice and measure PPL/KLD/router
   agreement recovery.
2. **Allocation.**  A ranking of components by recovered loss tells us where to
   spend mixed precision, which layers deserve fp16 sidecars, and where the
   correction branches should be placed.  This is a finer-grained version of
   `moe/branch_sensitivity.py` and the RCO allocation map.
3. **Checkpoint merging.**  The paper's WP-guided component fusion can be tested
   on Scion's existing checkpoint soups: score each component's contribution,
   merge per component instead of uniform averaging, and compare PPL/KLD.  This
   is cheap and uses checkpoints we already have.
4. **Narrow release fixes.**  If a released model has a specific loop/refusal/
   format bug, use the probe to find the direction, then implement the fix as a
   small adapter or scale update rather than a raw ternary edit.

## 5. First Scion results (OLMoE, 2026-11-06)

The student is the Lloyd/g128 ternary body; attention and routers are already
FP, so every patch below is an **expert-bank** patch.  Layer-level exact
patching copied one layer's `gate_up_proj` + `down_proj` from the FP teacher into
the ternary student:

| layer patched | PPL | gap recovered |
|---|---:|---:|
| L2 | 120.77 | **86.0%** |
| L0 | 124.57 | **85.5%** |
| L3 | 127.21 | **85.2%** |
| L8 | 166.70 | 80.2% |
| L1 | 195.86 | 76.4% |
| L9 | 219.22 | 73.5% |
| L10 | 300.35 | 63.1% |
| L11 | 335.63 | 58.6% |
| L12 | 515.70 | 35.7% |
| L15 | 607.34 | 24.0% |

Baseline ternary PPL is 795.62; teacher is 11.02.  Restoring **one early layer's
expert banks** recovers ~85% of the gap.  Late layers recover much less.  Router
agreement rises only to ~0.77 even when PPL recovers 85%, so the expert fix
repairs the hidden-state cascade faster than it repairs routing.

Splitting the expert bank (same probe, early layers):

| layer | both | `gate_up_proj` | `down_proj` |
|---|---:|---:|---:|
| L0 | 85.5% | 71.6% | 22.9% |
| L1 | 76.4% | 38.7% | **70.9%** |
| L2 | **86.0%** | **79.5%** | 29.8% |
| L3 | 85.2% | **80.7%** | 14.5% |
| L8 | 80.2% | 69.5% | 17.4% |

`gate_up_proj` carries most of the early-layer recovery; `down_proj` is mostly
weak, with an odd L1 spike.  The practical read: the recoverable quantization
damage is concentrated in the first few layers' expert gate/up matrices, which is
a candidate target for mixed precision or targeted correction capacity.  A full
gate/up bank per layer is still large, so the open question is whether a subset
of experts inside those layers recovers most of it.

The per-expert group screen answered that question with a **negative** for the
simple rankings.  Ranking experts by quantization error and patching the top
8/16/32, then the bottom 8/16/32 as a control, gives layer-dependent and often
inverted results:

| layer | top-8 by error | top-16 | top-32 | bottom-8 | bottom-16 | bottom-32 |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 0.1% | 56.9% | 78.7% | 2.8% | 12.6% | 20.8% |
| L1 | 67.7% | 69.8% | 78.0% | 1.8% | 3.6% | 17.5% |
| L2 | 78.3% | 80.8% | 83.3% | 10.1% | 10.2% | 25.9% |
| L3 | 23.4% | 27.7% | 36.1% | **76.0%** | **77.7%** | **80.7%** |
| L8 | 23.9% | 36.6% | 48.3% | 3.9% | 61.3% | 65.5% |

Ranking by router load instead of error is no better: L2's bottom-16 by load
recovers 78.0% while its top-16 recovers 22.0%, and L0/L1 go the other way.
The mechanism is not "the high-error experts are the important ones"; patching a
group changes the hidden state, routing and expert interaction in a way that the
error/load proxy does not capture.  A targeted expert patch therefore needs a
proper gradient/activation attribution, not a weight-error ranking.  The robust
result is the layer/module map, not an expert ranking.

The obvious follow-up — give the early layers more correction rank and the late
layers less at matched total rank — also failed:

| allocation | ranks (layers 0-3 / 4-7 / 8-9 / 10-11 / 12-15) | PPL |
|---|---|---:|
| uniform frozen | 512 everywhere | **16.068** |
| early-heavy | 1024 / 512 / 512 / 256 / 128 | 16.277 |
| late-heavy (reverse control) | 128 / 512 / 512 / 256 / 1024 | 16.209 |

Both non-uniform maps are worse than uniform at the same total rank.  The
weight-patch sensitivity says where restoring **expert weights** recovers PPL;
it does not say where the **residual-stream branch** needs more capacity.  The
two interventions are not interchangeable.  Uniform rank 512 remains the right
sidecar allocation on this evidence.

## 6. Proposed next experiment (OLMoE, cheap)

- Teacher: FP OLMoE.  Student: Lloyd/g128 ternary body, no corrections.
- Held-out: 8-32 wikitext windows.
- Compute `dL/dtheta` for the student on the held-out loss, and the first-order
  score `(theta_FP - theta_q) * grad` for every attention head and MLP neuron
  (and, as a coarser unit, every layer and expert).
- Validate the top-k components by exact patching: copy the FP slice into the
  ternary student, measure PPL and router agreement.  Compare against random
  components of the same size.
- If a small set recovers a meaningful fraction of the 795 -> 11 PPL gap, test a
  targeted artifact: restore those slices at higher precision (or add a branch
  only there) and compare with the rank-512 all-layer sidecar at matched size.

The key question is not "can an LLM write ternary weights?" but "which parameter
slices carry the quantization damage, and is a targeted patch cheaper than the
uniform sidecar?"  That is a Scion-shaped question, and the arXiv method gives a
principled way to ask it.

## 7. Verdict

- **Weight-space programming:** interesting as a diagnostic and a narrow
  frozen-model patch; not a replacement for correction training, and not
  practical on a 35B/176B ternary body.
- **Weight patching:** the useful part for Scion.  It gives a parameter-space
  sensitivity map that can guide mixed precision, branch placement, and
  checkpoint merging.  Worth a cheap OLMoE experiment before any release work.
