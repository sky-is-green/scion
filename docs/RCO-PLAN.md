# RCO (exact-budget allocation): application plan — 2026-10-13

**Sources.** Paper: *Model Compression with Exact Budget Constraints via Riemannian
Manifolds*, Helcig & Alistarh, arXiv:2605.00649v3 (last revised 2026-09-27). Reference
code: `github.com/IST-DASLab/RCO` (Apache-2.0), cloned to `work/rco` at `9a1e09c` on
2026-10-13. Every number in this doc is labelled: **theirs** = as reported in the paper
(their hardware is not stated per experiment; wall-clocks as reported), **ours** =
measured on this box. Nothing in the "staged plan" has been run yet.

## What it is

The recurring decision: pick one of K options for each of N groups (layers, experts,
tables) under an exact total budget, minimizing the model's real loss - which does not
decompose across groups, so sensitivity + DP/ILP solvers can only optimize proxies, and
evolutionary search pays hundreds of full evaluations.

RCO relaxes the assignment to softmax logits over per-group options, which makes
`{C(alpha) = B}` a smooth Riemannian manifold with three cheap operations: a closed-form
normal, a monotone binary-search retraction along the cost vector, and one-inner-product
momentum transport. It wraps them around Adam; discrete feasible assignments come from
Gumbel-STE + multiple-choice-knapsack DP. The budget holds exactly at every iterate.

Theirs, measured:

| setting | RCO | best baseline | wall-clock |
| --- | --- | --- | --- |
| synthetic MCKP (N=1000, K=32) | within 1% of DP in 594 steps; \|C-B\| < 1e-8 | Lagrangian plateaus at 37.6% gap; \|C-B\| ~ 1e-1 | - |
| Qwen3-8B, 2.25 avg bits (GPTQ, KL objective) | FW PPL **20.47** | EvoPress 21.40, HIGGS 32.07 | 62 min vs 11-14 h |
| Qwen3-8B, 2.5 avg bits | FW PPL **15.43** | EvoPress 15.64 | 117 min vs 11-14 h |
| OLMoE-1B-7B, 25% expert pruning | 0.608 avg (300 steps) | EvoESAP 0.581 (5.8 h) | ~10 min at 50 steps |
| Qwen3-30B-A3B, 25% expert pruning | Avg **71.0** (96% recovery) | EvoESAP 66.5 | ~85 min vs 5.2 h |
| Qwen3-Coder-Next, 50% pruning (512 experts/layer) | HumanEval **72.0** non-uniform | 40.9 uniform; full model 74.4 | - |

The Coder-Next row is the headline for us: at 50% expert sparsity the gap between
uniform and non-uniform allocation is 31 points of HumanEval - allocation, not the
pruning criterion, is what matters. At 3.5-4 bits their advantage over surrogates
shrinks to wall-clock; the edge lives at high compression.

## Why it maps to this project

The shrink lane's closed experiments are exactly this class:

- **The prune lane is closed, and this is the retry it asks for.** P5 executed uniform
  25%/50% pruning end to end (REAP-instrumented fork patch, 2.4M-token coder map,
  paired pod gates; `CODER-VARIANT-GATES-25.md`). Verdict: 25% was code-neutral but
  cost general quality (PPL +11.6%, HS -2.75, WG -0.50, KLD +0.19); 50% uniform broke
  thinking - closed for the general release. The paper's central measured result is
  that *uniform allocation is the failure mode* (Coder-Next 50%: HumanEval 72.0 vs 40.9
  non-uniform vs uniform; Qwen3-30B-A3B 25%: 96% vs 90% recovery), so RCO is the
  non-uniform version of the same closed experiment. We already hold most of the
  ingredients: the 2.4M-token REAP map (a valid `--router-scores-file` initialization
  for RCO), the pod gate recipes, and the uniform baselines to compare against.
- A later non-uniform release (per-layer/per-expert type choices under a byte budget)
  is the same problem in a different option set. Our current practice (probe -> manual
  pick -> pod gates) is proxy-driven; the paper's premise - proxy allocation breaks at
  high compression - is why we gate everything.

Two instantiations, cheap to expensive:

- **A. Expert pruning** (keep/prune per expert). Needs NO quantized database: a bf16
  model + calibration data + cached reference log-probs. Produces a smaller model of
  the same architecture (masked, or physically pruned experts). This is the cheap path
  and it lands directly on the pruning seed already in `SHRINK-PLAN.md`.
- **B. Mixed-precision allocation** over per-(group, option) quantized weights. Needs a
  variant database like theirs (GPTQ 2..8 bits per linear, per-expert for MoE). For our
  containers this means a variant builder across our own types (Q2_0/PTQ1_0/Q4_0/PLE
  tables) plus pack work, and the database build is the expensive part (theirs: 8 GPUs
  for an 8B model).

## Requirements and gaps

- **ROCm**: pure PyTorch - the OBQ pipeline is `torch.linalg`/Cholesky, no CUDA
  extensions. The search is a plain process (`--device-map`, `--max-memory-per-gpu`);
  the DB build is `torchrun --nproc-per-node=N` (N=1 is fine). The box has
  torch 2.14.0+rocm7.2 in `.venv-rocm`; their `torch==2.11.0+cu126` pin is just a pin -
  to verify in S1.
- **Base checkpoints**: the 176B base is not local and not public; the 35B base is
  public (`empero-ai/Qwen3.8-35B-A3B-Distill`, the 35B release's `base_model`); small
  models come straight from the Hub.
- **Architecture support**: the pipeline is generic HF transformers + module hooks.
  Qwen3.8-Flash-Next support at transformers 5.7 is unverified - Qwen3-MoE support is
  what the repo pins for; S1 starts on a known-good model.
- **Calibration**: FineWeb-Edu is already in the HF cache; the code/math chat sets
  (`theblackcat102/evol-codealpaca-v1`, `allenai/tulu-3-sft-personas-math`) need
  downloads. Domain choice matters a lot (their ablation: coding calibration trades
  general knowledge away and vice versa) - for Scion, code-calibrated with a general
  holdout, matching our code-arm gate philosophy.
- **What does not fit**: costs must be linear in the assignment; inference-latency
  costs break the retraction monotonicity. RCO allocates size/quality, not serving
  speed.
- **Disk**: 225 GB free on this box. A 35B x 7-bitwidth DB would not fit and belongs on
  the pod; pruning needs no DB.

## Staged plan

**S1 - local small repro (queued; no pods; one free 7900 XT).** Goal: the toolchain
runs on ROCm and we have our own wall-clock; not a headline reproduction.

- S1a (pruning, no DB): `allenai/OLMoE-1B-7B-0125-Instruct` at 25%, 50 steps then 300
  steps, FineWeb-Edu calibration (the repo's `run_search_prune.sh` defaults). Theirs:
  ~10 min at 50 steps; ours to be measured. VRAM watch: default batch 4 x 2048 on a
  13 GB model in 20 GB - drop `--batch-size` if it bites.
- S1b (quantization, DB path): `Qwen/Qwen3-4B` DB at 2..8 bits with
  `torchrun --nproc-per-node=1`, then the 2.5-bit search. Theirs at 8B: 62-215 min.
- Record here: wall-clock, budget error, KL curve, ROCm fixes. Deliverable: this
  section with our numbers.

**S2 - Scion-35B-A3B pruning (pod; ask before booking).** Pruning needs no DB: bf16 35B
(~70 GB) on 2x A6000 (96 GB) with `--device-map balanced` or `--max-memory-per-gpu`,
200-300 steps, g=4, 64x2048 calibration, code + general mix. Theirs at 30B: ~85 min ->
a few GPU-hours. Then materialize the mask, convert to our container/pack, and run the
local PPL instrument; if it heads anywhere, our usual gates apply (`SHRINK-PLAN.md`
gate order; pod KLD/HS/WG; paired code arms).

**S3 - decide after S1/S2.**

- The direct prize: **Flash-Next 176B at 25% non-uniform pruning** - the P5 retry,
  initialized from the existing 2.4M-token REAP map and judged on the same paired
  gates. Gaps to close first: an HF-loadable bf16 checkpoint for the RCO pipeline (our
  artifact is the custom pack format) and a large pod.
- A 35B mixed-precision DB is a bigger pod job and needs the variant-builder work
  first. Neither starts before S1/S2 evidence exists.

## Risks / open questions

- STE bias: convergence is empirical (their own limitation); S1 must show the KL curve
  actually descends on our stack before S2 spends anything.
- Their wall-clocks are their hardware; our S1 numbers replace them for planning.
- Pruning quality is architecture-dependent; Scion's routers/experts differ from their
  models. The gates decide, not the paper.
- Instantiation B needs a variant-DB builder across our type set before the search can
  even run on our format.

## S1 recipe (queued - run when a card frees)

```sh
cd /run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/rco
# one-time env (no GPU needed). Match the box's ROCm wheel; see .venv-rocm's torch.
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
.venv/bin/pip install numpy==2.3.5 tqdm transformers==5.7.0 accelerate datasets
# (compressed-tensors only for the compressed-tensors I/O paths; lm_eval for benchmarks)

# S1a - OLMoE pruning, 50 steps first, then 300 for the paper's datapoint.
.venv/bin/python rco_search_prune.py --model allenai/OLMoE-1B-7B-0125-Instruct \
  --target-sparsity 0.25 --n-steps 50 --n-gumbel-samples 4 --antithetic \
  --batch-size 2 --calibration-data fineweb_edu --calibration-samples 64 \
  --save-mask /tmp/opencode/olmoe-25-step50.pt

# S1b - Qwen3-4B multi-bitwidth DB + 2.5-bit search.
.venv/bin/torchrun --nproc-per-node=1 run_quantize.py --model_name_or_path Qwen/Qwen3-4B \
  --quantizable_modules '.*(q|k|v|o|gate|up|down)_proj$' --pre_block_modules model.embed_tokens \
  --block_modules model.layers --post_block_modules model.norm lm_head \
  --calibration_data fineweb_edu --calibration_samples 256 --calibration_seq_length 2048 \
  --bitwidth_options 2 3 4 5 6 7 8 --calibration_bitwidth 4 --group_size 128 \
  --save_dir /run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/rco-data/qwen3_4b_db
.venv/bin/python rco_search_quant.py --model Qwen/Qwen3-4B \
  --layer-dir /run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/rco-data/qwen3_4b_db \
  --bitwidths 2,3,4,5,6,7,8 --target-avg-bits 2.5 --n-steps 200 --lr 0.1 --tau-min 0.01 \
  --n-gumbel-samples 4 --batch-size 1 --batches-per-step 1 --gradient-checkpointing \
  --weightstore-cache lazy --calibration-data fineweb_edu --calibration-samples 256 \
  --device-map balanced --save-json
```

Box note: both cards are normally held by `scion-flashnext` (see
`P4-STRATA-FIT.md`); run S1 only while that service is stopped.
