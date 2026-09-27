# MoE ternary harness

Experiment harness behind [`../docs/MOE-EXTENSION.md`](../docs/MOE-EXTENSION.md):
in-place ternary quantisation of a pretrained Mixture-of-Experts model, the
router-drift probe, and the low-rank correction-branch trainers.

Two targets:

- **`allenai/OLMoE-1B-7B-0924`** — the frozen proxy body (16 layers, 64 experts,
  top-8). Everything trains here.
- **a `qwen3_5_moe`-class 35B-A3B prefix** — the real target; only the target
  census, role map, and routing-drift probe run against it in this harness.

## Layout

| file | role |
|---|---|
| `gguf_header.py` | remote GGUF header census over HTTP range requests |
| `role_map.py` | MoE role map + ternary projection (8.82 GiB / 2.186 bpw for the 35B distill) |
| `probe_router.py` | E1 routing-drift prefix probe |
| `olmoe_proxy.py` | in-place ternary QAT + teacher cache + KD loop + eval |
| `moe_proxy.py` | MoTE-style up-cycle proxy (Qwen3-1.7B) |
| `olmoe_rotate_rtn.py` | rotation vs RTN quantizer comparison |
| `olmoe_corrections.py` | per-layer residual-stream correction branches (route A); STE branch formats and mixed-precision sidecars |
| `olmoe_experts.py` | per-expert correction branches (placement control) |
| `eval_ckpts.py` | checkpoint trajectory + router diagnostics |
| `save_ternary_olmoe.py` | materialise a ternary build to an HF dir |
| `qwen35_moe_proxy.py` | 35B-A3B (`qwen3_5_moe`) port: fused-bank STE patch, prefix smoke, cache/train/eval stages |
| `branch_sensitivity.py` | per-layer sidecar sensitivity scan (which layers a mixed sidecar should keep fp16) |
| `export_branches_lora.py` | pack a branch checkpoint as a llama.cpp LoRA adapter GGUF |

`results/` holds the JSON evidence quoted in the write-up: per-run 8-window
evals, the E1 routing probe, the rotation comparison, the AUTOGRID scans, and
the 35B role maps.

[`PORT-QWEN35.md`](PORT-QWEN35.md) is the plan for the real target
(`qwen3_5_moe`, 35B-A3B): validated target inventory, required harness
changes, cache/build/correction plan, and the local smoke-test sequence.

## Setup

All scripts read and write one artifact root, `$MOE_ARTIFACTS` (default:
`moe/artifacts/`, gitignored):

```
$MOE_ARTIFACTS/
  olmoe-hf/     FP OLMoE-1B-7B (HF dir)
  olmoe/        outputs: teacher-cache.pt, correction checkpoints, eval JSONs
  canary-hf/    Qwen3-1.7B dense base (only needed by moe_proxy.py)
  empero-hf/    35B prefix checkpoint (only needed by probe_router.py)
```

```sh
pip install -r ../requirements.txt -r ../requirements-gpu-lock.txt
pip install -e ..          # bonsai_forensics (rotation/quant helpers used here)
export MOE_ARTIFACTS=$PWD/artifacts
huggingface-cli download allenai/OLMoE-1B-7B-0924 \
    --local-dir $MOE_ARTIFACTS/olmoe-hf
```

## Run

```sh
# 1. teacher cache: top-50 logits + router top-8 for the KD loss
HIP_VISIBLE_DEVICES=1 python olmoe_proxy.py cache \
    --windows 4096 --corpus-chars 50000000 --device cuda:0

# 1b. once: precompute the teacher router refs for the in-run eval so training
#     never loads the teacher (single-card training; the 'both cards' path
#     still works when --ref-file is omitted)
HIP_VISIBLE_DEVICES=1 python olmoe_corrections.py ref --device cuda:0 \
    --ref-file $MOE_ARTIFACTS/olmoe/eval-ref-w2.pt

# 2. per-layer residual-stream corrections (route A); single card with
#    precomputed eval refs (step 1b).  --quant lloyd matches the deployable
#    Q1_0_g128 quantizer, so training sees the deployment quantizer exactly.
HIP_VISIBLE_DEVICES=1 python olmoe_corrections.py train --device-map cuda:0 \
    --ref-file $MOE_ARTIFACTS/olmoe/eval-ref-w2.pt \
    --rank 512 --quant lloyd --branch-quant g128 \
    --windows 4096 --corpus-chars 50000000 --epochs 2 \
    --lr-half-every 500 --lr-decay-start 2000 --router-weight 0

# 3. held-out 8-window eval; single card
HIP_VISIBLE_DEVICES=1 python olmoe_corrections.py eval --rank 512 \
    --quant lloyd --branch-quant g128 \
    --load $MOE_ARTIFACTS/olmoe/olmoe-corr-r512-g128-step8192.pt

# 4. pack the branches as a standard llama.cpp LoRA adapter
#    (defaults ternarise the factors with the deployed lloyd/g128 rule, so the
#    adapter matches the eval; --dtype q1_0_g128 packs them in the fork's
#    native format, 8.9 MB instead of 67 MB, bit-exact with the f16 export;
#    add --routers --base-model <olmoe-hf> to also ship the trained routers as
#    exact rank-64 ffn_gate_inp LoRA pairs)
PYTHONPATH=<llama.cpp fork>/gguf-py python export_branches_lora.py \
    --load $MOE_ARTIFACTS/olmoe/olmoe-corr-r512-g128-attnoutlloyd-step8000.pt \
    --arch olmoe --target attn_out --dtype q1_0_g128 --out branches.lora.gguf

# placement control: corrections inside the experts instead of the stream
HIP_VISIBLE_DEVICES=0,1 python olmoe_experts.py train --device-map auto ...
```

The cache build and the training run must use the same `--windows` /
`--corpus-chars` / `--seed`, or the cached teacher targets will not line up
with the student inputs.  Body/branch quantizer: `--quant {absmean,lloyd}`
(`lloyd` is the deployable Q1_0_g128 rule; scale rule only, same 2-bit
storage).  Branch formats: `--branch-quant {fp32,g128,rank}` (STE-trained when
not fp32; `rank` is the TAARDIS V3 per-rank-component scheme).

`export_branches_lora.py` supports the `attn_out` placement only, because a
LoRA attaches to a linear tensor; the `moe_out` placement (the best measured
one) needs a runtime op and is not expressible as a LoRA.

## Ops notes

- One heavy GPU process at a time. Training needs both cards only for the
  brief teacher/student coexistence (`device_map auto` with the `max_memory`
  caps in the script); single-card stages (cache, eval) should pin the free
  card, not the display card.
- Run heavy jobs under `systemd-run --user --scope -p MemoryMax=..`; kill by
  PID, never by pattern.
- Corrections and masters must be fp32: bf16 masters swallow adapter-scale
  updates (bulk update RMS ~5e-6 vs bf16 ULP ~6e-5).
- With `device_map="auto"` the pre-patch experts `forward` is bound onto each
  instance; rebind through `patch_experts(group, model)`.
- `F.kl_div(..., reduction="batchmean")` on a 3-D tensor divides by batch only;
  reshape to `[-1, k]` first.
- The router-KD term is redundant on this stack: a `--router-weight 0` control
  matched the KD-on trajectory within ~1.5% at every checkpoint.
