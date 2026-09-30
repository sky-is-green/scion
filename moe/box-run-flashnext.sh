#!/bin/bash
# Flash-Next rental — one-shot box runbook (2xH200 FP8 cache, then 1xH200 train).
#
# Runtime is PINNED: qwen4_exp exists only on transformers main at commit
# f339035b (our 5.5.0 rejects the arch), and it needs tokenizers 0.23.x.
# The FP8 teacher is the official fine-grained fp8 release; on H200 (cc >= 8.9)
# the loader keeps 8-bit weights native and the `kernels` package supplies the
# fp8 matmuls -- watch for a silent BF16 fallback in the preflight report
# (`native_fp8_kept` must be true).
#
# Stages (one at a time; the volume keeps every artifact):
#   bash box-run-flashnext.sh setup       # pinned runtime + teacher download
#   bash box-run-flashnext.sh preflight   # loader smoke + 1 fwd + cache dry-run
#   bash box-run-flashnext.sh smoke       # 2-layer real-weights prefix smoke
#   bash box-run-flashnext.sh cache       # full teacher cache (~1-2 h)
#   bash box-run-flashnext.sh ref         # eval router refs
#   bash box-run-flashnext.sh train       # cur05 mirror (primary)
#   bash box-run-flashnext.sh eval        # prefix PPL + router agreement
#   bash box-run-flashnext.sh ple-ab      # PLE precision A/B (release pricing)
#
# Budget discipline: Flash-Next cap $55, stop at 80% ($44) and report; release
# the second GPU before training (the cache is on the volume).
set -euo pipefail

export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"
for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda; do
    if [ -d "$cuda_dir" ]; then
        export PATH="$cuda_dir/bin:$PATH"
        export CUDA_HOME="${CUDA_HOME:-$cuda_dir}"
        break
    fi
done
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MOE_ARTIFACTS="${MOE_ARTIFACTS:-/workspace/artifacts}"
export Q4_MODEL="${Q4_MODEL:-/workspace/models/qwen38-flashnext-fp8}"
REPO="${REPO:-/workspace/scion}"
TRANSFORMERS_COMMIT="${TRANSFORMERS_COMMIT:-f339035b}"
Q="$MOE_ARTIFACTS/qwen4exp"
MIXFILE="$Q/curric-combo.jsonl"
CACHE="$Q/prefix-top512-tail64.pt"
REF="$Q/eval-ref-w2.pt"
LOCKDIR="$MOE_ARTIFACTS/.stage-lock"
DEVICE_MAP="${DEVICE_MAP:-auto}"   # cuda:0 forces all non-PLE weights onto the GPU

stage="${1:?usage: box-run-flashnext.sh <setup|preflight|smoke|cache|ref|train|eval|ple-ab>}"

acquire() {
    if ! mkdir "$LOCKDIR" 2>/dev/null; then
        echo "REFUSING: $LOCKDIR exists (holder $(cat "$LOCKDIR/pid" 2>/dev/null))" >&2
        echo "one heavy stage at a time; if the holder is dead, rm -rf it" >&2
        exit 3
    fi
    echo "$$" > "$LOCKDIR/pid"
    trap 'rm -rf "$LOCKDIR"' EXIT
}

case "$stage" in
setup)
    nvidia-smi
    python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
    free -g | head -2; df -h /workspace | tail -1

    pip install "transformers @ git+https://github.com/huggingface/transformers.git@$TRANSFORMERS_COMMIT"
    pip install "tokenizers>=0.23.1,<0.24.0" "accelerate==1.15.0" "datasets==4.3.0" \
                "safetensors==0.8.0" "huggingface_hub==1.33.0" "numpy==2.5.3"
    # `kernels` pulls sigstore -> cryptography; the base image's Debian
    # cryptography has no RECORD and pip refuses to uninstall it (uninstall-no-record-file).
    pip install --ignore-installed cryptography kernels setuptools wheel
    pip install --no-build-isolation causal-conv1d flash-linear-attention || \
        echo "WARN: fast-path install failed -- expect the torch fallback (slower)"
    pip install "triton==3.8.0" || echo "WARN: triton pin failed"

    python - <<'EOF'
import transformers
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel
print("transformers", transformers.__version__, "qwen4_exp OK")
EOF

    [ -d "$REPO/moe" ] || { echo "REFUSING: repo not at $REPO -- upload the tree first" >&2; exit 4; }
    mkdir -p "$Q"

    if [ ! -f "$Q4_MODEL/config.json" ]; then
        pip install "huggingface_hub[cli]"
        hf download Qwen/Qwen3.8-Flash-Next-FP8 --local-dir "$Q4_MODEL"
    fi
    du -sh "$Q4_MODEL"
    echo "REMINDER: upload $MIXFILE (curric-combo.jsonl) before the cache stage."
    ;;

preflight)
    cd "$REPO"
    acquire
    python moe/qwen4exp_proxy.py preflight --model-dir "$Q4_MODEL" \
        --layers 1 --ple none --device cuda:0 --device-map "$DEVICE_MAP" \
        --top-logits 512 --tail-logits 64 --seq 512 \
        --out "$Q/preflight.json"
    echo "CHECK: native_fp8_kept=true, forward_s sane, cache_record_keys include"
    echo "idx/val/w/tidx/tlp. Any missing tensor = do not start the cache."
    ;;

smoke)
    cd "$REPO"
    acquire
    python moe/qwen4exp_proxy.py smoke --model-dir "$Q4_MODEL" --layers 2 \
        --ple rows --device cuda:0 --steps 10 --seq 256
    echo "check: hidden drift ~0.3-0.45, router top-10 agreement ~0.85-0.95"
    ;;

cache)
    cd "$REPO"
    acquire
    [ -f "$MIXFILE" ] || { echo "REFUSING: $MIXFILE missing -- upload it first;" >&2
        echo "the cache corpus must match the trained corpus exactly." >&2; exit 4; }
    time python moe/qwen4exp_proxy.py cache --model-dir "$Q4_MODEL" \
        --device-map "$DEVICE_MAP" --device cuda:0 --windows 4096 --corpus-chars 50000000 \
        --seq 512 --seed 0 --corpus-file "$MIXFILE" --agentic-frac 0.05 \
        --top-logits 512 --tail-logits 64 --fast-indexer --force-gpu \
        --save-every 500 --cache-file "$CACHE"
    ls -la "$CACHE"
    ;;

ref)
    cd "$REPO"
    acquire
    time python moe/qwen4exp_proxy.py ref --model-dir "$Q4_MODEL" \
        --device cuda:0 --device-map "$DEVICE_MAP" --fast-indexer --force-gpu --ref-file "$REF"
    ;;

train)
    cd "$REPO"
    acquire
    # device_map=auto + --force-gpu: an explicit cuda:0 map materialises the
    # 47.7 GiB PLE table on the card and OOMs (P1 trap); auto keeps it host-side
    # and force-gpu pulls the accelerate-offloaded blocks back on.
    # --compact-banks: the frozen banks live as 2-bit codes + fp16 scales and the
    # forward decodes per-expert slices, so the cur05-mirror fits one H200.
    time python moe/qwen4exp_proxy.py train --model-dir "$Q4_MODEL" \
        --device cuda:0 --device-map auto \
        --cache-file "$CACHE" --ref-file "$REF" --compact-banks --grad-checkpoint \
        --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
        --kd-weight 2.0 --kd-tail-weight 2.0 --kd-tailcond-weight 3.0 --temp 2.0 \
        --balance bias --log-entropy --fast-indexer --force-gpu \
        --corpus-file "$MIXFILE" --agentic-frac 0.05 \
        --windows 4096 --corpus-chars 50000000 --seq 512 --seed 0 \
        --epochs 1 --steps 4096 --eval-every 1000 --ckpt-every 1000 \
        --log-every 100 --tag cur05
    echo "kill criteria: abort on NaN, or a >50% deviation from the 35B shape"
    echo "(lm 7-9, kd ~0.25-0.35, H 9.8-10.4, loadH 5.1-5.4); checkpoints every"
    echo "1000 steps land in $Q"
    ;;

train-kd5)
    cd "$REPO"
    acquire
    # one-variable secondary arm: --kd-weight 5.0 (the support-fit lever); the
    # cache, tail weights, temp, balance and curriculum are the signed values.
    time python moe/qwen4exp_proxy.py train --model-dir "$Q4_MODEL" \
        --device cuda:0 --device-map auto \
        --cache-file "$CACHE" --ref-file "$REF" --compact-banks --grad-checkpoint \
        --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
        --kd-weight 5.0 --kd-tail-weight 2.0 --kd-tailcond-weight 3.0 --temp 2.0 \
        --balance bias --log-entropy --fast-indexer --force-gpu \
        --corpus-file "$MIXFILE" --agentic-frac 0.05 \
        --windows 4096 --corpus-chars 50000000 --seq 512 --seed 0 \
        --epochs 1 --steps 4096 --eval-every 1000 --ckpt-every 1000 \
        --log-every 100 --tag kd5
    ;;

eval)
    cd "$REPO"
    acquire
    for tag in cur05 kd5; do
        ckpt="$Q/qwen4exp-corr-r512-g128-step4096-$tag.pt"
        [ -f "$ckpt" ] || { echo "missing $ckpt (run train/$tag first)"; continue; }
        time python moe/qwen4exp_proxy.py eval --model-dir "$Q4_MODEL" \
            --device cuda:0 --device-map cuda:0 --prefix-layers 2 \
            --eval-windows 8 --quant lloyd --branch-quant g128 \
            --branch-target both --rank 512 --load "$ckpt" --tag "eval-$tag"
    done
    ls -la "$Q"/qwen4exp-eval-*.json 2>/dev/null || true
    ;;

export)
    cd "$REPO"
    # dense f16 factors (no fork gguf-py needed on the pod); the substantive
    # merge into a release body is the local P3 step.
    for tag in cur05 kd5; do
        ckpt="$Q/qwen4exp-corr-r512-g128-step4096-$tag.pt"
        [ -f "$ckpt" ] || { echo "missing $ckpt (run train/$tag first)"; continue; }
        python moe/export_branches_lora.py \
            --load "$ckpt" --arch qwen4exp --target both --dtype f16 \
            --recipe "qwen4exp both r512 g128, lloyd compact banks, $tag, 4096 steps, 1 epoch" \
            --out "$Q/qwen4exp-adapter-$tag-final.gguf" || \
            echo "WARN: adapter export failed for $tag (do it locally in P3)"
    done
    ls -la "$Q"/*adapter*.gguf 2>/dev/null || true
    echo "fetch the adapters + eval JSONs before terminating"
    ;;

ple-ab)
    cd "$REPO"
    acquire
    time python moe/qwen4exp_proxy.py ple-ab --model-dir "$Q4_MODEL" \
        --layers 2 --device cuda:0 --eval-windows 8 --ple-bits 8,4,2 \
        --ple-group 32 --out "$Q/ple-ab-pod.json"
    ;;
esac
