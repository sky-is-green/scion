#!/bin/bash
# 35B v2 rental — one-shot box runbook (H100/H200 80 GB+, RunPod).
#
# Recipe signed 2026-09-30 (EXPECTED-IMPROVEMENT-MEMO.md §5):
#   primary  cur05    = D_KL2 3.0 + bias + hard-window curriculum 0.05
#   fallback pred2.0  = D_KL2 2.0 + bias, trained from the SAME cache
#   (one variable between arms: --kd-tailcond-weight 3.0 vs 2.0)
#
# Stage by stage; everything resumable (cache/ckpts land on the volume):
#   bash box-run-v2.sh setup
#   bash box-run-v2.sh smoke
#   bash box-run-v2.sh cache          # mixed-corpus master cache
#   bash box-run-v2.sh ref
#   bash box-run-v2.sh train          # cur05 (primary)
#   bash box-run-v2.sh train-fallback # pred2.0
#   bash box-run-v2.sh eval
#   bash box-run-v2.sh export
#   bash box-run-v2.sh gate           # ship gate (see docs/RENTAL-RUNBOOK-V2.md)
#
# Code delivery: the pod needs the session-8 tree (--alloc-file plumbing etc.).
# Either push scion-test and set SCION_COMMIT, or drop a tarball at
# $MOE_ARTIFACTS/scion-src.tar.gz (setup extracts it).  The corpus mix file
# $Q/curric-combo.jsonl must be uploaded before `cache` or the stage refuses.
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
REPO="${REPO:-/workspace/scion}"
SCION_COMMIT="${SCION_COMMIT:-scion-test}"     # branch or commit hash
SCION_TARBALL="${SCION_TARBALL:-$MOE_ARTIFACTS/scion-src.tar.gz}"
Q="$MOE_ARTIFACTS/qwen35"
MIXFILE="$Q/curric-combo.jsonl"
CACHE="$Q/prefix-top512-tail64-cur05.pt"
REF="$Q/eval-ref-w2.pt"
PRIMARY_TAG="cur05"
FALLBACK_TAG="pred2.0"
GGUF_PY="${GGUF_PY:-$REPO/taardis-llama.cpp/gguf-py}"

stage="${1:?usage: box-run-v2.sh <setup|smoke|cache|ref|train|train-fallback|eval|export|gate>}"

case "$stage" in
setup)
    nvidia-smi
    python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
    free -g | head -2; df -h "$MOE_ARTIFACTS" 2>/dev/null | tail -1 || df -h /workspace | tail -1

    pip install "transformers==5.5.0" "accelerate==1.15.0" "datasets==4.3.0" \
                "safetensors==0.8.0" "huggingface_hub==1.32.0" \
                "tokenizers==0.22.2" "numpy==2.5.2"
    pip install ninja setuptools wheel
    pip install --no-build-isolation causal-conv1d flash-linear-attention || \
        echo "WARN: fast-path install failed — expect the torch fallback"
    pip install "triton==3.8.0" || echo "WARN: triton pin failed"

    if [ -f "$SCION_TARBALL" ]; then
        mkdir -p "$REPO" && tar -xzf "$SCION_TARBALL" -C "$REPO" --strip-components=1
        echo "repo from tarball: $SCION_TARBALL"
    else
        [ -d "$REPO" ] || git clone https://github.com/sky-is-green/scion "$REPO"
        git -C "$REPO" fetch --all && git -C "$REPO" checkout "$SCION_COMMIT"
    fi
    [ -d "$REPO/taardis-llama.cpp" ] || git clone -b q1_0_g128-port \
        https://github.com/CodeMasterCody3D/prism-ml-llama.cpp "$REPO/taardis-llama.cpp"
    mkdir -p "$Q"

    [ -f "$MOE_ARTIFACTS/empero-hf/config.json" ] || \
        hf download empero-ai/Qwen3.8-35B-A3B-Distill --local-dir "$MOE_ARTIFACTS/empero-hf"
    du -sh "$MOE_ARTIFACTS/empero-hf"
    echo "REMINDER: upload $MIXFILE before the cache stage (or that stage fails hard)."
    ;;
smoke)
    cd "$REPO"
    python moe/qwen35_moe_proxy.py smoke --layers 4 --device cuda:0
    echo "check: hidden drift ~0.3-0.45, router top-8 agreement ~0.83, and NO"
    echo "'fast path is not available' warning (install fla+causal_conv1d if you see it)"
    ;;
cache)
    cd "$REPO"
    [ -f "$MIXFILE" ] || { echo "REFUSING: $MIXFILE missing — upload it first; the"
        echo "cache corpus must match the trained corpus exactly." >&2; exit 4; }
    time python moe/qwen35_moe_proxy.py cache --device cuda:0 \
        --windows 4096 --corpus-chars 50000000 --seq 512 --seed 0 \
        --corpus-file "$MIXFILE" --agentic-frac 0.05 \
        --top-logits 512 --tail-logits 64 \
        --cache-file "$CACHE"
    ;;
ref)
    cd "$REPO"
    time python moe/qwen35_moe_proxy.py ref --device cuda:0 --ref-file "$REF"
    ;;
train)
    cd "$REPO"
    # optional RCO allocation (only if the local arm won): add
    #   --alloc-file "$Q/rco-alloc-budget125.map.json"
    time python moe/qwen35_moe_proxy.py train --device-map cuda:0 --device cuda:0 \
        --cache-file "$CACHE" --ref-file "$REF" \
        --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
        --kd-weight 2.0 --kd-tail-weight 2.0 --kd-tailcond-weight 3.0 --temp 2.0 \
        --balance bias --log-entropy \
        --corpus-file "$MIXFILE" --agentic-frac 0.05 \
        --windows 4096 --corpus-chars 50000000 --seq 512 --seed 0 \
        --epochs 1 --steps 4096 --eval-every 1000 --ckpt-every 1000 --log-every 100 \
        --tag "$PRIMARY_TAG"
    ;;
train-fallback)
    cd "$REPO"
    time python moe/qwen35_moe_proxy.py train --device-map cuda:0 --device cuda:0 \
        --cache-file "$CACHE" --ref-file "$REF" \
        --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
        --kd-weight 2.0 --kd-tail-weight 2.0 --kd-tailcond-weight 2.0 --temp 2.0 \
        --balance bias --log-entropy \
        --corpus-file "$MIXFILE" --agentic-frac 0.05 \
        --windows 4096 --corpus-chars 50000000 --seq 512 --seed 0 \
        --epochs 1 --steps 4096 --eval-every 1000 --ckpt-every 1000 --log-every 100 \
        --tag "$FALLBACK_TAG"
    ;;
eval)
    cd "$REPO"
    for tag in "$PRIMARY_TAG" "$FALLBACK_TAG"; do
        ckpt="$Q/qwen35-corr-r512-g128-step4096-$tag.pt"
        [ -f "$ckpt" ] || { echo "missing $ckpt (run train/$tag first)"; continue; }
        time python moe/qwen35_moe_proxy.py eval --device-map cuda:0 --device cuda:0 \
            --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
            --load "$ckpt" --tag "eval-$tag"
    done
    ;;
export)
    cd "$REPO"
    for tag in "$PRIMARY_TAG" "$FALLBACK_TAG"; do
        CKPT="$Q/qwen35-corr-r512-g128-step4096-$tag.pt"
        [ -f "$CKPT" ] || { echo "missing $CKPT"; continue; }
        SOUP="$Q/qwen35-corr-r512-g128-soup-$tag.pt"
        CKPTS=$(ls $Q/qwen35-corr-r512-g128-step*-$tag.pt 2>/dev/null || true)
        N=$(echo "$CKPTS" | grep -c . || true)
        if [ "$N" -ge 2 ]; then python moe/soup_checkpoints.py "$SOUP" $CKPTS; fi
        for src in "$CKPT" "$SOUP"; do
            [ -f "$src" ] || continue
            case "$src" in *soup*) t="soup" ;; *) t="final" ;; esac
            PYTHONPATH="$GGUF_PY" python moe/export_branches_lora.py \
                --load "$src" --arch qwen35moe --target both --dtype q1_0_g128 \
                --routers --base-model "$MOE_ARTIFACTS/empero-hf" \
                --recipe "qwen35moe both r512 g128, lloyd banks, $tag, 4096 steps, 1 epoch, $t" \
                --eval-json "$Q/qwen35-eval-$tag.json" \
                --out "$Q/qwen35-adapter-$tag-$t.gguf"
        done
    done
    ls -la "$Q"/*.lora*.gguf "$Q"/*adapter*.gguf 2>/dev/null || true
    echo "download the adapters for both arms (and the eval JSONs) before terminating"
    ;;
gate)
    echo "Ship gate (memo §3): the full-model, full-vocab community KLD vs the"
    echo "uncorrected body and the v1 class, HellaSwag/Winogrande 400 unchanged,"
    echo "and the 1.7B canary set.  Commands and artifact conventions:"
    echo "  docs/RELEASE-35B-MODEL-CARD.md + docs/QUANT-RETENTION-35B.md"
    echo "  and the v1 workspace logs (rental/workspace/eval-*.log)."
    echo "Run this against BOTH arms; the memo's acceptance clauses are in §3."
    ;;
*)
    echo "unknown stage: $stage" >&2; exit 2 ;;
esac
