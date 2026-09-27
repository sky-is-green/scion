#!/bin/bash
# 35B rental — one-shot box runbook (A100/H100 80 GB, RunPod).
# Run stage by stage; every stage is resumable (cache/ckpts land on the volume).
# The deployment body is built LOCALLY already; the box only trains + exports.
#
#   bash box-run.sh setup     # deps, repo, model download, sanity
#   bash box-run.sh smoke
#   bash box-run.sh cache
#   bash box-run.sh ref
#   bash box-run.sh train
#   bash box-run.sh eval
#   bash box-run.sh export
#
# RunPod: activate the template's python env first if it uses one (conda/venv),
# so `python` and `pip` refer to the same interpreter as the CUDA torch.
set -euo pipefail

# RunPod PyTorch images: the system python is externally managed (PEP 668) and
# the CUDA toolkit is not on PATH; make the plain `pip install` calls below work.
export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"
for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda; do
    if [ -d "$cuda_dir" ]; then
        export PATH="$cuda_dir/bin:$PATH"
        export CUDA_HOME="${CUDA_HOME:-$cuda_dir}"
        break
    fi
done

# less fragmentation on the 80 GB card (~67 GiB weights + optimiser + acts)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MOE_ARTIFACTS="${MOE_ARTIFACTS:-/workspace/artifacts}"
REPO="${REPO:-/workspace/scion}"
Q="$MOE_ARTIFACTS/qwen35"
CACHE="$Q/teacher-cache.pt"
REF="$Q/eval-ref-w2.pt"
CKPT="$Q/qwen35-corr-r512-g128-step4096.pt"
ADAPTER="$Q/qwen35-adapter.lora.gguf"
# TAARDIS fork's gguf-py is what writes the compact q1_0_g128 adapter tensors
GGUF_PY="${GGUF_PY:-$REPO/taardis-llama.cpp/gguf-py}"

stage="${1:?usage: box-run.sh <setup|smoke|cache|ref|train|eval|export>}"

case "$stage" in
setup)
    nvidia-smi
    python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
    free -g | head -2; df -h "$MOE_ARTIFACTS" 2>/dev/null | tail -1 || df -h /workspace | tail -1

    # exact versions validated on the dev box (transformers 5.5.0 ran the smoke)
    pip install "transformers==5.5.0" "accelerate==1.15.0" "datasets==4.3.0" \
                "safetensors==0.8.0" "huggingface_hub==1.32.0" \
                "tokenizers==0.22.2" "numpy==2.5.2"
    # fast GDN path: without these the 30 linear-attention layers use the slow
    # torch fallback. If the install fails, training still works (slower).
    # causal-conv1d has no wheel for torch 2.9/cu128 — build it from source with
    # the image's nvcc (now on PATH); no build isolation so it links against the
    # installed torch instead of downloading another one.
    pip install ninja setuptools wheel
    pip install --no-build-isolation causal-conv1d flash-linear-attention || \
        echo "WARN: fast-path install failed — expect the torch fallback"
    # Hopper GDN kernels: fla's gated chunk_bwd_dqkwg refuses triton 3.4-3.7.0
    # (fla issue #640) and triton 3.7.1 fails to launch kernels when the 80 GB
    # card is nearly full (the 35B student peaks at ~80.6 GiB); 3.8.0 validated.
    pip install "triton==3.8.0" || echo "WARN: triton pin failed"

    [ -d "$REPO" ] || git clone https://github.com/sky-is-green/scion "$REPO"
    [ -d "$REPO/taardis-llama.cpp" ] || git clone -b q1_0_g128-port \
        https://github.com/CodeMasterCody3D/prism-ml-llama.cpp "$REPO/taardis-llama.cpp"

    mkdir -p "$Q"
    [ -f "$MOE_ARTIFACTS/empero-hf/config.json" ] || \
        hf download empero-ai/Qwen3.8-35B-A3B-Distill --local-dir "$MOE_ARTIFACTS/empero-hf"
    du -sh "$MOE_ARTIFACTS/empero-hf"
    ;;
smoke)
    cd "$REPO"
    python moe/qwen35_moe_proxy.py smoke --layers 4 --device cuda:0
    echo "check: hidden drift ~0.3-0.45 (depends on which tensors the local load finds), router top-8 agreement ~0.83, and NO"
    echo "'fast path is not available' warning (install fla+causal_conv1d if you see it)"
    ;;
cache)
    cd "$REPO"
    time python moe/qwen35_moe_proxy.py cache --device cuda:0 \
        --windows 4096 --corpus-chars 50000000 --top-logits 50 --cache-file "$CACHE"
    ;;
ref)
    cd "$REPO"
    time python moe/qwen35_moe_proxy.py ref --device cuda:0 --ref-file "$REF"
    ;;
train)
    cd "$REPO"
    # if the pod was interrupted, append: --resume "$Q/qwen35-corr-r512-g128-step<last>.pt"
    time python moe/qwen35_moe_proxy.py train --device-map cuda:0 --device cuda:0 \
        --cache-file "$CACHE" --ref-file "$REF" \
        --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
        --kd-weight 1.0 --temp 2.0 \
        --windows 4096 --corpus-chars 50000000 --epochs 1 --steps 4096 \
        --eval-every 1000 --ckpt-every 1000 --log-every 100
    ;;
eval)
    cd "$REPO"
    time python moe/qwen35_moe_proxy.py eval --device-map cuda:0 --device cuda:0 \
        --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
        --load "$CKPT"
    ;;
export)
    cd "$REPO"
    # SWA: soup the checkpoints of this run (measured +0.5-1% on OLMoE) and
    # export BOTH the final and the soup; pick locally on the runtime metric.
    SOUP="$Q/qwen35-corr-r512-g128-soup.pt"
    CKPTS=$(ls $Q/qwen35-corr-r512-g128-step*.pt 2>/dev/null || true)
    N=$(echo "$CKPTS" | grep -c . || true)
    if [ "$N" -ge 2 ]; then
        python moe/soup_checkpoints.py "$SOUP" $CKPTS
    fi
    for src in "$CKPT" "$SOUP"; do
        [ -f "$src" ] || continue
        case "$src" in
            *soup*) tag="soup" ;;
            *)      tag="final" ;;
        esac
        PYTHONPATH="$GGUF_PY" python moe/export_branches_lora.py \
            --load "$src" --arch qwen35moe --target both --dtype q1_0_g128 \
            --routers --base-model "$MOE_ARTIFACTS/empero-hf" \
            --recipe "qwen35moe both r512 g128, lloyd banks, kd1.0/t2.0, 4096 steps, 1 epoch, $tag" \
            --eval-json "$Q/qwen35-eval.json" \
            --out "${ADAPTER%.gguf}-$tag.gguf"
    done
    ls -la "$Q"/*.lora*.gguf
    echo "download both adapters (plus the eval JSON); pick on the local runtime metric"
    ;;
*)
    echo "unknown stage: $stage" >&2; exit 2 ;;
esac
