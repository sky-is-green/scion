#!/bin/bash
# v2 pilot on a 2x48 GB (or 1x80 GB) CUDA pod.
#
# Stages:
#   bash moe/runpod-pilot.sh setup     # repo, pinned runtime, models, CUDA fork
#   bash moe/runpod-pilot.sh teacher   # 30 teacher traces (10 per family)
#   bash moe/runpod-pilot.sh score     # 512-token logit dump -> cache.pt
#   bash moe/runpod-pilot.sh train     # seeded 35B corrections, 200 steps
#   bash moe/runpod-pilot.sh export    # adapter + merged GGUF for local eval
#
# Inputs that must be uploaded before `teacher`:
#   /workspace/scion/v2/corpus/rollout-prompts.jsonl   (gitignored corpus)
#
# The teacher is Scion-FlashNext-176B-A6B Q6_K (60.2 GB); the student body is
# seeded from the deployed 35B GGUF so training-time codes equal the shipped
# codes (see v2/export-audit/README.md).  All artifacts stay under /workspace
# so they can be downloaded before the pod is terminated.
set -euo pipefail

export PIP_BREAK_SYSTEM_PACKAGES=1
export MOE_ARTIFACTS="${MOE_ARTIFACTS:-/workspace/artifacts}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
REPO="${REPO:-/workspace/scion}"
MODELS="${MODELS:-/workspace/models}"
Q="$MOE_ARTIFACTS/qwen35"
TEACHER="$MODELS/flashnext/Scion-FlashNext-176B-A6B-Q6_K.gguf"
BODY="$MODELS/Scion-35B-A3B-PQ2_0-corr.gguf"
FORK="${FORK:-/workspace/prism-qwen4exp}"
WORK="${WORK:-/workspace/pilot}"

stage="${1:?usage: runpod-pilot.sh <setup|teacher|score|train|export>}"

case "$stage" in
setup)
    nvidia-smi
    python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
    free -g | head -2; df -h /workspace | tail -1
    mkdir -p "$MODELS" "$WORK" "$Q" "$HF_HOME"

    if [ ! -d "$REPO/moe" ]; then
        git clone -b scion-test https://github.com/sky-is-green/scion "$REPO"
    fi
    cd "$REPO" && git fetch origin scion-test && git checkout -q scion-test && git pull -q
    pip install -q "transformers==5.5.0" "accelerate==1.15.0" "datasets==4.3.0" \
        "safetensors==0.8.0" "huggingface_hub[cli]==1.32.0" \
        "tokenizers==0.22.2" "numpy==2.5.2" ninja setuptools wheel
    pip install -q --no-build-isolation causal-conv1d flash-linear-attention || \
        echo "WARN: fast kernels unavailable; torch fallback"

    hf download empero-ai/Qwen3.8-35B-A3B-Distill --local-dir "$MOE_ARTIFACTS/empero-hf"
    hf download SkyIsNotGreen/Scion-35B-A3B \
        --include "Scion-35B-A3B-PQ2_0-corr.gguf" --local-dir "$MODELS"
    hf download SkyIsNotGreen/Scion-FlashNext-176B-A6B \
        --include "Scion-FlashNext-176B-A6B-Q6_K.gguf" --local-dir "$MODELS/flashnext"

    if [ ! -x "$FORK/build/bin/llama-server" ]; then
        [ -d "$FORK/.git" ] || git clone -b qwen4exp-proto \
            https://github.com/sky-is-green/prism-ml-llama.cpp "$FORK"
        cd "$FORK"
        cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF
        cmake --build build -j"$(nproc)" --target llama-server llama-perplexity
    fi
    ls -lh "$TEACHER" "$BODY" "$MOE_ARTIFACTS/empero-hf/config.json"
    sha256sum "$TEACHER" "$BODY"
    ;;

teacher)
    cd "$REPO"
    [ -f "$TEACHER" ] || { echo "missing $TEACHER" >&2; exit 2; }
    [ -f v2/corpus/rollout-prompts.jsonl ] || {
        echo "missing v2/corpus/rollout-prompts.jsonl; upload the corpus first" >&2
        exit 4; }
    nohup "$FORK/build/bin/llama-server" -m "$TEACHER" \
        --host 127.0.0.1 --port 8090 -ngl 99 -fa on -c 8192 -np 1 --jinja \
        > "$WORK/teacher-server.log" 2>&1 &
    echo $! > "$WORK/teacher.pid"
    for i in $(seq 1 120); do
        curl -sf -m 5 http://127.0.0.1:8090/health >/dev/null && break
        sleep 5
        kill -0 "$(cat "$WORK/teacher.pid")" 2>/dev/null || {
            tail -40 "$WORK/teacher-server.log"; exit 5; }
    done
    python v2/scripts/capture_rollouts.py --port 8090 \
        --model scion-flashnext-176b-a6b-q6k --per-family "${PER_FAMILY:-10}" \
        --max-tokens "${TEACHER_MAX_TOKENS:-4096}" \
        --thinking-budget "${TEACHER_MAX_TOKENS:-4096}" \
        --out-dir v2/traces/flashnext 2>&1 | tee "$WORK/teacher-traces.log"
    kill "$(cat "$WORK/teacher.pid")" 2>/dev/null || true
    sleep 5
    ;;

score)
    cd "$REPO"
    python - "$REPO" > "$WORK/traces.txt" <<'EOF'
import glob, json, sys
repo = sys.argv[1]
texts = []
for p in sorted(glob.glob(repo + "/v2/traces/flashnext/*.json")):
    d = json.load(open(p))
    msg = ((d.get("response") or {}).get("choices") or [{}])[0].get("message", {})
    body = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
    texts.append(d["prompt"].strip() + "\n\n" + body.strip())
print("\n\n".join(texts))
EOF
    wc -c "$WORK/traces.txt"
    # -c 4096: each chunk is a 4k window (experiment 1's window length); the
    # fork scores the second half of each chunk (2047 rows) - that half-chunk
    # layout is what build_trace_cache.py detects and decodes (v2/REAL-CACHE).
    # The dump is ~1.02 GB per chunk: keep it on the volume, delete after the
    # cache is built.
    "$FORK/build/bin/llama-perplexity" -m "$TEACHER" -f "$WORK/traces.txt" \
        -c 4096 --chunks -1 -ngl 99 -fa on --save-all-logits "$WORK/logits.bin" \
        > "$WORK/perplexity.log" 2>&1
    python moe/build_trace_cache.py --logits "$WORK/logits.bin" \
        --out "$WORK/cache.pt" --top-k 512 --seq 4096 --tail-logits 64 --seed 0
    rm -f "$WORK/logits.bin"
    ls -lh "$WORK/cache.pt"
    ;;

train)
    cd "$REPO"
    [ -f "$WORK/cache.pt" ] || { echo "missing $WORK/cache.pt" >&2; exit 2; }
    [ -f "$BODY" ] || { echo "missing $BODY" >&2; exit 2; }
    # FlashNext P2 memory switches (v2/MEMORY-PORT.md): the frozen banks live
    # as deployed codes+scales read verbatim from --bank-gguf, the decode and
    # the decoder layers are recomputed in backward (group of 64), and the
    # vocab-side CE is chunked.  --verify-banks runs the bit-exactness gate at
    # build time.  The cache carries tidx/tlp (score stage: --tail-logits 64),
    # so the full cur05 recipe incl. D_KL2 runs.  --log-entropy stays off on a
    # tight card (monitoring only; ~1 GiB of fp32 logits transients).
    time python moe/qwen35_moe_proxy.py train \
        --device-map auto --device cuda:0 \
        --bank-gguf "$BODY" \
        --cache-file "$WORK/cache.pt" \
        --compact-banks --grad-checkpoint --checkpoint-mode group --expert-group-size 64 \
        --verify-banks --chunked-ce \
        --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
        --kd-weight 2.0 --kd-tail-weight 2.0 --kd-tailcond-weight 3.0 --temp 2.0 \
        --balance bias \
        --epochs 1 --steps "${PILOT_STEPS:-200}" --eval-every 0 \
        --ckpt-every "${PILOT_CKPT_EVERY:-100}" --log-every 10 \
        --tag pilot 2>&1 | tee "$WORK/train.log"
    ls -lh "$Q"/*pilot*.pt
    ;;

export)
    cd "$REPO"
    CKPT="$(ls -t "$Q"/*pilot*.pt | head -1)"
    [ -f "$CKPT" ] || { echo "no pilot checkpoint" >&2; exit 2; }
    export PYTHONPATH="$FORK/gguf-py:${PYTHONPATH:-}"
    python moe/export_branches_lora.py --load "$CKPT" --arch qwen35moe \
        --target both --dtype q1_0_g128 --routers \
        --base-model "$MOE_ARTIFACTS/empero-hf" \
        --recipe "v2 pilot: FlashNext traces, seeded body, rank512 g128, 200 steps" \
        --out "$WORK/qwen35-v2-pilot-adapter.gguf"
    python moe/merge_adapter_into_body.py --body "$BODY" \
        --adapter "$WORK/qwen35-v2-pilot-adapter.gguf" \
        --out "$WORK/Scion-35B-A3B-v2-pilot.gguf"
    sha256sum "$WORK"/*.gguf
    ;;

*)
    echo "unknown stage: $stage" >&2; exit 2 ;;
esac
