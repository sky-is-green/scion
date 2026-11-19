#!/bin/bash
# FlashNext 176B validation, queued behind the 35B subset30 eval.
#
# Runs the same held-out subset against the 176B so the two engines are measured
# with one harness.  The 176B cannot take the tensor-parallel split at this
# context: at 0.6/0.4 the spare already holds 19.0 GiB at 4k, and a 40k KV cache
# pushes it over.  Layer split 0.55/0.45 keeps both cards at ~16-17 GiB.
#
# Every GPU command goes through gpu_guard.sh and gpu_run.sh, same as the 35B.

set -u

W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work
S=$W/scion-llama.cpp
R=$S/scripts/scion/gpu_run.sh
G=$S/scripts/scion/gpu_guard.sh
B=$S/build-hip/bin/llama-server
M=$W/models-p310/flashnext/Scion-FlashNext-176B-A6B-Q6_K.gguf
OUT=$W/scion/v2/eval/hard-subset30-flashnext-v1
RESULT=/tmp/opencode/scion-flashnext.result

echo "flashnext queue: waiting for the 35B eval to finish" >&2
while systemctl --user is-active --quiet scion-35b-eval.service; do
    sleep 60
done
echo "flashnext queue: 35B eval is $(systemctl --user show -p Result --value scion-35b-eval.service)" >&2

# Free the GPUs and wait for the 35B server to release them.
systemctl --user stop scion-35b-server.service
for _ in $(seq 1 60); do
    pgrep -x llama-server >/dev/null || break
    sleep 2
done
if pgrep -x llama-server >/dev/null; then
    echo "flashnext queue: 35B server did not stop; refusing to start" >&2
    echo "blocked" > "$RESULT"
    exit 1
fi

export LD_LIBRARY_PATH=$W/rocm-7.2.4/opt/rocm/lib:$W/rocm-7.2.4/usr/lib
export HIP_VISIBLE_DEVICES=1,0
export GPU_RUN_LOG=/tmp/opencode/flashnext-server.run
export GPU_RUN_PROGRESS_S=900

# No -ctxcp 0: it is the 176B's 20-40 s/token switch, never use it here.
$R $G $B -m "$M" --device ROCm0,ROCm1 -ts 0.55,0.45 -c 40960 -ngl 99 \
    -ncmoe 0 --moe-cache-mib 0 -ot per_layer_token_embd=CPU \
    -np 1 -t 8 -fit off --no-context-shift --no-cache-idle-slots -sps 0 \
    --host 127.0.0.1 --port 8083 > /dev/null 2>&1 &
spid=$!

healthy=0
for _ in $(seq 1 300); do
    if curl -sf http://127.0.0.1:8083/health >/dev/null 2>&1; then
        healthy=1
        break
    fi
    if ! kill -0 "$spid" 2>/dev/null; then
        break
    fi
    sleep 2
done

if [ "$healthy" -ne 1 ]; then
    echo "flashnext queue: server never became healthy" >&2
    echo "server-failed" > "$RESULT"
    kill -TERM "$spid" 2>/dev/null
    wait "$spid" 2>/dev/null
    exit 1
fi

echo "flashnext queue: server healthy, starting the eval" >&2
python3 "$W/scion/v2/scripts/hard_subset.py" \
    --port 8083 \
    --label "Scion-FlashNext-176B v1" \
    --out-dir "$OUT" \
    --max-tokens 16384 \
    --thinking-budget 16384
rc=$?

kill -TERM "$spid" 2>/dev/null
wait "$spid" 2>/dev/null

if [ "$rc" -eq 0 ]; then
    echo "ok" > "$RESULT"
else
    echo "eval-failed-$rc" > "$RESULT"
fi
echo "flashnext queue: done rc=$rc" >&2
exit "$rc"
