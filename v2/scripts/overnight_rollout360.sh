#!/bin/bash
# Overnight 360-trace corpus pass (run when the machine is free - it owns both
# cards and the NVMe for ~10 h total).
#
#   teacher rollouts (resumable, skips already-captured traces)
#   -> probe chunks (70,656-token teacher-forced chunks, serving text format)
#   -> head-input dump (peer mode: spare primary, display peer)
#   -> 512-token windows for the trainer (v2/traces/mtp-probe-gpu2)
#
# Every stage is idempotent: stop it any time and rerun; finished traces,
# chunks and dump files are reused.  Log: /tmp/opencode/overnight-360.log
#
#   nohup v2/scripts/overnight_rollout360.sh &
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work
. $W/scion/v2/scripts/gpu_roles.sh
export HIP_VISIBLE_DEVICES=$HIP_ORDER_SPARE_FIRST
LOG=/tmp/opencode/overnight-360.log
exec >>"$LOG" 2>&1
echo "=== start $(date) ==="

TRACES=$W/scion/v2/traces/rollout360
CHUNKS=$W/scion/v2/traces/mtp-probe-360/chunks
HEADIN=$W/scion/v2/traces/mtp-probe-360/headin
OUT=$W/scion/v2/traces/mtp-probe-gpu2
mkdir -p "$TRACES" "$CHUNKS" "$HEADIN"

have=$(ls "$TRACES"/*.json 2>/dev/null | wc -l)
if [ "$have" -lt 360 ]; then
    echo "--- rollouts: $have/360 captured, starting the teacher server"
    SRV=""
    if ! curl -sf -m 3 http://127.0.0.1:8092/health >/dev/null 2>&1; then
        cd "$W/strata"
        .venv/bin/python -m serve.server --engine strata \
            --config "$W/scion/v2/configs/flashnext-q6k-176b-peer.json" --port 8092 &
        SRV=$!
        ok=0
        for _ in $(seq 1 120); do
            if curl -sf -m 3 http://127.0.0.1:8092/health >/dev/null 2>&1; then ok=1; break; fi
            sleep 5
        done
        if [ "$ok" -ne 1 ]; then
            echo "--- teacher server did not become ready, aborting"
            exit 1
        fi
    fi
    .venv/bin/python "$W/scion/v2/scripts/capture_teacher_corpus.py" \
        --port 8092 --max-tokens 2048 --out-dir "$TRACES"
    echo "--- capture exit $?, stopping the teacher server"
    if [ -n "$SRV" ]; then
        kill -TERM "$SRV" 2>/dev/null
        wait "$SRV" 2>/dev/null
    fi
    # also stop a server this script did not start (the dump needs the cards)
    for p in $(ps -eo pid,args | grep "python -m serve.server" | grep -v grep | awk '{print $1}'); do
        kill -TERM "$p" 2>/dev/null
    done
    # let the engine release the cards before the dump
    for _ in $(seq 1 30); do
        used=$(curl -sf -m 2 http://127.0.0.1:8092/health >/dev/null 2>&1 && echo yes || echo no)
        [ "$used" = "no" ] && break
        sleep 2
    done
    sleep 5
fi

echo "--- building probe chunks"
cd "$W/strata"
.venv/bin/python "$W/scion/v2/scripts/build_probe_chunks.py" \
    --traces "$TRACES" \
    --tokenizer "$W/models-p310/scion-strata/pack-q6k/tokenizer" \
    --out "$CHUNKS" --per-chunk 70656 || exit 1

echo "--- head-input dump"
"$W/scion/v2/scripts/dump_headin_360.sh" || exit 1

echo "--- assembling 512-token windows"
n=$(ls "$CHUNKS"/chunk*-cont.txt 2>/dev/null | wc -l)
if [ "$n" -lt 1 ]; then
    echo "--- no chunks to assemble"
    exit 1
fi
"$W/.envs/scion-rocm/bin/python" "$W/scion/v2/scripts/assemble_headin.py" \
    --chunks "$n" --per-chunk 70656 \
    --in-dir "$HEADIN" --out "$OUT" --tokens "$CHUNKS/tokens.bin" || exit 1

echo "=== done $(date): $n chunks -> $OUT ==="
