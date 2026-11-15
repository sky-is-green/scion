#!/bin/bash
# The 360-trace corpus head-input dump: same engine recipe as dump_headin.sh,
# reading the chunks built by build_probe_chunks.py.
#
# Run ONLY when the teacher server is stopped (it wants both cards).
# CARD POLICY (v2/scripts/gpu_roles.sh): SPARE is the primary (strata device 0 via
# HIP_VISIBLE_DEVICES); DISPLAY carries the lesser strain as a capped peer tier.
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work
. $W/scion/v2/scripts/gpu_roles.sh
export HIP_VISIBLE_DEVICES=$HIP_ORDER_SPARE_FIRST
cd $W/strata
CHUNK_DIR=$W/scion/v2/traces/mtp-probe-360/chunks
OUT_DIR=$W/scion/v2/traces/mtp-probe-360/headin
mkdir -p "$OUT_DIR"
STRATA="build-hip/strata"
PER_CHUNK=70656
NEED=$(( PER_CHUNK * 10256 ))
COMMON=(--pack $W/models-p310/scion-strata/pack-q6k
  --native $W/models-p310/flashnext/Scion-FlashNext-176B-A6B-Q6_K.gguf
  --ple-gguf $W/models-p310/flashnext/Scion-FlashNext-176B-A6B-Q6_K.gguf
  --embd-gguf $W/models-p310/scion-strata/scion-embd-bf16.gguf
  --expert-profile $W/strata/data/expert-profile.bin
  --expert-cache auto --prefill 4096 --max-context 72704
  --ple-io direct --mmap-experts --vram-reserve-mib 1024
  --peer-device 1 --peer-reserve-mib 3072 --peer-slots 5000
  --pool-workers 6 --pcie-frac 0
  --spec 4 --greedy)

N=$(ls "$CHUNK_DIR"/chunk*-cont.txt 2>/dev/null | wc -l)
echo "=== $N chunks in $CHUNK_DIR ==="
for ((i=0; i<N; i++)); do
  P=$CHUNK_DIR/chunk$i-prompt.txt
  F=$CHUNK_DIR/chunk$i-cont.txt
  D=$OUT_DIR/headin-chunk$i.bin
  got=$(stat -c %s "$D" 2>/dev/null || echo 0)
  if [ "$got" -ge "$NEED" ]; then
    echo "=== chunk $i already complete ($got) - skip ==="
    continue
  fi
  for attempt in 1 2 3; do
    echo "=== chunk $i attempt $attempt start $(date +%H:%M:%S) ==="
    "$STRATA" "${COMMON[@]}" \
      --tokens-file "$P" --max-new $((PER_CHUNK - 1)) \
      --spec-oracle "$F" --spec-follow "$F" \
      --dump-head-input "$D" \
      > /tmp/opencode/strata-headin360-chunk$i.log 2>&1
    rc=$?
    got=$(stat -c %s "$D" 2>/dev/null || echo 0)
    echo "=== chunk $i attempt $attempt exit $rc dump $got / need $NEED ==="
    if [ "$rc" -eq 0 ] && [ "$got" -ge "$NEED" ]; then
      echo "=== chunk $i OK ==="; break
    fi
    sleep 5
  done
done
echo "ALL CHUNKS DONE $(date +%H:%M:%S)"
