#!/bin/bash
# The FlashNext head-input dump, chunked, 2-card PEER mode: the display card
# (strata device 0) runs the model with the big reserve; the free card (device
# 1) holds a second expert tier (STRATA_PEER_HOT) - at 72728 max-context the
# whole 24576-pair profile fits across both cards, so the CPU pool stays idle.
# Each chunk runs as ONE continuous teacher-forced sequence (1-token prompt +
# --spec-oracle/--spec-follow with the chunk's own tokens), dumping
# {pos, tok, 2560 floats} per verified position.  A GPU hang (#267) kills a
# chunk mid-run; the retry loop reruns it from scratch (up to 3x).
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work
cd $W/strata
mkdir -p $W/scion/v2/traces/mtp-probe/headin
STRATA="build-hip/strata"
COMMON=(--pack $W/models-p310/scion-strata/pack-q6k
  --native $W/models-p310/flashnext/Scion-FlashNext-176B-A6B-Q6_K.gguf
  --ple-gguf $W/models-p310/flashnext/Scion-FlashNext-176B-A6B-Q6_K.gguf
  --embd-gguf $W/models-p310/scion-strata/scion-embd-bf16.gguf
  --expert-profile $W/strata/data/expert-profile.bin
  --expert-cache auto --prefill 4096 --max-context 72704
  --ple-io direct --mmap-experts --vram-reserve-mib 3072
  --peer-device 1 --peer-reserve-mib 600
  --pool-workers 6 --pcie-frac 0
  --spec 4 --greedy)
for i in 0 1 2 3 4 5 6 7; do
  P=$W/scion/v2/traces/mtp-probe/chunks/chunk$i-prompt.txt
  F=$W/scion/v2/traces/mtp-probe/chunks/chunk$i-cont.txt
  D=$W/scion/v2/traces/mtp-probe/headin/headin-chunk$i.bin
  need=$(( 70656 * 10256 ))
  for attempt in 1 2 3; do
    echo "=== chunk $i attempt $attempt start $(date +%H:%M:%S) ==="
    "$STRATA" "${COMMON[@]}" \
      --tokens-file $P --max-new 70655 \
      --spec-oracle $F --spec-follow $F \
      --dump-head-input $D \
      > /tmp/opencode/strata-headin-chunk$i.log 2>&1
    rc=$?
    got=$(stat -c %s $D 2>/dev/null || echo 0)
    echo "=== chunk $i attempt $attempt exit $rc dump $got / need $need ==="
    if [ "$rc" -eq 0 ] && [ "$got" -ge "$need" ]; then
      echo "=== chunk $i OK ==="; break
    fi
    sleep 5
  done
done
echo "ALL CHUNKS DONE $(date +%H:%M:%S)"
