#!/usr/bin/env bash
# Compare the v1 body and the hot/cold body on the same text, then generate.
# This is a kernel/coherence check, not a capability gate: the hot/cold body
# has no correction adapter yet; the reference is the merged v2 model (the
# older v1 body file is rejected by the current llama.cpp GGUF reader).
set -euo pipefail

W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work
BIN="$W/scion-llama.cpp/build-hip/bin"
V1="$W/scion/v2/full/Scion-35B-A3B-v2-cur05.gguf"
HC="$W/scion/v2/hotcold/Scion-35B-A3B-v2-down4.gguf"
TXT=/tmp/opencode/ppl.txt
OUT=/tmp/opencode/hotcold-body-check
mkdir -p "$OUT"

ppl() {
  local name="$1" model="$2"
  echo "== PPL $name"
  HIP_VISIBLE_DEVICES=1 "$BIN/llama-perplexity" -m "$model" -f "$TXT" \
    -c 512 --chunks 4 -ngl 99 --device ROCm0 -t 8 2>&1 \
    | tee "$OUT/ppl-$name.log" | tail -n 8
}

gen() {
  local name="$1" model="$2"
  echo "== GEN $name"
  HIP_VISIBLE_DEVICES=1 "$BIN/llama-cli" -m "$model" \
    -p "A train leaves at 3 pm and travels 60 mph. It arrives at 6 pm. How far did it travel? Answer briefly." \
    -n 64 -ngl 99 --device ROCm0 -t 8 --temp 0.6 --top-p 0.95 \
    --no-context-shift --single-turn 2>&1 | tee "$OUT/gen-$name.log" | tail -n 20
}

ppl v1 "$V1"
ppl down4 "$HC"
gen down4 "$HC"
echo "logs in $OUT"
