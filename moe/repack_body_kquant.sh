#!/bin/bash
# repack_body_kquant.sh — Tier-1 size repack of a Flash-Next release GGUF:
# rewrite the non-expert body tensor types to k-quants/Q8_0 while copying the
# already-deployed containers VERBATIM (same-type overrides make llama-quantize
# take the copy path, ~src/llama-quant.cpp "quantize = cur_type != new_type"):
#   - experts  ffn_{gate,up,down}_exps.weight : PTQ1_0 (1.75 bpw, bit-copied)
#   - PLE table per_layer_token_embd.weight   : Q4_0   (bit-copied)
#   - adapters *.lora_a / *.lora_b            : F16    (bit-copied)
# Everything else takes <body-type> (default Q8_0; Q6_K / Q5_K_M / Q4_K_M
# shrink further at a quality cost measured by the gate).
#
# Usage: repack_body_kquant.sh <in.gguf> <out.gguf> [body-type]
#   BIN=/path/to/llama-quantize  (default: the fork's CPU build)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${SCION_WORKSPACE:-$(dirname "$ROOT")}"
QWEN4EXP_BIN="${SCION_QWEN4EXP_BIN:-$WORKSPACE/llama-qwen4exp/build-q4exp-proto/bin}"

IN="${1:?usage: repack_body_kquant.sh <in.gguf> <out.gguf> [body-type]}"
OUT="${2:?usage: repack_body_kquant.sh <in.gguf> <out.gguf> [body-type]}"
TYPE="${3:-Q8_0}"
BIN="${BIN:-$QWEN4EXP_BIN/llama-quantize}"
[ -x "$BIN" ] || { echo "llama-quantize not executable: $BIN" >&2; exit 2; }
[ -s "$IN" ] || { echo "input missing: $IN" >&2; exit 2; }
time "$BIN" \
    --tensor-type "per_layer_token_embd=Q4_0" \
    --tensor-type "ffn_.*_exps=PTQ1_0" \
    --tensor-type ".*lora_[ab]=F16" \
    "$IN" "$OUT" "$TYPE"
ls -la "$OUT"
sha256sum "$OUT"
