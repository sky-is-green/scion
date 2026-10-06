#!/bin/sh
# Mixed-precision ablation sweep on the V2 rotated base.
#
# Each variant folds+ternarizes everything as V2 does, except the named target
# set which is copied as F16 (and not listed in the rotation metadata).  PPL is
# the wikitext-2 test, c512, 100 chunks, same settings as the V2 baseline.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="${SCION_WORKSPACE:-$(dirname "$ROOT")}"
MODELS="${SCION_MODELS:-$WORKSPACE/models}"
CLEF_MODEL="${SCION_CLEF_MODEL:-$MODELS/clef-flash-ternary}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin}"
GGUF_PY="${GGUF_PY:-$HOME/llama.cpp/gguf-py}"
TMP="${TMPDIR:-/tmp}"

PY="${SCION_PYTHON:-python3}"
S="$ROOT"
T="$CLEF_MODEL"
V=$T/v2
export PYTHONPATH="$GGUF_PY"
export HIP_VISIBLE_DEVICES=1
cd "$S"

run() {
  tag=$1
  keep=$2
  echo "== $tag (keep f16: $keep)"
  "$PY" dense/clef_v2_convert.py --in "$T/clef-flash-f16.gguf" \
      --out "$V/clef-flash-v2-$tag.gguf" --keep-f16 "$keep"
  ( cd "$V" && "$LLAMA_BIN/llama-perplexity" \
      -m "clef-flash-v2-$tag.gguf" -f wiki.test.raw -c 512 --chunks 100 -ngl 99 \
      > "ppl-$tag.log" 2>&1 )
  grep -E "Final estimate" "$V/ppl-$tag.log"
}

run nomlp  '\.ffn_(gate|up|down)\.weight$'
run nodown '\.ffn_down\.weight$'
run noqkv  '\.attn_qkv\.weight$'
run noedge '^blk\.(0|1|30|31)\.'
