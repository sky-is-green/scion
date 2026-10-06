#!/bin/sh
# Mixed-precision sweep on the recovered GPU.  Runs the six variants
# back-to-back (minimises the idle runtime-PM window that wedged the card) and
# aborts loudly if ROCm disappears rather than falling into the CPU path.
set -u

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
PPL="$LLAMA_BIN/llama-perplexity"
cd "$S"

if [ ! -f "$V/clef-flash-v2-noedge.gguf" ]; then
  echo "== convert noedge"
  "$PY" dense/clef_v2_convert.py --in "$T/clef-flash-f16.gguf" \
      --out "$V/clef-flash-v2-noedge.gguf" --keep-f16 '^blk\.(0|1|30|31)\.'
fi

run() {
  file=$1; tag=$2
  echo "== ppl100 $tag"
  ( cd "$V" && "$PPL" -m "$file" -f wiki.test.raw -c 512 --chunks 100 -ngl 99 \
      > "ppl100-$tag.log" 2>&1 )
  if grep -q "failed to initialize ROCm" "$V/ppl100-$tag.log"; then
    echo "!! GPU LOST (ROCm init failed)"; exit 1
  fi
  grep -E "Final estimate" "$V/ppl100-$tag.log" || echo "  (no estimate)"
}

run clef-flash-v2-pq2_0-rot.gguf  v2
run clef-flash-v2-ctrl-norot.gguf ctrl
run clef-flash-v2-nomlp.gguf      nomlp
run clef-flash-v2-nodown.gguf     nodown
run clef-flash-v2-noqkv.gguf      noqkv
run clef-flash-v2-noedge.gguf     noedge
echo "== done"
