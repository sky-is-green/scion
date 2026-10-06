#!/bin/sh
# CPU fallback for the mixed-precision sweep (ROCm is down on this host).
# Same variants, 25 chunks, all through the fixed build-cpu rotation path.
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
PPL="${LLAMA_BIN_CPU:-$WORKSPACE/llama.cpp/build-cpu/bin}/llama-perplexity"
cd "$S"

conv() {
  tag=$1; keep=$2
  if [ ! -f "$V/clef-flash-v2-$tag.gguf" ]; then
    echo "== convert $tag (keep f16: $keep)"
    "$PY" dense/clef_v2_convert.py --in "$T/clef-flash-f16.gguf" \
        --out "$V/clef-flash-v2-$tag.gguf" --keep-f16 "$keep"
  fi
}

ppl() {
  file=$1; tag=$2
  echo "== ppl25 $tag"
  ( cd "$V" && "$PPL" -m "$file" -f wiki.test.raw -c 512 --chunks 25 \
      > "ppl25-$tag.log" 2>&1 )
  grep -E "Final estimate" "$V/ppl25-$tag.log" || echo "  (no estimate; see ppl25-$tag.log)"
}

conv nomlp  '\.ffn_(gate|up|down)\.weight$'
conv nodown '\.ffn_down\.weight$'
conv noqkv  '\.attn_qkv\.weight$'
conv noedge '^blk\.(0|1|30|31)\.'

ppl clef-flash-v2-pq2_0-rot.gguf   v2
ppl clef-flash-v2-ctrl-norot.gguf  ctrl
ppl clef-flash-v2-nomlp.gguf       nomlp
ppl clef-flash-v2-nodown.gguf      nodown
ppl clef-flash-v2-noqkv.gguf       noqkv
ppl clef-flash-v2-noedge.gguf      noedge
