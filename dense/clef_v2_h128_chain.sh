#!/bin/sh
# Goal-B continuation chain, run detached from the OpenCode app scope:
#   128-window Hessian capture -> GPTQ + Lloyd + act-order -> PPL.
# Started via: systemd-run --user --unit=clef-h128 ...
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
EXT="${SCION_STORAGE:-$WORKSPACE/storage}"
export PYTHONPATH="$GGUF_PY"
export HIP_VISIBLE_DEVICES=1

echo "== capture 128 windows $(date -Is)"
"$PY" "$S/dense/clef_v2_hessians.py" --out "$EXT/clef-v2-hessians-128" \
    --windows 128 --seq 512 --group 4 --device cuda:0

echo "== gptq act-order $(date -Is)"
"$PY" "$S/dense/clef_v2_convert.py" --in "$T/clef-flash-f16.gguf" \
    --out "$T/v2/clef-flash-v2-signs-gptq-lloyd-ao-h128.gguf" \
    --sign-seed 1337 --hessian-dir "$EXT/clef-v2-hessians-128" \
    --gptq-lloyd-scales --gptq-act-order

echo "== ppl $(date -Is)"
cd "$T/v2"
"$LLAMA_BIN/llama-perplexity" \
    -m clef-flash-v2-signs-gptq-lloyd-ao-h128.gguf \
    -f wiki.test.raw -c 512 --chunks 100 -ngl 99 \
    > ppl100-gptq-lloyd-ao-h128.log 2>&1
grep -E "Final estimate" ppl100-gptq-lloyd-ao-h128.log
echo "== done $(date -Is)"
