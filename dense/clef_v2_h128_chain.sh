#!/bin/sh
# Goal-B continuation chain, run detached from the OpenCode app scope:
#   128-window Hessian capture -> GPTQ + Lloyd + act-order -> PPL.
# Started via: systemd-run --user --unit=clef-h128 ...
set -e
PY=/home/penis/Desktop/work/.venv-rocm/bin/python
S=/home/penis/Desktop/work/scion
T=/home/penis/Desktop/work/models/clef-flash-ternary
EXT=/run/media/penis/30CE2C97CE2C577E/storage
export PYTHONPATH=/home/penis/llama.cpp/gguf-py
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
/home/penis/llama.cpp/build/bin/llama-perplexity \
    -m clef-flash-v2-signs-gptq-lloyd-ao-h128.gguf \
    -f wiki.test.raw -c 512 --chunks 100 -ngl 99 \
    > ppl100-gptq-lloyd-ao-h128.log 2>&1
grep -E "Final estimate" ppl100-gptq-lloyd-ao-h128.log
echo "== done $(date -Is)"
