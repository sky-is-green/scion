#!/usr/bin/env bash
# 4-layer hot/cold training -> export -> merge -> audit smoke (local, no pod).
#
# Proves the training-time bits=4 path matches the Q4_0_G128 deployment codec
# and that the exporter/merge/audit still work with a mixed body.  It is not a
# capability result: the cache is synthetic and the prefix is 4 layers.
set -euo pipefail

W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work
SCION="$W/scion"
PY="$W/.envs/scion-rocm/bin/python"
GG="$W/scion-llama.cpp/gguf-py"
BODY="$W/models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf"
CACHE=/tmp/opencode/qwen35-hotcold-cache.pt
TAG=hotcold4l
CKPT="$SCION/moe/artifacts/qwen35/qwen35-corr-r512-g128-step2-${TAG}.pt"
ADAPTER=/tmp/opencode/${TAG}-adapter.gguf
MERGED=/tmp/opencode/${TAG}-merged.gguf

cd "$SCION"

echo "== train 2 steps (Q4 layers 0-1, TQ1 layers 2-3)"
MOE_ARTIFACTS="$W/models-p310/empero-prefix" \
  "$PY" moe/qwen35_moe_proxy.py train --prefix-layers 4 --device cuda:0 \
  --alloc-file v2/hotcold/alloc-4l.json --cache-file "$CACHE" \
  --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
  --steps 2 --tag "$TAG"

echo "== export"
PYTHONPATH="$GG" "$PY" moe/export_branches_lora.py \
  --load "$CKPT" --arch qwen35moe --target both --dtype q1_0_g128 \
  --recipe "hotcold 4-layer smoke" --out "$ADAPTER"

echo "== merge into the v1 body (replace the 4-layer branches only)"
PYTHONPATH="$GG" "$PY" moe/merge_adapter_into_body.py \
  --body "$BODY" --adapter "$ADAPTER" --out "$MERGED" --replace

echo "== audit"
PYTHONPATH="$GG" "$PY" v2/scripts/export_audit.py \
  --deployed "$MERGED" --branches-only

ls -lh "$ADAPTER" "$MERGED"
