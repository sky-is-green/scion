#!/bin/bash
# Placement sweep 2026-09-27: expert-offload cost curve on 2x RX 7900 XT + 7800X3D.
#
# Goal: measure what it costs to keep N layers' MoE experts on the CPU while the
# rest stays on the GPUs. This is the static lever llama.cpp exposes today
# (`-ncmoe`); the adaptive per-expert cache Strata/FreeToken have would have to
# beat this baseline to be worth engine work.
#
# Usage: ./run-bench.sh
set -u
cd "$(dirname "$0")/.."          # repo root (adjust if nested)
BIN=build-hip/bin/llama-bench
PPL=build-hip/bin/llama-perplexity
OUT=placement-sweep-20260927
MIX=olmoe-q1exp-q8rest.gguf
F16=olmoe-f16.gguf

mkdir -p "$OUT"

echo "### mixed model (q1exp-q8rest, 2.1G), dual-GPU layer split" | tee "$OUT/bench-mixed-2gpu.log"
for n in 0 4 8 12 16; do
  echo "--- ncmoe=$n ---" | tee -a "$OUT/bench-mixed-2gpu.log"
  $BIN -m $MIX -ngl 99 -ncmoe $n -p 512 -n 128 -r 3 -t 8 2>&1 \
    | tee -a "$OUT/bench-mixed-2gpu.log" | grep -E "^\| (model|AMD|qwen|olmoe)|^\| +[0-9]"
done

echo "### mixed model, CPU-tail thread scaling at ncmoe=16" | tee -a "$OUT/bench-mixed-2gpu.log"
$BIN -m $MIX -ngl 99 -ncmoe 16 -p 512 -n 128 -r 3 -t 16 2>&1 \
  | tee -a "$OUT/bench-mixed-2gpu.log" | grep -E "^\| +[0-9]"

echo "### mixed model, single GPU (sm=none, mg=0)" | tee "$OUT/bench-mixed-1gpu.log"
for n in 0 16; do
  echo "--- ncmoe=$n ---" | tee -a "$OUT/bench-mixed-1gpu.log"
  $BIN -m $MIX -ngl 99 -ncmoe $n -sm none -mg 0 -p 512 -n 128 -r 3 -t 8 2>&1 \
    | tee -a "$OUT/bench-mixed-1gpu.log" | grep -E "^\| +[0-9]"
done

echo "### f16 model (13G), dual-GPU, expert bytes x7.5 vs ternary" | tee "$OUT/bench-f16-2gpu.log"
for n in 0 16; do
  echo "--- ncmoe=$n ---" | tee -a "$OUT/bench-f16-2gpu.log"
  $BIN -m $F16 -ngl 99 -ncmoe $n -p 512 -n 128 -r 2 -t 8 2>&1 \
    | tee -a "$OUT/bench-f16-2gpu.log" | grep -E "^\| +[0-9]"
done

echo "### PPL equivalence check (placement must not change the math), 128 chunks" | tee "$OUT/ppl.log"
for n in 0 16; do
  echo "--- ncmoe=$n ---" | tee -a "$OUT/ppl.log"
  $PPL -m $MIX -f wiki.test.raw -c 512 -ngl 99 -ncmoe $n -t 8 --chunks 128 2>&1 \
    | tail -8 | tee -a "$OUT/ppl.log"
done

echo "### done $(date -Iseconds)"
