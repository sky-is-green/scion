#!/bin/sh
# Clef V2: PPL A/B/C (rotated V2 vs deployed unrotated PQ2_0 vs f16) on the
# local wikitext-2 test file, plus the 48-token hidden-cos probe.  One GPU at a
# time (non-display card 0 after HIP_VISIBLE_DEVICES=1).
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="${SCION_WORKSPACE:-$(dirname "$ROOT")}"
MODELS="${SCION_MODELS:-$WORKSPACE/models}"
CLEF_MODEL="${SCION_CLEF_MODEL:-$MODELS/clef-flash-ternary}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin}"
GGUF_PY="${GGUF_PY:-$HOME/llama.cpp/gguf-py}"
TMP="${TMPDIR:-/tmp}"

P="$LLAMA_BIN/llama-perplexity"
B="${SCION_BRIDGE:-$WORKSPACE/hivebench/tools/clef-bridge/clef_embed}"
T="$CLEF_MODEL"
V=$T/v2
export HIP_VISIBLE_DEVICES=1
cd "$V"

echo "[v2 rotated]"
"$P" -m clef-flash-v2-pq2_0-rot.gguf -f wiki.test.raw -c 512 --chunks 100 -ngl 99 > ppl-v2.log 2>&1
grep -E "Final estimate" ppl-v2.log || tail -3 ppl-v2.log

echo "[deployed pq2_0 unrotated]"
"$P" -m "$T/clef-flash-PQ2_0.gguf" -f wiki.test.raw -c 512 --chunks 100 -ngl 99 > ppl-pq2.log 2>&1
grep -E "Final estimate" ppl-pq2.log || tail -3 ppl-pq2.log

echo "[f16]"
if ! "$P" -m "$T/clef-flash-f16.gguf" -f wiki.test.raw -c 512 --chunks 100 -ngl 99 > ppl-f16.log 2>&1; then
  echo "  full offload failed, retrying -ngl 80"
  "$P" -m "$T/clef-flash-f16.gguf" -f wiki.test.raw -c 512 --chunks 100 -ngl 80 > ppl-f16.log 2>&1
fi
grep -E "Final estimate" ppl-f16.log || tail -3 ppl-f16.log

echo "[hidden-cos probe: 48 tokens]"
HIP_VISIBLE_DEVICES= "$B" "$T/clef-flash-f16.gguf" "$T/corrections/probe/tokens.txt" "$TMP/f16_probe2.bin" 4096 2>&1 | tail -1
HIP_VISIBLE_DEVICES= "$B" clef-flash-v2-pq2_0-rot.gguf "$T/corrections/probe/tokens.txt" "$TMP/v2_probe.bin" 4096 2>&1 | tail -1
python3 - <<EOF
import numpy as np
f16 = np.fromfile('$TMP/f16_probe2.bin', dtype=np.float32).reshape(-1, 4096)
v2 = np.fromfile('$TMP/v2_probe.bin', dtype=np.float32).reshape(-1, 4096)
cos = (f16*v2).sum(1)/(np.linalg.norm(f16,axis=1)*np.linalg.norm(v2,axis=1)+1e-9)
print('hidden cos f16 vs v2: min %.4f mean %.4f max %.4f' % (cos.min(), cos.mean(), cos.max()))
EOF
grep -m1 "Hadamard" ppl-v2.log || true
