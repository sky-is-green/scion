#!/bin/sh
# Goal-B gate: community-standard ternary baselines vs the QAT artifact.
#
#   imatrix -> TQ1_0/PTQ1_0/TQ2_0/Q2_K (+no-imatrix control) -> PPL ->
#   free-generation probes (llama-server + /completion, same suite as the QAT
#   acceptance test) for f16, the QAT artifact, and each baseline.
#
# Local only (no rental); the non-display card is card0 / PCI 07.  Run detached:
#   systemd-run --user --unit=clef-gate /bin/sh \
#       ./dense/clef_gate_chain.sh
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="${SCION_WORKSPACE:-$(dirname "$ROOT")}"
MODELS="${SCION_MODELS:-$WORKSPACE/models}"
CLEF_MODEL="${SCION_CLEF_MODEL:-$MODELS/clef-flash-ternary}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin}"
GGUF_PY="${GGUF_PY:-$HOME/llama.cpp/gguf-py}"
TMP="${TMPDIR:-/tmp}"

B="$LLAMA_BIN"
T="$CLEF_MODEL"
V=$T/v2
D="$ROOT/dense"
export HIP_VISIBLE_DEVICES=1
export PATH=/usr/local/bin:/usr/bin:/bin
cd "$V" || exit 1

stamp() { echo "== $1 [$(date -Is)]"; }
FAIL=""

RS=$(cat /sys/bus/pci/devices/0000:07:00.0/power/runtime_status 2>/dev/null)
if [ "$RS" != "active" ]; then
    echo "GPU 0000:07:00.0 runtime_status=$RS; aborting before GPU work"
    exit 1
fi
[ -s "$T/clef-flash-f16.gguf" ] || { echo "missing f16"; exit 1; }
[ -s "$V/wiki.test.raw" ] || { echo "missing wiki.test.raw"; exit 1; }

# --- 1. importance matrix (GPU) -------------------------------------------
stamp "imatrix (f16, wiki.test.raw)"
if "$B/llama-imatrix" -m "$T/clef-flash-f16.gguf" -f "$V/wiki.test.raw" \
        -o "$V/clef-imatrix.dat" -c 512 -ngl 99 > "$V/gate-imatrix.log" 2>&1; then
    grep -E "Final estimate|writing" "$V/gate-imatrix.log" | tail -2
    IM=1
else
    echo "FAILED imatrix"; tail -3 "$V/gate-imatrix.log"; IM=0; FAIL="$FAIL imatrix"
fi

# --- 2. quantize ----------------------------------------------------------
# quant <tag> <type> <use-im 0|1>; falls back to no-imatrix if the imatrix
# attempt dies (e.g. a codec that ignores importance matrices).
quant() {
    tag=$1; type=$2; im=$3
    stamp "quantize $tag ($type, imatrix=$im)"
    if [ "$im" = 1 ] && [ "$IM" = 1 ]; then
        "$B/llama-quantize" --imatrix "$V/clef-imatrix.dat" \
            "$T/clef-flash-f16.gguf" "$V/clef-flash-$tag.gguf" "$type" \
            > "$V/gate-quantize-$tag.log" 2>&1
    else
        "$B/llama-quantize" "$T/clef-flash-f16.gguf" \
            "$V/clef-flash-$tag.gguf" "$type" \
            > "$V/gate-quantize-$tag.log" 2>&1
    fi
    if [ ! -s "$V/clef-flash-$tag.gguf" ] && [ "$im" = 1 ]; then
        echo "  imatrix attempt failed, retrying without"
        "$B/llama-quantize" "$T/clef-flash-f16.gguf" \
            "$V/clef-flash-$tag.gguf" "$type" \
            > "$V/gate-quantize-$tag.log" 2>&1
    fi
    if [ -s "$V/clef-flash-$tag.gguf" ]; then
        grep -E "quant size" "$V/gate-quantize-$tag.log" | tail -1
    else
        echo "FAILED quantize $tag"; FAIL="$FAIL quant-$tag"
    fi
}

quant tq1_0      TQ1_0   1
quant tq1_0-noim TQ1_0   0
quant ptq1_0     PTQ1_0  1
quant tq2_0      TQ2_0   1
quant q2_k       Q2_K    1

# --- 3. PPL (c512, 100 chunks, GPU) ----------------------------------------
ppl() { # <tag> <model>
    tag=$1; model=$2
    stamp "ppl $tag"
    "$B/llama-perplexity" -m "$model" -f wiki.test.raw -c 512 --chunks 100 \
        -ngl 99 > "ppl100-$tag.log" 2>&1
    grep -E "Final estimate" "ppl100-$tag.log" | tail -1 \
        || { echo "FAILED ppl $tag"; FAIL="$FAIL ppl-$tag"; }
}

for tag in tq1_0 tq1_0-noim ptq1_0 tq2_0 q2_k; do
    [ -s "$V/clef-flash-$tag.gguf" ] && ppl "$tag" "$V/clef-flash-$tag.gguf"
done
echo "reference: f16 12.5894 (ppl-f16.log), QAT 23.2508 (ppl100-qat-a1.log)"

# --- 4. generation probes (GPU, serialized) --------------------------------
gen() { # <tag> <model>
    tag=$1; model=$2
    stamp "gen $tag"
    python3 "$D/clef_gen_probe.py" --model "$model" --name "gate-$tag" \
        --outdir "$V/gate-gen" --port 8896 \
        || { echo "FAILED gen $tag"; FAIL="$FAIL gen-$tag"; }
}

gen f16  "$T/clef-flash-f16.gguf"
gen qat  "$V/clef-flash-v2-qat-a1.gguf"
for tag in tq1_0 ptq1_0 tq2_0 q2_k; do
    [ -s "$V/clef-flash-$tag.gguf" ] && gen "$tag" "$V/clef-flash-$tag.gguf"
done

# --- 5. f16-text NLL axis (QAT test used teacher-forced NLL 2.02 vs 1.24) ---
if [ -s "$V/gate-gen/gate-f16.json" ]; then
    python3 - "$V" <<'EOF'
import json, sys
from pathlib import Path
v = Path(sys.argv[1])
recs = json.loads((v / "gate-gen/gate-f16.json").read_text())
text = "\n\n".join(r["text"] for r in recs)
# repeat to exceed one 512-token block so perplexity has a full chunk
(v / "gate-f16-gen.txt").write_text((text + "\n\n") * 3)
print(f"[f16 clean text: {len(text)} chars -> 3 repeats for PPL]")
EOF
    nll() { # <tag> <model>
        tag=$1; model=$2
        [ -s "$V/gate-gen/gate-$tag.json" ] || return 0
        stamp "nll-on-f16-text $tag"
        "$B/llama-perplexity" -m "$model" -f "$V/gate-f16-gen.txt" \
            -c 512 --chunks 1 -ngl 99 > "$V/gate-nll-$tag.log" 2>&1 \
            || echo "  (nll axis failed for $tag)"
        grep -E "Final estimate" "$V/gate-nll-$tag.log" | tail -1 | sed 's/^/  /'
    }
    nll f16 "$T/clef-flash-f16.gguf"
    nll qat "$V/clef-flash-v2-qat-a1.gguf"
    for tag in tq1_0 ptq1_0 tq2_0 q2_k; do
        nll "$tag" "$V/clef-flash-$tag.gguf"
    done
fi

# --- 6. summary ------------------------------------------------------------
stamp "summary"
python3 - "$V" <<'EOF'
import json, sys
from pathlib import Path
v = Path(sys.argv[1])
print(f"{'tag':10s} {'PPL(wiki)':>12s}  gen loop frac (prose/code/math)")
for tag, log in [("f16", "ppl-f16.log"), ("qat", "ppl100-qat-a1.log")] + \
        [(t, f"ppl100-{t}.log") for t in ("tq1_0", "tq1_0-noim", "ptq1_0", "tq2_0", "q2_k")]:
    ppl = "-"
    lp = v / log
    if lp.exists():
        for line in lp.read_text().splitlines():
            if "Final estimate" in line:
                ppl = line.split("PPL = ")[1].split(" ")[0]
    scores = []
    gp = v / f"gate-gen/gate-{tag}.json"
    if gp.exists():
        recs = json.loads(gp.read_text())
        by = {r["name"]: r["repeated_8gram_frac"] for r in recs}
        scores = [f"{by.get(n, '-')}" for n in ("prose", "code", "math")]
    print(f"{tag:10s} {ppl:>12s}  {'/'.join(scores) if scores else '-'}")
EOF

if [ -n "$FAIL" ]; then
    echo "FAILURES:$FAIL"
else
    echo "all steps completed"
fi
stamp "done"
