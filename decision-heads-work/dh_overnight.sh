#!/bin/sh
# Decision-head community arms on the 176B, overnight (see docs/DECISION-HEADS-PORT.md).
#
#   systemd-run --user --unit=dh-overnight /bin/sh <this>
#
# Stops scion-flashnext, serves the 176B through the decision-head build (server.py on 8897), runs
# the study's community protocol (gen 3 samples/task, then the resumable embed, then the eval and
# the veto arms) and puts the user's service back whichever way it ends.
set -u
WORKSPACE=${SCION_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SCION_MODELS=${SCION_MODELS:-$WORKSPACE/models}
SCION_REPO=${SCION_REPO:-$WORKSPACE/scion}
STRATA_DH=${STRATA_DH_ROOT:-$WORKSPACE/strata-dh}
STRATA_DATA=${SCION_STRATA_DATA:-$WORKSPACE/strata-data}
PACK176=${SCION_176B_PACK:-$SCION_MODELS/scion-strata/pack-q6k}
GGUF176=${SCION_176B_GGUF:-$SCION_MODELS/scion-strata/Scion-FlashNext-176B-A6B-Q6K.gguf}
EMB176=${SCION_176B_EMBD:-$SCION_MODELS/scion-strata/scion-embd-bf16.gguf}
PROFILE=${SCION_EXPERT_PROFILE:-$WORKSPACE/strata/data/expert-profile.bin}
MTP=${SCION_MTP_DIR:-$STRATA_DATA/mtp/rt}
W=$WORKSPACE
PY=$W/hivebench/.venv/bin/python
OUT=${SCION_DH_OUT:-$SCION_REPO/decision-heads-work/overnight}
HIVE_EXP=${SCION_HIVEBENCH:-$WORKSPACE/hivebench}/experiments/cascade
TASKS=${SCION_TASKS:-$HIVE_EXP/tasks-community.json}
CASC=$HIVE_EXP
R=$CASC/results/jev-readout-20261006
PREFIX=scion176b-community
SERVICE=scion-flashnext
mkdir -p "$OUT"
exec >> "$OUT/run.log" 2>&1
echo "== $(date -Is) start (pid $$)"

SRV=
cleanup() {
    echo "== $(date -Is) stopping the decision-head server"
    if [ -n "$SRV" ]; then
        kill "$SRV" 2>/dev/null
        wait "$SRV" 2>/dev/null
    fi
    echo "== $(date -Is) restarting $SERVICE"
    systemctl --user start "$SERVICE"
    sleep 5
    systemctl --user is-active "$SERVICE"
    echo "== $(date -Is) done"
}
trap cleanup EXIT INT TERM

echo "== $(date -Is) stopping $SERVICE"
systemctl --user stop "$SERVICE" || true
sleep 3

cat > "$OUT/dh-server.json" <<EOF
{
 "exe": "$STRATA_DH/build-hip/strata",
 "args": [
  "--pack", "$PACK176",
  "--native", "$GGUF176",
  "--ple-gguf", "$GGUF176",
  "--embd-gguf", "$EMB176",
  "--expert-profile", "$PROFILE",
  "--expert-cache", "auto",
  "--prefill", "8192",
  "--spec", "4",
  "--spec-min-p", "0.5",
  "--mtp", "$MTP",
  "--max-context", "4096",
  "--ple-io", "direct",
  "--mmap-experts",
  "--vram-reserve-mib", "3072",
  "--pool-workers", "15",
  "--pcie-frac", "0",
  "--pipeline-windows", "2"
 ],
 "cwd": "$STRATA_DH",
 "tokenizer": "$PACK176/tokenizer",
 "model_name": "scion-flashnext-176b-a6b-q6k-dh",
 "log": "$OUT/engine.log",
 "lib_dirs": ["/opt/rocm/lib"],
 "env": {"STRATA_PREFILL_MMQ": "1", "STRATA_PREFILL_RING": "96", "STRATA_IO_THREADS": "32"},
 "port": 8897,
 "backend": "hip",
 "gpu": [0, 1],
 "gpus_asked": true,
 "layer_split": "auto",
 "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.05}
}
EOF

echo "== $(date -Is) starting the decision-head server (exe: $W/strata-dh/build-hip/strata)"
"$PY" "$STRATA_DH/serve/server.py" --engine strata --config "$OUT/dh-server.json" --port 8897 \
    > "$OUT/server.log" 2>&1 &
SRV=$!
up=0
for i in $(seq 1 240); do
    if curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1; then up=1; break; fi
    kill -0 "$SRV" 2>/dev/null || break
    sleep 5
done
if [ "$up" != 1 ]; then echo "== the server did not come up"; exit 1; fi
echo "== $(date -Is) server up"

ce() {   # ce <phase args...>: community_eval with the run's common flags
    "$PY" "$CASC/community_eval.py" --server http://127.0.0.1:8897 --out-dir "$OUT" --tasks "$TASKS" \
        --model-label "SkyIsNotGreen/Scion-FlashNext-176B-A6B (local Q6K)" --model-name scion176b \
        --runtime "Strata 0.1.44 + emb= (main engine), 2x7900XT, layer split auto + --pipeline-windows 2" \
        --jbench-prefix main176b "$@"
}

# ---- smoke: two tasks through gen + embed + eval; a failure here stops before the long phases
echo "== $(date -Is) smoke: 2 tasks, 1 sample"
ce --prefix dh-smoke --phase gen --limit 2 --samples 1 || { echo "== smoke gen FAILED"; exit 1; }
ce --prefix dh-smoke --phase embed || { echo "== smoke embed FAILED"; exit 1; }
ce --prefix dh-smoke --phase eval --samples 1 || { echo "== smoke eval FAILED"; exit 1; }
echo "== $(date -Is) smoke ok"

# the frozen J-Bench states of this body (the eval's J-Bench arm)
# the frozen J-Bench states of this body, for the eval's J-Bench arm: the study's npz key (L1), not ours (X)
"$PY" - "$W/scion/decision-heads-work/dh-probe-states-main.npz" "$OUT/probe-states-main176b.npz" <<'PYEOF'
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
np.savez_compressed(sys.argv[2], L1=z['X'], labels=z['labels'], splits=z['splits'], item_ids=z['item_ids'])
print('wrote', sys.argv[2], z['X'].shape)
PYEOF

# ---- the full run
echo "== $(date -Is) gen: the 664 tasks x 3 samples"
ce --prefix "$PREFIX" --phase gen || { echo "== gen FAILED"; exit 1; }
echo "== $(date -Is) embed (resumable)"
"$PY" "$W/scion/decision-heads-work/dh_embed.py" --prefix "$PREFIX" --out-dir "$OUT" \
    || { echo "== embed FAILED"; exit 1; }
echo "== $(date -Is) eval"
ce --prefix "$PREFIX" --phase eval || { echo "== eval FAILED"; exit 1; }

# veto_eval reads the samples from the study's results dir
cp "$OUT/$PREFIX-samples.jsonl" "$R/$PREFIX-samples.jsonl"

echo "== $(date -Is) veto: in-domain head"
"$PY" "$CASC/veto_eval.py" --prefix "$PREFIX" --states "$OUT/$PREFIX-states.npz" --tasks "$TASKS" \
    --out "$OUT/veto-indomain.json" || echo "== veto (in-domain) FAILED"
echo "== $(date -Is) veto: the frozen J-Bench head"
"$PY" "$CASC/veto_eval.py" --prefix "$PREFIX" --states "$OUT/$PREFIX-states.npz" --tasks "$TASKS" \
    --head-json "$W/scion/decision-heads-work/scion-176b-a6b-decision-head.json" \
    --out "$OUT/veto-frozen.json" || echo "== veto (frozen) FAILED"

echo "== $(date -Is) all phases finished"
