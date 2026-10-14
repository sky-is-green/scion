#!/usr/bin/env bash
# FFN part-by-part byte comparison at layer 0, first pass (the prompt's first chunk, identical inputs on both runs).
#   run A: the split config with STRATA_DENSE_SPLIT_OFF=1 -- the plain FFN on card 0, ffn_r0 = 9216, so its traces
#          are the reference values at the split's offsets.
#   run B: the split config, the peer path.
# Usage: bonsai-parts.sh [layer]
set -u
# Working-tree locations come from the environment; the defaults are the lane's layout, where the engine
# worktrees are siblings of this repository's parent (../strata-bonsai2, ../strata).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OCTMP=${STRATA_TMP:-$OCTMP}
W=${STRATA_BONSAI2_WORKTREE:-$ROOT/strata-bonsai2}
VENV=${STRATA_VENV_PY:-$ROOT/strata/.venv/bin/python}
UNIT=strata-dense-bonsai
ENGLOG=$OCTMP/strata-dense-bonsai.log
LOG=$OCTMP/parts
mkdir -p "$LOG"
LAYER=${1:-0}

restart_serve() {
  systemctl --user stop $UNIT 2>/dev/null
  systemctl --user start scion-flashnext 2>/dev/null
  for i in $(seq 1 60); do
    c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null || echo 000)
    [ "$c" = "200" ] && { echo "serve back"; return; }
    sleep 3
  done
  echo "serve check failed"
}
trap restart_serve EXIT

run_case() {
  local tag=$1 extra=$2
  echo "== $tag =="
  python3 - "$W/strata-bonsai-2-27b-split.json" "$LOG/$tag.json" "$extra" "$LAYER" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d.setdefault('env', {}).update({"STRATA_DENSE_FFN_TRACE": sys.argv[4]})
d['env'].update(json.loads(sys.argv[3]))
json.dump(d, open(sys.argv[2], 'w'))
PY
  systemctl --user stop scion-flashnext 2>/dev/null
  sleep 4
  : > "$ENGLOG"
  systemd-run --user --unit=$UNIT --collect --property=Nice=5 --property=WorkingDirectory=$W \
    --property=StandardOutput=append:$ENGLOG --property=StandardError=append:$ENGLOG \
    $VENV serve/server.py --engine strata --config "$LOG/$tag.json" --port 8082
  for i in $(seq 1 120); do
    c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8082/v1/models 2>/dev/null || echo 000)
    [ "$c" = "200" ] && { echo "  ready: yes"; break; }
    systemctl --user is-active --quiet $UNIT || { echo "  engine died"; break; }
    sleep 3
  done
  curl -s http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"bonsai-2-27b","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":1,"temperature":0}' \
    > "$LOG/$tag.reply.json"
  sleep 1
  cp "$ENGLOG" "$LOG/$tag.engine.log" 2>/dev/null
  systemctl --user stop $UNIT 2>/dev/null
  sleep 2
}

run_case ref '{"STRATA_DENSE_SPLIT_OFF":"1"}'
run_case peer '{}'

python3 - "$LOG" "$LAYER" <<'PY'
import sys, os
log, layer = sys.argv[1], sys.argv[2]
tag = f"dbg layer={layer} "
def heads(path):
    out = {}
    for ln in open(path, errors='replace'):
        if not ln.startswith(tag):
            continue
        p = ln.split()
        # dbg layer=0 kind name head: v v v bits: x x x
        name = p[3]
        if name in out:
            continue                      # first pass only
        try:
            i = p.index('bits:')
        except ValueError:
            continue
        out[name] = (' '.join(p[i+1:]), ln.strip())
    return out
A = heads(os.path.join(log, 'ref.engine.log'))
B = heads(os.path.join(log, 'peer.engine.log'))
pairs = [
    ("card0 gate rows        ", "ffn_g",      "s0_g"),
    ("card0 rotated half     ", "ffn_hrot",   "s0_hrot"),
    ("card0 assembled act    ", "ffn_hrot",   "s0_hfull"),
    ("card0 assembled act r0 ", "ffn_hrot_r0","s0_hfull_r0"),
    ("card1 gate rows        ", "ffn_g_r0",   "s1_g"),
    ("card1 rotated half     ", "ffn_hrot_r0","s1_hrot"),
    ("card1 assembled act    ", "ffn_hrot",   "s1_hfull"),
    ("card1 assembled act r0 ", "ffn_hrot_r0","s1_hfull_r0"),
    ("card0 down output      ", "ffn_mix",    "s0_mix"),
    ("card1 down output r0   ", "ffn_mix_r0", "s1_mix"),
]
print("part                       ref-name/split-name     same")
for label, a, b in pairs:
    if a not in A or b not in B:
        print(f"{label} {a}/{b}: missing ({'ref' if a not in A else 'split'})")
        continue
    same = 'YES' if A[a][0] == B[b][0] else 'NO'
    print(f"{label} {a}/{b}: {same}")
    if same == 'NO':
        print(f"   ref   {A[a][1]}")
        print(f"   split {B[b][1]}")
PY
echo "== parts done =="
