#!/usr/bin/env bash
# ref (split config + SPLIT_OFF) vs peer (split config) with a position debug range, diffing every "dbg " line.
# Usage: bonsai-dbg.sh <A:B pos range> [extra-json-for-both]
set -u
# Working-tree locations come from the environment; the defaults are the lane's layout, where the engine
# worktrees are siblings of this repository's parent (../strata-bonsai2, ../strata).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OCTMP=${STRATA_TMP:-$OCTMP}
W=${STRATA_BONSAI2_WORKTREE:-$ROOT/strata-bonsai2}
VENV=${STRATA_VENV_PY:-$ROOT/strata/.venv/bin/python}
UNIT=strata-dense-bonsai
ENGLOG=$OCTMP/strata-dense-bonsai.log
LOG=$OCTMP/dbg
mkdir -p "$LOG"
RANGE=$1
EXTRA=${2:-{}}

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
  local tag=$1 forcesplit=$2
  python3 - "$W/strata-bonsai-2-27b-split.json" "$LOG/$tag.json" "$RANGE" "$forcesplit" "$EXTRA" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
env = {"STRATA_DENSE_DEBUG": sys.argv[3]}
if sys.argv[4] == 'off':
    env["STRATA_DENSE_SPLIT_OFF"] = "1"
env.update(json.loads(sys.argv[5]))
d.setdefault('env', {}).update(env)
json.dump(d, open(sys.argv[2], 'w'))
PY
  echo "== $tag =="
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

run_case ref off
run_case peer on

python3 - "$LOG" <<'PY'
import sys, os, re
log = sys.argv[1]
def dbg(path):
    out = []
    for ln in open(path, errors='replace'):
        if ln.startswith('dbg '):
            out.append(ln.rstrip('\n'))
    return out
A = dbg(os.path.join(log, 'ref.engine.log'))
B = dbg(os.path.join(log, 'peer.engine.log'))
print(f"dbg lines: ref {len(A)} peer {len(B)}")
n = 0
for i in range(max(len(A), len(B))):
    a = A[i] if i < len(A) else '<none>'
    b = B[i] if i < len(B) else '<none>'
    if a != b:
        # drop the pos/layer prefix for the comparison note? no, show both
        print(f"DIFF {i}:")
        print(f"  ref  {a}")
        print(f"  peer {b}")
        n += 1
        if n >= 12:
            print("  ... (stopping at 12)")
            break
if n == 0:
    print("ALL dbg lines identical")
PY
echo "== dbg comparison done =="
