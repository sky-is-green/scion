#!/usr/bin/env bash
# One split-engine window: run <src-config> + <extra env json> as <tag>, ask the two acceptance questions,
# compare the replies against the stored reference (default: the fresh base run in /tmp/opencode/exact).
# Usage: bonsai-bisect.sh <tag> <src-config> '<env json>' [ref-tag]
set -u
# Working-tree locations come from the environment; the defaults are the lane's layout, where the engine
# worktrees are siblings of this repository's parent (../strata-bonsai2, ../strata).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OCTMP=${STRATA_TMP:-$OCTMP}
W=${STRATA_BONSAI2_WORKTREE:-$ROOT/strata-bonsai2}
VENV=${STRATA_VENV_PY:-$ROOT/strata/.venv/bin/python}
UNIT=strata-dense-bonsai
ENGLOG=$OCTMP/strata-dense-bonsai.log
LOG=$OCTMP/bisect
REF=$OCTMP/exact
mkdir -p "$LOG"
TAG=$1; SRC=$2; ENVJSON=$3; REFTAG=${4:-base}

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

python3 - "$W/$SRC" "$LOG/$TAG.json" "$ENVJSON" <<'PY'
import json, sys
src, dst, extra = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
d = json.load(open(src)); d.setdefault('env', {}).update(extra)
json.dump(d, open(dst, 'w'))
PY

echo "== $TAG =="
systemctl --user stop scion-flashnext 2>/dev/null
sleep 4
: > "$ENGLOG"
systemd-run --user --unit=$UNIT --collect --property=Nice=5 --property=WorkingDirectory=$W \
  --property=StandardOutput=append:$ENGLOG --property=StandardError=append:$ENGLOG \
  $VENV serve/server.py --engine strata --config "$LOG/$TAG.json" --port 8082
for i in $(seq 1 120); do
  c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8082/v1/models 2>/dev/null || echo 000)
  [ "$c" = "200" ] && { echo "  ready: yes"; break; }
  systemctl --user is-active --quiet $UNIT || { echo "  engine died"; break; }
  sleep 3
done
curl -s http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-2-27b","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64,"temperature":0}' \
  > "$LOG/$TAG-fr.json"
curl -s http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-2-27b","messages":[{"role":"user","content":"Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall."}],"max_tokens":512,"temperature":0}' \
  > "$LOG/$TAG-long.json"
sleep 1
cp "$ENGLOG" "$LOG/$TAG.engine.log" 2>/dev/null
systemctl --user stop $UNIT 2>/dev/null
sleep 2
python3 - "$LOG" "$REF" "$TAG" "$REFTAG" <<'PY'
import json, sys, os
log, ref, tag, reftag = sys.argv[1:5]
def load(d, t, w):
    j = json.load(open(os.path.join(d, f'{t}-{w}.json')))
    m = j['choices'][0]['message']
    return (m.get('reasoning_content') or '') + '\x00' + (m.get('content') or ''), j.get('usage', {})
for w in ('fr', 'long'):
    try:
        a, ua = load(ref, reftag, w)
    except Exception as e:
        print(f"-- {w}: no reference ({e})"); continue
    b, ub = load(log, tag, w)
    same = 'YES' if a == b else 'NO'
    print(f"-- {w}: ref {ua.get('completion_tokens')} tok, {tag} {ub.get('completion_tokens')} tok, identical: {same}")
    if a != b:
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"   first diff at char {i}: ref {a[max(0,i-30):i+30]!r} vs {tag} {b[max(0,i-30):i+30]!r}")
prof = []
for ln in open(os.path.join(log, tag + '.engine.log'), errors='replace'):
    s = ln.strip()
    if 'generated in' in s or 'FFN split ready' in s:
        print('  ', s)
    if 'strata-dense prof' in s:
        prof.append(s)
for s in prof[-2:]:
    print('  ', s)
PY
echo "== $TAG done =="
