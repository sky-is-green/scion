#!/usr/bin/env bash
# A/B on the current kernel: single card vs the FFN split with direct peer DMA (share 0.5).
set -u
# Working-tree locations come from the environment; the defaults are the lane's layout, where the engine
# worktrees are siblings of this repository's parent (../strata-bonsai2, ../strata).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OCTMP=${STRATA_TMP:-$OCTMP}
W=${STRATA_BONSAI2_WORKTREE:-$ROOT/strata-bonsai2}
VENV=${STRATA_VENV_PY:-$ROOT/strata/.venv/bin/python}
UNIT=strata-dense-bonsai
LOGDIR=$OCTMP
mkdir -p "$LOGDIR"

restart_serve() {
  systemctl --user stop $UNIT 2>/dev/null
  systemctl --user start scion-flashnext 2>/dev/null
  for i in $(seq 1 60); do
    c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null || echo 000)
    [ "$c" = "200" ] && { echo "serve back: HTTP 200 (after ~$((i*3))s)"; return; }
    sleep 3
  done
  echo "serve check failed"
}
trap restart_serve EXIT

ask() {
  curl -s http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"bonsai-2-27b","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":24,"temperature":0}' \
    > "$LOGDIR/ab-$1-fr.json" 2>/dev/null
  echo -n "  france: "
  "$VENV" - <<PY
import json
d = json.load(open('$LOGDIR/ab-$1-fr.json'))
print((d['choices'][0]['message'].get('content') or d.get('error', {}).get('message', ''))[:80].strip())
PY
  curl -s http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"bonsai-2-27b","messages":[{"role":"user","content":"Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall."}],"max_tokens":512,"temperature":0}' \
    > "$LOGDIR/ab-$1-long.json" 2>/dev/null
  "$VENV" - <<PY
import json
d = json.load(open('$LOGDIR/ab-$1-long.json'))
print('  usage:', d.get('usage'))
PY
}

run_cfg() {
  echo "== $1 =="
  systemctl --user stop scion-flashnext 2>/dev/null
  sleep 4
  systemd-run --user --unit=$UNIT --collect --property=Nice=5 \
    --property=WorkingDirectory=$W \
    --property=StandardOutput=append:$LOGDIR/strata-dense-bonsai-server.log \
    --property=StandardError=append:$LOGDIR/strata-dense-bonsai-server.log \
    $VENV serve/server.py --engine strata --config "$2" --port 8082
  local ready=no
  for i in $(seq 1 120); do
    c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8082/v1/models 2>/dev/null || echo 000)
    [ "$c" = "200" ] && { ready=yes; break; }
    systemctl --user is-active --quiet $UNIT || { echo "  engine unit died"; break; }
    sleep 3
  done
  echo "  ready: $ready (~$((i*3))s)"
  if [ "$ready" = "yes" ]; then ask "$1"; fi
  systemctl --user stop $UNIT 2>/dev/null
  sleep 2
}

echo "== window opens =="; date
run_cfg base strata-bonsai-2-27b.json
run_cfg peer strata-bonsai-2-27b-split.json
echo "--- engine timing / split lines:"
grep -aE 'generated|FFN split|handoff' "$LOGDIR/strata-dense-bonsai.log" | tail -8
echo "== window closes =="; date
