#!/usr/bin/env bash
# FFN-split part trace: base vs split, layer 0 only.
set -u
# Working-tree locations come from the environment; the defaults are the lane's layout, where the engine
# worktrees are siblings of this repository's parent (../strata-bonsai2, ../strata).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OCTMP=${STRATA_TMP:-$OCTMP}
W=${STRATA_BONSAI2_WORKTREE:-$ROOT/strata-bonsai2}
VENV=${STRATA_VENV_PY:-$ROOT/strata/.venv/bin/python}
UNIT=strata-dense-bonsai
ENGLOG=$OCTMP/strata-dense-bonsai.log
LOG=$OCTMP/ffntr
mkdir -p "$LOG"

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

run_cfg() {
  local tag=$1 cfg=$2
  echo "== $tag =="
  systemctl --user stop scion-flashnext 2>/dev/null
  sleep 4
  : > "$ENGLOG"
  systemd-run --user --unit=$UNIT --collect --property=Nice=5 \
    --property=WorkingDirectory=$W \
    --property=StandardOutput=append:$ENGLOG \
    --property=StandardError=append:$ENGLOG \
    $VENV serve/server.py --engine strata --config "$cfg" --port 8082
  local ready=no
  for i in $(seq 1 120); do
    c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8082/v1/models 2>/dev/null || echo 000)
    [ "$c" = "200" ] && { ready=yes; break; }
    systemctl --user is-active --quiet $UNIT || { echo "  engine died"; break; }
    sleep 3
  done
  echo "  ready: $ready"
  if [ "$ready" = "yes" ]; then
    curl -s http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"bonsai-2-27b","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":1,"temperature":0}' \
      > "$LOG/$tag.reply.json"
    sleep 1
  fi
  cp "$ENGLOG" "$LOG/$tag.engine.log" 2>/dev/null
  systemctl --user stop $UNIT 2>/dev/null
  sleep 2
}

echo "== window opens =="; date
run_cfg base strata-bonsai-2-27b.json
run_cfg split strata-bonsai-2-27b-split.json
echo "== ffn traces (layer 0) =="
for t in base split; do
  echo "-- $t"
  grep -a "ffn_\|dbg layer" "$LOG/$t.engine.log" | head -40
done
echo "== window closes =="; date
