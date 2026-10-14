#!/usr/bin/env bash
# Acceptance: the split's token stream must be byte-identical to the single card's, and be faster.
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/strata-bonsai2
VENV=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/strata/.venv/bin/python
UNIT=strata-dense-bonsai
ENGLOG=/tmp/opencode/strata-dense-bonsai.log
LOG=/tmp/opencode/exact
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
      -d '{"model":"bonsai-2-27b","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64,"temperature":0}' \
      > "$LOG/$tag-fr.json"
    curl -s http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"bonsai-2-27b","messages":[{"role":"user","content":"Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall."}],"max_tokens":512,"temperature":0}' \
      > "$LOG/$tag-long.json"
    sleep 1
  fi
  cp "$ENGLOG" "$LOG/$tag.engine.log" 2>/dev/null
  systemctl --user stop $UNIT 2>/dev/null
  sleep 2
}

echo "== window opens =="; date
run_cfg base strata-bonsai-2-27b.json
run_cfg split strata-bonsai-2-27b-split.json
echo "== compare =="
python3 - "$LOG" <<'PY'
import json, sys, os
log = sys.argv[1]
def txt(tag, which):
    d = json.load(open(os.path.join(log, f'{tag}-{which}.json')))
    m = d['choices'][0]['message']
    return (m.get('reasoning_content') or '') + '\x00' + (m.get('content') or ''), d.get('usage', {})
for which in ('fr', 'long'):
    a, ua = txt('base', which)
    b, ub = txt('split', which)
    print(f"-- {which}: base {ua.get('completion_tokens')} tokens, split {ub.get('completion_tokens')}")
    print(f"   identical: {'YES' if a == b else 'NO'}")
    if a != b:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"   first diff at char {i}: base {a[max(0,i-30):i+30]!r} vs split {b[max(0,i-30):i+30]!r}")
                break
        else:
            print(f"   length differs: {len(a)} vs {len(b)}")
print("-- timings:")
for tag in ('base', 'split'):
    for ln in open(os.path.join(log, tag + '.engine.log'), errors='replace'):
        if 'generated in' in ln or 'FFN split ready' in ln:
            print(f"   [{tag}] {ln.strip()}")
PY
echo "== window closes =="; date
