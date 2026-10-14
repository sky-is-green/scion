#!/usr/bin/env bash
# Debug run: base vs split-dma vs split-staged, each with its own engine log, per-layer traces compared.
set -u
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/strata-bonsai2
VENV=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/strata/.venv/bin/python
UNIT=strata-dense-bonsai
ENGLOG=/tmp/opencode/strata-dense-bonsai.log
LOG=/tmp/opencode/dbg2
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
      -d '{"model":"bonsai-2-27b","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":2,"temperature":0}' \
      > "$LOG/$tag.reply.json"
    sleep 1
  fi
  cp "$ENGLOG" "$LOG/$tag.engine.log" 2>/dev/null
  systemctl --user stop $UNIT 2>/dev/null
  sleep 2
}

echo "== window opens =="; date
python3 - "$W" <<'PY'
import json, sys, shutil
w = sys.argv[1]
d = json.load(open(w + '/strata-bonsai-2-27b-split.json'))
d.setdefault('env', {})['STRATA_DENSE_SPLIT_STAGE'] = '1'
json.dump(d, open(w + '/strata-bonsai-2-27b-split-staged.json', 'w'), indent=1)
d2 = json.load(open(w + '/strata-bonsai-2-27b-split.json'))
d2['env'].pop('STRATA_DENSE_SPLIT_STAGE', None)
json.dump(d2, open(w + '/strata-bonsai-2-27b-split.json', 'w'), indent=1)
PY
run_cfg base strata-bonsai-2-27b.json
run_cfg dma strata-bonsai-2-27b-split.json
run_cfg staged strata-bonsai-2-27b-split-staged.json
echo "== traces =="
python3 - "$LOG" <<'PY'
import json, re, sys, os
log = sys.argv[1]
pat = re.compile(r'dbg pos=(\d+) layer=(-?\d+)\s+(\S+)\s+(\S+)\s+max\|v\|=(\S+)\s+rms=(\S+)\s+nan=(\d+) inf=(\d+)')
def parse(tag):
    out = {}
    p = os.path.join(log, tag + '.engine.log')
    for ln in open(p, errors='replace'):
        m = pat.match(ln)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = (float(m.group(5)), float(m.group(6)))
    return out
runs = {t: parse(t) for t in ('base', 'dma', 'staged')}
print("entries:", {t: len(r) for t, r in runs.items()})
layers = sorted({k[1] for k in runs['base'] if k[0] == 0})
for L in layers[:16] + layers[-3:]:
    k = (0, L)
    if all(k in r for r in runs.values()):
        b, d, s = runs['base'][k], runs['dma'][k], runs['staged'][k]
        print(f"layer {L:3d}: base {b[0]:10.5g} | dma {d[0]:10.5g} ({(d[0]-b[0])/b[0]*100:+7.3f}%)"
              f" | staged {s[0]:10.5g} ({(s[0]-b[0])/b[0]*100:+7.3f}%)   rms b/d/s {b[1]:.5g}/{d[1]:.5g}/{s[1]:.5g}")
for t in ('base', 'dma', 'staged'):
    try:
        d = json.load(open(os.path.join(log, t + '.reply.json')))
        print(t, "reply:", (d['choices'][0]['message'].get('reasoning_content') or '')[:60].replace('\n', ' '))
    except Exception as e:
        print(t, "reply: ?", e)
PY
echo "== window closes =="; date
