#!/usr/bin/env bash
# moe-pin-cycle.sh <tag> - the pinned-tier identity arm: the live serve config + --adapt-every 1000000
# (the fixed expert tier, the documented exactness setting), run in a serve-down window on 8081.
# Protocol: the fixed bench warm-up (4 requests) then 8 stored bodies, same as moe-cycle.sh.
set -eu
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/strata
VENV=$W/.venv/bin/python
TAG=$1
D=/tmp/opencode/scion-moe-exact
mkdir -p "$D"
systemctl --user stop scion-flashnext
sleep 3
systemd-run --user --unit=scion-flashnext-pin --collect --property=WorkingDirectory=$W \
  "$VENV" serve/server.py --engine strata --config strata-q6k-176b-pin.json --port 8081
for i in $(seq 1 100); do
  c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null || echo 000)
  [ "$c" = "200" ] && break
  systemctl --user is-active --quiet scion-flashnext-pin || { echo "pinned engine died"; break; }
  sleep 3
done
echo "== $TAG: pinned serve ready at $(date +%H:%M:%S)"
bash /run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/scion/serving/bench_strata.sh "$TAG-bench" | tee "$D/$TAG-bench.txt"
bash /run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/scion/serving/moe-exact.sh "$TAG" 2>&1 | sed "s/^/  /"
systemctl --user stop scion-flashnext-pin
systemctl --user start scion-flashnext
python3 - "$D" "$TAG" <<'PY'
import json, sys, os, hashlib
d, tag = sys.argv[1:3]
for i in range(1, 9):
    p = os.path.join(d, f'{tag}-g{i}.json')
    if not os.path.exists(p): continue
    m = json.load(open(p))['choices'][0]['message']
    r = (m.get('reasoning_content') or '') + (m.get('content') or '')
    print(f"  traj {tag} g{i}: len {len(r)} md5 {hashlib.md5(r.encode()).hexdigest()[:10]}")
PY
