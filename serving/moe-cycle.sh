#!/usr/bin/env bash
# moe-cycle.sh <tag> - cold-restart the live serve and run the identical warm-up protocol:
# fixed bench (4 requests) then 8 more of the same prompt, storing traj md5s.
# Used to separate engine changes from restart-path nondeterminism.
set -eu
W=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work
TAG=$1
D=/tmp/opencode/scion-moe-exact
mkdir -p "$D"
systemctl --user restart scion-flashnext
for i in $(seq 1 80); do
  c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null || echo 000)
  [ "$c" = "200" ] && break
  sleep 3
done
[ "$c" = "200" ] || { echo "serve did not come up"; exit 1; }
echo "== $TAG: serve ready at $(date +%H:%M:%S)"
bash "$W/scion/serving/bench_strata.sh" "$TAG-bench" | tee "$D/$TAG-bench.txt"
bash "$W/scion/serving/moe-exact.sh" "$TAG" 2>&1 | sed "s/^/  /"
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
