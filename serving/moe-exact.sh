#!/usr/bin/env bash
# moe-exact.sh <tag> [port] - the Scion-MoE lane's warmed greedy identity capture.
# 4 requests of the fixed bench prompt (run 1 warms; 2-4 report) against the live serve;
# stores the full JSON per run, then reports which warmed bodies are byte-identical to
# the stored reference in /tmp/opencode/scion-moe-exact (default tag: base).
# The body compared is reasoning_content + content (the fork/Strata chat shape).
set -u
PORT=${2:-8081}
DIR=/tmp/opencode/scion-moe-exact
mkdir -p "$DIR"
TAG=$1
REFTAG=${3:-base}
P='You are helping tune a local model. Write a detailed, factual paragraph about the history of computing. Begin now and do not stop early.'
for i in 1 2 3 4 5 6 7 8; do
  curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":128,\"temperature\":0}" \
    > "$DIR/$TAG-g$i.json"
done
python3 - "$DIR" "$TAG" "$REFTAG" <<'PY'
import json, sys, os
d, tag, reftag = sys.argv[1:4]
def body(p):
    j = json.load(open(p))
    m = j['choices'][0]['message']
    return (m.get('reasoning_content') or '') + '\x00' + (m.get('content') or '')
for i in (4, 5, 6, 7, 8):
    a = os.path.join(d, f'{reftag}-g{i}.json'); b = os.path.join(d, f'{tag}-g{i}.json')
    if not os.path.exists(b):
        print(f"-- g{i}: missing {b}"); continue
    if not os.path.exists(a):
        print(f"-- g{i}: no reference ({a})"); continue
    x, y = body(a), body(b)
    same = 'YES' if x == y else 'NO'
    print(f"-- g{i}: ref {len(x)} chars, {tag} {len(y)} chars, identical: {same}")
    if x != y:
        k = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        print(f"   first diff at char {k}: ref {x[max(0,k-30):k+30]!r} vs {tag} {y[max(0,k-30):k+30]!r}")
PY
