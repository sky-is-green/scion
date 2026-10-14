#!/bin/bash
# ep-ab-window.sh - the served A/B for the EP HIP port, in one exclusive window.
# Arm P: scion-flashnext's pipelined config (layer split + --pipeline-windows 2), from the same
#        strata-ep build as arm E, so only the EP change differs.
# Arm E: --expert-parallel-device 1 (strata-ep-b.json).
# For each arm: the fixed bench (4 requests), a 400-token decode with both cards' busy sampled,
# and one 128-token greedy request saved for the output comparison.
# Prereq: the box is free (stop scion-flashnext / other engines first); restores nothing itself.
set -u
ROOT=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work
PY=$ROOT/strata/.venv/bin/python
EP=$ROOT/strata-ep
PROMPT='You are helping tune a local model. Write a detailed, factual paragraph about the history of computing. Begin now and do not stop early.'
OUT=/tmp/opencode

busy() {  # $1 seconds; prints avg/max busy + avg power per drm card
  local secs=$1 c h
  rm -f $OUT/ab-busy.0 $OUT/ab-busy.1 $OUT/ab-pow.0 $OUT/ab-pow.1
  for _ in $(seq 1 $((secs * 10))); do
    for c in 0 1; do
      cat /sys/class/drm/card$c/device/gpu_busy_percent >> $OUT/ab-busy.$c 2>/dev/null
      h=$(echo /sys/class/drm/card$c/device/hwmon/hwmon*/power1_average)
      cat $h >> $OUT/ab-pow.$c 2>/dev/null
    done
    sleep 0.1
  done
  for c in 0 1; do
    awk -v card=$c '{s+=$1; if($1>m)m=$1; n++} END {printf "  drm card%d (HIP %d): busy avg %.1f%% max %d%%", card, (card==1?0:1), s/n, m} END{}' $OUT/ab-busy.$c
    awk '{s+=$1; if($1>m)m=$1; n++} END {printf " | power avg %.0f W max %.0f W\n", s/n/1e6, m/1e6}' $OUT/ab-pow.$c
  done
}

wait_ready() {
  for _ in $(seq 1 60); do
    if curl -s --max-time 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"loaded": *true'; then return 0; fi
    sleep 5
  done
  echo "  ARM FAILED TO BECOME READY"; return 1
}

run_arm() {
  local name=$1 cfg=$2
  echo "================ arm $name ================"
  systemctl --user reset-failed "ab-$name" 2>/dev/null
  systemd-run --user --unit="ab-$name" --collect --property=WorkingDirectory=$EP \
    $PY serve/server.py --engine strata --config "$cfg" --port 8081 >/dev/null 2>&1
  wait_ready || { systemctl --user stop "ab-$name"; return 1; }
  echo "--- fixed bench (4 requests) ---"
  (cd $ROOT/scion && bash serving/bench_strata.sh "ab-$name" 8081) 2>&1
  echo "--- 400-token decode + card busy ---"
  ( curl -s --max-time 300 "http://127.0.0.1:8081/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Write a detailed, factual essay about the history of computing from the abacus to the GPU. Do not stop early."}],"max_tokens":400,"temperature":0}' \
      > $OUT/ab-$name-long.json ) &
  local bg=$!
  sleep 4
  busy 12
  wait $bg
  python3 -c "
import json
d=json.load(open('$OUT/ab-$name-long.json'))
t=d.get('timings',{})
print('  long 400: prompt_n', t.get('prompt_n'), 'gen_tps', round(t.get('predicted_per_second',0),1), 'tokens', t.get('predicted_n'))
" 2>/dev/null
  echo "--- 128-token output capture (the bench prompt) ---"
  curl -s --max-time 240 "http://127.0.0.1:8081/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":128,\"temperature\":0}" > $OUT/ab-$name-128.json
  python3 -c "
import json
d=json.load(open('$OUT/ab-$name-128.json'))
m=d['choices'][0]['message']
txt=(m.get('reasoning_content') or '')+'\n===CONTENT===\n'+(m.get('content') or '')
open('$OUT/ab-$name-128.txt','w').write(txt)
t=d.get('timings',{})
print('  128: gen_tps', round(t.get('predicted_per_second',0),1), 'len', len(txt))
" 2>/dev/null
  systemctl --user stop "ab-$name"
  sleep 3
  grep -E 'expert parallel|zero-doorbell|VRAM free|pre-filled' $OUT/strata_ep_$( [ "$name" = "pipe" ] && echo pipe || echo b ).engine.log 2>/dev/null | tail -6
}

run_arm pipe $EP/strata-ep-pipe.json
run_arm ep   $EP/strata-ep-b.json
echo "================ output comparison ================"
if cmp -s $OUT/ab-pipe-128.txt $OUT/ab-ep-128.txt; then
  echo "  128-token outputs byte-identical"
else
  echo "  outputs differ; first difference:"
  cmp $OUT/ab-pipe-128.txt $OUT/ab-ep-128.txt | head -2
fi
echo "done"
