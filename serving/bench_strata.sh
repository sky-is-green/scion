#!/bin/bash
# bench_strata.sh <label> - fixed 4-request decode benchmark against the local Strata server.
# Run 1 is warm-up (prompt cache/graphs); runs 2-4 report.
set -u
PORT=8081
LABEL=${1:-run}
P='You are helping tune a local model. Write a detailed, factual paragraph about the history of computing. Begin now and do not stop early.'
for i in 1 2 3 4; do
  curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":128,\"temperature\":0}" \
  | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    t=d.get('timings',{}) or {}
    print('$LABEL run $i: prompt_n', t.get('prompt_n'), 'prompt_tps', round(t.get('prompt_per_second',0),1),
          'gen_tps', round(t.get('predicted_per_second',0),1),
          'drafts', t.get('draft_n_accepted'), '/', t.get('draft_n'))
except Exception as e:
    print('$LABEL run $i: parse error', e)
"
done
