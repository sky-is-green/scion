#!/bin/bash
# w1w2-revive.sh — restart the long-lived spend/pull guards if the local
# machine/session restarted while the pod is still alive. Never touches the
# spend state file (the cap math continues), never archives watch state.
set -u
R=/home/penis/Desktop/work/hivebench/artifacts/ternary/moe/qwen4exp/pod/remote-w1w2
WS=/home/penis/Desktop/work/scion/moe/pod_watch
POD=qhuqr08i6aldeg
IP=64.247.206.76
PORT=10547
RATE=0.82

K=$(cat ~/.runpod-api-key 2>/dev/null) || exit 0
PODS=$(curl -s --max-time 30 https://rest.runpod.io/v1/pods -H "Authorization: Bearer $K" 2>/dev/null) || exit 0
echo "$PODS" | grep -q "$POD" || exit 0   # pod gone: nothing to guard

if ! systemctl --user is-active --quiet w1w2-spend; then
    systemctl --user reset-failed w1w2-spend >/dev/null 2>&1
    systemd-run --user --unit=w1w2-spend --collect \
      --property=StandardOutput=append:"$R/w1w2-spend.log" \
      --property=StandardError=append:"$R/w1w2-spend.log" \
      --setenv=PW_POD="$POD" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
      --setenv=PW_DST="$R" --setenv=PW_WINDOW_CAP=6.5 --setenv=PW_RATE_PER_HR="$RATE" \
      --setenv=PW_FETCH_MIN=15 --setenv=PW_MARGIN_MIN=10 --setenv=PW_RESERVE_FLOOR=0.5 \
      --setenv=PW_CKPTS=qwen4exp-eval-kld-48l-step3000.json \
      --setenv=PW_KILL_SIBLING=pod_watch/puller.sh \
      bash "$WS/spend_guard.sh" >/dev/null 2>&1
fi
if ! systemctl --user is-active --quiet w1w2-pull; then
    systemctl --user reset-failed w1w2-pull >/dev/null 2>&1
    systemd-run --user --unit=w1w2-pull --collect \
      --property=StandardOutput=append:"$R/w1w2-pull.log" \
      --property=StandardError=append:"$R/w1w2-pull.log" \
      --setenv=PW_POD="$POD" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
      --setenv=PW_DST="$R" --setenv=PW_VERIFY=1 \
      bash "$WS/puller.sh" >/dev/null 2>&1
fi
