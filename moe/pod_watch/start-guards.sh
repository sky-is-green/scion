#!/bin/bash
# start-guards.sh <POD> <IP> <PORT> <RATE> — start the two long-lived guards as
# systemd user transient units (survive harness restarts).
set -u
POD="${1:?usage: start-guards.sh <POD> <IP> <PORT> <RATE>}"
IP="${2:?}"
PORT="${3:?}"
RATE="${4:?}"
DST=/home/penis/Desktop/work/hivebench/artifacts/ternary/moe/qwen4exp/pod/remote-w1w2
WS=/home/penis/Desktop/work/scion/moe/pod_watch
mkdir -p "$DST"
TS=$(date -u +%Y%m%dT%H%M%SZ)
# fresh window: archive stale guard/watchdog state (the old spend state would
# make the cap math count the previous window's spend)
for f in "$DST/.spend-guard-state" "$DST"/.stage-watch-*; do
    [ -e "$f" ] && mv "$f" "$f.prev-$TS"
done

systemctl --user reset-failed w1w2-spend w1w2-pull >/dev/null 2>&1 || true

systemd-run --user --unit=w1w2-spend --collect \
  --property=StandardOutput=append:"$DST/w1w2-spend.log" \
  --property=StandardError=append:"$DST/w1w2-spend.log" \
  --setenv=PW_POD="$POD" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
  --setenv=PW_DST="$DST" --setenv=PW_WINDOW_CAP=6 --setenv=PW_RATE_PER_HR="$RATE" \
  --setenv=PW_FETCH_MIN=15 --setenv=PW_MARGIN_MIN=10 --setenv=PW_RESERVE_FLOOR=0.5 \
  --setenv=PW_CKPTS=qwen4exp-eval-kld-48l-step3000.json \
  --setenv=PW_KILL_SIBLING=pod_watch/puller.sh \
  bash "$WS/spend_guard.sh"

systemd-run --user --unit=w1w2-pull --collect \
  --property=StandardOutput=append:"$DST/w1w2-pull.log" \
  --property=StandardError=append:"$DST/w1w2-pull.log" \
  --setenv=PW_POD="$POD" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
  --setenv=PW_DST="$DST" --setenv=PW_VERIFY=1 \
  bash "$WS/puller.sh"

sleep 2
systemctl --user --no-pager --plain status w1w2-spend w1w2-pull 2>&1 | grep -E "●|Active:|Main PID" | head -12
