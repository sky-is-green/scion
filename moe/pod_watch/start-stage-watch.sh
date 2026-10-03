#!/bin/bash
# start-stage-watch.sh <UNIT> <STAGE> <IP> <PORT> <IDLE_S> — start one stage
# watchdog as a systemd user transient unit (unique unit per attempt).
set -u
UNIT="${1:?usage: start-stage-watch.sh <UNIT> <STAGE> <IP> <PORT> <IDLE_S>}"
STAGE="${2:?}"
IP="${3:?}"
PORT="${4:?}"
IDLE="${5:?}"
DST=/home/penis/Desktop/work/hivebench/artifacts/ternary/moe/qwen4exp/pod/remote-w1w2
WS=/home/penis/Desktop/work/scion/moe/pod_watch
mkdir -p "$DST"

systemctl --user stop "$UNIT" >/dev/null 2>&1 || true
systemctl --user reset-failed "$UNIT" >/dev/null 2>&1 || true

systemd-run --user --unit="$UNIT" --collect \
  --property=StandardOutput=append:"$DST/$UNIT.log" \
  --property=StandardError=append:"$DST/$UNIT.log" \
  --setenv=PW_STAGE="$STAGE" --setenv=PW_SSH_HOST="root@$IP" --setenv=PW_SSH_PORT="$PORT" \
  --setenv=PW_DST="$DST" --setenv=PW_LOG_IDLE_S="$IDLE" \
  --setenv=PW_KILL_PATTERN=qwen4exp_eval.py \
  bash "$WS/stage_watchdog.sh"

sleep 2
systemctl --user --no-pager --plain status "$UNIT" 2>&1 | grep -E "●|Active:|Main PID" | head -5
