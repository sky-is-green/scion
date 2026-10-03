#!/bin/bash
# pod_watch/stage-exec.sh — run ONE heavy stage detached, with the three files
# the stage watchdog + spend guard read (see stage_watchdog.sh):
#
#   /workspace/logs/<stage>.log   stdout+stderr of the stage
#   /workspace/logs/<stage>.pid   the setsid session leader pid
#   /workspace/logs/<stage>.rc    exit code, written when the stage ends
#
# Usage (ON the pod, from the repo root):
#   bash scion/moe/pod_watch/stage-exec.sh <stage> <cmd...>
#
# The stage runs in a new session (setsid) so the watchdog can SIGTERM the
# whole process group; the rc file is written by the session leader itself,
# so a completed stage is never mistaken for a stall.
#
# Config (environment):
#   PW_REMOTE_LOGS  remote log dir (default /workspace/logs)
#
# The caller is responsible for the repo cwd and for not running two stages
# at once (box-run-flashnext.sh's stage lock does that).
set -u

STAGE="${1:?usage: stage-exec.sh <stage> <cmd...>}"
shift
LOGS="${PW_REMOTE_LOGS:-/workspace/logs}"
mkdir -p "$LOGS"
LOG="$LOGS/$STAGE.log"
PID="$LOGS/$STAGE.pid"
RC="$LOGS/$STAGE.rc"

rm -f "$RC" "$PID"
# setsid keeps the pid: background jobs in a non-interactive shell are not
# process-group leaders, so util-linux setsid execs in place instead of
# forking (the watchdog's `kill -TERM -$pid` then hits the whole group).
export PW_STAGE_RC="$RC"
setsid bash -c '"$@"; rc=$?; echo "$rc" > "$PW_STAGE_RC"; exit "$rc"' _ "$@" \
    >"$LOG" 2>&1 &
echo "$!" >"$PID"
echo "stage=$STAGE pid=$(cat "$PID") log=$LOG"
