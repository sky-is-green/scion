#!/bin/bash
# pod_watch/stage_watchdog.sh — kill a stalled training stage, fetch partials.
#
# New in the cost-control protocol (no P2b equivalent — attempt 8 ground for
# ~5.5 h with no watchdog and crossed the $44 stop).  Watches ONE detached
# stage (see stage-exec.sh: /workspace/logs/<stage>.{log,rc,pid}):
#   - no new log content for PW_LOG_IDLE_S (default 600) → STALL;
#   - no `^step N` progress for PW_STEP_IDLE_S (default 900) once a step has
#     been seen → STALL (covers "log moves but training doesn't", e.g. a slow
#     offload path; the step guard arms only after the first step, so the
#     ~40-min load+build is governed by the log-line guard alone).
# On STALL: TERM the remote process group (setsid session leader in the pid
# file; pkill fallback), rsync partials + logs, exit 2.  Otherwise exit 0
# after PW_MAX_ITERS (or loop forever when 0).
#
# Remote clock is used for both timestamps (no local/remote skew issue).
# All remote commands are single simple commands (see common.sh rule 2).
#
# Additional config (see common.sh for the shared PW_*):
#   PW_STAGE        stage name (default train)
#   PW_REMOTE_LOGS  remote log dir (default /workspace/logs)
#   PW_REMOTE_ART   remote artifact dir (default /workspace/artifacts/qwen4exp)
#   PW_LOG_IDLE_S   seconds without new log content before STALL (default 600)
#   PW_STEP_IDLE_S  seconds without step progress before STALL (default 900)
#   PW_TAIL_BYTES   remote log bytes inspected per cycle (default 65536)
#   PW_KILL_PATTERN pkill fallback pattern (default qwen4exp_proxy.py <stage>)

set -u
HERE=$(dirname "${BASH_SOURCE[0]}")
# shellcheck source=common.sh
source "$HERE/common.sh"

PW_STAGE="${PW_STAGE:-train}"
PW_REMOTE_LOGS="${PW_REMOTE_LOGS:-/workspace/logs}"
PW_REMOTE_ART="${PW_REMOTE_ART:-/workspace/artifacts/qwen4exp}"
PW_LOG_IDLE_S="${PW_LOG_IDLE_S:-600}"
PW_STEP_IDLE_S="${PW_STEP_IDLE_S:-900}"
PW_TAIL_BYTES="${PW_TAIL_BYTES:-65536}"
PW_KILL_PATTERN="${PW_KILL_PATTERN:-qwen4exp_proxy.py $PW_STAGE}"
STATE_FILE="${PW_DST:?PW_DST is required}/.stage-watch-$PW_STAGE"

RLOG="$PW_REMOTE_LOGS/$PW_STAGE.log"
RPID="$PW_REMOTE_LOGS/$PW_STAGE.pid"
RRC="$PW_REMOTE_LOGS/$PW_STAGE.rc"

mkdir -p "$PW_DST"

last_step="" last_step_t=0 last_sig="" last_sig_t=0
if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
fi
save_state() {
    printf 'last_step=%s\nlast_step_t=%s\nlast_sig=%s\nlast_sig_t=%s\n' \
        "$last_step" "$last_step_t" "$last_sig" "$last_sig_t" > "$STATE_FILE"
}

fetch_partials() {
    local -a opts
    mapfile -t opts < <(pw_ssh_opts)
    rsync -a --partial --timeout=600 -e "ssh ${opts[*]}" \
        --include='qwen4exp-corr-*.pt' --include='qwen4exp-eval-*.json' \
        --include='*.log' --include='*.rc' --exclude='*' \
        "${PW_SSH_HOST:?}:$PW_REMOTE_ART/" "$PW_DST/" >/dev/null 2>&1 || \
        pwlog "WARN: partial artifact rsync rc=$?"
    mkdir -p "$PW_DST/logs"
    rsync -a --partial --timeout=600 -e "ssh ${opts[*]}" \
        "${PW_SSH_HOST:?}:$PW_REMOTE_LOGS/" "$PW_DST/logs/" >/dev/null 2>&1 || \
        pwlog "WARN: partial log rsync rc=$?"
    return 0
}

kill_remote_stage() {
    # TERM the whole process group (stage-exec runs under setsid, so the pid
    # file holds a session leader); fall back to pkill on the stage pattern.
    local pid
    pid=$(pw_remote cat "$RPID" 2>/dev/null | tr -cd '0-9') || pid=""
    if [ -n "$pid" ]; then
        pw_remote kill -TERM "-$pid" 2>/dev/null || \
            pw_remote kill -TERM "$pid" 2>/dev/null || true
        pwlog "sent SIGTERM to remote pgid/pid $pid"
    else
        pwlog "no remote pid file; pkill fallback: $PW_KILL_PATTERN"
    fi
    pw_remote pkill -f "$PW_KILL_PATTERN" 2>/dev/null || true
}

stall() {
    # stall <reason> — kill, fetch, exit 2 (distinct from guard exit 0/1 so a
    # supervisor can tell "killed a stall" from "spend stop" from "config").
    pwlog "STALL: $1"
    kill_remote_stage
    fetch_partials
    save_state
    exit 2
}

pwlog "stage_watchdog start stage=$PW_STAGE host=${PW_SSH_HOST:?PW_SSH_HOST is required} log_idle=${PW_LOG_IDLE_S}s step_idle=${PW_STEP_IDLE_S}s"

iter=0
while true; do
    ts=$(date -u +%FT%TZ)
    if ! rnow=$(pw_remote date +%s 2>/dev/null) || [ -z "$rnow" ]; then
        pwlog "$ts remote clock unreadable (ssh down?) — skipping cycle"
        iter=$((iter + 1))
        pw_should_stop_after "$iter" && break
        continue
    fi
    tail_text=$(pw_remote tail -c "$PW_TAIL_BYTES" "$RLOG" 2>/dev/null) || tail_text=""
    if [ -z "$tail_text" ]; then
        pwlog "$ts remote log $RLOG unreadable — skipping cycle"
        iter=$((iter + 1))
        pw_should_stop_after "$iter" && break
        continue
    fi

    sig=$(printf '%s' "$tail_text" | sha256sum | awk '{print $1}')
    step=$(printf '%s\n' "$tail_text" | grep -oE '^step [0-9]+' | tail -1 | awk '{print $2}')

    if [ -z "$last_sig" ]; then
        last_sig="$sig"; last_sig_t="$rnow"   # first observation arms the clock
    elif [ "$sig" != "$last_sig" ]; then
        last_sig="$sig"; last_sig_t="$rnow"
    elif [ $((rnow - last_sig_t)) -gt "$PW_LOG_IDLE_S" ]; then
        stall "no new log content for $((rnow - last_sig_t))s (limit ${PW_LOG_IDLE_S}s)"
    fi

    if [ -n "$step" ]; then
        if [ -z "$last_step" ]; then
            last_step="$step"; last_step_t="$rnow"   # arms the step guard
        elif [ "$step" != "$last_step" ]; then
            pwlog "$ts progress: step $last_step -> $step"
            last_step="$step"; last_step_t="$rnow"
        elif [ $((rnow - last_step_t)) -gt "$PW_STEP_IDLE_S" ]; then
            stall "no step progress for $((rnow - last_step_t))s at step $step (limit ${PW_STEP_IDLE_S}s)"
        fi
    fi
    save_state

    # A finished stage (rc file present) is not a stall — hand over to the
    # spend guard by exiting 0.
    if rrc=$(pw_remote cat "$RRC" 2>/dev/null) && [ -n "$rrc" ]; then
        pwlog "$ts stage $PW_STAGE exited rc=$rrc — watchdog done"
        break
    fi

    pwlog "$ts ok step=${step:-none} sig=${sig:0:12}"
    iter=$((iter + 1))
    pw_should_stop_after "$iter" && break
done
save_state
exit 0
