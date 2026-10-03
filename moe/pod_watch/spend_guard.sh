#!/bin/bash
# pod_watch/spend_guard.sh — per-window spend watchdog for rental runs.
#
# Replaces /tmp/opencode/p2b-stopper.sh.  Fixes from the P2b post-mortem:
#   1. The stopper called `ssh $SSHO '<cmd>'` with NO destination host, so the
#      remote-sha check failed on every cycle and stderr was swallowed —
#      step3000 sat on the pod ~45 min while the guard reported
#      "step3000 absent".  All remote calls go through pw_remote(), which
#      structurally requires PW_SSH_HOST.  A per-cycle `true` probe separates
#      "ssh is down" (loud ALERT + consecutive counter) from "file absent"
#      (normal while training).
#   2. The $0.50 balance floor was checked only in the else-branch (i.e. never
#      when the sha/rc probes "succeeded") and is a fixed constant.  The guard
#      now checks the balance EVERY cycle against a COMPUTED reserve:
#        reserve = max(PW_RESERVE_FLOOR,
#                      rate * (fetch_min + poll_min + margin_min) / 60)
#      so a $4.59/hr card with a 15-min fetch stops with dollars left, not cents.
#   3. The old low-balance path called terminate WITHOUT a final fetch.
#      Every stop path here fetches first, then terminates (order tested).
#   4. Per-window CAP: stop when (start_balance - now) >= cap - reserve.
#      The start balance is recorded in $PW_DST/.spend-guard-state on first run.
#   5. Auth via Authorization header, never ?api_key= in the URL.
#
# Additional config (see common.sh for the shared PW_*):
#   PW_WINDOW_CAP    per-window spend cap in $ (0 = disabled, default 0)
#   PW_RATE_PER_HR   pod burn rate $/hr (default 1.09, L40S)
#   PW_FETCH_MIN     worst-case final-fetch minutes (default 20)
#   PW_MARGIN_MIN    safety margin minutes (default 15)
#   PW_RESERVE_FLOOR absolute balance floor $ (default 2.0)
#   PW_CKPTS         space-separated target artifact filenames in
#                    $PW_REMOTE_ART (default: none — balance/cap only)
#   PW_REMOTE_ART    remote artifact dir (default /workspace/artifacts/qwen4exp)
#   PW_REMOTE_LOGS   remote log dir (default /workspace/logs)
#   PW_SSH_FAIL_MAX  consecutive ssh failures before ALERT (default 6; the
#                    guard keeps polling — it never terminates blind)
#   PW_KILL_SIBLING  pkill pattern for a sibling puller on stop (default: none)
#
# Exit 0 after a stop (pod termination requested); 1 on config error.

set -u
HERE=$(dirname "${BASH_SOURCE[0]}")
# shellcheck source=common.sh
source "$HERE/common.sh"

PW_WINDOW_CAP="${PW_WINDOW_CAP:-0}"
PW_RATE_PER_HR="${PW_RATE_PER_HR:-1.09}"
PW_FETCH_MIN="${PW_FETCH_MIN:-20}"
PW_MARGIN_MIN="${PW_MARGIN_MIN:-15}"
PW_RESERVE_FLOOR="${PW_RESERVE_FLOOR:-2.0}"
PW_CKPTS="${PW_CKPTS:-}"
PW_REMOTE_ART="${PW_REMOTE_ART:-/workspace/artifacts/qwen4exp}"
PW_REMOTE_LOGS="${PW_REMOTE_LOGS:-/workspace/logs}"
PW_SSH_FAIL_MAX="${PW_SSH_FAIL_MAX:-6}"
PW_KILL_SIBLING="${PW_KILL_SIBLING:-}"
STATE_FILE="${PW_DST:?PW_DST is required}/.spend-guard-state"

mkdir -p "$PW_DST"

compute_reserve() {
    # Prints the stop reserve in $.  awk handles the float math.
    awk -v rate="$PW_RATE_PER_HR" -v fetch="$PW_FETCH_MIN" \
        -v poll="$PW_POLL" -v margin="$PW_MARGIN_MIN" -v floor="$PW_RESERVE_FLOOR" \
        'BEGIN { r = rate * (fetch + poll/60 + margin) / 60; print (r > floor ? r : floor) }'
}

below() {
    # below <a> <b> — true (rc 0) iff a < b (floats).
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'
}

fetch_all() {
    # Best-effort full artifact+log sync.  Never fails the guard (rc always 0).
    local -a opts
    mapfile -t opts < <(pw_ssh_opts)
    rsync -a --partial --timeout=600 -e "ssh ${opts[*]}" \
        --include='qwen4exp-corr-*.pt' --include='qwen4exp-eval-*.json' \
        --include='*.log' --include='*.rc' --exclude='*' \
        "${PW_SSH_HOST:?}:$PW_REMOTE_ART/" "$PW_DST/" >/dev/null 2>&1 || \
        pwlog "WARN: artifact rsync rc=$?"
    mkdir -p "$PW_DST/logs"
    rsync -a --partial --timeout=600 -e "ssh ${opts[*]}" \
        "${PW_SSH_HOST:?}:$PW_REMOTE_LOGS/" "$PW_DST/logs/" >/dev/null 2>&1 || \
        pwlog "WARN: log rsync rc=$?"
    return 0
}

stop() {
    # stop <reason> — fetch everything, terminate the pod, exit 0.
    local reason="$1"
    pwlog "STOPPING: $reason"
    fetch_all
    pw_terminate_pod "$reason"
    if [ -n "$PW_KILL_SIBLING" ]; then
        pkill -f "$PW_KILL_SIBLING" >/dev/null 2>&1 || true
    fi
    pwlog "final local artifacts:"; ls -la "$PW_DST" | tail -8 >> "$PW_LOG"
    exit 0
}

remote_sha() {
    # remote_sha <filename> — remote sha256 of one artifact, or empty when the
    # file is absent.  The remote side runs ONE simple command; field-splitting
    # happens HERE, so there is no remote quoting to get wrong.
    pw_remote sha256sum "$PW_REMOTE_ART/$1" 2>/dev/null | awk '{print $1}'
}

RESERVE=$(compute_reserve)
pwlog "spend_guard start pod=${PW_POD:?PW_POD is required} host=${PW_SSH_HOST:?PW_SSH_HOST is required} cap=$PW_WINDOW_CAP reserve=$RESERVE ckpts=[$PW_CKPTS]"

ssh_fails=0
iter=0
while true; do
    ts=$(date -u +%FT%TZ)

    # 0. Connectivity probe: separates "ssh down" from "file absent".
    if probe_err=$(pw_remote true 2>&1); then
        ssh_fails=0
        ssh_ok=1
    else
        ssh_fails=$((ssh_fails + 1))
        ssh_ok=0
        # Loud, not silent: the P2b failure hid here behind `2>/dev/null || true`.
        pwlog "$ts SSH CHECK FAILED (consecutive $ssh_fails): $probe_err"
        if [ "$ssh_fails" -ge "$PW_SSH_FAIL_MAX" ]; then
            pwlog "$ts ALERT: ssh down $ssh_fails cycles; still polling (balance guard protects spend); NOT terminating blind"
        fi
    fi

    if [ "$ssh_ok" = "1" ]; then
        # 1. Target checkpoints: remote sha -> fetch -> local sha -> stop on match.
        for ckpt in $PW_CKPTS; do
            rsha=$(remote_sha "$ckpt")
            if [ -n "$rsha" ]; then
                pw_rsync_from "$PW_REMOTE_ART/$ckpt" "$PW_DST/" --timeout=900 \
                    >/dev/null 2>&1 || pwlog "WARN: ckpt rsync rc=$? for $ckpt"
                lsha=$(sha256sum "$PW_DST/$ckpt" 2>/dev/null | awk '{print $1}')
                if [ -n "$lsha" ] && [ "$rsha" = "$lsha" ]; then
                    stop "$ckpt fetched + sha-verified (${rsha:0:12})"
                fi
                ldis="${lsha:0:12}"; ldis="${ldis:-none}"
                pwlog "$ts $ckpt present but fetch/sha pending (remote ${rsha:0:12}, local $ldis)"
            fi
        done

        # 2. Train exit marker (rc file = the detached runner finished).
        if rc=$(pw_remote cat "$PW_REMOTE_LOGS/train.rc" 2>/dev/null); then
            if [ -n "$rc" ]; then
                stop "train process exited rc=$rc"
            fi
        fi
    else
        pwlog "$ts skipping ssh-dependent checks (ssh down)"
    fi

    # 3. Balance + window cap — EVERY cycle, independent of ssh (API path).
    if bal=$(pw_balance) && [ -n "$bal" ]; then
        if [ ! -f "$STATE_FILE" ]; then
            echo "start_balance=$bal" > "$STATE_FILE"
            pwlog "$ts window start_balance=$bal"
        fi
        # shellcheck disable=SC1090
        source "$STATE_FILE"
        spent=$(awk -v s="$start_balance" -v b="$bal" 'BEGIN { print s - b }')
        cap_line=""
        if below "$bal" "$RESERVE"; then
            stop "balance \$$bal < reserve \$$RESERVE"
        fi
        if awk -v c="$PW_WINDOW_CAP" 'BEGIN { exit !(c > 0) }'; then
            stop_at=$(awk -v c="$PW_WINDOW_CAP" -v r="$RESERVE" 'BEGIN { print c - r }')
            cap_line=" spent=$spent (cap-stop at $stop_at)"
            if awk -v s="$spent" -v t="$stop_at" 'BEGIN { exit !(s >= t) }'; then
                stop "window spend \$$spent >= cap-reserve \$$stop_at (cap \$$PW_WINDOW_CAP)"
            fi
        fi
        pwlog "$ts running; ckpts pending; balance=$bal reserve=$RESERVE$cap_line"
    else
        pwlog "$ts running; ckpts pending; balance query FAILED"
    fi

    iter=$((iter + 1))
    pw_should_stop_after "$iter" && break
done
pwlog "spend_guard: max iterations reached without a stop condition — pod left up"
