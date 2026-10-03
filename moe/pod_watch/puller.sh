#!/bin/bash
# pod_watch/puller.sh — 5-minute artifact puller (survival sync).
#
# Keeps local copies of checkpoints/logs current so a zero-balance stop or a
# watchdog kill loses at most one poll interval.  The SPEND GUARD (not this
# script) owns sha-verification and termination; this script only fetches.
# Optional verify pass (PW_VERIFY=1): compares each local qwen4exp-corr-*.pt
# against its remote sha and logs OK/MISMATCH (remote reads are sequential
# and cheap; default off to keep the hot loop to one rsync pair).
#
# Additional config (see common.sh for the shared PW_*):
#   PW_REMOTE_ART  remote artifact dir (default /workspace/artifacts/qwen4exp)
#   PW_REMOTE_LOGS remote log dir (default /workspace/logs)
#   PW_VERIFY      0/1 (default 0)
# Exits 0 when the remote train.rc reads 0 (final fetch done), like the P2b
# puller; the spend guard handles every other stop condition.

set -u
HERE=$(dirname "${BASH_SOURCE[0]}")
# shellcheck source=common.sh
source "$HERE/common.sh"

PW_REMOTE_ART="${PW_REMOTE_ART:-/workspace/artifacts/qwen4exp}"
PW_REMOTE_LOGS="${PW_REMOTE_LOGS:-/workspace/logs}"
PW_VERIFY="${PW_VERIFY:-0}"

mkdir -p "$PW_DST" "$PW_DST/logs"
pwlog "puller start host=${PW_SSH_HOST:?PW_SSH_HOST is required} verify=$PW_VERIFY"

sync_once() {
    local -a opts
    mapfile -t opts < <(pw_ssh_opts)
    rsync -a --partial --timeout=120 -e "ssh ${opts[*]}" \
        --include='qwen4exp-corr-*.pt' --include='qwen4exp-eval-*.json' \
        --include='*.log' --include='*.rc' --exclude='*' \
        "${PW_SSH_HOST:?}:$PW_REMOTE_ART/" "$PW_DST/" || return 1
    rsync -a --partial --timeout=120 -e "ssh ${opts[*]}" \
        "${PW_SSH_HOST:?}:$PW_REMOTE_LOGS/" "$PW_DST/logs/" || return 1
    return 0
}

verify_once() {
    local f rsha lsha
    for f in "$PW_DST"/qwen4exp-corr-*.pt; do
        [ -e "$f" ] || continue
        rsha=$(pw_remote sha256sum "$PW_REMOTE_ART/$(basename "$f")" \
            2>/dev/null | awk '{print $1}')
        lsha=$(sha256sum "$f" | awk '{print $1}')
        if [ -n "$rsha" ] && [ "$rsha" = "$lsha" ]; then
            pwlog "verify OK $(basename "$f") ${lsha:0:12}"
        else
            rdis="${rsha:0:12}"; rdis="${rdis:-absent}"
            pwlog "verify PENDING $(basename "$f") (remote $rdis, local ${lsha:0:12})"
        fi
    done
}

fails=0
iter=0
while true; do
    ts=$(date -u +%FT%TZ)
    if sync_once >/dev/null 2>&1; then
        fails=0
        pwlog "$ts ok $(ls -1 "$PW_DST" | tr '\n' ' ')"
        [ "$PW_VERIFY" = "1" ] && verify_once
    else
        fails=$((fails + 1))
        pwlog "$ts sync FAILED (consecutive $fails)"
        if [ "$fails" -ge 24 ]; then
            pwlog "$ts giving up after 2 h of failures"
            exit 1
        fi
    fi
    if rc=$(pw_remote cat "$PW_REMOTE_LOGS/train.rc" 2>/dev/null) && [ "$rc" = "0" ]; then
        pwlog "$ts train rc=0 — final fetch done, exiting"
        exit 0
    fi
    iter=$((iter + 1))
    pw_should_stop_after "$iter" && break
done
pwlog "puller: max iterations reached — exiting"
exit 0
