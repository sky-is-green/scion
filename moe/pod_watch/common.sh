#!/bin/bash
# pod_watch/common.sh — shared helpers for the Flash-Next rental watchdogs.
#
# Sourced, never executed.  Two rules that the P2b post-mortem paid for:
#   1. ssh is ALWAYS called with the destination host (the P2b stopper called
#      bare `ssh $SSHO '<cmd>'` with no host, so every remote check silently
#      failed and stderr was swallowed by `2>/dev/null || true`).
#   2. Remote commands are single simple commands with NO pipes/quotes; all
#      parsing happens locally.  There is no quoting to get wrong.
#
# Config (environment, PW_ prefix):
#   PW_SSH_HOST   user@ip of the pod (required)
#   PW_SSH_KEY    ssh key path (default $HOME/.ssh/id_runpod)
#   PW_SSH_PORT   ssh port (required — RunPod assigns it per pod)
#   PW_POD        RunPod pod id (required for terminate)
#   PW_KEY_FILE   RunPod API key file (default $HOME/.runpod-api-key)
#   PW_DST        local artifact dir (required)
#   PW_LOG        log file (default $PW_DST/pod-watch.log)
#   PW_POLL       seconds between cycles (default 300)
#   PW_MAX_ITERS  cycle cap, 0 = forever (default 0)
#
# The API key is read from PW_KEY_FILE and NEVER printed to any log/file.
# Auth uses the Authorization header, never ?api_key= in the URL (the old
# stopper put the key in the GraphQL URL, which lands in proxy/server logs).

# shellcheck disable=SC2034
PW_SSH_KEY="${PW_SSH_KEY:-$HOME/.ssh/id_runpod}"
PW_KEY_FILE="${PW_KEY_FILE:-$HOME/.runpod-api-key}"
PW_POLL="${PW_POLL:-300}"
PW_MAX_ITERS="${PW_MAX_ITERS:-0}"
PW_LOG="${PW_LOG:-${PW_DST:-/tmp/opencode}/pod-watch.log}"

pwlog() {
    # pwlog <msg...> — timestamped line to PW_LOG and stdout.  Callers must
    # never pass secrets here (nothing in this file logs $K).
    local ts line
    ts=$(date -u +%FT%TZ)
    line="$ts $*"
    echo "$line" >> "$PW_LOG"
    echo "$line"
}

pw_ssh_opts() {
    # Prints the ssh option prefix, one token per line, for mapfile.
    printf '%s\n' -i "$PW_SSH_KEY" -p "${PW_SSH_PORT:?PW_SSH_PORT is required}" \
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20
}

pw_remote() {
    # pw_remote <simple-command...> — run one simple remote command.
    # The host is MANDATORY (structural fix for the P2b missing-$SRC bug:
    # there is no code path that calls ssh without a destination).
    # No pipes, no redirects, no quotes in the remote command — parse locally.
    local host="${PW_SSH_HOST:?PW_SSH_HOST is required}"
    local -a opts
    mapfile -t opts < <(pw_ssh_opts)
    ssh "${opts[@]}" "$host" "$@"
}

pw_api_key() {
    # Prints the API key to stdout (for command substitution only).
    cat "${PW_KEY_FILE:?PW_KEY_FILE is required}"
}

pw_balance() {
    # Prints the current RunPod client balance (a float), or empty on failure.
    local k resp
    k=$(pw_api_key)
    resp=$(curl -s --max-time 30 -X POST https://api.runpod.io/graphql \
        -H 'Content-Type: application/json' \
        -H "Authorization: Bearer $k" \
        -d '{"query":"query { myself { clientBalance } }"}' 2>/dev/null) || return 0
    python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)["data"]["myself"]["clientBalance"])
except Exception:
    pass' <<< "$resp" 2>/dev/null
}

pw_terminate_pod() {
    # pw_terminate_pod <reason> — DELETE the pod via the REST API.
    local reason="$1" k out
    k=$(pw_api_key)
    out=$(curl -s --max-time 60 -X DELETE \
        "https://rest.runpod.io/v1/pods/${PW_POD:?PW_POD is required}" \
        -H "Authorization: Bearer $k" 2>&1) || out="curl rc=$?"
    pwlog "terminate response: $(head -c 300 <<< "$out")"
    pwlog "pod ${PW_POD} termination requested: $reason"
}

pw_rsync_from() {
    # pw_rsync_from <remote-path> <local-dst> [extra rsync args...] —
    # single-path fetch that always names the host (same structural fix).
    local src="$1" dst="$2"
    shift 2
    local host="${PW_SSH_HOST:?PW_SSH_HOST is required}"
    local -a opts
    mapfile -t opts < <(pw_ssh_opts)
    rsync -a --partial -e "ssh ${opts[*]}" "$@" "$host:$src" "$dst"
}

pw_should_stop_after() {
    # Loop helper: returns 0 (stop) when PW_MAX_ITERS iters are done AND
    # sleeps PW_POLL between iterations otherwise.  Usage:
    #   iter=0; while true; do ...; iter=$((iter+1)); pw_should_stop_after "$iter" && break; done
    # With PW_POLL=0 there is no sleep (unit tests).
    local iter="$1"
    if [ "$PW_MAX_ITERS" -gt 0 ] && [ "$iter" -ge "$PW_MAX_ITERS" ]; then
        return 0
    fi
    if [ "$PW_POLL" -gt 0 ]; then
        sleep "$PW_POLL"
    fi
    return 1
}
