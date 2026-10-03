"""Tests for the rental watchdog suite (``moe/pod_watch/``).

P2b post-mortem pins (all reproduced here against fakes — no pods, no spend):
  - the stopper called bare ``ssh $SSHO '<cmd>'`` with NO destination host, so
    every remote check failed and ``2>/dev/null || true`` swallowed it: step3000
    sat on the pod ~45 min while the guard logged "step3000 absent";
  - the $0.50 balance floor fired only in the else-branch and was a constant;
  - the low-balance path terminated WITHOUT a final fetch;
  - there was no stage watchdog (attempt 8 ground ~5.5 h unwatched).

The scripts only touch the network through ``ssh``/``curl``/``rsync`` found on
PATH, so every test runs them against fake implementations driven by a fake
pod filesystem + a fake clock.  Nothing here spends money or needs a GPU.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

POD_WATCH = Path(__file__).resolve().parents[1] / "pod_watch"
SCRIPTS = ["common.sh", "spend_guard.sh", "stage_watchdog.sh", "puller.sh",
           "stage-exec.sh"]

FAKE_SSH = """\
#!/bin/bash
# Fake ssh: first non-flag *@* arg is the host, the rest is the remote command.
# Exits 99 when no host is given (real ssh fails the same way) — this is what
# makes the P2b missing-host regression test bite.
args=("$@")
host=""; ci=-1
for i in "${!args[@]}"; do
  case "${args[$i]}" in
    -*) continue;;
  esac
  # NOTE: plain glob, quoted literal — this box's bash mis-matches `@*`
  # inside case patterns (matches empty only), so `@` never appears in a
  # pattern here.
  if [[ "${args[$i]}" == *"$FAKE_HOSTFrag"* ]]; then
    host="${args[$i]}"; ci=$((i + 1)); break
  fi
done
echo "SSH host=[$host] cmd=[${args[@]:$ci}]" >> "$CALL_LOG"
if [ -z "$host" ]; then echo "ssh: no destination (fake usage failure)" >&2; exit 99; fi
if [ "${FAKE_SSH_DOWN:-0}" = "1" ]; then
  echo "ssh: connect to host $host port 22: Connection timed out" >&2; exit 255
fi
cmd=("${args[@]:$ci}")
R="$FAKE_REMOTE_ROOT"
case "${cmd[0]}" in
  true) exit 0;;
  echo) echo "${cmd[@]:1}";;
  cat)
    f="$R${cmd[1]}"
    [ -f "$f" ] && { cat "$f"; exit 0; }
    echo "cat: ${cmd[1]}: No such file" >&2; exit 1;;
  sha256sum)
    f="$R${cmd[1]}"
    [ -f "$f" ] && { sha256sum "$f" | sed "s|$f|${cmd[1]}|"; exit 0; }
    echo "sha256sum: ${cmd[1]}: No such file" >&2; exit 1;;
  date) cat "$FAKE_CLOCK";;
  stat) stat -c %Y "$R${cmd[3]}";;
  tail) tail -c "${cmd[2]}" "$R${cmd[3]}";;
  kill|pkill) echo "KILLCMD ${cmd[*]}" >> "$CALL_LOG"; exit 0;;
  *) echo "fake-ssh: unknown cmd ${cmd[*]}" >&2; exit 2;;
esac
"""

FAKE_CURL = """\
#!/bin/bash
# Fake curl: records argv (Bearer token redacted), serves balance + DELETE.
clean=()
for a in "$@"; do
  case "$a" in
    *Bearer*) clean+=("Bearer-REDACTED");;
    *) clean+=("$a");;
  esac
done
echo "CURL ${clean[*]}" >> "$CALL_LOG"
for a in "$@"; do
  case "$a" in
    *api_key*) echo "CURL-LEAK api_key in argv" >> "$CALL_LOG";;
  esac
done
if [[ "$*" == *graphql* ]]; then
  printf '{"data":{"myself":{"clientBalance": %s}}}' "$(cat "$FAKE_BALANCE")"
  exit 0
fi
echo '{}'
exit 0
"""

FAKE_RSYNC = """\
#!/bin/bash
# Fake rsync: copies host:path -> local dst (file or dir contents).
echo "RSYNC $*" >> "$CALL_LOG"
src=""; n=$#
for a in "$@"; do
  # NOTE: see the `@*` glob caveat above — no `@` in patterns.
  if [[ "$a" == *"$FAKE_HOSTFrag"*":"* ]]; then src="$a"; fi
done
dst="${!n}"
if [ -z "$src" ]; then echo "rsync: no host in src" >&2; exit 99; fi
rpath="${src#*:}"
rf="$FAKE_REMOTE_ROOT$rpath"
if [ "${FAKE_RSYNC_CORRUPT:-0}" = "1" ]; then
  echo "RSYNC-CORRUPT (test-induced bit flip)" >> "$CALL_LOG"
fi
if [ -f "$rf" ]; then
  [ -d "$dst" ] && dst="$dst/$(basename "$rf")"
  mkdir -p "$(dirname "$dst")"; cp "$rf" "$dst"
elif [ -d "$rf" ]; then
  mkdir -p "$dst"; cp -a "$rf/." "$dst/"
else
  echo "rsync: missing $rpath" >&2; exit 1
fi
if [ "${FAKE_RSYNC_CORRUPT:-0}" = "1" ] && [ -f "$dst" ]; then
  printf 'X' >> "$dst"
fi
exit 0
"""


@pytest.fixture()
def fakeworld(tmp_path, monkeypatch):
    """A fake pod + fake network + env to run the scripts against."""
    w = tmp_path / "w"
    bindir = w / "bin"
    remote = w / "remote"
    rart = remote / "workspace" / "artifacts" / "qwen4exp"
    rlogs = remote / "workspace" / "logs"
    dst = w / "dst"
    for d in (bindir, rart, rlogs, dst):
        d.mkdir(parents=True)
    for name, body in (("ssh", FAKE_SSH), ("curl", FAKE_CURL), ("rsync", FAKE_RSYNC)):
        p = bindir / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (w / "key").write_text("TESTKEY-SENTINEL-abcdef\n")
    (w / "clock").write_text("1700000000\n")
    (w / "balance").write_text("10.0\n")
    (w / "calls.log").write_text("")
    env = {
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "PW_SSH_HOST": "root@fake",
        "PW_SSH_PORT": "2222",
        "PW_SSH_KEY": str(w / "nokey"),
        "PW_KEY_FILE": str(w / "key"),
        "PW_POD": "fakepod1",
        "PW_DST": str(dst),
        "PW_LOG": str(dst / "watch.log"),
        "PW_POLL": "0",
        "PW_MAX_ITERS": "2",
        "FAKE_REMOTE_ROOT": str(remote),
        "FAKE_CLOCK": str(w / "clock"),
        "FAKE_BALANCE": str(w / "balance"),
        "CALL_LOG": str(w / "calls.log"),
        "FAKE_SSH_DOWN": "0",
        "FAKE_RSYNC_CORRUPT": "0",
        "FAKE_HOSTFrag": "root@fake",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return {
        "w": w, "bin": bindir, "remote": remote, "rart": rart, "rlogs": rlogs,
        "dst": dst, "env": env, "balance": w / "balance", "clock": w / "clock",
    }


def run(script, *args, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", str(POD_WATCH / script), *args],
                          capture_output=True, text=True, timeout=120, env=env)


def calls(w):
    return (w["w"] / "calls.log").read_text()


def ssh_calls(w):
    return [ln for ln in calls(w).splitlines() if ln.startswith("SSH ")]


def log(w):
    p = w["dst"] / "watch.log"
    return p.read_text() if p.exists() else ""


# --- script hygiene -------------------------------------------------------

@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_syntax(script):
    proc = subprocess.run(["bash", "-n", str(POD_WATCH / script)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"{script} has a syntax error:\n{proc.stderr}"


# --- spend guard ----------------------------------------------------------

def test_spend_guard_ssh_always_names_host(fakeworld):
    """P2b regression: every ssh call carries the destination host.

    The fake ssh exits 99 (like real ssh with no host) when the host is
    missing, so a script with the old bare-`ssh $SSHO '<cmd>'` bug would log
    SSH CHECK FAILED here instead of passing silently.
    """
    w = fakeworld
    proc = run("spend_guard.sh", extra_env={"PW_MAX_ITERS": "2"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert ssh_calls(w), "expected ssh calls, got none"
    for ln in ssh_calls(w):
        assert ln.startswith("SSH host=[root@fake]"), f"ssh without host: {ln}"
    assert "SSH CHECK FAILED" not in log(w)


def test_spend_guard_ckpt_match_fetches_then_terminates(fakeworld):
    w = fakeworld
    blob = bytes(range(256)) * 400  # 100 KiB
    (w["rart"] / "ckpt-step3000.pt").write_bytes(blob)
    proc = run("spend_guard.sh", extra_env={
        "PW_MAX_ITERS": "10", "PW_CKPTS": "ckpt-step3000.pt"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "sha-verified" in log(w)
    assert (w["dst"] / "ckpt-step3000.pt").read_bytes() == blob
    # Termination happened, and the fetch preceded it (the old low-balance
    # path terminated first and fetched never).
    cl = calls(w)
    assert "/v1/pods/" in cl, "pod was never terminated"
    assert cl.index("RSYNC") < cl.index("/v1/pods/"), \
        "terminate must come after the fetch"


def test_spend_guard_sha_mismatch_does_not_terminate(fakeworld):
    w = fakeworld
    (w["rart"] / "ckpt-step3000.pt").write_bytes(b"remote-bytes")
    proc = run("spend_guard.sh", extra_env={
        "PW_MAX_ITERS": "2", "PW_CKPTS": "ckpt-step3000.pt",
        "FAKE_RSYNC_CORRUPT": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "pending" in log(w)
    assert "/v1/pods/" not in calls(w), "terminated on an UNVERIFIED fetch"


def test_spend_guard_low_balance_fetches_before_terminate(fakeworld):
    """The old stopper deleted the pod with no final fetch on this path."""
    w = fakeworld
    (w["rlogs"] / "train.log").write_text("step 100 lm 1.0\n")
    (w["balance"]).write_text("0.5\n")  # under the default $2 floor
    proc = run("spend_guard.sh", extra_env={"PW_MAX_ITERS": "3"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "balance" in log(w) and "reserve" in log(w)
    cl = calls(w)
    assert "/v1/pods/" in cl
    assert cl.index("RSYNC") < cl.index("/v1/pods/"), \
        "low-balance stop must fetch BEFORE terminating"
    assert (w["dst"] / "logs" / "train.log").exists()


def test_spend_guard_reserve_math_and_cap(fakeworld):
    w = fakeworld
    # rate 4.59, fetch 15 + poll 0 + margin 10 -> 4.59*25/60 = 1.9125 -> floor 2.0
    proc = run("spend_guard.sh", extra_env={
        "PW_MAX_ITERS": "1", "PW_RATE_PER_HR": "4.59",
        "PW_FETCH_MIN": "15", "PW_MARGIN_MIN": "10"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "reserve=2" in log(w)
    # margin 30 -> 4.59*45/60 = 3.4425 (above the floor)
    proc = run("spend_guard.sh", extra_env={
        "PW_MAX_ITERS": "1", "PW_RATE_PER_HR": "4.59",
        "PW_FETCH_MIN": "15", "PW_MARGIN_MIN": "30",
        "PW_LOG": str(w["dst"] / "watch2.log")})
    assert "reserve=3.4425" in (w["dst"] / "watch2.log").read_text()
    # window cap: start 10, now 5, cap 6, reserve 2 -> spent 5 >= 4 -> stop
    (w["dst"] / ".spend-guard-state").write_text("start_balance=10.0\n")
    (w["balance"]).write_text("5.0\n")
    proc = run("spend_guard.sh", extra_env={
        "PW_MAX_ITERS": "3", "PW_WINDOW_CAP": "6",
        "PW_LOG": str(w["dst"] / "watch3.log")})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "window spend" in (w["dst"] / "watch3.log").read_text()
    assert "/v1/pods/" in calls(w)
    # ...but balance 7.5 (spent 2.5 < 4) keeps running.
    (w["w"] / "calls.log").write_text("")
    (w["balance"]).write_text("7.5\n")
    proc = run("spend_guard.sh", extra_env={
        "PW_MAX_ITERS": "1", "PW_WINDOW_CAP": "6",
        "PW_LOG": str(w["dst"] / "watch4.log")})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "spent=2.5" in (w["dst"] / "watch4.log").read_text()
    assert "/v1/pods/" not in calls(w)


def test_spend_guard_ssh_failure_is_loud_never_blind(fakeworld):
    """P2b failure mode: silent `2>/dev/null || true` on every probe.

    ssh down + healthy balance must ALERT loudly, keep polling, and never
    terminate (there is nothing verified to terminate for).
    """
    w = fakeworld
    proc = run("spend_guard.sh", extra_env={
        "PW_MAX_ITERS": "2", "PW_SSH_FAIL_MAX": "2", "FAKE_SSH_DOWN": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    lg = log(w)
    assert "SSH CHECK FAILED" in lg
    assert "Connection timed out" in lg, "ssh stderr must be logged, not swallowed"
    assert "ALERT" in lg
    assert "pod left up" in lg
    assert "/v1/pods/" not in calls(w)


def test_key_never_lands_in_logs_or_urls(fakeworld):
    """API key: Authorization header only; never in a URL, log, or state file."""
    w = fakeworld
    (w["rart"] / "ckpt-a.pt").write_bytes(b"data")
    run("spend_guard.sh", extra_env={
        "PW_MAX_ITERS": "10", "PW_CKPTS": "ckpt-a.pt"})
    cl = calls(w)
    assert "api_key" not in cl, "key-in-URL leak (the old GraphQL call)"
    assert "CURL-LEAK" not in cl
    assert "Bearer-REDACTED" in cl, "expected Authorization-header auth"
    # calls.log records test-harness argv (like ps) — the key-hygiene target
    # is everything the SCRIPTS write: logs, state, artifacts.
    for p in w["dst"].rglob("*"):
        if p.is_file() and p.name != "ckpt-a.pt":
            assert "TESTKEY-SENTINEL" not in p.read_text(errors="replace"), \
                f"API key leaked into {p}"


# --- stage watchdog -------------------------------------------------------

def set_clock(w, t):
    (w["w"] / "clock").write_text(f"{t}\n")


def test_stage_watchdog_stale_log_kills_and_fetches(fakeworld):
    w = fakeworld
    (w["rlogs"] / "train.log").write_text("step 2950 lm 2.4 kd 0.7\n")
    (w["rlogs"] / "train.pid").write_text("3411\n")
    (w["rart"] / "partial.pt").write_bytes(b"partial-ckpt")
    set_clock(w, 1700000000)
    proc = run("stage_watchdog.sh", extra_env={"PW_MAX_ITERS": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    set_clock(w, 1700000000 + 700)  # past the 600 s log-idle limit
    proc = run("stage_watchdog.sh", extra_env={"PW_MAX_ITERS": "1"})
    assert proc.returncode == 2, proc.stderr[-2000:] + log(w)[-2000:]
    lg = log(w)
    assert "STALL" in lg and "no new log content" in lg
    cl = calls(w)
    assert "KILLCMD" in cl, "stalled stage was never killed"
    assert "-3411" in cl, "expected process-group kill of the stage pid"
    assert (w["dst"] / "partial.pt").read_bytes() == b"partial-ckpt"
    assert (w["dst"] / "logs" / "train.log").exists()


def test_stage_watchdog_step_stall_with_moving_log(fakeworld):
    """Log lines arrive (eval chatter) but no `^step` progress: still a stall."""
    w = fakeworld
    t0 = 1700000000
    (w["rlogs"] / "train.log").write_text("step 100 lm 3.0\n")
    set_clock(w, t0)
    assert run("stage_watchdog.sh", extra_env={"PW_MAX_ITERS": "1"}).returncode == 0
    with open(w["rlogs"] / "train.log", "a") as f:
        f.write("[eval] step 100 ppl 5.8\n")
    set_clock(w, t0 + 500)
    assert run("stage_watchdog.sh", extra_env={"PW_MAX_ITERS": "1"}).returncode == 0
    with open(w["rlogs"] / "train.log", "a") as f:
        f.write("[eval] still evaluating\n")
    set_clock(w, t0 + 1000)  # step idle > 900 s
    proc = run("stage_watchdog.sh", extra_env={"PW_MAX_ITERS": "1"})
    assert proc.returncode == 2, proc.stderr[-2000:] + log(w)[-2000:]
    assert "no step progress" in log(w)


def test_stage_watchdog_progress_no_action(fakeworld):
    w = fakeworld
    t0 = 1700000000
    for i, step in enumerate((100, 200, 300)):
        (w["rlogs"] / "train.log").write_text(f"step {step} lm 2.0\n")
        set_clock(w, t0 + i * 300)
        proc = run("stage_watchdog.sh", extra_env={"PW_MAX_ITERS": "1"})
        assert proc.returncode == 0, proc.stderr[-2000:]
    assert "progress: step 200 -> 300" in log(w)
    assert "KILLCMD" not in calls(w)


def test_stage_watchdog_load_phase_not_killed(fakeworld):
    """No `^step` yet (40-min load+build) with a growing log: never a stall."""
    w = fakeworld
    t0 = 1700000000
    (w["rlogs"] / "train.log").write_text("loading shard 1/39\n")
    set_clock(w, t0)
    assert run("stage_watchdog.sh", extra_env={"PW_MAX_ITERS": "1"}).returncode == 0
    with open(w["rlogs"] / "train.log", "a") as f:
        f.write("loading shard 39/39\n")
    set_clock(w, t0 + 3600)  # far past the step limit, but no step ever seen
    proc = run("stage_watchdog.sh", extra_env={"PW_MAX_ITERS": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "KILLCMD" not in calls(w)


def test_stage_watchdog_finished_stage_hands_over(fakeworld):
    w = fakeworld
    (w["rlogs"] / "train.log").write_text("step 100 lm 1.0\ndone\n")
    (w["rlogs"] / "train.rc").write_text("0\n")
    proc = run("stage_watchdog.sh", extra_env={"PW_MAX_ITERS": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "exited rc=0" in log(w)
    assert "KILLCMD" not in calls(w)


# --- puller ---------------------------------------------------------------

def test_puller_survival_sync_and_rc_exit(fakeworld):
    w = fakeworld
    blob = b"ckpt-bytes" * 1000
    (w["rart"] / "qwen4exp-corr-r512-g128-step1000-cur05.pt").write_bytes(blob)
    (w["rlogs"] / "train.log").write_text("step 1000 lm 2.4\n")
    proc = run("puller.sh", extra_env={"PW_MAX_ITERS": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (w["dst"] / "qwen4exp-corr-r512-g128-step1000-cur05.pt").read_bytes() == blob
    assert (w["dst"] / "logs" / "train.log").exists()
    # rc=0 ends the puller with the final-fetch message.
    (w["rlogs"] / "train.rc").write_text("0\n")
    proc = run("puller.sh", extra_env={
        "PW_MAX_ITERS": "5", "PW_LOG": str(w["dst"] / "pull.log")})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "final fetch done" in (w["dst"] / "pull.log").read_text()


def test_puller_verify_flag(fakeworld):
    w = fakeworld
    (w["rart"] / "qwen4exp-corr-x.pt").write_bytes(b"v1")
    proc = run("puller.sh", extra_env={
        "PW_MAX_ITERS": "1", "PW_VERIFY": "1",
        "PW_LOG": str(w["dst"] / "vpull.log")})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "verify OK qwen4exp-corr-x.pt" in (w["dst"] / "vpull.log").read_text()


# --- stage-exec (the detached runner the watchdog reads) ------------------

def test_stage_exec_writes_log_pid_rc(tmp_path):
    """Local run: the three files the watchdog/guard consume appear."""
    logs = tmp_path / "logs"
    script = POD_WATCH / "stage-exec.sh"
    env = dict(os.environ, PW_REMOTE_LOGS=str(logs))
    proc = subprocess.run(
        ["bash", str(script), "demo", "bash", "-c",
         "echo hello-stage; exit 7"],
        env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert "stage=demo pid=" in proc.stdout
    # the runner detaches; wait for the rc file to appear
    rc = logs / "demo.rc"
    for _ in range(100):
        if rc.exists() and rc.read_text().strip():
            break
        __import__("time").sleep(0.1)
    assert rc.read_text().strip() == "7"
    assert "hello-stage" in (logs / "demo.log").read_text()
    pid = (logs / "demo.pid").read_text().strip()
    assert pid.isdigit() and int(pid) > 0
