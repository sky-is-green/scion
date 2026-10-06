"""The runner's lock and CARD wiring, checked for real.

The lock exists because its absence was expensive: two copies of the same
--kd-weight arm ran concurrently, both wrote the same checkpoint path, and the
box OOM'd before either finished, leaving a file that was neither one's. The
guard has to fail *loudly* to be worth anything, so these tests run the actual
script rather than re-implementing the logic.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MOE = Path(__file__).resolve().parents[1]
RUNNER = MOE / "phase1-w1.sh"


def _run(args, env=None, timeout=120):
    e = dict(os.environ)
    e.setdefault("LOCKDIR", os.path.join(tempfile.gettempdir(), "test-lock"))
    e.update(env or {})
    return subprocess.run(["bash", str(RUNNER), *args], capture_output=True,
                          text=True, env=e, timeout=timeout)


def test_runner_is_valid_bash():
    proc = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_every_stage_takes_the_lock():
    """Any stage that loads the FP prefix must be exclusive.

    Walked structurally: find the case arm whose body calls acquire(), collect
    the stage names in its pattern list, and require every real stage label to
    appear. A regex over the whole file previously matched nothing and failed
    silently, which is exactly the failure this suite exists to catch.
    """
    lines = RUNNER.read_text().splitlines()
    locked: set[str] = set()
    found = False
    for i, line in enumerate(lines):
        if re.match(r"^case\b.*\bin\s*$", line):
            # the pattern list runs until a line ending in ')'
            j = i + 1
            while j < len(lines) and not lines[j].rstrip().endswith(")"):
                j += 1
            body = "\n".join(lines[j + 1:j + 6])
            if "acquire" in body:
                for grp in lines[i + 1:j + 1]:
                    for name in grp.rstrip(")").split("|"):
                        if re.fullmatch(r"[a-z][a-z0-9-]*", name.strip()):
                            locked.add(name.strip())
                found = True
            break
    assert found, "no case arm calls acquire(); the lock is not wired in"
    stages = {s for s in re.findall(r"^([a-z][a-z0-9-]*)\)\s*$", "\n".join(lines), re.M)
              if s != "*"}
    assert stages, "no stages found"
    assert stages <= locked, f"stages not guarded: {sorted(stages - locked)}"


def test_a_second_stage_is_refused(tmp_path):
    """Holding the lock must make a second invocation exit non-zero."""
    lock = tmp_path / "lock"
    lock.mkdir()
    (lock / "pid").write_text("999999")          # a pid that is not running
    proc = _run(["steer"], env={"LOCKDIR": str(lock)})
    assert proc.returncode == 3, f"expected refusal, got {proc.returncode}"
    assert "REFUSING" in proc.stderr
    assert "STALE_PID" in proc.stderr, "the refusal must say how to reclaim"


def test_a_stale_lock_can_be_reclaimed(tmp_path):
    """A lock left by a killed run must not block the box forever.

    PY is pointed at /bin/true so the stage returns immediately: the point is
    whether it got *past* the lock, and a real python would try to load the
    model and hang the test.
    """
    lock = tmp_path / "lock"
    lock.mkdir()
    (lock / "pid").write_text("999999")          # not a running pid
    proc = _run(["steer"], env={"LOCKDIR": str(lock), "STALE_PID": "1",
                                "PY": "/bin/true"}, timeout=60)
    assert "REFUSING" not in proc.stderr
    assert "reclaiming stale lock" in proc.stderr


def test_the_lock_is_released_on_exit(tmp_path):
    """A stage that fails must not leave the box locked for the next run."""
    lock = tmp_path / "lock"
    shutil.rmtree(lock, ignore_errors=True)
    proc = _run(["steer"], env={"LOCKDIR": str(lock), "PY": "/bin/false"},
                timeout=60)
    assert not lock.exists(), f"lock leaked: {list(lock.iterdir()) if lock.exists() else ''}"


def test_the_lock_reports_its_holder():
    text = RUNNER.read_text()
    assert "$LOCKDIR/pid" in text, "the refusal must name the holding pid"
    assert "another stage is running" in text.lower()


def test_card_is_honoured_and_logged():
    """Two cards are available, but memory -- not cards -- is the limit.

    The comment has to keep saying that, or the next person runs three jobs.
    """
    text = RUNNER.read_text()
    assert 'CARD="${CARD:-1}"' in text
    assert "HIP_VISIBLE_DEVICES=$CARD" in text
    assert "### CARD=$CARD" in text, "each stage must log which card it used"
    assert re.search(r"17 GB|~17 GB", text), "the host-RAM warning was dropped"
    assert "One job at a TIME, not one job per card" in text


def test_sweep_stages_still_pass_their_tags():
    """The lock is additive; the per-weight tag must survive it."""
    text = RUNNER.read_text()
    assert '--tag "kdw$w"' in text, "the sweep must still tag by weight"
    assert '--tag "lmonly"' in text
    assert '--tag "arm$arm"' in text
