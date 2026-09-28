"""Every stage in phase1-w1.sh must pass flags the stage's script accepts.

This fired for real: the ``kld`` and ``kld-body`` stages forwarded the training
recipe array, which carries ``--kd-weight``/``--temp``.  ``kld_eval.py`` has no
such flags, so both stages died on argparse before touching the GPU -- after two
~90-minute training runs had already completed.  The runner and its targets are
now checked against each other so a flag change cannot silently break a stage.
"""
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

MOE = Path(__file__).resolve().parents[1]
RUNNER = MOE / "phase1-w1.sh"

# CPU-only venv: enough for the argparse imports, no GPU touched
sys.path.insert(0, str(MOE))

# stage name in the runner -> the script it invokes
TARGETS = {"kld": "kld_eval.py", "kld-body": "kld_eval.py", "steer": "steer_probe.py"}


def _stage_blocks() -> dict[str, list[str]]:
    """Command lines per stage, from the ``case`` arms of the runner.

    Parsed by hand rather than with a regex over the whole file: the arms are
    indented two spaces and end at ``;;``, and an earlier regex silently matched
    nothing (returned ``{}``) instead of failing, which is how three stages
    ended up untested for a while.
    """
    text = RUNNER.read_text()
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        # arms are top-level case labels: `ref)`, `kld-body)`, `*)`
        arm = re.match(r"^([a-z][a-z|-]*)\)\s*$", line)
        if arm:
            current = None if arm.group(1) == "*" else arm.group(1)
            if current:
                blocks[current] = []
            continue
        if current and line.strip() == ";;":
            current = None
            continue
        if current:
            blocks[current].append(line.rstrip())
    return {k: v for k, v in blocks.items() if v}


def test_runner_exists_and_is_valid_bash():
    proc = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_shared_and_recipe_arrays_are_declared():
    """The arrays exist and the KD-only knobs are kept out of RECIPE_EVAL."""
    text = RUNNER.read_text()
    assert "SHARED=(" in text
    assert "RECIPE=(" in text
    assert "RECIPE_EVAL=(" in text
    m = re.search(r"RECIPE_EVAL=\((.*?)\)", text, re.S)
    assert m, "RECIPE_EVAL not found"
    # these are loss knobs of the trainer; kld_eval.py has no such flags
    for bad in ("--kd-weight", "--temp", "--kd-filter-frac"):
        assert bad not in m.group(1), f"{bad} leaked into RECIPE_EVAL"


def _flags_of(stage: str) -> list[str]:
    """The ``--flag value`` pairs a stage hands its script, shell arrays expanded.

    Two things this has to get right, both learned the hard way:

    - Comments are stripped first; prose like ``--windows/--corpus-chars`` is not
      a flag and would have made the checks pass for the wrong reason.
    - Every flag gets a value.  Bare ``--flag`` tokens make argparse fail with
      "expected one argument", which is a *different* error from "unrecognized
      arguments" -- so a bare-flag check would never notice a flag the script
      does not define.
    """
    body = "\n".join(_stage_blocks()[stage])
    body = body.replace("\\\n", " ")
    body = "\n".join(ln for ln in body.splitlines()
                     if not ln.strip().startswith("#"))
    for name in ("SHARED", "RECIPE", "RECIPE_EVAL"):
        m = re.search(rf"^{name}=\((.*?)\)", RUNNER.read_text(), re.S | re.M)
        if m:
            body = body.replace(f'"${{{name}[@]}}"', m.group(1))
    # Live shell vars ($ckpt, $MOE/..., $PY) become a placeholder; values that
    # came out of the arrays above are real and must survive, because a choice
    # flag like --quant rejects a dummy ("invalid choice: '1'") before argparse
    # ever reaches the "unrecognized arguments" check we care about.
    body = re.sub(r"\$\{[^}]*\}|\$\w+", "1", body)

    out: list[str] = []
    tokens = shlex.split(body)
    for i, tok in enumerate(tokens):
        if not tok.startswith("--") or len(tok) < 3:
            continue
        out.append(tok)
        nxt = tokens[i + 1] if i + 1 < len(tokens) else "1"
        out.append(nxt if not nxt.startswith("--") else "1")
    return out


@pytest.mark.parametrize("stage,script", sorted(TARGETS.items()))
def test_stage_flags_are_accepted(stage, script):
    blocks = _stage_blocks()
    assert stage in blocks, f"stage {stage} not found in runner"
    flags = _flags_of(stage)
    assert flags, f"stage {stage} passes no flags; parser is probably stale"
    # Parse against the script's real parser.  `--help` would exit 0 before
    # argparse ever complains about an unknown flag, which is how the original
    # version of this test passed while the stages were broken.
    mod = __import__(Path(script).stem)
    parser = mod.build_parser()
    try:
        parser.parse_args(flags)
    except SystemExit:
        pytest.fail(f"stage {stage} passes flags {script} rejects: {flags}")


def test_train_stage_passes_a_tag():
    """Both arms share rank/branch-quant/steps, so --tag is what keeps them apart."""
    train = "\n".join(_stage_blocks()["train"])
    assert "--tag" in train, \
        "train stage must pass --tag or arms overwrite each other"


def test_the_flag_check_would_catch_the_regression(tmp_path):
    """Guard against a check that passes for the wrong reason.

    Put the training recipe array back on the kld stage -- the exact mistake that
    killed the first gate run after two completed trainings -- and confirm
    kld_eval.py does reject it.
    """
    global RUNNER
    good = RUNNER.read_text()
    bad_text = good.replace('"${RECIPE_EVAL[@]}"', '"${RECIPE[@]}"')
    assert bad_text != good, "patch did not apply; the check is not exercised"
    bad = tmp_path / "phase1-w1.sh"
    bad.write_text(bad_text)

    orig, RUNNER = RUNNER, bad
    try:
        flags = _flags_of("kld")
    finally:
        RUNNER = orig
    assert any(f in flags for f in ("--kd-weight", "--temp")), \
        "patch did not change the flags; the check is not exercised"
    import kld_eval
    with pytest.raises(SystemExit):
        kld_eval.build_parser().parse_args(flags)


def test_every_case_arm_is_covered_by_a_test():
    """No stage may exist in the runner without a flag-compatibility check."""
    blocks = _stage_blocks()
    # ref/cache/train are validated by the proxy's own flag set at run time;
    # these three are the ones the runner composes from RECIPE_EVAL / PYTHONPATH
    assert set(TARGETS) <= set(blocks), \
        f"runner stages not under test: {sorted(set(blocks) - set(TARGETS))}"
