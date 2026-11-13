#!/usr/bin/env python3
"""Re-create the heldout harness clone and make it runnable on this Linux box.

The harness grades coding items by executing candidate code through
``/usr/bin/sandbox-exec -p <profile> <PYTHON_FOR_TESTS> -I candidate.py`` -
both macOS paths.  The original hard-90 runs on this box used local shims
for those paths (unsandboxed execution, same isolated CPython); the shims
are gone and /usr/bin is not writable without a password, so this script
patches the fresh clone instead, with the same semantics:

  - grade.py: drop the sandbox-exec wrapper (plain subprocess; the runner
    already executes in a tempfile dir with a 10 s timeout);
  - config.py: PYTHON_FOR_TESTS -> the local isolated CPython 3.12.

Idempotent: each patch is applied only when the macOS line is still there.
Run after every fresh clone (see EVAL.md; the clone lives in /tmp and is
wiped by reboots).

Usage: python v2/scripts/setup_harness.py [--clone-dir /tmp/opencode/bonsai2-27b-test]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

CLONE = "https://github.com/heldoutlab/bonsai2-27b-test"
LOCAL_PY = shutil.which("python3.12") or ""


def patch(path: Path, old: str, new: str, label: str) -> bool:
    text = path.read_text()
    if new in text:
        print(f"  {label}: already patched")
        return False
    if old not in text:
        raise SystemExit(f"  {label}: expected line not found in {path}")
    path.write_text(text.replace(old, new, 1))
    print(f"  {label}: patched")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone-dir", default="/tmp/opencode/bonsai2-27b-test")
    args = ap.parse_args()
    d = Path(args.clone_dir)

    if not (d / "harness/grade.py").exists():
        print(f"cloning {CLONE} -> {d}")
        subprocess.run(["git", "clone", "-q", CLONE, str(d)], check=True)
    grade = d / "harness/grade.py"
    config = d / "harness/config.py"

    if not LOCAL_PY:
        raise SystemExit("no python3.12 on PATH for PYTHON_FOR_TESTS")
    patch(grade,
          '["/usr/bin/sandbox-exec", "-p", SANDBOX, PYTHON_FOR_TESTS, "-I", path],',
          '[PYTHON_FOR_TESTS, "-I", path],',
          "grade.py sandbox-exec")
    patch(config,
          'PYTHON_FOR_TESTS = "/opt/homebrew/bin/python3.12"',
          f'PYTHON_FOR_TESTS = "{LOCAL_PY}"',
          "config.py PYTHON_FOR_TESTS")
    print(f"harness ready at {d}")


if __name__ == "__main__":
    main()
