"""Every GPU script must be runnable as a script, not just importable.

A missing ``if __name__ == "__main__"`` block exits 0 and prints nothing, which
on a GPU stage looks exactly like success -- it cost a wasted card slot once
already, so it is pinned here.  Both modules keep their heavy imports inside
main(), so ``--help`` works on the CPU-only test venv.
"""
import subprocess
import sys
from pathlib import Path

import pytest

MOE = Path(__file__).resolve().parents[1]
SCRIPTS = ["kld_eval.py", "steer_probe.py", "build_ayot_prompts.py",
           "ayot_gen.py", "qwen35_moe_proxy.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help_exits_zero_with_usage(script):
    proc = subprocess.run([sys.executable, str(MOE / script), "--help"],
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"{script} --help failed:\n{proc.stderr[-2000:]}"
    assert "usage:" in proc.stdout, f"{script} printed no usage (no __main__ guard?)"


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_compiles(script):
    proc = subprocess.run([sys.executable, "-m", "py_compile", str(MOE / script)],
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-2000:]
