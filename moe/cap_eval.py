"""Capability suite runner (item 6): KLD/PPL cannot see intelligence loss.

Runs each task in ``cap_tasks.json`` through a GGUF model and scores the
completion with a deterministic local checker
(contains/exact/numeric/regex/python).  Two backends:

- ``cli``    : one llama-cli process per task (simple, but reloads the model
              20 times -- this OOMed the 30 GB box once; kept for tiny files).
- ``server`` : a SINGLE llama-server load, one ``/completion`` POST per task.
  ``--server-bin`` starts and stops the server for the run; ``--server-url``
  uses an already-running one.

Own tasks only for the python checkers (they exec model output with a
timeout).  Positive control for the CHECKERS is unit tests with hand-written
answers; the pipeline's negative control is a broken model (the 4-layer
no-PLE prototype should sit at the floor).  Calibration on capable models
happens at W1+W2; the harness is ready before it is needed.

Usage:
    python moe/cap_eval.py --model <file.gguf> --cli <llama-cli>
        --tasks moe/cap_tasks.json --out report.json [--max-tokens 40]
    python moe/cap_eval.py --model <file.gguf> --backend server \
        --server-bin <llama-server> --out report.json
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def numbers(s: str):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?",
                                         s.replace(",", ""))]


def strip_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m[0] if m else text


def check_completion(check: dict, completion: str) -> tuple[bool, str]:
    """Returns (passed, detail). All matching is case-insensitive."""
    t = check["type"]
    if t == "contains":
        missing = [v for v in check["values"] if v.lower() not in completion.lower()]
        return (not missing, f"missing={missing}" if missing else "all present")
    if t == "contains_any":
        hit = [v for v in check["values"] if v.lower() in completion.lower()]
        return (bool(hit), f"hit={hit}" if hit else "no match")
    if t == "exact":
        return (norm(completion) == norm(check["value"]),
                f"got={completion.strip()[:60]!r}")
    if t == "numeric":
        nums = numbers(completion)
        ok = any(abs(n - check["value"]) <= check.get("tol", 0.5) for n in nums)
        return (ok, f"numbers={nums[:5]}")
    if t == "regex":
        m = re.search(check["pattern"], completion, re.IGNORECASE)
        return (m is not None, f"match={bool(m)}")
    if t == "python_output":
        code = strip_code(completion).strip()
        try:
            p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return (False, "exec timeout")
        if p.returncode != 0:
            return (False, f"exec rc={p.returncode}: {p.stderr.strip()[:120]}")
        return (p.stdout.strip() == check["value"],
                f"stdout={p.stdout.strip()[:60]!r}")
    if t == "python_value":
        code = strip_code(completion).strip()
        try:
            p = subprocess.run([sys.executable, "-c", f"print({code})"],
                               capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return (False, "eval timeout")
        if p.returncode != 0:
            return (False, f"eval rc={p.returncode}")
        try:
            got = float(p.stdout.strip())
        except ValueError:
            return (False, f"not a number: {p.stdout.strip()[:60]!r}")
        return (abs(got - check["value"]) <= 1e-9, f"value={got}")
    raise ValueError(f"unknown checker {t}")


def run_task(cli: str, model: str, prompt: str, max_tokens: int,
             timeout: int) -> str:
    """One completion via llama-cli batch mode (stdin closed: no chat loop)."""
    p = subprocess.run(
        [cli, "-m", model, "-p", prompt, "-n", str(max_tokens),
         "-c", "512", "--temp", "0", "--no-display-prompt",
         "--reasoning", "off"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=timeout)
    out = p.stdout
    # drop llama-cli timing footers like "[ Prompt: 1.2 t/s | ... ]".
    lines = [ln for ln in out.splitlines()
             if not re.match(r"\s*\[.*t/s.*\]\s*$", ln)]
    return "\n".join(lines).strip()


def run_task_http(server_url: str, prompt: str, max_tokens: int,
                  timeout: int) -> str:
    """One completion against a resident llama-server ``/completion``."""
    body = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 0,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        server_url.rstrip("/") + "/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    return (data.get("content") or "").strip()


def wait_health(server_url: str, timeout: float = 300.0) -> None:
    """Block until llama-server answers /health (model loaded)."""
    url = server_url.rstrip("/") + "/health"
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - still loading
            last = repr(exc)[:120]
        time.sleep(2.0)
    raise TimeoutError(f"llama-server not healthy after {timeout:.0f}s: {last}")


def start_server(server_bin: str, model: str, port: int, ctx: int = 512,
                 extra: str = "", log_path: str = "") -> subprocess.Popen:
    """Start one llama-server (single load); caller stops it in a finally."""
    cmd = [server_bin, "-m", model, "-c", str(ctx), "--host", "127.0.0.1",
           "--port", str(port), "--temp", "0"]
    cmd += shlex.split(extra)
    log = open(log_path, "wb") if log_path else subprocess.DEVNULL
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)


def evaluate(cli: str, model: str, tasks: list, max_tokens: int,
             timeout: int, backend: str = "cli", server_url: str = "") -> dict:
    def run_one(prompt: str) -> str:
        if backend == "server":
            return run_task_http(server_url, prompt, max_tokens, timeout)
        return run_task(cli, model, prompt, max_tokens, timeout)

    results = []
    for t in tasks:
        try:
            completion = run_one(t["prompt"])
            passed, detail = check_completion(t["check"], completion)
            err = ""
        except Exception as exc:  # noqa: BLE001 - one bad task must not kill the suite
            completion, passed, detail, err = "", False, "harness error", repr(exc)[:160]
        results.append({"id": t["id"], "category": t["category"],
                        "passed": passed, "detail": detail, "error": err,
                        "completion": completion[:300]})
        print(f"{t['id']:8s} [{'PASS' if passed else 'FAIL'}] {detail}",
              flush=True)
    by_cat: dict[str, dict] = {}
    for r in results:
        c = by_cat.setdefault(r["category"], {"n": 0, "ok": 0})
        c["n"] += 1
        c["ok"] += 1 if r["passed"] else 0
    total = sum(c["n"] for c in by_cat.values())
    ok = sum(c["ok"] for c in by_cat.values())
    return {"model": model, "n": total, "ok": ok, "backend": backend,
            "accuracy": ok / total if total else 0.0,
            "by_category": {k: {"n": v["n"], "ok": v["ok"],
                                "acc": v["ok"] / v["n"]} for k, v in by_cat.items()},
            "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cli", default="", help="llama-cli path (cli backend)")
    ap.add_argument("--backend", choices=["cli", "server"], default="cli")
    ap.add_argument("--server-url", default="",
                    help="existing llama-server base URL (server backend)")
    ap.add_argument("--server-bin", default="",
                    help="start one llama-server for the run (single load)")
    ap.add_argument("--server-args", default="", help="extra llama-server args")
    ap.add_argument("--server-log", default="", help="llama-server log path")
    ap.add_argument("--port", type=int, default=8797)
    ap.add_argument("--tasks", default=str(
        Path(__file__).resolve().parent / "cap_tasks.json"))
    ap.add_argument("--out", default="")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    if args.backend == "cli" and not args.cli:
        ap.error("--cli is required for the cli backend")

    proc = None
    server_url = args.server_url
    try:
        if args.backend == "server" and args.server_bin:
            server_url = f"http://127.0.0.1:{args.port}"
            print(f"starting llama-server on {server_url} ...", flush=True)
            proc = start_server(args.server_bin, args.model, args.port,
                                ctx=512, extra=args.server_args,
                                log_path=args.server_log)
            wait_health(server_url)
            print("server healthy", flush=True)
        tasks = json.loads(Path(args.tasks).read_text())["tasks"]
        report = evaluate(args.cli, args.model, tasks, args.max_tokens,
                          args.timeout, backend=args.backend,
                          server_url=server_url)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    print(f"\n{report['ok']}/{report['n']} = {report['accuracy']:.3f}")
    for k, v in report["by_category"].items():
        print(f"  {k:12s} {v['ok']}/{v['n']} = {v['acc']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
