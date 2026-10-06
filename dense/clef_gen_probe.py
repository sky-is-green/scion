#!/usr/bin/env python3
"""Goal-B gate: free-generation probes for a Clef-Flash GGUF via llama-server.

Mirrors the QAT acceptance suite (qat_derisk.py --gen-samples): prose / code /
math, same prompts and token budgets, but served through llama-server +
/completion (the fork's llama-cli is an interactive REPL) with a fixed seed so
every model sees an identical sampler.

Writes <outdir>/<name>.{json,txt,server.log} and prints one JSON line per probe
with a loop/repetition score (repeated 8-gram fraction and longest repeat
count), so degeneration is measurable, not just eyeballed.

Usage:
  clef_gen_probe.py --model M.gguf --name tag --outdir DIR [--port 8896]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

PROBES = [
    {"name": "prose", "prompt": "The history of the Roman Empire begins with",
     "temperature": 0.7, "n_predict": 120},
    {"name": "code", "prompt": "def fibonacci(n):",
     "temperature": 0.7, "n_predict": 120},
    {"name": "math", "prompt": "Question: A train travels 240 km in 3 hours. "
     "What is its average speed? Answer:",
     "temperature": 0.0, "n_predict": 100},
]
SEED = 1337
# pinned sampler (server defaults) so runs are comparable across models
SAMPLER = {"seed": SEED, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
           "repeat_penalty": 1.0, "cache_prompt": False}


def wait_health(port: int, proc: subprocess.Popen, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200 and json.load(r).get("status") == "ok":
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    return False


def completion(port: int, body: dict):
    """Return (response_json, None) or (None, error_dict) on HTTP failure."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")
        except Exception:
            detail = ""
        return None, {"code": e.code, "detail": detail[:2000]}


def loop_score(text: str, n: int = 8) -> dict:
    toks = text.split()
    if len(toks) < n:
        return {"distinct_8gram_frac": 1.0, "repeated_8gram_frac": 0.0,
                "max_8gram_count": 0, "tokens": len(toks)}
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    total = sum(grams.values())
    distinct = len(grams) / total
    return {"distinct_8gram_frac": round(distinct, 4),
            "repeated_8gram_frac": round(1.0 - distinct, 4),
            "max_8gram_count": max(grams.values()), "tokens": len(toks)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--server", default="/home/penis/llama.cpp/build/bin/llama-server")
    ap.add_argument("--server-arg", action="append", default=[],
                    help="extra llama-server arg (repeatable), e.g. --no-jinja "
                         "to keep /completion raw on models whose output the "
                         "jinja/chat parser rejects")
    ap.add_argument("--port", type=int, default=8896)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    server_log = open(out / f"{args.name}.server.log", "w")
    cmd = [args.server, "-m", args.model, "-c", str(args.ctx),
           "-ngl", str(args.ngl), "--host", "127.0.0.1",
           "--port", str(args.port)] + args.server_arg
    print(f"[{time.strftime('%H:%M:%S')}] starting {args.name}: "
          f"{' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        if not wait_health(args.port, proc, args.timeout):
            print(f"FATAL: server for {args.name} did not become healthy",
                  file=sys.stderr)
            return 2
        results = []
        had_error = False
        for probe in PROBES:
            body = {"prompt": probe["prompt"],
                    "temperature": probe["temperature"],
                    "n_predict": probe["n_predict"], **SAMPLER}
            t0 = time.time()
            resp, err = completion(args.port, body)
            if err is not None:
                had_error = True
                rec = {"name": probe["name"], "prompt": probe["prompt"],
                       "temperature": probe["temperature"],
                       "n_predict": probe["n_predict"], "seed": SEED,
                       "elapsed_s": round(time.time() - t0, 2),
                       "error": err, "text": ""}
                results.append(rec)
                print(json.dumps({"name": rec["name"], "error": err["code"],
                                  "detail": err["detail"][:120]}), flush=True)
                continue
            text = resp.get("content", "")
            rec = {"name": probe["name"], "prompt": probe["prompt"],
                   "temperature": probe["temperature"],
                   "n_predict": probe["n_predict"], "seed": SEED,
                   "sampler": {k: v for k, v in SAMPLER.items() if k != "seed"},
                   "elapsed_s": round(time.time() - t0, 2),
                   "timings": resp.get("timings"),
                   "text": text, **loop_score(text)}
            results.append(rec)
            print(json.dumps({k: rec[k] for k in
                              ("name", "temperature", "n_predict", "tokens",
                               "repeated_8gram_frac", "max_8gram_count")}),
                  flush=True)
        (out / f"{args.name}.json").write_text(json.dumps(results, indent=1))
        lines = []
        for rec in results:
            if rec.get("text"):
                lines.append(f"===== {rec['name']} (temp {rec['temperature']}, "
                             f"{rec['n_predict']} tok, seed {SEED}) =====\n"
                             f"{rec['prompt']}\n{rec['text']}\n")
            else:
                lines.append(f"===== {rec['name']} =====\nERROR: "
                             f"{rec.get('error')}\n")
        (out / f"{args.name}.txt").write_text("\n".join(lines))
        return 1 if had_error else 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        server_log.close()


if __name__ == "__main__":
    sys.exit(main())
