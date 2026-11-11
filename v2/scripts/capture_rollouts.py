#!/usr/bin/env python3
"""Capture rollouts from the frozen Scion body on the pilot corpus.

This is the on-policy / failure-rollout arm of v2 (Right On Q, SteerDistill):
the frozen body generates on prompts that are NOT hard-90 or easy-30.  The raw
responses are kept for steering and for length/loop analysis; nothing is
trained here.

Usage:
  python v2/scripts/capture_rollouts.py --port 8082 --per-family 4 \
      --max-tokens 16384 --out-dir v2/rollouts/body
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="scion-35b-a3b-pq2_0-corr")
    ap.add_argument("--prompts", default=str(REPO / "v2/corpus/rollout-prompts.jsonl"))
    ap.add_argument("--out-dir", default=str(REPO / "v2/rollouts/body"))
    ap.add_argument("--per-family", type=int, default=4,
                    help="take the first N of each family (0 = all)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--thinking-budget", type=int, default=16384)
    ap.add_argument("--timeout", type=float, default=3 * 3600)
    ap.add_argument("--families", default="reasoning,coding,tools")
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            open(args.prompts, encoding="utf-8") if line.strip()]
    by_family = {f: [] for f in ("reasoning", "coding", "tools")}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    wanted = [f.strip() for f in args.families.split(",") if f.strip()]
    chosen = []
    for f in wanted:
        take = by_family[f] if args.per_family <= 0 else by_family[f][:args.per_family]
        chosen.extend(take)
    chosen.sort(key=lambda r: r["id"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out_dir.glob("*.json")}
    todo = [r for r in chosen if r["id"] not in done]
    print(f"{len(chosen)} chosen, {len(done)} done, {len(todo)} todo", flush=True)

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    summary = []
    for k, r in enumerate(todo, 1):
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": r["prompt"]},
            ],
            "temperature": args.temperature,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.05,
            "seed": args.seed + k - 1,
            "max_tokens": args.max_tokens,
            "stream": False,
            "thinking_budget_tokens": args.thinking_budget,
        }
        if r.get("tools"):
            body["tools"] = r["tools"]
        t0 = time.time()
        try:
            resp = post(url, body, args.timeout)
            err = None
        except Exception as e:  # keep the run going; record the failure
            resp, err = None, f"{type(e).__name__}: {e}"
        wall = time.time() - t0
        rec = {"id": r["id"], "family": r["family"], "source": r["source"],
               "prompt": r["prompt"], "tools": r.get("tools", []),
               "reference": r.get("reference", ""),
               "request": {k: v for k, v in body.items() if k != "messages"},
               "response": resp, "error": err,
               "wall_s": round(wall, 1)}
        (out_dir / f"{r['id']}.json").write_text(json.dumps(rec, indent=1))
        usage = (resp or {}).get("usage", {}) or {}
        toks = usage.get("completion_tokens", 0)
        finish = ((resp or {}).get("choices") or [{}])[0].get("finish_reason")
        summary.append({"id": r["id"], "family": r["family"],
                        "completion_tokens": toks, "wall_s": round(wall, 1),
                        "finish_reason": finish, "error": err})
        print(f"[{k}/{len(todo)}] {r['id']} {r['family']} "
              f"tok={toks} {wall:.0f}s finish={finish} err={err}", flush=True)

    all_files = sorted(out_dir.glob("*.json"))
    recs = [json.loads(p.read_text()) for p in all_files]
    toks = [((r.get("response") or {}).get("usage", {}) or
             {}).get("completion_tokens", 0) for r in recs]
    walls = [r["wall_s"] for r in recs]
    summary_out = {
        "n": len(recs),
        "by_family": {
            f: {"n": sum(1 for r in recs if r["family"] == f)}
            for f in ("reasoning", "coding", "tools")
        },
        "mean_completion_tokens": round(statistics.mean(toks), 1) if toks else 0,
        "median_wall_s": round(statistics.median(walls), 1) if walls else 0,
        "truncated": sum(1 for r in recs
                         if ((r.get("response") or {}).get("choices") or
                             [{}])[0].get("finish_reason") == "length"),
        "errors": sum(1 for r in recs if r.get("error")),
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_out, indent=2) + "\n")
    print(json.dumps(summary_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
