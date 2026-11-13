#!/usr/bin/env python3
"""Run heldoutlab's 90 round-2 hard items through Hivebench against Scion-35B-A3B.

The 90 item definitions and their graders come from the heldoutlab repo; this
script is the Hivebench-side runner. It talks to the Strata OpenAI endpoint
(port 8082 by default), with a 32k thinking budget and a 32k max-token cap.

Usage:
  python3.12 experiments/hard90_scion.py [--limit N] [--port 8082]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Must be set before importing the heldoutlab config/run modules.
os.environ.setdefault("BONSAI_ROUND", "2")

REPO = Path(__file__).resolve().parents[1]
HELDOUT = Path("/tmp/opencode/bonsai2-27b-test")
HARNESS = HELDOUT / "harness"
ITEMS = HELDOUT / "items_r2"
for p in (str(HARNESS), str(ITEMS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run              # noqa: E402  heldoutlab runner/item selection/graders
import config as cfg    # noqa: E402

OUT = REPO / "experiments" / "hard90-scion-32k"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    cfg.PORT = args.port
    cfg.MAX_TOKENS = {"medium": 32768, "xhigh": 32768}
    cfg.MAX_TOOL_TURNS = 30
    # recovery "recover" mode needs the template's xhigh effort sentence in
    # the first system message (see v2/scripts/hard_subset.py XHIGH_SENTENCE);
    # without it the mode is a silent no-op
    from hard_subset import XHIGH_SENTENCE
    cfg.SYSTEM_PROMPT = cfg.SYSTEM_PROMPT.rstrip() + "\n\n" + XHIGH_SENTENCE
    print("recovery sentence appended to the harness system prompt", flush=True)

    arm = {
        "label": "Scion-35B-A3B PQ2_0-corr (Strata, 32k)",
        "sampler": {"temperature": args.temperature, "top_p": 0.95,
                    "top_k": 20, "min_p": 0.05},
        "effort_field": False,
        "extra_body": {"thinking_budget_tokens": 32768},
    }

    items, envs = run.load_items()
    order = run.select("main", items)
    if args.limit:
        order = order[: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "results.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text().splitlines() if line.strip()}
    todo = [i for i in order if i not in done]
    print(f"hard90 scion-35b: {len(order)} items, {len(done)} already done, {len(todo)} todo", flush=True)

    for k, item_id in enumerate(todo, 1):
        family, item = items[item_id]
        t0 = time.time()
        if family == "tools":
            rec = run.run_tools(arm, item, envs, "medium", cfg.SEEDS["main"])
        else:
            rec = run.run_single(arm, family, item, "medium", cfg.SEEDS["main"])
        wall = time.time() - t0
        rec.update(id=item_id, family=family, wall_s=round(wall, 1),
                   completion_tokens=run.tokens(rec["turns"]),
                   seed=cfg.SEEDS["main"], temperature=args.temperature,
                   arm=arm["label"])
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        tps = rec["completion_tokens"] / wall if wall > 0 else 0.0
        print(f"[{k}/{len(todo)}] {item_id} {family} correct={rec['correct']} "
              f"tok={rec['completion_tokens']} {wall:.0f}s {tps:.1f} tok/s", flush=True)

    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    walls = [r["wall_s"] for r in rows]
    toks = [r["completion_tokens"] for r in rows]
    print(json.dumps({
        "n": n,
        "correct": correct,
        "accuracy": round(100 * correct / n, 2) if n else None,
        "median_s": round(statistics.median(walls), 2) if walls else None,
        "p95_s": round(sorted(walls)[max(0, int(0.95 * len(walls)) - 1)], 2) if walls else None,
        "mean_completion_tokens": round(statistics.mean(toks), 1) if toks else None,
        "total_completion_tokens": sum(toks),
        "mean_decode_tok_s": round(sum(t / w for t, w in zip(toks, walls) if w > 0) / n, 2) if n else None,
        "by_family": {
            fam: {
                "n": sum(1 for r in rows if r["family"] == fam),
                "correct": sum(1 for r in rows if r["family"] == fam and r["correct"]),
            } for fam in ("reasoning", "coding", "tools")
        },
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
