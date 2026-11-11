#!/usr/bin/env python3
"""Run a fixed item subset through an OpenAI-compatible llama.cpp server.

Same harness, graders and protocol as EVAL.md; the only change is the item
list so external baselines can be compared item-for-item with the saved Scion
run.  Output is resumable JSONL plus a summary in the same shape as
v2/eval/*/summary.json.

Usage:
  python v2/scripts/hard_subset.py --port 8080 --label "Ternary Nemotron" \
      --out-dir v2/eval/hard-subset30-nemotron
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("BONSAI_ROUND", "2")

REPO = Path(__file__).resolve().parents[2]
HELDOUT = Path("/tmp/opencode/bonsai2-27b-test")
HARNESS = HELDOUT / "harness"
ITEMS = HELDOUT / "items_r2"
for p in (str(HARNESS), str(ITEMS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run  # noqa: E402
import config as cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--items-file", default=str(REPO / "v2/eval/subset30.json"))
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--thinking-budget", type=int, default=32768)
    ap.add_argument("--effort", default="medium",
                    help="sent as reasoning_effort only with --effort-field")
    ap.add_argument("--effort-field", action="store_true")
    ap.add_argument("--extra-body", default="{}",
                    help="JSON object merged into every chat request")
    args = ap.parse_args()

    cfg.PORT = args.port
    cfg.MAX_TOKENS = {"medium": args.max_tokens, "xhigh": args.max_tokens}
    cfg.MAX_TOOL_TURNS = 30

    spec = json.loads(Path(args.items_file).read_text())
    order = spec["ids"] if isinstance(spec, dict) else spec
    item_map, envs = run.load_items()
    missing = [i for i in order if i not in item_map]
    if missing:
        raise SystemExit(f"unknown ids: {missing}")

    arm = {
        "label": args.label,
        "sampler": {"temperature": args.temperature, "top_p": 0.95,
                    "top_k": 20, "min_p": 0.05},
        "effort_field": args.effort_field,
        "extra_body": json.loads(args.extra_body),
    }
    if args.thinking_budget:
        arm["extra_body"].setdefault("thinking_budget_tokens",
                                     args.thinking_budget)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text().splitlines()
                if line.strip()}
    todo = [i for i in order if i not in done]
    print(f"{args.label}: {len(order)} items, {len(done)} done, {len(todo)} todo",
          flush=True)

    for k, item_id in enumerate(todo, 1):
        family, item = item_map[item_id]
        t0 = time.time()
        if family == "tools":
            rec = run.run_tools(arm, item, envs, args.effort, args.seed)
        else:
            rec = run.run_single(arm, family, item, args.effort, args.seed)
        wall = time.time() - t0
        rec.update(id=item_id, family=family, wall_s=round(wall, 1),
                   completion_tokens=run.tokens(rec["turns"]),
                   seed=args.seed, temperature=args.temperature,
                   arm=args.label)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        tps = rec["completion_tokens"] / wall if wall > 0 else 0.0
        print(f"[{k}/{len(todo)}] {item_id} {family} correct={rec['correct']} "
              f"tok={rec['completion_tokens']} {wall:.0f}s {tps:.1f} tok/s",
              flush=True)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()
            if line.strip()]
    n = len(rows)
    correct = sum(bool(r["correct"]) for r in rows)
    walls = [r["wall_s"] for r in rows]
    fam = {f: {"n": sum(1 for r in rows if r["family"] == f),
               "correct": sum(1 for r in rows
                              if r["family"] == f and r["correct"])}
           for f in ("reasoning", "coding", "tools")}
    summary = {
        "label": args.label,
        "n": n,
        "correct": correct,
        "accuracy": round(100 * correct / n, 2) if n else None,
        "by_family": fam,
        "truncated": sum(1 for r in rows if r.get("truncated")),
        "median_wall_s": round(statistics.median(walls), 1) if walls else None,
        "mean_decode_tok_s": round(
            statistics.mean(r["completion_tokens"] / r["wall_s"]
                            for r in rows if r["wall_s"] > 0), 2) if rows else None,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
