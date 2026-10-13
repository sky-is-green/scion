#!/usr/bin/env python3
"""Build a scion-prune-plan JSON from a router-stats usage map.

Thin wrapper around the exporter's plan builder so the plan can be produced
without a source GGUF (the pod gate windows read the release's header
remotely).  Also prints the audit numbers the prune decision needs: the share
of each scoring signal carried by the dropped experts, per layer, and the
keep-set overlap against alternative rankings.

    work/.venv-rocm/bin/python moe/make_prune_plan.py \
        --map moe/artifacts/qwen4exp/router-stats-coder.json \
        --keep-frac 0.25 --rank-by reap \
        --out moe/artifacts/qwen4exp/prune-plan-25.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "hivebench/tools/strata-bridge"))

import repack_experts_q2_0 as rp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-frac", type=float, default=0.25)
    ap.add_argument("--rank-by", choices=list(rp.RANK_FIELDS), default="reap")
    ap.add_argument("--used-count", type=int, default=0,
                    help="default: the map's n_expert_used")
    ap.add_argument("--compare", default="mass,counts,mass_dense,norm_sum",
                    help="rankings to print keep-set overlap against")
    args = ap.parse_args()

    usage = json.loads(Path(args.map).read_text())
    used = args.used_count or int(usage.get("n_expert_used") or 1)
    plan = rp.plan_from_usage(usage, args.keep_frac, args.rank_by, used)

    by_il = {int(L["il"]): L for L in usage["layers"]}
    print(f"map: {usage['n_layer']} layers x {usage['n_expert']} experts, "
          f"{usage['tokens']} tokens, rank_by {args.rank_by}, keep {plan['keep_n']}")

    total = {k: 0.0 for k in rp.RANK_FIELDS}
    dropped = {k: 0.0 for k in rp.RANK_FIELDS}
    per_layer = []
    for L in plan["layers"]:
        il = int(L["il"])
        src = by_il[il]
        drop = set(L["dropped"])
        d_share = {}
        for k in rp.RANK_FIELDS:
            vals = src[k]
            t = sum(float(v) for v in vals)
            d = sum(float(vals[e]) for e in drop)
            total[k] += t
            dropped[k] += d
            if t > 0:
                d_share[k] = d / t
        per_layer.append((il, d_share))
    print("dropped share of each signal (all layers):")
    for k in rp.RANK_FIELDS:
        print(f"  {k:10s} {100.0 * dropped[k] / total[k]:6.2f}%")
    worst = max(per_layer, key=lambda x: x[1].get(args.rank_by, 0.0))
    print(f"worst layer by dropped {args.rank_by} share: il {worst[0]} "
          f"{100.0 * worst[1][args.rank_by]:.2f}%")

    # keep-set overlap against alternative rankings
    for other in [c.strip() for c in args.compare.split(",") if c.strip()]:
        if other == args.rank_by or other not in rp.RANK_FIELDS:
            continue
        other_plan = rp.plan_from_usage(usage, args.keep_frac, other, used)
        ov = []
        kept_ours = {int(L["il"]): set(L["keep"]) for L in plan["layers"]}
        for L in other_plan["layers"]:
            il = int(L["il"])
            kept = set(L["keep"])
            ov.append(len(kept & kept_ours[il]) / len(kept))
        print(f"keep-set overlap vs {other:10s}: mean {100.0 * sum(ov) / len(ov):5.1f}% "
              f"(min layer {100.0 * min(ov):5.1f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
