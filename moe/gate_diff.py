#!/usr/bin/env python3
"""Gate-to-gate worst-token comparison for ``kld_eval.py`` JSONs.

The aggregate table (mean/p99/max) says whether a candidate is better *on
average*; it does not say whether the candidate fixed the tokens the baseline
actually fails on.  That distinction is the whole tail argument: the Phase-1
arms fail on a small, shared set of positions (windows 2/3/6/7, adjacent
positions), so a candidate that lowers the mean without touching those tokens
has not fixed the tail.

``kld_eval.py`` parks only the 16-token worst list per run, so "fixed" here
means "fell out of the candidate's worst 16" -- the strongest statement the
parked data supports.  A per-token KLD dump would make this exact; that is a
follow-up instrument change, not something this script can recover.

usage:
    python moe/gate_diff.py base.json cand.json [cand2.json ...]

Prints the metric table, the baseline-worst persistence per candidate, the new
spikes each candidate introduces, and the pairwise overlap of the worst sets.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path


def load(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    if "worst_tokens" not in d:
        raise SystemExit(f"{path}: no worst_tokens (not a kld_eval gate JSON?)")
    return d


def worst_set(d: dict) -> list[tuple[int, int]]:
    return [(t["window"], t["pos"]) for t in d["worst_tokens"]]


def tag(path: str) -> str:
    return Path(path).stem.removeprefix("kld-")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", help="baseline gate JSON")
    ap.add_argument("candidates", nargs="+", help="candidate gate JSONs")
    ap.add_argument("--list", type=int, default=8,
                    help="max positions to list per cell (default 8)")
    args = ap.parse_args(argv)

    runs = [(tag(args.base), load(args.base))]
    runs += [(tag(p), load(p)) for p in args.candidates]
    for _, d in runs:
        if d.get("n_tokens") != runs[0][1].get("n_tokens"):
            print(f"warning: n_tokens differ ({d.get('n_tokens')} vs "
                  f"{runs[0][1].get('n_tokens')}) -- positions are not "
                  f"comparable across a different eval set", file=sys.stderr)

    hdr = f"{'run':16} {'mean':>7} {'p99':>7} {'max':>7} {'ent':>6} {'peak':>7}  sharper"
    print(hdr)
    for t, d in runs:
        print(f"{t:16} {d['mean']:7.4f} {d['p99']:7.4f} {d['max']:7.4f} "
              f"{d['student_entropy_nats']:6.2f} {d['student_top1_prob']:7.4f}  "
              f"{'yes' if d['sharper_than_teacher'] else 'no'}")

    base_t, base_d = runs[0]
    base = worst_set(base_d)
    base_s = set(base)

    print(f"\n{base_t} worst-16 persistence (still worst in candidate):")
    for t, d in runs[1:]:
        cur = worst_set(d)
        keep = [p for p in base if p in set(cur)]
        new = [p for p in cur if p not in base_s]
        keep_s = " ".join(f"w{w}:{p}" for w, p in keep[:args.list])
        new_s = " ".join(f"w{w}:{p}" for w, p in new[:args.list])
        print(f"  {t:14} {len(keep):2d}/16 keep  [{keep_s}]")
        if new:
            print(f"  {'':14} new spikes: [{new_s}]")

    if len(runs) > 2:
        print("\npairwise worst-16 overlap (count/16):")
        sets = {t: set(worst_set(d)) for t, d in runs}
        for (a, da), (b, db) in combinations(runs, 2):
            print(f"  {a:14} x {b:14}: {len(sets[a] & sets[b]):2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
