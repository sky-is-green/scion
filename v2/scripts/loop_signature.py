#!/usr/bin/env python3
"""Loop-signature analysis over captured traces (the teacher-gate metric).

For every trace: 10-gram duplicate fraction (the same statistic that put the
hard-90 long items at 0.65-0.97 duplication when they degenerate, and the v1
recovered replies at ~0.18), plus the longest repeated 10-gram run and the
finish reason.  A trace "shows the loop signature" when its duplicate
fraction is >= 0.35 (degenerate repetition zone; healthy reasoning lands
under ~0.2 in the earlier measurements).

Usage: python v2/scripts/loop_signature.py [--dir v2/traces/flashnext]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dup_stats(text: str, n: int = 10):
    ws = text.split()
    grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    if not grams:
        return 0.0, 0, 0
    seen, rep, best_run = set(), 0, 0
    cur = 0
    for g in grams:
        if g in seen:
            rep += 1
            cur += 1
            best_run = max(best_run, cur)
        else:
            cur = 0
        seen.add(g)
    return rep / len(grams), best_run, len(grams)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="v2/traces/flashnext")
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="duplicate fraction above which a trace shows the "
                         "loop signature")
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.dir).glob("*.json")):
        d = json.load(open(p))
        msg = ((d.get("response") or {}).get("choices") or [{}])[0].get("message", {})
        body = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
        if not body.strip():
            print(f"{d.get('id', p.stem)}: EMPTY body (err={d.get('error')})")
            continue
        frac, run, n = dup_stats(body)
        finish = ((d.get("response") or {}).get("choices") or [{}])[0].get("finish_reason")
        tok = ((d.get("response") or {}).get("usage") or {}).get("completion_tokens")
        rows.append((d.get("id", p.stem), d.get("family"), tok, finish, frac, run))

    bad = [r for r in rows if r[4] >= args.threshold]
    fracs = sorted(r[4] for r in rows)
    print(f"{'id':10s} {'family':10s} {'tok':>6s} {'finish':7s} {'dup10':>6s} {'run':>4s}")
    for r in rows:
        flag = " LOOP" if r[4] >= args.threshold else ""
        print(f"{r[0]:10s} {str(r[1]):10s} {str(r[2]):>6s} {str(r[3]):7s} {r[4]:6.3f} {r[5]:4d}{flag}")
    if fracs:
        print(f"\nn={len(rows)}  median dup10={fracs[len(fracs)//2]:.3f}  "
              f"max={fracs[-1]:.3f}  >= {args.threshold}: {len(bad)}")
    print("TEACHER GATE: " + ("FAILED - the teacher's own traces loop"
          if bad else "PASSED - clean traces (no loop signature)"))
    return 0


if __name__ == "__main__":
    main()
