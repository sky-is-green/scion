#!/usr/bin/env python3
"""Pack the captured teacher rollouts into the curriculum trace JSONL.

The 360-prompt corpus (`v2/traces/rollout360/*.json`) was captured from the
resident 176B teacher.  This turns it into the `{"text": ...}` rows that
`build_curriculum.py` appends to the hard-window mix, and applies the teacher
loop gate (10-gram duplicate fraction >= 0.35 = degenerate zone) so a looping
teacher trace never becomes a training target.

The prompt is part of the text: the trainer learns the assistant continuation
in the same serving format the cache scores.

Usage:
  python v2/scripts/pack_teacher_traces.py \
      --dir v2/traces/rollout360 \
      --out v2/curriculum/teacher-traces-v2.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from loop_signature import dup_stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="v2/traces/rollout360")
    ap.add_argument("--out", default="v2/curriculum/teacher-traces-v2.jsonl")
    ap.add_argument("--max-dup", type=float, default=0.35,
                    help="drop a trace at or above this 10-gram duplicate "
                         "fraction (the teacher gate)")
    ap.add_argument("--min-words", type=int, default=20)
    args = ap.parse_args()

    paths = sorted(Path(args.dir).glob("*.json"))
    if not paths:
        raise SystemExit(f"no traces in {args.dir}")

    kept, dropped = [], []
    fam = collections.Counter()
    for p in paths:
        d = json.loads(p.read_text())
        body = ((d.get("reasoning") or "") + "\n"
                + (d.get("content") or "")).strip()
        if d.get("error") or len(body.split()) < args.min_words:
            dropped.append((d.get("id", p.stem), d.get("family"),
                            d.get("error") or "short"))
            continue
        frac, run, _ = dup_stats(body)
        if frac >= args.max_dup:
            dropped.append((d.get("id", p.stem), d.get("family"),
                            f"loop dup10={frac:.3f} run={run}"))
            continue
        text = (d.get("prompt") or "").strip() + "\n\n" + body
        kept.append({"text": text, "id": d.get("id", p.stem),
                     "family": d.get("family")})
        fam[d.get("family")] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    words = sum(len(r["text"].split()) for r in kept)
    print(f"kept {len(kept)}/{len(paths)} traces  {words} words  "
          f"families {dict(fam)}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if dropped:
        print(f"dropped {len(dropped)}:")
        for row in dropped:
            print("  ", *row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
