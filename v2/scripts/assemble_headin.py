#!/usr/bin/env python3
"""Assemble the strata --dump-head-input dumps into fork-probe window files.

Each chunk's dump holds records {pos int64, tok int64, 2560 floats} for the
chunk's own positions (0..len-1, chunk-local; the run was one continuous
teacher-forced sequence over the chunk's tokens).  Pipelined verify stages can
write a position twice - the LAST record wins.  The trainer's windows are
512-position blocks: chunk c contributes windows c*138 + k for k in 0..137
(70,656 = 138 x 512), written as win_<idx>_h.bin [512, 2560] fp32 and
win_<idx>_argmax.bin [512] int32 - the fork probe's exact layout, so
moe/mtp_release_train.py consumes them unchanged.

Usage:
  python v2/scripts/assemble_headin.py --chunks 8 --per-chunk 70656 \
      --in-dir v2/traces/mtp-probe/headin --out v2/traces/mtp-probe-gpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

N = 2560
REC = 16 + N * 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--per-chunk", type=int, default=70656)
    ap.add_argument("--in-dir", default="v2/traces/mtp-probe/headin")
    ap.add_argument("--out", default="v2/traces/mtp-probe-gpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wins_per = args.per_chunk // 512
    total_rows = 0
    for c in range(args.chunks):
        path = Path(args.in_dir) / f"headin-chunk{c}.bin"
        raw = np.fromfile(path, dtype=np.uint8)
        n_recs = len(raw) // REC
        by_pos: dict[int, np.ndarray] = {}
        toks: dict[int, int] = {}
        for k in range(n_recs):
            rec = raw[k * REC:(k + 1) * REC]
            p = int(np.frombuffer(rec[:8].tobytes(), dtype="<i8")[0])
            by_pos[p] = rec[16:].copy()
            toks[p] = int(np.frombuffer(rec[8:16].tobytes(), dtype="<i8")[0])
        missing = [p for p in range(args.per_chunk) if p not in by_pos]
        if missing:
            print(f"chunk {c}: {len(missing)} positions missing (e.g. {missing[:5]}) - windows still written")
        for k in range(wins_per):
            idx = c * wins_per + k
            h = np.zeros((512, N), dtype="<f4")
            am = np.zeros(512, dtype="<i4")
            for r in range(512):
                p = k * 512 + r
                if p in by_pos:
                    h[r] = np.frombuffer(by_pos[p], dtype="<f4")
                    am[r] = toks[p]
                else:                     # mirror the previous row (a hole in a
                    h[r] = h[r - 1]       # dead chunk run; the trainer sees one
                    am[r] = am[r - 1]     # duplicated step, not a wrong one)
            h.tofile(out / f"win_{idx}_h.bin")
            am.tofile(out / f"win_{idx}_argmax.bin")
            total_rows += 512
        print(f"chunk {c}: {n_recs} records -> {wins_per} windows (win_{c*wins_per}..win_{(c+1)*wins_per-1})")
    print(f"total: {total_rows} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
