#!/usr/bin/env python3
"""Combine two MTP probe corpora into one trainer layout, keeping both held-out sets.

The trainer reads a single contiguous probe dir and splits it as
[0, n_train) train / [n_train, n_train+n_eval) eval.  To keep the old held-out
100 windows from leaking into training while adding a second corpus, the layout
is:

    old train (0..999) | new train (first n_new-n_new_eval) | old eval (last 100) | new eval (last n_new_eval)

and the trainer is run with --n-train = 1000 + (n_new - n_new_eval) and
--n-eval = 100 + n_new_eval.  Only symlinks are created; no window bytes are
copied.

Usage:
  python v2/scripts/combine_probes.py \
      --old v2/traces/mtp-probe-gpu --new v2/traces/mtp-probe-gpu2 \
      --new-eval 100 --out v2/traces/mtp-probe-combined
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

OLD_N = 1104     # old corpus windows; the last 100 were the v1 held-out set
OLD_EVAL = 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--new-eval", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    old, new, out = Path(args.old), Path(args.new), Path(args.out)
    out_h = out / "h"
    out_h.mkdir(parents=True, exist_ok=True)

    n_old = len(list((old / "h").glob("*_h.bin")))
    n_new = len(list((new / "h").glob("*_h.bin")))
    n_new_eval = min(args.new_eval, n_new)
    n_new_train = n_new - n_new_eval

    order = [(old, w) for w in range(n_old - OLD_EVAL)] + \
            [(new, w) for w in range(n_new_train)] + \
            [(old, n_old - OLD_EVAL + w) for w in range(OLD_EVAL)] + \
            [(new, n_new_train + w) for w in range(n_new_eval)]

    for i, (root, w) in enumerate(order):
        for suf in ("h.bin", "argmax.bin"):
            dst = out_h / f"win_{i:02d}_{suf}"
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink((root / "h" / f"win_{w:02d}_{suf}").resolve(), dst)

    toks = [np.fromfile(root / "tokens.bin", dtype=np.int32).reshape(-1, 512)[w]
            for root, w in order]
    np.stack(toks).tofile(out / "tokens.bin")

    print(f"combined {len(order)} windows: train {n_old - OLD_EVAL}+{n_new_train} "
          f"eval {OLD_EVAL}+{n_new_eval} -> {out}")
    print(f"run the trainer with --n-train {len(order) - OLD_EVAL - n_new_eval} "
          f"--n-eval {OLD_EVAL + n_new_eval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
