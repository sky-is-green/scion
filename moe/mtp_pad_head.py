"""Zero-pad a trained k=1 drafter head to a wider one (moe/mtp_pad_head.py).

WHY.  A wider intermediate can only be trained from scratch with the trainer (the shapes differ), and a
from-scratch CPU run is ~9 h here.  Growing the head around a trained one keeps the function exactly:
the new fc1 rows start at zero (their gelu outputs are zero, so fc2 sees nothing new), and the new fc2
columns start at zero.  The first fine-tune step then uses the new capacity only where it helps.

    python moe/mtp_pad_head.py --in head-op2.pt --out head-op2-w15.pt --width-mult 1.5
    python moe/mtp_release_train.py --probe-dir ... --width-mult 1.5 --init head-op2-w15.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="trained head checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width-mult", type=float, required=True)
    args = ap.parse_args()

    sd = torch.load(args.src, map_location="cpu")
    fc1 = sd["_mtp_head.fc1.weight"].float()
    fc2 = sd["_mtp_head.fc2.weight"].float()
    width, two_h = fc1.shape
    hidden, w2 = fc2.shape
    if w2 != width or two_h != 2 * hidden:
        raise SystemExit(f"not a k=1 head: fc1 {tuple(fc1.shape)}, fc2 {tuple(fc2.shape)}")
    want = int(round(args.width_mult * two_h))
    if want <= width:
        raise SystemExit(f"--width-mult must grow the head: {width} -> {want}")

    g1 = torch.zeros(want, two_h)
    g1[:width] = fc1
    g2 = torch.zeros(hidden, want)
    g2[:, :width] = fc2
    torch.save({"_mtp_head.fc1.weight": g1, "_mtp_head.fc2.weight": g2}, args.out)
    print(f"{args.src}: width {width} -> {want} (hidden {hidden}); "
          f"fc1 {fc1.numel()} -> {g1.numel()} params, fc2 {fc2.numel()} -> {g2.numel()}; wrote {args.out}")


if __name__ == "__main__":
    main()