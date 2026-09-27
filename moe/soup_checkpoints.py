"""SWA-style checkpoint soup for correction branches.

Averages the fp32 branch/gate masters of several checkpoints from the SAME
run into one checkpoint.  Measured on OLMoE (runtime PPL, 563 chunks): the
4-way soup of steps 6000-8192 scored 14.438 vs 14.584 for the final step and
14.485 for the 1-epoch winner — a free ~0.5-1% within-run gain with no
training change.

Do not average across runs: two same-config runs differ enough (14.48 vs
14.56 at step 4000) that a cross-run soup collapsed to 21.7 harness.

Usage:
  soup_checkpoints.py out.pt ckpt1.pt ckpt2.pt ...
"""

from __future__ import annotations

import sys

import torch


def soup(out: str, paths: list[str]) -> None:
    acc: dict[str, torch.Tensor] | None = None
    for p in paths:
        sd = torch.load(p, map_location="cpu")
        if acc is None:
            acc = {k: v.float().clone() for k, v in sd.items()}
        else:
            if set(sd) != set(acc):
                raise SystemExit(f"{p}: key set differs from the first checkpoint")
            for k in acc:
                acc[k] += sd[k].float()
    assert acc is not None
    n = len(paths)
    acc = {k: v / n for k, v in acc.items()}
    torch.save(acc, out)
    print(f"wrote {out} ({n}-way mean of {len(acc)} tensors)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    soup(sys.argv[1], sys.argv[2:])
