#!/usr/bin/env python3
"""Compare ternary codebook rules on real FP expert weights.

The classical-math lane (RESEARCH-HANDOFF §7.7 B) says three things we can
measure offline, on the actual weight distribution, with no model forward:

1. **Data-free vs per-tensor adaptivity.**  The shipped TurboQuant recipe
   rotates (Hadamard), RMS-scales per group, and applies a *fixed* 3-level
   codebook optimal for N(0,1) (`c ≈ 1.224`).  Our deployed rule is per-group
   Lloyd (TAARDIS Q1_0_g128).  The gap between them is the value of per-tensor
   adaptivity — i.e. part of the calibration-free edge.
2. **Rotation.**  TurboQuant's Hadamard is what makes the fixed codebook work;
   measuring it raw vs rotated isolates how much of the gap is the codebook and
   how much is the rotation.
3. **Companding.**  High-rate theory (Bennett/Panter-Dite/Zador) says a
   monotone compander with exponent p = 1/3 is MSE-optimal for heavy tails;
   measuring p ∈ {1, 1/2, 1/3} on real groups says whether our weights are
   heavy-tailed enough for it to matter.

Group size 64 vs 128 is measured too (the shipped recipe uses 64; ours 128).

Pure functions, CPU-only.  The loader reads a slice of one expert tensor from
the FP safetensors so the numbers are from the real distribution, not a toy.

usage:
    python moe/codebook_probe.py --layer 10 --experts 4
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moe_proxy import ternary_absmean, ternary_lloyd  # noqa: E402

#: Lloyd-Max reconstruction point for a 3-level quantizer on N(0,1)
GAUSS_C = 1.224


def normalized_mse(w: torch.Tensor, wq: torch.Tensor) -> float:
    """Scale-free reconstruction error: MSE / mean(w²)."""
    w = w.float()
    return float(((w - wq.float()) ** 2).mean() / w.pow(2).mean().clamp_min(1e-30))


def per_group_nmse(w: torch.Tensor, wq: torch.Tensor, group: int) -> dict:
    """p50/p90 of per-group normalized MSE (where the tail groups live)."""
    w = w.float().reshape(-1, group)
    wq = wq.float().reshape(-1, group)
    err = (w - wq).pow(2).mean(-1) / w.pow(2).mean(-1).clamp_min(1e-30)
    return {"p50": float(err.median()), "p90": float(err.quantile(0.9))}


def hadamard(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform along the last axis (power of 2).

    Orthonormal and symmetric, so applying it again inverts it.  This is the
    cheap Gaussianizing rotation the shipped data-free recipe uses.
    """
    n = x.shape[-1]
    if n & (n - 1) or n == 0:
        raise ValueError(f"last dim must be a power of two, got {n}")
    y = x.float().clone()
    h = 1
    while h < n:
        y = y.reshape(-1, 2 * h)
        a, b = y[:, :h], y[:, h:]
        y = torch.cat([a + b, a - b], dim=-1)
        h *= 2
    return y.reshape(x.shape) / math.sqrt(n)


def fixed_gauss_ternary(w: torch.Tensor, group: int = 128, c: float = GAUSS_C,
                        rotate: bool = False) -> torch.Tensor:
    """Data-free codebook: per-group RMS scale + fixed N(0,1) 3-level shape."""
    x = hadamard(w) if rotate else w.float()
    g = x.reshape(*x.shape[:-1], x.shape[-1] // group, group)
    s = g.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    q = torch.zeros_like(g)
    q = torch.where(g > 0.5 * c * s, c * s, q)
    q = torch.where(g < -0.5 * c * s, -c * s, q)
    q = q.reshape(x.shape)
    return hadamard(q) if rotate else q


def companded_absmean(w: torch.Tensor, group: int = 128,
                      p: float = 1.0 / 3.0) -> torch.Tensor:
    """Monotone power compander, ternary in the companded domain, inverted.

    ``p=1`` is plain absmean; high-rate theory's MSE-optimal exponent for
    heavy-tailed densities is ``1/3`` (the f^{1/3} point density).
    """
    if p <= 0:
        raise ValueError("p must be positive")
    u = w.sign() * w.abs().pow(p)
    uq = ternary_absmean(u, group)
    return (uq.sign() * uq.abs().pow(1.0 / p)).to(w.dtype)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", default=str(Path.home() / "Desktop/work/hivebench"
                                              / "artifacts/ternary/moe"))
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--experts", type=int, default=4,
                    help="how many experts of the tensor to load (slice)")
    ap.add_argument("--which", choices=["gate_up_proj", "down_proj"],
                    default="gate_up_proj")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.artifacts) / "empero-hf"
    import json
    index = json.loads((root / "model.safetensors.index.json").read_text())
    key = f"model.language_model.layers.{args.layer}.mlp.experts.{args.which}"
    shard = root / index["weight_map"][key]
    from safetensors import safe_open
    with safe_open(shard, framework="pt") as f:
        w = f.get_slice(key)[:args.experts].contiguous().to(torch.float32)
    print(f"{key}: {tuple(w.shape)} from {shard.name}; "
          f"kurtosis {float((w - w.mean()).pow(4).mean() / w.var().pow(2)):.2f}")

    variants = [
        ("absmean g128", lambda: ternary_absmean(w, 128)),
        ("lloyd g128 (ours)", lambda: ternary_lloyd(w, 128)),
        ("lloyd g64", lambda: ternary_lloyd(w, 64)),
        ("fixed gauss g128 raw", lambda: fixed_gauss_ternary(w, 128)),
        ("fixed gauss g128 rot", lambda: fixed_gauss_ternary(w, 128, rotate=True)),
        ("fixed gauss g64 rot", lambda: fixed_gauss_ternary(w, 64, rotate=True)),
        ("compander p=1/3 g128", lambda: companded_absmean(w, 128, 1.0 / 3.0)),
        ("compander p=1/2 g128", lambda: companded_absmean(w, 128, 0.5)),
    ]
    print(f"{'rule':24} {'NMSE':>8} {'p50':>8} {'p90':>8}")
    for name, fn in variants:
        wq = fn()
        stats = per_group_nmse(w, wq, 128)
        print(f"{name:24} {normalized_mse(w, wq):8.4f} "
              f"{stats['p50']:8.4f} {stats['p90']:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
