#!/usr/bin/env python
"""The decision head's runtime form for the engines, and its parity check.

The study's head JSON stays the exchange artifact; the engine reads this binary:

    magic  "SJH1"  4 bytes
    dim    uint32
    flags  uint32   (0)
    w      float32[dim]     ridge weights (on standardised features)
    mu     float32[dim]
    sd     float32[dim]
    b      float32          the ridge's bias
    a, c   float32          Platt: p = sigmoid(a * (w.(h-mu)/sd + b) + c)

Usage:
    python head_bin.py --from-json head.json --out head.bin
    python head_bin.py --verify head.json head.bin states.npz    # engine-side parity in Python
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"SJH1"


def from_json(head_json: Path, out: Path) -> None:
    h = json.loads(head_json.read_text(encoding="utf-8"))
    w = np.asarray(h["weights"]["w"], dtype=np.float32)
    mu = np.asarray(h["weights"]["mu"], dtype=np.float32)
    sd = np.asarray(h["weights"]["sd"], dtype=np.float32)
    b = float(h["weights"]["b"])
    a, c = (float(v) for v in h["weights"]["platt"])
    with out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", w.size, 0))
        f.write(w.tobytes())
        f.write(mu.tobytes())
        f.write(sd.tobytes())
        f.write(struct.pack("<fff", b, a, c))
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, dim {w.size}, "
          f"platt a={a:.3f} c={c:.3f}, bias {b:.4f})")


def read_bin(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    raw = path.read_bytes()
    assert raw[:4] == MAGIC, f"{path}: not a head binary"
    dim, _flags = struct.unpack_from("<II", raw, 4)
    off = 12
    w = np.frombuffer(raw, dtype="<f4", count=dim, offset=off).astype(np.float64); off += dim * 4
    mu = np.frombuffer(raw, dtype="<f4", count=dim, offset=off).astype(np.float64); off += dim * 4
    sd = np.frombuffer(raw, dtype="<f4", count=dim, offset=off).astype(np.float64); off += dim * 4
    b, a, c = struct.unpack_from("<fff", raw, off)
    return w, mu, sd, float(b), float(a), float(c)


def prob(head: Path, vec: np.ndarray) -> float:
    """The engine's own arithmetic: float32 inputs, double math, sigmoid last."""
    w, mu, sd, b, a, c = read_bin(head)
    assert vec.size == w.size, f"vector {vec.size} vs head dim {w.size}"
    z = float(np.dot(w, (vec.astype(np.float64) - mu) / sd)) + b
    return float(1.0 / (1.0 + np.exp(-(a * z + c))))


def verify(head_json: Path, head_bin: Path, states: Path) -> int:
    """Apply both forms to the same vectors: the engine's binary must match the JSON exactly."""
    hj = json.loads(head_json.read_text(encoding="utf-8"))
    w = np.asarray(hj["weights"]["w"], dtype=np.float64)
    mu = np.asarray(hj["weights"]["mu"], dtype=np.float64)
    sd = np.asarray(hj["weights"]["sd"], dtype=np.float64)
    b = float(hj["weights"]["b"])
    a, c = (float(v) for v in hj["weights"]["platt"])
    z = np.load(states, allow_pickle=False)
    X = z["X"].astype(np.float64)
    p_json = 1.0 / (1.0 + np.exp(-(a * (((X - mu) / sd) @ w + b) + c)))
    w2, mu2, sd2, b2, a2, c2 = read_bin(head_bin)
    p_bin = np.array([prob(head_bin, row.astype(np.float32)) for row in X])
    worst = float(np.max(np.abs(p_json - p_bin)))
    print(f"verify: n={X.shape[0]} dim={X.shape[1]} worst |JSON - binary| = {worst:.3e}")
    return 0 if worst < 1e-6 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-json", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("head.bin"))
    ap.add_argument("--verify", nargs=2, metavar=("HEAD_JSON", "HEAD_BIN"), default=None)
    ap.add_argument("states", nargs="?", type=Path, default=None)
    args = ap.parse_args()
    if args.from_json is not None:
        from_json(args.from_json, args.out)
        return 0
    if args.verify is not None and args.states is not None:
        return verify(Path(args.verify[0]), Path(args.verify[1]), args.states)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
