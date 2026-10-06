"""Per-tensor sweep of an exported qwen4exp GGUF against the ISTA reference.

Dequantizes one tensor at a time (memory-safe: never materialise a whole
GGUF), compares values and shapes, and flags anything beyond quantization
noise (rel > 0.2).  This is the tool that found the missing V-head reorder,
the -exp(A_log) and the 1+w norm conventions (2026-10-03).

Usage:
    python moe/gguf_ref_sweep.py <ours.gguf> [ref.gguf] [--big]
    ref.gguf defaults to the symlinked ISTA reference on a mounted backup
    drive (override with SCION_STORAGE or pass it explicitly).
    --big also checks token_embd/output (needs ~4 GB transient).
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

from scion_paths import QWEN4EXP_GGUF_PY, STORAGE

sys.path.insert(0, str(QWEN4EXP_GGUF_PY))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

DEFAULT_REF = str(STORAGE / "flashnext-backup" /
                  "models/qwen38-q2_0/Q2_0" /
                  "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf")
BIG = ("token_embd.weight", "output.weight")


def load(t):
    a = t.data
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        a = a.view(np.uint16).astype(np.uint32) << 16
        return a.view(np.float32)
    if t.tensor_type in (gguf.GGMLQuantizationType.F32,
                         gguf.GGMLQuantizationType.F16):
        return np.asarray(a, np.float32)
    try:
        return np.asarray(dequantize(a, t.tensor_type), np.float32)
    except NotImplementedError:
        return None


def main():
    args = [a for a in sys.argv[1:] if a != "--big"]
    check_big = "--big" in sys.argv
    ours = gguf.GGUFReader(args[0])
    ref = gguf.GGUFReader(args[1] if len(args) > 1 else DEFAULT_REF)
    refmap = {t.name: t for t in ref.tensors}
    checked = 0
    flagged = []
    for t in ours.tensors:
        if "_exps" in t.name or ".lora_" in t.name:
            continue
        if t.name in BIG and not check_big:
            continue
        rt = refmap.get(t.name)
        if rt is None:
            flagged.append((t.name, "no reference tensor"))
            continue
        a, b = load(t), load(rt)
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            flagged.append((t.name, f"shape {a.shape} vs {b.shape}"))
        else:
            rel = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))
            if rel > 0.2:
                one = float(np.abs((1 + a) - b).max() /
                            (np.abs(b).max() + 1e-9))
                flagged.append((t.name, f"rel {rel:.3f} (1+x rel {one:.3f})"))
        checked += 1
        del a, b
        if checked % 150 == 0:
            gc.collect()
    print(f"checked {checked} tensors")
    for name, why in flagged[:40]:
        print(f"  FLAG {name}: {why}")
    print("FLAGGED:", len(flagged))


if __name__ == "__main__":
    main()
