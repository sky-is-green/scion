"""Cross-check exported qwen4exp tensors against the working ISTA reference.

Dequantizes the same tensor from both GGUFs and reports relative differences
plus a row-permutation best match (a wrong V-head reorder shows as "direct
diff huge, permuted match excellent").  The reference is a different
quantization, so small numeric noise is expected; a layout bug is not.

Usage: python moe/gguf_crosscheck.py <ours.gguf> <ref.gguf>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from scion_paths import QWEN4EXP_GGUF_PY

sys.path.insert(0, str(QWEN4EXP_GGUF_PY))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def load(reader, name):
    t = next((x for x in reader.tensors if x.name == name), None)
    if t is None:
        return None
    a = t.data
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        a = a.view(np.uint16).astype(np.uint32) << 16
        a = a.view(np.float32)
    elif t.tensor_type not in (gguf.GGMLQuantizationType.F32,
                               gguf.GGMLQuantizationType.F16):
        try:
            a = dequantize(a, t.tensor_type)
        except NotImplementedError:
            return "RAW"
    a = np.asarray(a, dtype=np.float32)
    return a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)


def rel(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))


NAMES = [
    "blk.0.attn_qkv.weight",
    "blk.0.attn_gate.weight",
    "blk.0.ssm_alpha.weight",
    "blk.0.ssm_beta.weight",
    "blk.0.ssm_out.weight",
    "blk.0.ssm_conv1d.weight",
    "blk.0.ssm_norm.weight",
    "blk.3.attn_q.weight",
    "blk.3.attn_k.weight",
    "blk.3.attn_v.weight",
    "blk.3.attn_output.weight",
    "blk.0.ffn_gate_shexp.weight",
    "blk.0.ffn_up_shexp.weight",
    "blk.0.ffn_down_shexp.weight",
    "blk.0.ffn_gate_inp.weight",
    "blk.0.ffn_gate_inp_shexp.weight",
    "blk.0.hc_attn_down.weight",
    "blk.0.hc_attn_up.weight",
    "blk.0.hc_attn_inject.weight",
    "blk.0.hc_ffn_down.weight",
    "blk.0.hc_ffn_up.weight",
    "blk.0.hc_ffn_inject.weight",
]


def main():
    ours = gguf.GGUFReader(sys.argv[1])
    ref = gguf.GGUFReader(sys.argv[2])
    flags = []
    for name in NAMES:
        a, b = load(ours, name), load(ref, name)
        if a is None or b is None:
            print(f"{name}: MISSING ours={a is not None} ref={b is not None}")
            continue
        if isinstance(a, str) or isinstance(b, str):
            print(f"{name}: skipped (custom quant type not dequantizable)")
            continue
        if a.shape != b.shape:
            print(f"{name}: SHAPE {a.shape} vs {b.shape}")
            flags.append(name)
            continue
        d = rel(a, b)
        note = ""
        if d > 0.25:
            # row permutation probe: for each row of a, best-matching ref row
            n = min(a.shape[0], 512)
            aa = a[:n] / (np.linalg.norm(a[:n], axis=1, keepdims=True) + 1e-9)
            bb = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
            sim = aa @ bb.T  # (n, rows)
            best = sim.max(axis=1)
            perm = sim.argmax(axis=1)
            frac = float((best > 0.999).mean())
            note = (f" permuted-match frac {frac:.3f} "
                    f"(max sim {best.max():.4f})")
            if frac > 0.9:
                note += f" -> ROW PERMUTATION! first rows -> {perm[:8].tolist()}"
            flags.append(name)
        print(f"{name}: rel {d:.4f}{note}")
    print("FLAGGED:", flags if flags else "none")


if __name__ == "__main__":
    main()
