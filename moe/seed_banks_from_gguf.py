"""Seed the frozen HF expert banks from the deployed GGUF ternary codes.

Why this exists
---------------
The 35B release body was packed by the C++ ``GGUF_PQ2_0_LLOYD`` quantizer while
the correction pipeline quantises the BF16 master with the Python Lloyd g128
rule.  The two rules normally agree, but fp16 scale rounding flips a small
fraction of codes (the v2 export audit measures it per tensor).  Training on
the raw BF16 master therefore sees codes that are not exactly the deployed
ones.

The fix is to make the deployed GGUF the source of truth for the frozen body:
copy each expert bank's dequantised deployed values into the HF model *before*
the training-time quantiser runs.  ``ternary_lloyd`` is idempotent on those
values (re-quantising them reproduces the deployed codes exactly), so the
training and deployment codes then agree by construction.  The body is not
modified; only the frozen weights handed to correction training are aligned.

Usage (inside the training process, after ``from_pretrained`` and before
``quantize_bank_inplace``):

    from seed_banks_from_gguf import seed_model_banks
    seed_model_banks(model, "/path/Scion-35B-A3B-PQ2_0-corr.gguf")

CLI shape check:
    python moe/seed_banks_from_gguf.py <model.gguf> --layers 0,39
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

TYPE_Q2_0_G128 = 142
TYPE_Q1_0_G128 = 43


def _rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def _string(f):
    return f.read(_rd(f, "Q")).decode("utf-8", "replace")


def _value(f, t):
    if t == 0:
        return _rd(f, "B")
    if t == 1:
        return _rd(f, "b")
    if t == 2:
        return _rd(f, "H")
    if t == 3:
        return _rd(f, "h")
    if t == 4:
        return _rd(f, "I")
    if t == 5:
        return _rd(f, "i")
    if t == 6:
        return _rd(f, "f")
    if t == 7:
        return bool(_rd(f, "B"))
    if t == 8:
        return _string(f)
    if t == 9:
        et = _rd(f, "I")
        return [_value(f, et) for _ in range(_rd(f, "Q"))]
    if t == 10:
        return _rd(f, "Q")
    if t == 11:
        return _rd(f, "q")
    if t == 12:
        return _rd(f, "d")
    raise ValueError(f"GGUF KV type {t}")


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", path
        _rd(f, "I")
        n_tensors, n_kv = _rd(f, "Q"), _rd(f, "Q")
        kv = {}
        for _ in range(n_kv):
            k = _string(f)
            kv[k] = _value(f, _rd(f, "I"))
        tensors = []
        for _ in range(n_tensors):
            name = _string(f)
            nd = _rd(f, "I")
            dims = [_rd(f, "Q") for _ in range(nd)]
            ttype = _rd(f, "I")
            offset = _rd(f, "Q")
            n = 1
            for d in dims:
                n *= d
            tensors.append({"name": name, "dims": dims, "type": ttype,
                            "offset": offset, "n": n})
        align = int(kv.get("general.alignment", 32))
        data_start = (f.tell() + align - 1) // align * align
    return {"kv": kv, "tensors": tensors, "data_start": data_start}


def _decode_ternary_blocks(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Q2_0_g128 / Q1_0_g128: fp16 scale + 128 2-bit codes per 34-byte block."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scales = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    qb = blocks[:, 2:]
    codes = np.empty((blocks.shape[0], 128), dtype=np.int8)
    for j in range(4):
        codes[:, j::4] = ((qb >> (2 * j)) & 0x03).astype(np.int8) - 1
    return codes, scales


def read_bank_blocks(path: Path, layer: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """The deployed container verbatim: (codes, scales) per 128-weight group.

    Returns ``codes`` uint8 ``[E, out, ng, 32]`` (four 2-bit codes per byte,
    ``q + 1`` little-endian within the byte -- the same layout
    ``qwen35_moe_proxy.pack_ternary_codes`` writes) and ``scales`` fp16
    ``[E, out, ng]``, where ``ng = in / 128``.  Groups run along the last
    (contiguous) dim, so a Q2_0_g128 block sequence maps 1:1 onto the HF
    ``[E, out, in]`` bank layout with the group axis fastest.  Decoding these
    buffers is therefore *bit-identical* to the deployed dequantisation: this
    is the source for the compact-bank training container.
    """
    h = read_header(path)
    name = f"blk.{layer}.ffn_{kind}_exps.weight"
    t = next((x for x in h["tensors"] if x["name"] == name), None)
    if t is None:
        raise KeyError(f"{name} not in {path}")
    if t["type"] not in (TYPE_Q2_0_G128, TYPE_Q1_0_G128):
        raise ValueError(f"{name} has type {t['type']}, expected ternary g128")
    if t["n"] % 128:
        raise ValueError(f"{name}: {t['n']} weights is not a multiple of 128")
    ne0, ne1, ne2 = t["dims"]          # [in, out, E]
    if ne0 % 128:
        raise ValueError(f"{name}: last dim {ne0} is not a multiple of 128")
    ng = ne0 // 128
    nb = t["n"] // 128 * 34
    with open(path, "rb") as f:
        f.seek(h["data_start"] + t["offset"])
        raw = f.read(nb)
    if len(raw) != nb:
        raise SystemExit(f"{name}: short read ({len(raw)} of {nb} bytes)")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scales = blocks[:, :2].copy().view("<f2").reshape(ne2, ne1, ng)
    qb = blocks[:, 2:].reshape(ne2, ne1, ng, 32)
    return qb.copy(), scales.copy()


def read_bank(path: Path, layer: int, kind: str) -> np.ndarray:
    """Dequantised expert bank as [experts, out, in] float32."""
    h = read_header(path)
    name = f"blk.{layer}.ffn_{kind}_exps.weight"
    t = next((x for x in h["tensors"] if x["name"] == name), None)
    if t is None:
        raise KeyError(f"{name} not in {path}")
    if t["type"] not in (TYPE_Q2_0_G128, TYPE_Q1_0_G128):
        raise ValueError(f"{name} has type {t['type']}, expected ternary g128")
    nb = t["n"] // 128 * 34
    with open(path, "rb") as f:
        f.seek(h["data_start"] + t["offset"])
        raw = f.read(nb)
    codes, scales = _decode_ternary_blocks(raw)
    ne0, ne1, ne2 = t["dims"]
    vals = codes.astype(np.float32) * scales[:, None]
    return vals.reshape(ne2, ne1, ne0)


def seed_model_banks(model, gguf: str, layers: list[int] | None = None,
                     verbose: bool = True) -> int:
    """Copy deployed ternary banks into a loaded ``qwen3_5_moe`` model.

    Returns the number of layers seeded.  The model's expert parameters keep
    their device and dtype; the ternary values are exact before the cast.
    """
    import torch
    from qwen35_moe_proxy import text_layers

    path = Path(gguf)
    h = read_header(path)
    if h["kv"].get("general.architecture") != "qwen35moe":
        raise SystemExit(f"{gguf}: not a qwen35moe model")
    n_layer = int(h["kv"].get("qwen35moe.block_count", 0))
    want = set(layers if layers is not None else range(n_layer))
    done = 0
    for il, layer in enumerate(text_layers(model)):
        if il not in want:
            continue
        experts = layer.mlp.experts
        gate = read_bank(path, il, "gate")
        up = read_bank(path, il, "up")
        down = read_bank(path, il, "down")
        gu = torch.from_numpy(np.concatenate([gate, up], axis=1))
        dn = torch.from_numpy(down)
        with torch.no_grad():
            p = experts.gate_up_proj
            p.copy_(gu.to(device=p.device, dtype=p.dtype))
            p = experts.down_proj
            p.copy_(dn.to(device=p.device, dtype=p.dtype))
        done += 1
        if verbose:
            print(f"seeded layer {il}: gate{tuple(gate.shape)} "
                  f"up{tuple(up.shape)} down{tuple(down.shape)}", flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--layers", default="0",
                    help="comma-separated layer indices for the shape check")
    args = ap.parse_args()
    for il in (int(x) for x in args.layers.split(",") if x):
        for kind in ("gate", "up", "down"):
            a = read_bank(Path(args.gguf), il, kind)
            codes, scales = read_bank_blocks(Path(args.gguf), il, kind)
            print(f"layer {il} {kind}: {a.shape} {a.dtype} "
                  f"min={a.min():.4g} max={a.max():.4g}; blocks "
                  f"codes {codes.shape} {codes.dtype} scales {scales.shape} "
                  f"{scales.dtype}")


if __name__ == "__main__":
    main()
