"""Validate a hot/cold body against the bf16 source.

Checks, for sample layers:
  - every expert block scale is finite (the first exporter bug produced Inf
    TQ1 scales by reading the source 11 MB past the data section);
  - the body's dequantised expert values equal a fresh encode of the source
    tensor (Q4_0_G128 for 147, Lloyd/TQ1 for 143).

Usage:
  PYTHONPATH=.../gguf-py python v2/hotcold/validate_body.py \
    --body v2/hotcold/Scion-35B-A3B-v2-down4.gguf \
    --src .../Qwen3.8-35B-A3B-Distill-BF16.gguf --layers 0,20,39
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "moe"))
sys.path.insert(0, str(HERE.parent.parent / "scion-llama.cpp" / "gguf-py"))

import q4_0_g128 as Q4  # noqa: E402
import ptq1_0  # noqa: E402
from qwen35_moe_proxy import _encode_slice  # noqa: E402
from seed_banks_from_gguf import read_bank, read_header, _decode_tq1_blocks  # noqa: E402


def source_tensor(src: Path, h: dict, name: str) -> np.ndarray:
    t = next(x for x in h["tensors"] if x["name"] == name)
    ne0, ne1, ne2 = t["dims"]
    with open(src, "rb") as f:
        f.seek(h["data_start"] + t["offset"])
        raw = f.read(t["n"] * 2)
    a = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return a.view(np.float32).reshape(ne2, ne1, ne0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--layers", default="0,20,39")
    ap.add_argument("--experts", type=int, default=2)
    args = ap.parse_args()

    body = Path(args.body)
    src = Path(args.src)
    hb = read_header(body)
    hs = read_header(src)
    layers = [int(x) for x in args.layers.split(",") if x]
    failures = 0
    for il in layers:
        for kind in ("gate", "up", "down"):
            name = f"blk.{il}.ffn_{kind}_exps.weight"
            t = next((x for x in hb["tensors"] if x["name"] == name), None)
            if t is None:
                print(f"missing {name}")
                failures += 1
                continue
            bpg = 66 if t["type"] == 147 else 28
            nb = t["n"] // 128 * bpg
            with open(body, "rb") as f:
                f.seek(hb["data_start"] + t["offset"])
                raw = f.read(nb)
            blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, bpg)
            if t["type"] == 147:
                sc = blocks[:, :2].copy().view("<f2").astype(np.float32)
            else:
                sc = blocks[:, 26:28].copy().view("<f2").astype(np.float32)
            bad = int((~np.isfinite(sc)).sum())
            if bad:
                print(f"{name}: {bad} non-finite scales")
                failures += 1
                continue

            got = read_bank(body, il, kind)
            w = source_tensor(src, hs, name)
            e = args.experts
            if t["type"] == 147:
                ref = Q4.decode_np(Q4.encode_np(w[:e])).reshape(e, *w.shape[1:])
            else:
                codes, scales = _encode_slice(torch.from_numpy(w[:e]), 128,
                                               "lloyd")
                qs, qh, s = ptq1_0.repack_bank_ptq1_0(codes, scales, 128)
                blk = ptq1_0.pack_block_bytes(qs, qh, s).numpy()
                trits, s2 = _decode_tq1_blocks(blk.tobytes())
                ne0, ne1, ne2 = t["dims"]
                ng = ne0 // 128
                ref = (trits.astype(np.float32).reshape(e, ne1, ng, 128)
                       * s2.reshape(e, ne1, ng, 1)).reshape(e, ne1, ne0)
            diff = float(np.abs(got[:e] - ref).max())
            print(f"{name}: {t['type']} scales_finite max_diff={diff:.3e}")
            if diff > 1e-5:
                failures += 1
    if failures:
        raise SystemExit(f"{failures} failures")
    print("body validation PASS")


if __name__ == "__main__":
    main()
