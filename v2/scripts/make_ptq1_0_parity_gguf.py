#!/usr/bin/env python3
"""Build a minimal GGUF with layer-0 expert banks transcoded Q2_0 -> PTQ1_0.

Used by the PLAN section 13.6 kernel-parity check: the three layer-0 expert
tensors are decoded from the shipped Q2_0_g128 blocks, re-packed base-3 in the
PTQ1_0 (type 143) layout, and written to a small GGUF that `ptq1_0_parity`
can load.  The bytes come from the deployed artifact, so the test exercises
the real weights, not synthetic data.

Usage:
  PYTHONPATH=<fork>/gguf-py python v2/scripts/make_ptq1_0_parity_gguf.py \
      --model <deployed.gguf> --layer 0 --out /tmp/ptq1_0-parity.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "moe"))
import export_audit as E  # noqa: E402
import ptq1_0  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default="")
    args = ap.parse_args()

    if args.gguf_py:
        sys.path.insert(0, args.gguf_py)
    import gguf
    from gguf import GGMLQuantizationType, GGUFWriter

    model = Path(args.model)
    h = E.read_header(model)
    writer = GGUFWriter(args.out, arch="qwen35moe")
    writer.add_name("Scion-35B-A3B PTQ1_0 parity (layer 0)")
    writer.add_block_count(40)
    writer.add_embedding_length(2048)
    writer.add_expert_count(256)
    writer.add_expert_feed_forward_length(512)
    writer.add_file_type(7)

    for kind in ("gate", "up", "down"):
        name = f"blk.{args.layer}.ffn_{kind}_exps.weight"
        t = next(x for x in h["tensors"] if x["name"] == name)
        codes, scales, _ = E.decode_q2_0_g128(E.read_tensor(model, h, t))
        ne0, ne1, ne2 = t["dims"]                 # [in, out, experts]
        ng_row = ne0 // 128
        assert codes.shape[0] == ne2 * ne1 * ng_row, codes.shape
        c = torch.from_numpy(codes).reshape(ne2, ne1, ng_row, 128)
        s = torch.from_numpy(scales).reshape(ne2, ne1, ng_row)
        qs, qh = ptq1_0.pack_trits_ptq1_0(c)
        raw = ptq1_0.pack_block_bytes(qs, qh, s)   # [E, out, ng, 28]
        raw = raw.reshape(ne2, ne1, ng_row * 28).numpy().astype(np.uint8)
        print(f"{name}: {t['dims']} -> {raw.shape} ({raw.nbytes/1e6:.1f} MB)")
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.PTQ1_0)
        del codes, scales, c, s, qs, qh, raw

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("wrote", args.out, Path(args.out).stat().st_size)


if __name__ == "__main__":
    main()
