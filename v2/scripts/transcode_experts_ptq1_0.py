#!/usr/bin/env python3
"""Transcode every expert bank in a qwen35moe GGUF from Q2_0_g128 to PTQ1_0.

Same trits, tighter packing: 34 -> 28 bytes per 128 weights.  Non-expert
tensors and all metadata are copied byte-for-byte.  This is the size-track
prototype from PLAN section 13; it is not the release path until the engine
logits match the Q2_0 file.

Usage:
  PYTHONPATH=<fork>/gguf-py python v2/scripts/transcode_experts_ptq1_0.py \
      --in <deployed.gguf> --out /tmp/scion-ptq1_0.gguf
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


def copy_metadata(reader, writer) -> None:
    import gguf
    from gguf import GGUFValueType, Keys
    for field in reader.fields.values():
        if field.name.startswith("GGUF."):
            continue
        if field.name == Keys.General.ARCHITECTURE:
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type,
                             sub_type=sub_type)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default="")
    args = ap.parse_args()
    if args.gguf_py:
        sys.path.insert(0, args.gguf_py)
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

    src = Path(args.src)
    reader = GGUFReader(src)
    arch = reader.fields["general.architecture"].contents()
    writer = GGUFWriter(args.out, arch=arch)
    copy_metadata(reader, writer)

    payloads = []
    for t in reader.tensors:
        name = t.name
        if name.endswith("_exps.weight"):
            ne2, ne1, row_bytes = [int(x) for x in t.data.shape]
            ng_row = row_bytes // 34
            ne0 = ng_row * 128
            dims = [ne0, ne1, ne2]
            raw = E.read_tensor(src, {"data_start": 0, "tensors": []},
                                {"name": name, "n": ne0 * ne1 * ne2,
                                 "type": 142, "dims": dims,
                                 "offset": t.data_offset})
            codes, scales, _ = E.decode_q2_0_g128(raw)
            c = torch.from_numpy(codes).reshape(ne2, ne1, ng_row, 128)
            s = torch.from_numpy(scales).reshape(ne2, ne1, ng_row)
            qs, qh = ptq1_0.pack_trits_ptq1_0(c)
            blk = ptq1_0.pack_block_bytes(qs, qh, s)     # [E,out,ng,28]
            blk = blk.reshape(ne2, ne1, ng_row * 28).numpy().astype(np.uint8)
            writer.add_tensor_info(name, blk.shape, blk.dtype,
                                   blk.nbytes,
                                   raw_dtype=GGMLQuantizationType.PTQ1_0)
            payloads.append(blk)
            print(f"transcoded {name}: {dims} -> {blk.nbytes/1e6:.1f} MB",
                  flush=True)
        else:
            writer.add_tensor_info(name, t.data.shape, t.data.dtype,
                                   t.data.nbytes, raw_dtype=t.tensor_type)
            payloads.append(t.data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for payload in payloads:
        writer.write_tensor_data(payload, tensor_endianess=reader.endianess)
    writer.close()
    print("wrote", args.out, Path(args.out).stat().st_size)


if __name__ == "__main__":
    main()
