"""Build a per-tensor hot/cold qwen35moe body from the bf16 base GGUF.

GGUF stores one quantization type per tensor, so the hot/cold split here is
**per expert tensor** (layer x gate/up/down), not per expert row.  The
allocation map is the same ``{tensor: bits}`` file the trainer's
``--alloc-file`` consumes: 4 = Q4_0_G128 (4.125 bpw), 2 = TQ1_0_G128
(1.75 bpw, the deployed ternary codes).

The exporter streams one tensor at a time; it never materialises the 65 GB
source.  Non-expert tensors are copied byte-for-byte.  Expert rows are
requantised from the bf16 master with the deployable rules:

  bits=4  q4_0_g128.encode_np      (symmetric amax/7, fp16 scale, halves)
  bits=2  lloyd ternary + ptq1_0   (same trits/scales as Q2_0_g128, packed)

Usage:
  PYTHONPATH=<fork>/gguf-py python moe/build_hotcold_body.py \
      --src .../Qwen3.8-35B-A3B-Distill-BF16.gguf \
      --out .../Scion-35B-A3B-v2-hotcold.gguf \
      --policy down4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "scion-llama.cpp" / "gguf-py"))
sys.path.insert(0, str(HERE.parent / "v2" / "scripts"))

import gguf  # noqa: E402
import q4_0_g128 as Q4  # noqa: E402
import ptq1_0  # noqa: E402
from qwen35_moe_proxy import pack_ternary_codes  # noqa: E402
from seed_banks_from_gguf import read_header  # noqa: E402


def tensor_bits(name: str, alloc: dict, policy: str) -> int | None:
    """Return 4/2 for an expert tensor, or None for a copied tensor."""
    if not name.endswith("_exps.weight"):
        return None
    if name in alloc:
        return int(alloc[name])
    m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
    if not m:
        return None
    layer, kind = int(m.group(1)), m.group(2)
    if policy == "all4":
        return 4
    if policy == "all2":
        return 2
    if policy == "down4":
        return 4 if kind == "down" else 2
    if policy == "gateup4":
        return 4 if kind in ("gate", "up") else 2
    if policy == "half":
        return 4 if layer < 20 else 2
    if policy == "third":
        return 4 if layer < 14 else 2
    raise ValueError(f"unknown policy {policy}")


def bf16_bytes_to_f32(raw: bytes, src_type: int) -> np.ndarray:
    if src_type == gguf.GGMLQuantizationType.BF16:
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32)
    if src_type == gguf.GGMLQuantizationType.F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    raise ValueError(f"expert source type {src_type} is not BF16/F16")


def quantize_tensor(w: np.ndarray, bits: int, group: int = 128) -> np.ndarray:
    """Quantise ``[E, out, in]`` float32 to block bytes ``[E, out, ng*b]``."""
    e, rows, cols = w.shape
    if cols % group:
        raise ValueError(f"row width {cols} is not a multiple of {group}")
    ng = cols // group
    if bits == 4:
        out = np.empty((e, rows, ng * 66), dtype=np.uint8)
        chunk = max(1, (64 << 20) // max(rows * cols * 4, 1))
        for s in range(0, e, chunk):
            out[s:s + chunk] = Q4.encode_np(w[s:s + chunk]).reshape(
                -1, rows, ng * 66)
        return out
    if bits == 2:
        from qwen35_moe_proxy import _encode_slice
        out = np.empty((e, rows, ng * 28), dtype=np.uint8)
        chunk = max(1, (64 << 20) // max(rows * cols * 4, 1))
        for s in range(0, e, chunk):
            codes, scales = _encode_slice(
                torch.from_numpy(w[s:s + chunk]), group, "lloyd")
            qs, qh, sc = ptq1_0.repack_bank_ptq1_0(codes, scales, group)
            blk = ptq1_0.pack_block_bytes(qs, qh, sc)
            out[s:s + chunk] = blk.reshape(-1, rows, ng * 28).numpy()
        return out
    raise ValueError(f"unsupported bits {bits}")


def copy_metadata(reader, writer) -> None:
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
    ap.add_argument("--src", required=True,
                    help="bf16 base GGUF; expert rows are read from here")
    ap.add_argument("--dense-src", default="",
                    help="optional qwen35moe body GGUF whose non-expert tensors "
                         "are copied verbatim (e.g. the v1 deployed body).  This "
                         "keeps the v1 router/dense quantization and avoids "
                         "shipping 5 GiB of BF16 non-expert weights.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alloc-file", default="")
    ap.add_argument("--policy", default="down4",
                    choices=["down4", "gateup4", "half", "third", "all4", "all2"])
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--only-tensor", default="",
                    help="write just this tensor (codec/layout check)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N source tensors (debug)")
    args = ap.parse_args()

    alloc = {}
    if args.alloc_file:
        alloc = json.loads(Path(args.alloc_file).read_text())
        print(f"alloc: {len(alloc)} entries", flush=True)

    src = Path(args.src)
    reader = gguf.GGUFReader(src)
    dense_reader = (gguf.GGUFReader(args.dense_src)
                    if args.dense_src else reader)
    src_by_name = {t.name: t for t in reader.tensors}
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(args.out, arch=arch)
    copy_metadata(reader, writer)

    infos = []
    for t in dense_reader.tensors:
        if args.only_tensor and t.name != args.only_tensor:
            continue
        # The dense source may be a deployed body with embedded corrections;
        # the new body starts without them (they are retrained).
        if ".lora_a" in t.name or ".lora_b" in t.name:
            continue
        bits = tensor_bits(t.name, alloc, args.policy)
        if bits is None:
            infos.append((t, None, t.data.shape, t.data.dtype, t.data.nbytes,
                          t.tensor_type, t))
        else:
            st = src_by_name[t.name]
            ne0, ne1, ne2 = [int(x) for x in st.shape]
            ng = ne0 // args.group
            b = 66 if bits == 4 else 28
            # numpy order is [E, out, row_bytes]; GGUF stores dims reversed
            shape = (ne2, ne1, ng * b)
            infos.append((st, bits, shape, np.dtype(np.uint8),
                          int(np.prod(shape)), 147 if bits == 4 else 143, st))
        if args.limit and len(infos) >= args.limit:
            break

    for _t, _bits, shape, dtype, nbytes, raw_type, _st in infos:
        writer.add_tensor_info(_t.name, shape, dtype, nbytes,
                               raw_dtype=gguf.GGMLQuantizationType(raw_type))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    total = 0
    for n, (t, bits, shape, dtype, nbytes, raw_type, st) in enumerate(infos):
        if bits is None:
            payload = t.data
        else:
            ne0, ne1, ne2 = [int(x) for x in st.shape]
            with open(src, "rb") as f:
                # GGUFReader's data_offset is absolute; adding reader.data_offset
                # again shifts every expert read by ~11 MB (the data section start).
                f.seek(st.data_offset)
                raw = f.read(ne0 * ne1 * ne2 * 2)
            if len(raw) != ne0 * ne1 * ne2 * 2:
                raise SystemExit(f"{t.name}: short read")
            w = bf16_bytes_to_f32(raw, st.tensor_type).reshape(ne2, ne1, ne0)
            blocks = quantize_tensor(w, bits, args.group)
            payload = blocks.reshape(-1)
            print(f"[{n+1}/{len(infos)}] {t.name}: {bits}-bit "
                  f"{payload.nbytes/1e6:.1f} MB", flush=True)
            del w, blocks
        writer.write_tensor_data(payload)
        total += nbytes
        if n % 10 == 0:
            print(f"  wrote {n+1}/{len(infos)} tensors, "
                  f"{total/2**30:.2f} GiB", flush=True)
    writer.close()
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/2**30:.2f} GiB)",
          flush=True)


if __name__ == "__main__":
    main()
