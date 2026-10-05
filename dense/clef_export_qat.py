"""Export trained QAT masters to the V2 rotated-basis GGUF (deployed container).

`dense/qat_9b.py` saves folded ternary masters keyed by HF module name.  This
script re-applies the deployed Lloyd g128 rule, packs PQ2_0, copies every
non-target tensor byte-for-byte from the f16 GGUF, and writes the
`prism.hadamard.*` metadata with the same sign table used in training
(seed 1337).  Targets missing from the masters (mixed-precision runs) are
copied F16 like the converter's `--keep-f16`.

Usage:
    python dense/clef_export_qat.py \
        --masters qat-run/masters-step800.pt \
        --in .../clef-flash-f16.gguf \
        --out .../v2/clef-flash-v2-qat.gguf
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from clef_v2_convert import (  # noqa: E402
    BLOCK, PQ2_0, ROTATED, UNROTATED_TERNARY, add_prism_metadata, copy_metadata,
    pack_ternary, sign_table)
from clef_dense_load import HK, HV, NK, NV, reorder_v_heads  # noqa: E402
from gguf import GGUFReader, GGUFWriter, Keys  # noqa: E402

_GGUF_TO_HF = {
    "attn_qkv": "linear_attn.in_proj_qkv",
    "attn_gate": "linear_attn.in_proj_z",
    "ssm_out": "linear_attn.out_proj",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def gguf_to_hf(name: str) -> str:
    m = re.match(r"^blk\.(\d+)\.([a-z_]+)\.weight$", name)
    if not m:
        raise ValueError(f"unexpected tensor name {name}")
    return f"layers.{m.group(1)}.{_GGUF_TO_HF[m.group(2)]}.weight"


def hf_to_gguf(kind: str, w: torch.Tensor) -> torch.Tensor:
    """Inverse of the reverse loader's GDN reorder (HF layout -> GGUF layout)."""
    rep = NV // NK
    if kind == "attn_qkv":
        qd = kd = HK * NK
        q, k, v = w[:qd], w[qd:qd + kd], w[qd + kd:]
        return torch.cat([q, k, reorder_v_heads(v, 0, NK, rep, HV)], dim=0)
    if kind == "attn_gate":
        return reorder_v_heads(w, 0, NK, rep, HV)
    if kind == "ssm_out":
        return reorder_v_heads(w, 1, NK, rep, HV)
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--masters", required=True)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--sign-seed", type=int, default=1337)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    masters = torch.load(args.masters, map_location="cpu")
    masters = {k[:-len(".weight")] if k.endswith(".weight") else k: v
               for k, v in masters.items()}
    print(f"masters: {len(masters)} tensors", flush=True)

    reader = GGUFReader(src)
    arch = reader.fields[Keys.General.ARCHITECTURE].contents()
    names = [t.name for t in reader.tensors]
    width_of = {t.name: int(t.data.shape[-1]) for t in reader.tensors}
    candidates = [n for n in names
                  if ROTATED.match(n) or UNROTATED_TERNARY.match(n)]
    targets = [n for n in candidates if gguf_to_hf(n)[:-len(".weight")] in masters]
    kept = [n for n in candidates if n not in set(targets)]
    rotated_names = [n for n in targets if ROTATED.match(n)]
    print(f"ternary targets {len(targets)} (rotated {len(rotated_names)}), "
          f"kept F16 {len(kept)}, copied {len(names) - len(targets)}", flush=True)

    writer = GGUFWriter(str(dst), arch=arch)
    copy_metadata(reader, writer)
    sign_widths, sign_values = sign_table(rotated_names, width_of, args.sign_seed)
    add_prism_metadata(writer, rotated_names, sign_widths, sign_values)

    # single pass in reader order: info order must match the data write order
    t0 = time.time()
    packed: dict[str, np.ndarray] = {}
    target_set = set(targets)
    for i, t in enumerate(reader.tensors):
        if t.name in target_set:
            kind = t.name.split(".")[2]
            wt = hf_to_gguf(kind, masters[gguf_to_hf(t.name)[:-len(".weight")]].float().cpu())
            w = wt.numpy()
            if w.shape[-1] != width_of[t.name]:
                raise SystemExit(
                    f"{t.name}: master width {w.shape[-1]} != {width_of[t.name]}")
            arr = pack_ternary(w, "lloyd")
            packed[t.name] = arr
            writer.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes,
                                   raw_dtype=PQ2_0)
            if len(packed) % 50 == 0:
                print(f"  packed {len(packed)}/{len(targets)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype,
                                   t.data.nbytes)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        if t.name in packed:
            writer.write_tensor_data(packed[t.name])
        else:
            writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"wrote {dst} ({dst.stat().st_size/2**30:.2f} GiB, "
          f"{time.time()-t0:.0f}s)", flush=True)

    check = GGUFReader(dst)
    got = {t.name for t in check.tensors}
    missing = set(names) - got
    if missing:
        raise SystemExit(f"verify: missing {len(missing)} tensors")
    types = {t.tensor_type for t in check.tensors if t.name in set(packed)}
    fields = {f.name: f for f in check.fields.values()}
    print(f"verify: {len(check.tensors)} tensors, block "
          f"{fields['prism.hadamard.block_size'].contents()}, "
          f"{len(rotated_names)} rotated, types {types}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
