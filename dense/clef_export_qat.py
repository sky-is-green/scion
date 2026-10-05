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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--masters", required=True)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--sign-seed", type=int, default=1337)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    masters = torch.load(args.masters, map_location="cpu")
    print(f"masters: {len(masters)} tensors", flush=True)

    reader = GGUFReader(src)
    arch = reader.fields[Keys.General.ARCHITECTURE].contents()
    names = [t.name for t in reader.tensors]
    width_of = {t.name: int(t.data.shape[-1]) for t in reader.tensors}
    candidates = [n for n in names
                  if ROTATED.match(n) or UNROTATED_TERNARY.match(n)]
    targets = [n for n in candidates if gguf_to_hf(n) in masters]
    kept = [n for n in candidates if n not in set(targets)]
    rotated_names = [n for n in targets if ROTATED.match(n)]
    print(f"ternary targets {len(targets)} (rotated {len(rotated_names)}), "
          f"kept F16 {len(kept)}, copied {len(names) - len(targets)}", flush=True)

    writer = GGUFWriter(str(dst), arch=arch)
    copy_metadata(reader, writer)
    sign_widths, sign_values = sign_table(rotated_names, width_of, args.sign_seed)
    add_prism_metadata(writer, rotated_names, sign_widths, sign_values)

    t0 = time.time()
    packed: dict[str, np.ndarray] = {}
    for i, name in enumerate(targets):
        w = masters[gguf_to_hf(name)].float().cpu().numpy()
        if w.shape[-1] != width_of[name]:
            raise SystemExit(f"{name}: master width {w.shape[-1]} != {width_of[name]}")
        packed[name] = pack_ternary(w, "lloyd")
        writer.add_tensor_info(name, packed[name].shape, packed[name].dtype,
                               packed[name].nbytes, raw_dtype=PQ2_0)
        if (i + 1) % 50 == 0:
            print(f"  packed {i+1}/{len(targets)} ({time.time()-t0:.0f}s)",
                  flush=True)
    for name in names:
        if name in set(targets):
            continue
        t = next(t for t in reader.tensors if t.name == name)
        writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes)

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
