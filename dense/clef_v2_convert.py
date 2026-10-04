"""Clef-Flash V2: rotated-basis PQ2_0 conversion from the f16 GGUF.

The deployed PQ2_0 body is unrotated Lloyd ternary; the forensics record (Bonsai
2 / TAARDIS) is that the block-Hadamard rotation is the quality lever.  This
script re-converts the *f16* Clef body into the rotated basis:

  * fold ``W' = W Rᵀ`` into every attention/MLP linear that the runtime routes
    through ``build_lora_mm`` (``attn_qkv``, ``attn_gate``, ``attn_q/k/v``,
    ``attn_output``, ``ffn_gate/up/down``) with ``R`` the normalized
    Sylvester-Walsh-Hadamard block transform (block 1024, identity signs);
  * ternarize with the deployed Lloyd rule (g128) and pack PQ2_0 (type 142);
  * leave ``ssm_out`` unrotated-ternary (the runtime's GDN V-head reorder is a
    separate contract; v1 does not touch it) and copy every other tensor
    byte-for-byte (norms, GDN controls, embeddings, output head);
  * write ``prism.hadamard.*`` metadata (identity sign mode) so the fork
    rotates activations to match, and refuse to emit a width that is not a
    multiple of the block.

The convention is pinned by ``--self-test``: ``absorb_input`` here must satisfy
``W' · (R x) == W x`` with ``R x`` computed by the same function the runtime
uses (``rotation.apply_rotation``).

Usage:
    .venv-rocm/bin/python dense/clef_v2_convert.py --self-test
    .venv-rocm/bin/python dense/clef_v2_convert.py \
        --in .../clef-flash-f16.gguf --out .../v2/clef-flash-v2-pq2_0-rot.gguf
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
WORKSPACE = Path("/home/penis/Desktop/work")
sys.path.insert(0, str(WORKSPACE / "bonsai2-ternary-forensics"))

from quant import ternary_absmean, ternary_lloyd  # noqa: E402
from clef_export import pack_q1_0_g128  # noqa: E402
from bonsai_forensics import rotation as bf_rotation  # noqa: E402

try:
    from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter, Keys
except ImportError:  # running without the fork's gguf-py on PYTHONPATH
    sys.path.insert(0, "/home/penis/llama.cpp/gguf-py")
    from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter, Keys

BLOCK = 1024
PQ2_0 = GGMLQuantizationType.PQ2_0

ROTATED = re.compile(
    r"^blk\.\d+\.(attn_qkv|attn_gate|attn_q|attn_k|attn_v|attn_output"
    r"|ffn_gate|ffn_up|ffn_down)\.weight$")
UNROTATED_TERNARY = re.compile(r"^blk\.\d+\.ssm_out\.weight$")


def rotations_for(width: int, block: int = BLOCK) -> list[np.ndarray]:
    if width % block:
        raise ValueError(f"width {width} is not a multiple of block {block}")
    return [np.ones(block, dtype=np.float64) for _ in range(width // block)]


def fold(w: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """``W' = W Rᵀ`` for the runtime's ``x' = R x`` transform."""
    return bf_rotation.absorb_input(w, rotations_for(w.shape[-1], block))


def pack_ternary(w: np.ndarray, rule: str = "lloyd", group: int = 128) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    q = ternary_lloyd(t, group) if rule == "lloyd" else ternary_absmean(t, group)
    return pack_q1_0_g128(q)


def runtime_transform(x: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Emulate the fork on an activation ``x`` shaped ``[input, tokens]``.

    ``llama_mul_mat_hadamard`` reshapes the contiguous input axis into blocks
    and applies ``H (S ⊙ ·) / √g`` per block; ``apply_rotation`` works on the
    last axis, so transpose in and out.
    """
    rots = rotations_for(x.shape[0], block)
    return bf_rotation.apply_rotation(
        np.ascontiguousarray(x.T), rots, transpose=False).T


def add_prism_metadata(writer: GGUFWriter, rotated_names: list[str]) -> None:
    writer.add_uint32("prism.hadamard.version", 1)
    writer.add_uint32("prism.hadamard.block_size", BLOCK)
    writer.add_string("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
    writer.add_string("prism.hadamard.axis", "input-last-dimension")
    writer.add_string("prism.hadamard.sign_mode", "identity")
    writer.add_array("prism.hadamard.weight_names", rotated_names)


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name.startswith("GGUF."):
            continue
        if field.name == Keys.General.ARCHITECTURE:
            continue  # the writer emits it from arch
        vtype = field.types[0]
        sub = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub)


def self_test() -> None:
    rng = np.random.default_rng(0)
    for width in (4096, 8192, 12288):
        w = rng.standard_normal((37, width)).astype(np.float32)
        x = rng.standard_normal((width, 5)).astype(np.float32)
        y = w @ x
        y_folded = fold(w) @ runtime_transform(x)
        rel = np.abs(y_folded - y).max() / max(np.abs(y).max(), 1e-9)
        assert rel < 1e-4, (width, rel)
        print(f"self-test width {width}: max rel err {rel:.2e} OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", default="")
    ap.add_argument("--out", dest="dst", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--rule", choices=("lloyd", "absmean"), default="lloyd")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.src or not args.dst:
        raise SystemExit("--in and --out are required (or use --self-test)")

    src, dst = Path(args.src), Path(args.dst)
    work = Path(args.work) if args.work else dst.parent / "v2-work"
    work.mkdir(parents=True, exist_ok=True)
    reader = GGUFReader(src)
    arch = reader.fields[Keys.General.ARCHITECTURE].contents()
    names = [t.name for t in reader.tensors]
    rotated_names = [n for n in names if ROTATED.match(n)]
    unrot_names = [n for n in names if UNROTATED_TERNARY.match(n)]
    if not rotated_names:
        raise SystemExit("no rotated targets matched")
    print(f"{src.name}: {len(names)} tensors, arch {arch}; "
          f"rotated {len(rotated_names)}, unrotated-ternary {len(unrot_names)}, "
          f"copied {len(names) - len(rotated_names) - len(unrot_names)}", flush=True)

    # pass 1: fold + quantize targets to temp files, register all tensor infos
    writer = GGUFWriter(str(dst), arch=arch)
    copy_metadata(reader, writer)
    add_prism_metadata(writer, rotated_names)

    packed_paths: dict[str, Path] = {}
    t0 = time.time()
    for i, t in enumerate(reader.tensors):
        name = t.name
        if ROTATED.match(name):
            w = np.asarray(t.data, dtype=np.float32)
            packed = pack_ternary(fold(w), args.rule)
            if packed.shape[-1] % 34 or packed.shape[0] != w.shape[0]:
                raise SystemExit(f"{name}: unexpected packed shape {packed.shape}")
            path = work / (name.replace(".", "_") + ".pq2_0.npy")
            np.save(path, packed)
            packed_paths[name] = path
            # add_tensor_info converts the byte shape using raw_dtype
            writer.add_tensor_info(name, packed.shape, packed.dtype, packed.nbytes,
                                   raw_dtype=PQ2_0)
        elif UNROTATED_TERNARY.match(name):
            packed = pack_ternary(np.asarray(t.data, dtype=np.float32), args.rule)
            path = work / (name.replace(".", "_") + ".pq2_0.npy")
            np.save(path, packed)
            packed_paths[name] = path
            writer.add_tensor_info(name, packed.shape, packed.dtype, packed.nbytes,
                                   raw_dtype=PQ2_0)
        else:
            writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes)
        if (i + 1) % 50 == 0 or i + 1 == len(reader.tensors):
            print(f"  prepared {i+1}/{len(reader.tensors)} ({time.time()-t0:.0f}s)",
                  flush=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        if t.name in packed_paths:
            writer.write_tensor_data(np.load(packed_paths[t.name]))
        else:
            writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"wrote {dst} ({dst.stat().st_size/2**30:.2f} GiB, {time.time()-t0:.0f}s)",
          flush=True)

    # verify: reopen, check the rotation contract and tensor inventory
    check = GGUFReader(dst)
    fields = {f.name: f for f in check.fields.values()}
    got = {t.name for t in check.tensors}
    missing = set(names) - got
    if missing:
        raise SystemExit(f"verify: missing {len(missing)} tensors, e.g. {sorted(missing)[:3]}")
    wl = fields["prism.hadamard.weight_names"].contents()
    if list(wl) != rotated_names:
        raise SystemExit("verify: prism.hadamard.weight_names mismatch")
    types = {t.tensor_type for t in check.tensors if t.name in set(packed_paths)}
    print(f"verify: {len(check.tensors)} tensors, prism version "
          f"{fields['prism.hadamard.version'].contents()}, "
          f"block {fields['prism.hadamard.block_size'].contents()}, "
          f"sign_mode {fields['prism.hadamard.sign_mode'].contents()}, "
          f"{len(wl)} rotated weights, ternary types {types}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
