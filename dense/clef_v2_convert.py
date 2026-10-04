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
from bonsai_forensics import gptq, rotation as bf_rotation  # noqa: E402
from bonsai_forensics.run_quant import rotate_hessian  # noqa: E402

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


def rotations_for(width: int, seed: int = -1, block: int = BLOCK) -> list[np.ndarray]:
    """Per-block sign vectors; ``seed < 0`` gives identity signs (plain WHT)."""
    if width % block:
        raise ValueError(f"width {width} is not a multiple of block {block}")
    if seed < 0:
        return [np.ones(block, dtype=np.float64) for _ in range(width // block)]
    return bf_rotation.rotations_for(width, int(seed), block=block)


def fold(w: np.ndarray, seed: int = -1, block: int = BLOCK) -> np.ndarray:
    """``W' = W Rᵀ`` for the runtime's ``x' = R x`` transform."""
    return bf_rotation.absorb_input(w, rotations_for(w.shape[-1], seed, block))


def pack_ternary(w: np.ndarray, rule: str = "lloyd", group: int = 128) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    q = ternary_lloyd(t, group) if rule == "lloyd" else ternary_absmean(t, group)
    return pack_q1_0_g128(q)


def load_hessian(path: Path) -> np.ndarray:
    """Load an ``.npy`` Hessian, tolerating raw fp32 dumps from early runs."""
    try:
        return np.load(path).astype(np.float32)
    except ValueError:
        raw = np.fromfile(path, dtype=np.float32)
        d = int(round(raw.size ** 0.5))
        if d * d != raw.size:
            raise
        return raw.reshape(d, d)


def quantize_target(name: str, w: np.ndarray, rots: list[np.ndarray] | None,
                    hessian_dir: Path | None, args) -> tuple[np.ndarray, bool]:
    """GPTQ when a Hessian exists for this tensor, else deployed Lloyd RTN.

    ``w`` is already folded when ``rots`` is not None; the hessian is rotated
    with the same basis (``R H Rᵀ``) before GPTQ optimises the folded weight.
    """
    hp = (hessian_dir / f"{name}.hessian.npy") if hessian_dir else None
    if hp is not None and hp.is_file():
        h = load_hessian(hp)
        if rots is not None:
            h = np.ascontiguousarray(rotate_hessian(h, rots), dtype=np.float32)
        res = gptq.gptq_quantize(
            torch.from_numpy(np.ascontiguousarray(w)),
            torch.from_numpy(h),
            group_size=128,
            damp=args.gptq_damp,
            act_order=args.gptq_act_order,
            block_size=args.gptq_block,
            refine_iters=args.gptq_refine,
        )
        scales = res.scales.float().repeat_interleave(res.group_size, dim=-1)
        values = (res.codes.float() * scales[:, : w.shape[1]]).numpy()
        return pack_q1_0_g128(torch.from_numpy(values)), True
    return pack_ternary(w, args.rule), False


def runtime_transform(x: np.ndarray, seed: int = -1, block: int = BLOCK) -> np.ndarray:
    """Emulate the fork on an activation ``x`` shaped ``[input, tokens]``.

    ``llama_mul_mat_hadamard`` reshapes the contiguous input axis into blocks
    and applies ``H (S ⊙ ·) / √g`` per block; ``apply_rotation`` works on the
    last axis, so transpose in and out.
    """
    rots = rotations_for(x.shape[0], seed, block)
    return bf_rotation.apply_rotation(
        np.ascontiguousarray(x.T), rots, transpose=False).T


def add_prism_metadata(writer: GGUFWriter, rotated_names: list[str],
                       sign_widths: list[int] | None = None,
                       sign_values: list[int] | None = None) -> None:
    writer.add_uint32("prism.hadamard.version", 1)
    writer.add_uint32("prism.hadamard.block_size", BLOCK)
    writer.add_string("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
    writer.add_string("prism.hadamard.axis", "input-last-dimension")
    if sign_widths:
        writer.add_string("prism.hadamard.sign_mode", "explicit")
        # the C++ loader reads these into std::vector<int32_t>; force the type
        writer.add_key_value("prism.hadamard.sign_widths", sign_widths,
                             GGUFValueType.ARRAY, sub_type=GGUFValueType.INT32)
        writer.add_key_value("prism.hadamard.sign_values", sign_values,
                             GGUFValueType.ARRAY, sub_type=GGUFValueType.INT32)
    else:
        writer.add_string("prism.hadamard.sign_mode", "identity")
    writer.add_array("prism.hadamard.weight_names", rotated_names)


def sign_table(rotated_names: list[str], width_of: dict[str, int],
               seed: int) -> tuple[list[int], list[int]]:
    """Serialise the per-width PRF sign vectors for the explicit metadata."""
    widths: list[int] = []
    values: list[int] = []
    for width in sorted({width_of[n] for n in rotated_names}):
        rots = rotations_for(width, seed)
        flat = np.concatenate([np.asarray(b, dtype=np.int64).reshape(-1) for b in rots])
        if flat.size != width or not np.all(np.isin(flat, (-1, 1))):
            raise SystemExit(f"bad sign table for width {width}")
        widths.append(width)
        values.extend(int(v) for v in flat)
    return widths, values


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
    for seed in (-1, 1337):
        for width in (4096, 8192, 12288):
            w = rng.standard_normal((37, width)).astype(np.float32)
            x = rng.standard_normal((width, 5)).astype(np.float32)
            y = w @ x
            y_folded = fold(w, seed) @ runtime_transform(x, seed)
            rel = np.abs(y_folded - y).max() / max(np.abs(y).max(), 1e-9)
            assert rel < 1e-4, (seed, width, rel)
        print(f"self-test seed {seed}: widths 4096/8192/12288 OK")

    w = rng.standard_normal((256, 512)).astype(np.float32)
    x = rng.standard_normal((512, 512)).astype(np.float32)
    h = (x.T @ x / x.shape[0]).astype(np.float32)
    res = gptq.gptq_quantize(torch.from_numpy(w), torch.from_numpy(h),
                             group_size=128, block_size=128, refine_iters=0)
    values = (res.codes.float()
              * res.scales.float().repeat_interleave(res.group_size, dim=-1)[:, :512]).numpy()
    rel = np.linalg.norm(values - w) / np.linalg.norm(w)
    print(f"self-test GPTQ smoke: rel err {rel:.3f} ({res.codes.shape} codes) OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", default="")
    ap.add_argument("--out", dest="dst", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--rule", choices=("lloyd", "absmean"), default="lloyd")
    ap.add_argument("--no-rotation", action="store_true",
                    help="control: same tensor selection, unrotated ternary, no prism metadata")
    ap.add_argument("--keep-f16", action="append", default=[],
                    help="regex of a target tensor to leave copied/F16 (repeatable)")
    ap.add_argument("--sign-seed", type=int, default=-1,
                    help="PRF seed for explicit sign vectors (-1 = identity signs)")
    ap.add_argument("--hessian-dir", default="",
                    help="directory of <tensor>.hessian.npy for GPTQ (plain RTN without)")
    ap.add_argument("--gptq-damp", type=float, default=0.01)
    ap.add_argument("--gptq-act-order", action="store_true")
    ap.add_argument("--gptq-block", type=int, default=128)
    ap.add_argument("--gptq-refine", type=int, default=0,
                    help="group-scale LS refinement iters (0 = absmean, the PQ2_0 rule)")
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
    width_of = {t.name: int(t.data.shape[-1]) for t in reader.tensors}
    keep = [re.compile(p) for p in args.keep_f16]
    targets = [n for n in names
               if (ROTATED.match(n) or UNROTATED_TERNARY.match(n))
               and not any(k.search(n) for k in keep)]
    rotated_names = [] if args.no_rotation else [n for n in targets if ROTATED.match(n)]
    unrot_names = [n for n in targets if UNROTATED_TERNARY.match(n)]
    if not targets:
        raise SystemExit("no ternary targets matched")
    kept = [n for n in names if (ROTATED.match(n) or UNROTATED_TERNARY.match(n))
            and n not in set(targets)]
    print(f"{src.name}: {len(names)} tensors, arch {arch}; "
          f"rotated {len(rotated_names)}, unrotated-ternary {len(unrot_names)}, "
          f"kept-f16 targets {len(kept)}, copied {len(names) - len(targets)}", flush=True)

    # pass 1: fold + quantize targets to temp files, register all tensor infos
    sign_widths: list[int] = []
    sign_values: list[int] = []
    if rotated_names and args.sign_seed >= 0:
        sign_widths, sign_values = sign_table(rotated_names, width_of, args.sign_seed)
        print(f"explicit signs: seed {args.sign_seed}, widths {sign_widths}", flush=True)

    writer = GGUFWriter(str(dst), arch=arch)
    copy_metadata(reader, writer)
    if not args.no_rotation:
        add_prism_metadata(writer, rotated_names, sign_widths, sign_values)

    target_set = set(targets)
    rotated_set = set(rotated_names)
    hessian_dir = Path(args.hessian_dir) if args.hessian_dir else None
    packed_paths: dict[str, Path] = {}
    n_gptq = 0
    t0 = time.time()
    for i, t in enumerate(reader.tensors):
        name = t.name
        if name in target_set:
            w = np.asarray(t.data, dtype=np.float32)
            rots = rotations_for(w.shape[-1], args.sign_seed) if name in rotated_set else None
            if rots is not None:
                w = bf_rotation.absorb_input(w, rots)
            packed, used_gptq = quantize_target(name, w, rots, hessian_dir, args)
            n_gptq += int(used_gptq)
            if packed.shape[-1] % 34 or packed.shape[0] != w.shape[0]:
                raise SystemExit(f"{name}: unexpected packed shape {packed.shape}")
            path = work / (name.replace(".", "_") + ".pq2_0.npy")
            np.save(path, packed)
            packed_paths[name] = path
            # add_tensor_info converts the byte shape using raw_dtype
            writer.add_tensor_info(name, packed.shape, packed.dtype, packed.nbytes,
                                   raw_dtype=PQ2_0)
        else:
            writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes)
        if (i + 1) % 50 == 0 or i + 1 == len(reader.tensors):
            print(f"  prepared {i+1}/{len(reader.tensors)} "
                  f"({time.time()-t0:.0f}s, {n_gptq} GPTQ)", flush=True)
    print(f"quantized: {n_gptq}/{len(target_set)} with GPTQ, "
          f"{len(target_set) - n_gptq} with {args.rule} RTN", flush=True)

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
    types = {t.tensor_type for t in check.tensors if t.name in set(packed_paths)}
    if args.no_rotation:
        if "prism.hadamard.version" in fields:
            raise SystemExit("verify: control output has prism metadata")
        print(f"verify: {len(check.tensors)} tensors, no rotation metadata, "
              f"{len(packed_paths)} ternary tensors, types {types}", flush=True)
    else:
        wl = fields["prism.hadamard.weight_names"].contents()
        if list(wl) != rotated_names:
            raise SystemExit("verify: prism.hadamard.weight_names mismatch")
        mode = fields["prism.hadamard.sign_mode"].contents()
        if sign_widths:
            got_w = [int(x) for x in fields["prism.hadamard.sign_widths"].contents()]
            got_v = [int(x) for x in fields["prism.hadamard.sign_values"].contents()]
            if mode != "explicit" or got_w != sign_widths or got_v != sign_values:
                raise SystemExit("verify: explicit sign metadata mismatch")
        elif mode != "identity":
            raise SystemExit(f"verify: expected identity signs, got {mode}")
        print(f"verify: {len(check.tensors)} tensors, prism version "
              f"{fields['prism.hadamard.version'].contents()}, "
              f"block {fields['prism.hadamard.block_size'].contents()}, "
              f"sign_mode {mode}, "
              f"{len(wl)} rotated weights, ternary types {types}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
