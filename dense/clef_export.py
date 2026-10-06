"""Export trained dense Clef correction branches as a llama.cpp LoRA adapter.

The dense sibling of ``moe/export_branches_lora.py``.  Two of the three taps
map to ordinary LoRAs on the deployed tensors, exactly as in the MoE route:

    ``.linear_attn.out_proj.branch``  -> ``blk.N.ssm_out.weight``      LoRA
    ``.self_attn.o_proj.branch``      -> ``blk.N.attn_output.weight``  LoRA
    ``.mlp.branch``                   -> ``blk.N.ffn_out.weight``      virtual

``ffn_out`` is a dense-Qwen3.5 fork extension (there is no MoE output to attach
to): the fork applies ``build_lora_branch("blk.N.ffn_out.weight", ffn_in)`` in
``qwen35.cpp::build_layer_ffn``, and ``llama-adapter.cpp`` anchors the virtual
target to ``blk.N.ffn_down.weight`` (mirroring ``ffn_moe_out`` -> gate_inp).

The deployed factors are ternarised with the same Lloyd/g128 rule the checkpoint
was trained with (``CorrectionBranch._weights``), so the adapter reproduces the
eval forward.  ``--dtype q1_0_g128`` packs them in the fork's native layout
(2.125 bpw); the default.

Usage:
    PYTHONPATH=$GGUF_PY \\
    .venv-rocm/bin/python dense/clef_export.py \\
        --load .../branches-r512-g128-step78.pt \\
        --out .../clef-flash-corr-r512.lora.gguf \\
        --target both --eval-json .../eval-corr.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from clef_paths import GGUF_PY

try:
    from gguf import GGUFWriter
except ImportError:  # running without the fork's gguf-py on PYTHONPATH
    sys.path.insert(0, str(GGUF_PY))
    from gguf import GGUFWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from quant import ternary_absmean, ternary_lloyd  # noqa: E402

# checkpoint-key suffix -> (GGUF target tensor name, slot)
_BRANCH_SUFFIXES = (
    (".self_attn.o_proj.branch.down.weight", "attn_output.weight", "down"),
    (".self_attn.o_proj.branch.up.weight", "attn_output.weight", "up"),
    (".linear_attn.out_proj.branch.down.weight", "ssm_out.weight", "down"),
    (".linear_attn.out_proj.branch.up.weight", "ssm_out.weight", "up"),
    (".mlp.branch.down.weight", "ffn_out.weight", "down"),
    (".mlp.branch.up.weight", "ffn_out.weight", "up"),
)

_TARGETS = {
    "attn_out": {"attn_output.weight", "ssm_out.weight"},
    "mlp_out": {"ffn_out.weight"},
    "both": {"attn_output.weight", "ssm_out.weight", "ffn_out.weight"},
}


def layer_index(key: str) -> int:
    return int(re.search(r"layers\.(\d+)\.", key).group(1))


def collect_branch_pairs(sd: dict, wanted: set) -> dict:
    pairs: dict = {}
    for key, tensor in sd.items():
        for suffix, target, slot in _BRANCH_SUFFIXES:
            if key.endswith(suffix):
                break
        else:
            continue
        if target not in wanted:
            continue
        pairs.setdefault((layer_index(key), target), {})[slot] = tensor
    bad = {k: sorted(v) for k, v in pairs.items() if set(v) != {"down", "up"}}
    if bad:
        raise SystemExit(f"incomplete branch pairs: {bad}")
    return pairs


def deploy_weights(down: torch.Tensor, up: torch.Tensor, quant: str,
                   kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce ``CorrectionBranch._weights`` for the deployed factors."""
    if quant == "none":
        return down, up
    fn = ternary_lloyd if kind == "lloyd" else ternary_absmean
    if quant == "g128":
        return fn(down, 128), fn(up, 128)
    # rank-component scales, folded from the down factor into the up factor
    s = down.abs().mean(dim=1).clamp_min(1e-8)          # [rank]
    qd = torch.clamp(torch.round(down / s[:, None]), -1, 1)
    t = up.abs().mean(dim=0).clamp_min(1e-8)            # [rank]
    qu = torch.clamp(torch.round(up / t[None, :]), -1, 1)
    return qd, qu * (s * t)[None, :]


def pack_q1_0_g128(w: torch.Tensor, group: int = 128) -> np.ndarray:
    """Pack deployed ternary values into the fork's Q1_0_g128 byte layout.

    Per 128 weights: fp16 scale (2 bytes) then 32 bytes of 2-bit codes,
    code = q + 1 stored at bits ``2 * (j % 4)`` of byte ``j // 4``.  The scale
    is recovered as ``max|w|`` of the group, exact because the deployed values
    are ``q * fp16(scale)`` (see ``quant.ternary_lloyd``).
    """
    assert w.shape[-1] % group == 0, w.shape
    g = w.float().reshape(*w.shape[:-1], w.shape[-1] // group, group)
    scale = g.abs().amax(-1).half()                       # [..., n_groups]
    q = torch.clamp(torch.round(g / scale.float().unsqueeze(-1).clamp_min(1e-12)),
                    -1, 1).to(torch.uint8) + 1            # {0,1,2}
    q = q.reshape(*q.shape[:-1], group // 4, 4)
    codes = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)
    sb = scale.numpy().view(np.uint8).reshape(*scale.shape, 2)
    out = np.concatenate([sb, codes.numpy().astype(np.uint8)], axis=-1)
    return np.ascontiguousarray(out.reshape(*w.shape[:-1], -1))


def add_factor(w: GGUFWriter, name: str, tensor: torch.Tensor, dtype: str,
               raw_qtype) -> np.ndarray:
    if dtype == "q1_0_g128":
        packed = pack_q1_0_g128(tensor)
        w.add_tensor(name, packed, raw_dtype=raw_qtype)
        return packed
    np_dtype = np.float16 if dtype == "f16" else np.float32
    data = np.ascontiguousarray(tensor.float().numpy()).astype(np_dtype)
    w.add_tensor(name, data)
    return data


def verify_export(out: Path, dtype: str, written: dict) -> None:
    """Reopen the adapter; compare every tensor's bytes with what was written."""
    from gguf import GGUFReader
    from gguf.constants import GGML_QUANT_SIZES
    from gguf import GGMLQuantizationType

    r = GGUFReader(out)
    fields = {f.name: f for f in r.fields.values()}
    if fields["general.architecture"].contents() != "qwen35":
        raise SystemExit("verify: wrong arch")
    if fields["adapter.lora.alpha"].contents() != 0.0:
        raise SystemExit("verify: alpha is not 0.0")
    got = {t.name: t for t in r.tensors}
    missing = [n for n in written if n not in got]
    if missing:
        raise SystemExit(f"verify: missing tensors {missing[:4]}")
    mismatched = []
    for name, expected in written.items():
        t = got[name]
        raw = t.data.tobytes()
        if raw != expected.tobytes():
            mismatched.append(name)
    if mismatched:
        raise SystemExit(f"verify: {len(mismatched)} tensor(s) differ, e.g. {mismatched[:3]}")
    n_pairs = len([n for n in written if n.endswith(".lora_a")])
    print(f"verify: {n_pairs} factor pairs, {len(got)} tensors, arch=qwen35, "
          f"alpha=0.0, bytes identical (block {GGML_QUANT_SIZES[GGMLQuantizationType.Q1_0_g128]}"
          f")" if dtype == "q1_0_g128" else
          f"verify: {n_pairs} factor pairs, {len(got)} tensors, arch=qwen35, alpha=0.0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--load", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="qwen35")
    ap.add_argument("--target", choices=sorted(_TARGETS), default="both")
    ap.add_argument("--dtype", choices=["f16", "f32", "q1_0_g128"], default="q1_0_g128")
    ap.add_argument("--deploy-quant", choices=["lloyd", "absmean"], default="lloyd")
    ap.add_argument("--branch-quant", choices=["g128", "rank", "none"], default="g128")
    ap.add_argument("--base", default="clef-flash-PQ2_0.gguf",
                    help="provenance only: the body the branches were trained on")
    ap.add_argument("--recipe", default="", help="free-text provenance")
    ap.add_argument("--eval-json", default="", help="pilot eval JSON for provenance")
    args = ap.parse_args()

    wanted = _TARGETS[args.target]
    sd = torch.load(args.load, map_location="cpu")
    pairs = collect_branch_pairs(sd, wanted)
    if not pairs:
        raise SystemExit(f"no branch tensors for target '{args.target}' in {args.load}")
    ranks = {t["down"].shape[0] for t in pairs.values()}
    if len(ranks) != 1:
        raise SystemExit(f"mixed ranks in checkpoint: {ranks}")
    rank = ranks.pop()
    by_target: dict = {}
    for (i, target), t in pairs.items():
        by_target.setdefault(target, []).append(i)
    print(f"checkpoint {args.load}: {len(sd)} keys -> {len(pairs)} branches, rank {rank}, "
          f"quant {args.deploy_quant}/{args.branch_quant}/{args.dtype}")
    for target, layers in sorted(by_target.items()):
        print(f"  {target:20s} layers {len(layers):2d} ({min(layers)}..{max(layers)})")
    if args.branch_quant == "g128":
        ins = {t["down"].shape[-1] for t in pairs.values()}
        outs = {t["up"].shape[0] for t in pairs.values()}
        print(f"  dims: in {sorted(ins)}, out {sorted(outs)}")

    w = GGUFWriter(args.out, arch=args.arch)
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", 0.0)
    w.add_string("adapter.recipe", args.recipe)
    w.add_string("adapter.base", args.base)
    if args.eval_json:
        try:
            ev = json.load(open(args.eval_json))
            for split in ("train", "test"):
                s = ev.get(split)
                if isinstance(s, dict) and "correct" in s and "n" in s:
                    w.add_float32(f"adapter.eval.{split}_correct", float(s["correct"]))
                    w.add_float32(f"adapter.eval.{split}_n", float(s["n"]))
            w.add_string("adapter.eval.protocol", "torch f32 recurrent proxy (pre-packaging)")
        except Exception as e:  # provenance must never break the export
            print(f"warning: could not read eval json: {e}")

    raw_qtype = None
    if args.dtype == "q1_0_g128":
        from gguf import GGMLQuantizationType
        raw_qtype = GGMLQuantizationType.Q1_0_g128

    written: dict[str, np.ndarray] = {}
    for (i, target), t in sorted(pairs.items()):
        down, up = deploy_weights(t["down"], t["up"], args.branch_quant, args.deploy_quant)
        # gguf-py reverses tensor dims on write: pass [rank, in] / [out, rank]
        # so the file stores ne [in, rank] / [rank, out].
        written[f"blk.{i}.{target}.lora_a"] = add_factor(
            w, f"blk.{i}.{target}.lora_a", down, args.dtype, raw_qtype)
        written[f"blk.{i}.{target}.lora_b"] = add_factor(
            w, f"blk.{i}.{target}.lora_b", up, args.dtype, raw_qtype)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = Path(args.out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    verify_export(out, args.dtype, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
