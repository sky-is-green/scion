"""Export trained correction branches as a llama.cpp LoRA adapter GGUF.

``attn_out`` checkpoints become a LoRA on the attention output projection,
matching the TAARDIS adapter convention:

    blk.N.attn_output.weight.lora_a   (gguf dims [in, rank]  <- down.weight [rank, in])
    blk.N.attn_output.weight.lora_b   (gguf dims [rank, out] <- up.weight [out, rank])

``moe_out`` checkpoints become a LoRA-only branch with no base tensor
(``blk.N.ffn_moe_out.weight``); the fork applies it to the MoE block output
(``build_lora_branch`` in llama-graph.cpp, hooked in models/olmoe.cpp).

and the metadata:

    general.type = adapter
    general.architecture = <arch>
    adapter.type = lora            (--taardis writes taardis-lora)
    adapter.lora.alpha = 0.0       (scale 1.0, as in the reference adapters)

The exported factors are ternarised with the same deployed quantizer used in
training/eval (``--deploy-quant`` / ``--branch-quant``, mirroring
``CorrectionBranch._weights``), so the adapter reproduces the eval numbers.
``--dtype q1_0_g128`` packs them in the fork's native 2.125 bpw layout
(8.9 MB for rank 512 x 16 layers, bit-exact with the f16 export); it needs the
fork's ``gguf-py`` on ``PYTHONPATH`` for the custom tensor type.

The ``moe_out`` placement has no linear tensor to attach to and is refused:
that placement needs a runtime op rather than a LoRA.

Usage:
  PYTHONPATH=<fork>/gguf-py python export_branches_lora.py \
      --load $MOE_ARTIFACTS/olmoe/olmoe-corr-r512-g128-attnoutlloyd-step8000.pt \
      --arch olmoe --target attn_out --out branches-attnout.lora.gguf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe_proxy import ternary_absmean, ternary_lloyd  # noqa: E402
from olmoe_corrections import fold_gates  # noqa: E402


# checkpoint-key suffix -> (GGUF target tensor, slot).  ``read``/``write`` are
# the Phase C gate tensors: they fold into the down/up factors at export, so
# the exported LoRA container keeps its rank and the runtime is unchanged.
_BRANCH_SUFFIXES = (
    (".self_attn.o_proj.branch.down.weight", "attn_output.weight", "down"),
    (".self_attn.o_proj.branch.up.weight", "attn_output.weight", "up"),
    (".self_attn.o_proj.branch.read_gate", "attn_output.weight", "read"),
    (".self_attn.o_proj.branch.write_gate", "attn_output.weight", "write"),
    (".linear_attn.out_proj.branch.down.weight", "ssm_out.weight", "down"),
    (".linear_attn.out_proj.branch.up.weight", "ssm_out.weight", "up"),
    (".linear_attn.out_proj.branch.read_gate", "ssm_out.weight", "read"),
    (".linear_attn.out_proj.branch.write_gate", "ssm_out.weight", "write"),
    (".mlp.branch.down.weight", "ffn_moe_out.weight", "down"),
    (".mlp.branch.up.weight", "ffn_moe_out.weight", "up"),
    (".mlp.branch.read_gate", "ffn_moe_out.weight", "read"),
    (".mlp.branch.write_gate", "ffn_moe_out.weight", "write"),
)


def layer_index(key: str) -> int:
    return int(re.search(r"layers\.(\d+)\.", key).group(1))


def collect_branch_pairs(sd: dict, wanted: set) -> tuple[dict, dict]:
    """Split a branch state dict into (down/up factor pairs, gate tensors).

    Gate tensors are returned separately because they are not exported: they
    fold into the factors (``fold_gates``), which is what keeps the exported
    LoRA rank-r and the serving runtime gate-free.
    """
    pairs: dict = {}
    gates: dict = {}
    for key, tensor in sd.items():
        for suffix, target, slot in _BRANCH_SUFFIXES:
            if key.endswith(suffix):
                break
        else:
            continue
        if target not in wanted:
            continue
        k = (layer_index(key), target)
        if slot in ("down", "up"):
            pairs.setdefault(k, {})[slot] = tensor
        else:
            gates.setdefault(k, {})[slot] = tensor
    pairs = {k: v for k, v in pairs.items() if "down" in v and "up" in v}
    return pairs, gates


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
    code = q + 1 stored at bits ``2 * (j % 4)`` of byte ``j // 4``
    (ggml-common.h ``block_q1_0_g128``).  The scale is recovered as
    ``max|w|`` of the group, which is exact because the deployed values are
    ``q * fp16(scale)`` (see ``ternary_lloyd``).
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
               raw_qtype) -> None:
    """Add one lora factor; quantised dtypes are packed, dense ones cast."""
    if dtype == "q1_0_g128":
        w.add_tensor(name, pack_q1_0_g128(tensor), raw_dtype=raw_qtype)
    else:
        np_dtype = np.float16 if dtype == "f16" else np.float32
        w.add_tensor(name, np.ascontiguousarray(tensor.float().numpy()).astype(np_dtype))


def router_canon(key: str) -> str:
    """Canonical router key: from ``layers.`` on, without the branch nesting.

    Handles the HF prefix (``model.language_model.`` / ``model.``), the
    wrapped gate (``...mlp.mlp.gate.weight``) and plain gates alike, so the
    checkpoint and the base model always compare like for like.
    """
    i = key.find("layers.")
    k = key[i:] if i >= 0 else key
    return k.replace(".mlp.mlp.gate.weight", ".mlp.gate.weight")


def load_base_routers(base_model: str) -> dict:
    """Load ``model.layers.N.mlp.gate.weight`` from an HF safetensors dir or .pt file."""
    path = Path(base_model)
    out = {}
    if path.is_dir():
        from safetensors import safe_open
        for f in sorted(path.glob("*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    if k.endswith(".mlp.gate.weight"):
                        out[router_canon(k)] = sf.get_tensor(k).float()
    else:
        sd = torch.load(path, map_location="cpu")
        out = {router_canon(k): v.float() for k, v in sd.items() if k.endswith(".mlp.gate.weight")}
    if not out:
        raise SystemExit(f"no router weights found under {base_model}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="olmoe")
    ap.add_argument("--target", choices=["attn_out", "moe_out", "both"], default="attn_out")
    ap.add_argument("--taardis", action="store_true",
                    help="declare taardis-lora instead of plain lora")
    ap.add_argument("--dtype", choices=["f16", "f32", "q1_0_g128"], default="f16",
                    help="'q1_0_g128' packs the deployed ternary factors in the fork's "
                         "native format (~2.125 bpw; needs the fork's gguf-py on PYTHONPATH)")
    ap.add_argument("--deploy-quant", choices=["lloyd", "absmean"], default="lloyd",
                    help="scale rule for ternarising the factors (default: the deployed rule)")
    ap.add_argument("--branch-quant", choices=["g128", "rank", "none"], default="g128",
                    help="deployed branch format the checkpoint was trained with "
                         "('none' exports the raw fp32 masters)")
    ap.add_argument("--routers", action="store_true",
                    help="also export the trained router deltas as exact rank<=64 LoRA "
                         "pairs on ffn_gate_inp (needs --base-model)")
    ap.add_argument("--base-model", default="",
                    help="HF dir (safetensors) holding the frozen-body routers the "
                         "checkpoint was trained against")
    ap.add_argument("--recipe", default="",
                    help="free-text provenance written into adapter.recipe")
    ap.add_argument("--eval-json", default="",
                    help="optional eval JSON (branches-eval format) to embed as "
                         "adapter.eval.* provenance")
    args = ap.parse_args()

    if args.routers and not args.base_model:
        raise SystemExit("--routers needs --base-model")
    wanted = {"attn_out": {"attn_output.weight", "ssm_out.weight"},
              "moe_out": {"ffn_moe_out.weight"},
              "both": {"attn_output.weight", "ssm_out.weight",
                       "ffn_moe_out.weight"}}[args.target]

    sd = torch.load(args.load, map_location="cpu")
    sd = {k.replace(".doctor.", ".branch."): v for k, v in sd.items()}
    pairs, gates = collect_branch_pairs(sd, wanted)
    if not pairs:
        raise SystemExit(f"no branch tensors for target '{args.target}' found in {args.load}")
    rank = next(iter(pairs.values()))["down"].shape[0]
    print(f"exporting {len(pairs)} branch tensors, rank {rank}, "
          f"deploy-quant {args.deploy_quant}/{args.branch_quant}, "
          f"{len(gates)} gated branches")

    w = GGUFWriter(args.out, arch=args.arch)
    w.add_type("adapter")
    w.add_string("adapter.type", "taardis-lora" if args.taardis else "lora")
    w.add_float32("adapter.lora.alpha", 0.0)
    # provenance (zero-risk): what was trained, from what, and how it scored
    w.add_string("adapter.recipe", args.recipe)
    w.add_string("adapter.base", args.base_model or "unknown")
    if args.eval_json:
        try:
            ev = json.load(open(args.eval_json))
            tr = ev.get("trained_branches", ev)   # harness format or stage_eval format
            if "ppl" in tr:
                w.add_float32("adapter.eval.ppl", float(tr["ppl"]))
            if "router_agree" in tr:
                w.add_float32("adapter.eval.router_agree", float(tr["router_agree"]))
            w.add_string("adapter.eval.protocol", "8-window wikitext-2, c512")
        except Exception as e:  # provenance must never break the export
            print(f"warning: could not read eval json: {e}")

    raw_qtype = None
    if args.dtype == "q1_0_g128":
        from gguf import GGMLQuantizationType
        raw_qtype = GGMLQuantizationType.Q1_0_g128
    dense_dtype = np.float32 if args.dtype == "f32" else np.float16

    for (i, target), t in sorted(pairs.items()):
        down, up = t["down"], t["up"]
        g = gates.get((i, target), {})
        if "read" in g or "write" in g:
            # Phase C: fold the gates before quantization; (D_w B)(A D_r) is
            # still rank-r, so the container below is unchanged.
            down, up = fold_gates(down, up, g.get("read"), g.get("write"))
        down, up = deploy_weights(down, up, args.branch_quant, args.deploy_quant)
        # gguf-py reverses dims on write: passing [rank, in] / [out, rank]
        # stores ne [in, rank] / [rank, out], matching the reference adapters.
        add_factor(w, f"blk.{i}.{target}.lora_a", down, args.dtype, raw_qtype)
        add_factor(w, f"blk.{i}.{target}.lora_b", up, args.dtype, raw_qtype)

    if args.routers:
        base = load_base_routers(args.base_model)
        n_routers = 0
        r_router = 0
        for key, trained in sd.items():
            if not key.endswith(".mlp.gate.weight"):
                continue
            # canonicalize both sides: the wrapped gate nests the block
            # (``...mlp.mlp.gate.weight``) and the base may carry an HF prefix
            base_key = router_canon(key)
            if base_key not in base:
                continue
            delta = trained.float() - base[base_key]      # [out, in]
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            a = vh.numpy()                               # [rank, in]
            b = (u * s[None, :]).numpy()                 # [out, rank]
            i = layer_index(key)
            w.add_tensor(f"blk.{i}.ffn_gate_inp.weight.lora_a",
                         np.ascontiguousarray(a).astype(dense_dtype))
            w.add_tensor(f"blk.{i}.ffn_gate_inp.weight.lora_b",
                         np.ascontiguousarray(b).astype(dense_dtype))
            r_router = a.shape[0]
            n_routers += 1
        print(f"exported {n_routers} router deltas (rank {r_router}, dense)")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = Path(args.out)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
